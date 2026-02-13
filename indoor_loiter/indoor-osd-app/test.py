#!/usr/bin/env python3
"""
Minimal OpenCV + YOLO tester.

Usage:
  python minimal_tracker.py --source 0 --model yolo11n.pt
  python minimal_tracker.py --source path/to/video.mp4 --model yolo11n.pt

Keys:
  R    - reset to detection mode
  Esc/Q- quit
"""

from __future__ import annotations
import argparse
import math
import sys
import time
from typing import List, Tuple, Optional, Dict

try:
    import cv2
    import numpy as np
except Exception as e:
    print("This script requires OpenCV (cv2) and NumPy.")
    raise

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

# ---------------------------- helpers ----------------------------

YELLOW = (0, 255, 255)
PURPLE = (255, 0, 255)
WHITE  = (255, 255, 255)
BLACK  = (0, 0, 0)

def clip_bbox(x: float, y: float, w: float, h: float, W: int, H: int) -> Tuple[int, int, int, int]:
    x0 = max(0, int(math.floor(x)))
    y0 = max(0, int(math.floor(y)))
    x1 = min(W, int(math.ceil(x + w)))
    y1 = min(H, int(math.ceil(y + h)))
    return x0, y0, max(0, x1 - x0), max(0, y1 - y0)

def bbox_center(b: Tuple[int, int, int, int]) -> Tuple[float, float]:
    x, y, w, h = b
    return x + w / 2.0, y + h / 2.0

def point_in_bbox(px: float, py: float, b: Tuple[int, int, int, int]) -> bool:
    x, y, w, h = b
    return (px >= x) and (py >= y) and (px <= x + w) and (py <= y + h)

def det_center_inside(det_bbox: Tuple[int, int, int, int], outer: Tuple[int, int, int, int]) -> bool:
    cx, cy = bbox_center(det_bbox)
    return point_in_bbox(cx, cy, outer)

def draw_label(img, x, y, text, color, bg=True):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.45
    thickness = 1
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    if bg:
        cv2.rectangle(img, (x, max(0, y - th - baseline - 4)), (x + tw + 4, y), color, -1)
        cv2.putText(img, text, (x + 2, y - 3), font, scale, BLACK, thickness, cv2.LINE_AA)
    else:
        cv2.putText(img, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)

def gftt_in_bbox(gray: np.ndarray, bbox: Tuple[int, int, int, int], max_corners=64) -> np.ndarray:
    x, y, w, h = bbox
    mask = np.zeros_like(gray, dtype=np.uint8)
    cv2.rectangle(mask, (x, y), (x + w, y + h), 255, thickness=-1)
    pts = cv2.goodFeaturesToTrack(gray, maxCorners=max_corners, qualityLevel=0.01,
                                  minDistance=7, mask=mask, blockSize=7, useHarrisDetector=False)
    if pts is None:
        return np.empty((0, 1, 2), dtype=np.float32)
    return pts.astype(np.float32)

def warp_bbox_with_affine(bbox: Tuple[int, int, int, int], A: np.ndarray, W: int, H: int) -> Tuple[int, int, int, int]:
    x, y, w, h = bbox
    corners = np.array([
        [x,     y,     1.0],
        [x + w, y,     1.0],
        [x,     y + h, 1.0],
        [x + w, y + h, 1.0],
    ], dtype=np.float32)
    warped = (A @ corners.T).T
    min_x = float(np.min(warped[:, 0]))
    min_y = float(np.min(warped[:, 1]))
    max_x = float(np.max(warped[:, 0]))
    max_y = float(np.max(warped[:, 1]))
    return clip_bbox(min_x, min_y, max_x - min_x, max_y - min_y, W, H)

# ---------------------------- mouse state ----------------------------

class ClickSelector:
    def __init__(self):
        self.click_xy: Optional[Tuple[int, int]] = None

    def on_mouse(self, event, x, y, flags, userdata):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.click_xy = (int(x), int(y))

    def consume_click(self) -> Optional[Tuple[int, int]]:
        xy = self.click_xy
        self.click_xy = None
        return xy

# ---------------------------- core app ----------------------------

def run(args):
    if YOLO is None:
        print("Ultralytics not installed or import failed. `pip install ultralytics`")
        sys.exit(1)

    # Video source
    source = args.source if not args.source.isdigit() else int(args.source)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Failed to open source: {args.source}")
        sys.exit(1)

    # YOLO model
    model_path = args.model
    model = YOLO(model_path)

    win = "minimal-tracker"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_EXPANDED)
    selector = ClickSelector()
    cv2.setMouseCallback(win, selector.on_mouse)

    tracking = False
    track_bbox: Optional[Tuple[int, int, int, int]] = None
    prev_gray: Optional[np.ndarray] = None
    prev_pts: Optional[np.ndarray] = None

    last_dets: List[Dict] = []  # list of dicts with bbox + meta for click selection

    # main loop
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        H, W = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # ------------------------------------------------------------------
        # YOLO predict
        # ------------------------------------------------------------------
        results = model.predict(
            source=frame,  # BGR numpy ok
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            verbose=False,
            max_det=args.max_det
        )

        dets: List[Dict] = []
        if results:
            r0 = results[0]
            names = getattr(r0, "names", {}) or {}
            if hasattr(r0, "boxes") and r0.boxes is not None:
                try:
                    boxes = r0.boxes
                    xyxy = boxes.xyxy
                    conf = boxes.conf
                    cls  = boxes.cls
                    if hasattr(xyxy, "cpu"): xyxy = xyxy.cpu()
                    if hasattr(conf, "cpu"): conf = conf.cpu()
                    if hasattr(cls,  "cpu"): cls  = cls.cpu()
                    xyxy = xyxy.numpy()
                    conf = conf.numpy()
                    cls  = cls.numpy().astype(int)
                    for i in range(min(len(xyxy), args.max_det)):
                        x1, y1, x2, y2 = [float(v) for v in xyxy[i]]
                        w = max(0.0, x2 - x1)
                        h = max(0.0, y2 - y1)
                        dets.append({
                            "bbox": (int(round(x1)), int(round(y1)), int(round(w)), int(round(h))),
                            "conf": float(conf[i]),
                            "cls":  int(cls[i]),
                            "label": str(names.get(int(cls[i]), int(cls[i]))),
                        })
                except Exception:
                    pass

        last_dets = dets

        # ------------------------------------------------------------------
        # Click select (only when NOT tracking)
        # ------------------------------------------------------------------
        if not tracking:
            click = selector.consume_click()
            if click is not None:
                cx, cy = click
                # choose the highest-confidence box that contains the click
                cand = [(d["conf"], d) for d in dets if point_in_bbox(cx, cy, d["bbox"])]
                if cand:
                    cand.sort(key=lambda t: t[0], reverse=True)
                    chosen = cand[0][1]
                    track_bbox = chosen["bbox"]
                    prev_pts   = gftt_in_bbox(gray, track_bbox, max_corners=args.max_corners)
                    prev_gray  = gray.copy()
                    tracking   = True

        # ------------------------------------------------------------------
        # Draw & Track
        # ------------------------------------------------------------------
        vis = frame.copy()

        if not tracking:
            # Draw ALL detections (yellow)
            for d in dets:
                x, y, w, h = d["bbox"]
                cv2.rectangle(vis, (x, y), (x + w, y + h), YELLOW, 2)
                draw_label(vis, x, y, f"{d['label']} {d['conf']:.2f}", YELLOW, bg=True)
            draw_label(vis, 8, 20, "Click a yellow box to start tracking", WHITE, bg=False)
        else:
            # Track with LK + affine
            assert track_bbox is not None
            if prev_pts is None or len(prev_pts) < 6:
                prev_pts = gftt_in_bbox(gray, track_bbox, max_corners=args.max_corners)

            # calc flow
            if prev_pts is not None and len(prev_pts) >= 6 and prev_gray is not None:
                next_pts, status, err = cv2.calcOpticalFlowPyrLK(
                    prev_gray, gray, prev_pts, None,
                    winSize=(args.lk_win, args.lk_win),
                    maxLevel=args.lk_levels,
                    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
                )

                if next_pts is not None and status is not None:
                    m = status.reshape(-1) == 1
                    p0 = prev_pts[m].reshape(-1, 2)
                    p1 = next_pts[m].reshape(-1, 2)

                    if len(p0) >= 3 and len(p1) >= 3:
                        A, inliers = cv2.estimateAffinePartial2D(
                            p0, p1, method=cv2.RANSAC,
                            ransacReprojThreshold=args.reproj_px,
                            confidence=0.99
                        )
                        if A is not None:
                            track_bbox = warp_bbox_with_affine(track_bbox, A, W, H)

                    # update feature set for next round
                    prev_pts = p1[:, None, :].astype(np.float32)
                prev_gray = gray.copy()

            # draw tracked box (purple)
            x, y, w, h = track_bbox
            cv2.rectangle(vis, (x, y), (x + w, y + h), PURPLE, 2)
            draw_label(vis, x, y, "TRACK", PURPLE, bg=True)

            # For debug: draw YOLO detections whose centers fall inside the tracked bbox (yellow)
            for d in dets:
                if det_center_inside(d["bbox"], track_bbox):
                    dx, dy, dw, dh = d["bbox"]
                    cv2.rectangle(vis, (dx, dy), (dx + dw, dy + dh), YELLOW, 1)
                    draw_label(vis, dx, dy, f"{d['label']} {d['conf']:.2f}", YELLOW, bg=True)

            draw_label(vis, 8, 20, "Tracking. Press 'R' to reset to detection mode.", WHITE, bg=False)

        cv2.imshow(win, vis)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q'), ord('Q')):  # Esc/Q
            break
        elif key in (ord('r'), ord('R')):
            tracking   = False
            track_bbox = None
            prev_pts   = None
            prev_gray  = None

    cap.release()
    cv2.destroyAllWindows()

# ---------------------------- entry ----------------------------

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=str, default="0", help="Video source: camera index or path")
    ap.add_argument("--model",  type=str, default="yolo11n.pt", help="Ultralytics model path")
    ap.add_argument("--device", type=str, default="", help="Torch device spec, e.g. 'cpu' or 'cuda:0'")
    ap.add_argument("--imgsz",  type=int, default=640)
    ap.add_argument("--conf",   type=float, default=0.30)
    ap.add_argument("--iou",    type=float, default=0.45)
    ap.add_argument("--max_det", type=int, default=100)

    # LK / affine minimal knobs
    ap.add_argument("--lk_win", type=int, default=21)
    ap.add_argument("--lk_levels", type=int, default=3)
    ap.add_argument("--reproj_px", type=float, default=3.0)
    ap.add_argument("--max_corners", type=int, default=64)

    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run(args)
