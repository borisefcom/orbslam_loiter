#!/usr/bin/env python3
# recorder.py — Per-frame JPEG dumper with rich overlay (no video writer, no GUI).
#
# Expected config (under "recorder"):
#   enable: true
#   every_n_frames: 1
#   jpeg_quality: 90
#   root_img_directory: "./img"
#
# Viz dict keys used here (all optional):
#   algo, mode, hud_line, bg_status, obj_status, pid_src
#   detections: [{bbox or bbox_px, label, conf, ...}, ...]
#   bg_pts: Nx3 [x, y, flag]  (flag>=0.5 => "good" -> green; else orange)
#   obj_pts: Nx3 or Nx2 object feature points (orange; third channel ignored)
#   ghost: 4x2 quad (cyan)
#   search_quad: 4x2 quad (yellow)
#   search_envelope: (x,y,w,h)
#   raw_pt: (x,y), filt_pt: (x,y)
#   track_bbox: (x,y,w,h)
#   cam_fps: float, proc_fps: float (if provided by tracker)
#   frame_seq, ts_ms

import os, time, cv2
from typing import Dict, Any, List, Tuple, Optional

try:
    import numpy as np
except Exception:
    np = None

# Optional NVJPEG fast path (if available at runtime)
try:
    import nvjpeg  # type: ignore
    _nvjpeg_available = True
except Exception:
    nvjpeg = None  # type: ignore
    _nvjpeg_available = False

# Optional GStreamer (nvjpegenc) path
try:
    import gi  # type: ignore
    gi.require_version("Gst", "1.0")  # type: ignore[attr-defined]
    from gi.repository import Gst  # type: ignore

    Gst.init(None)
    _gst_available = True
except Exception:
    Gst = None  # type: ignore
    _gst_available = False


class Recorder:
    def __init__(self, cfg: Dict[str, Any]):
        self.enabled = bool(cfg.get("enable", True))
        self.root_dir = cfg.get("root_img_directory", "./img")
        self.every_n = int(cfg.get("every_n_frames", 1))
        self.jpeg_quality = int(cfg.get("jpeg_quality", 90))
        self.encoder = str(cfg.get("encoder", "nvjpeg")).lower()
        self._frame_idx = 0
        self._nvjpeg = None
        self._nvjpeg_warned = False
        self._gst_pipeline = None
        self._gst_appsrc = None
        self._gst_frame_idx = 0

        if _nvjpeg_available:
            try:
                self._nvjpeg = nvjpeg.NvJpeg()  # type: ignore[attr-defined]
            except Exception as e:
                self._nvjpeg = None
                print(f"[REC] nvjpeg unavailable, falling back to OpenCV JPEG: {e}", flush=True)

        # place each run into its own timestamped folder (only if enabled)
        stamp = time.strftime("%H-%M-%S--%d-%m-%Y")
        self.out_dir = os.path.join(self.root_dir, stamp)
        if self.enabled:
            try:
                os.makedirs(self.out_dir, exist_ok=True)
            except Exception as e:
                print(f"[REC] mkdir error: {e}", flush=True)

    def close(self):
        try:
            if self._gst_appsrc is not None:
                self._gst_appsrc.end_of_stream()  # type: ignore[attr-defined]
        except Exception:
            pass
        try:
            if self._gst_pipeline is not None and Gst is not None:
                self._gst_pipeline.set_state(Gst.State.NULL)  # type: ignore[attr-defined]
        except Exception:
            pass
        self._gst_pipeline = None
        self._gst_appsrc = None

    # ------- small helpers -------
    def _draw_text(self, img, text: str, y: int, color=(255, 255, 255)):
        if not text:
            return
        cv2.putText(img, text, (8, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(img, text, (8, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    color, 1, cv2.LINE_AA)

    def _draw_plus_pixel(self, img, x: int, y: int, color):
        """Draw a 5-pixel '+' for visibility."""
        h, w = img.shape[:2]
        coords = [(x, y), (x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
        for cx, cy in coords:
            if 0 <= cx < w and 0 <= cy < h:
                img[cy, cx] = color

    def _draw_cross(self, img, pt, color, size: int = 8, thickness: int = 2):
        if pt is None:
            return
        try:
            x, y = int(pt[0]), int(pt[1])
        except Exception:
            return
        cv2.drawMarker(img, (x, y), color,
                       markerType=cv2.MARKER_TILTED_CROSS,
                       markerSize=size, thickness=thickness,
                       line_type=cv2.LINE_AA)

    def _draw_rect(self, img, rect, color, thickness=2):
        if not rect:
            return
        try:
            x, y, w, h = map(int, rect)
        except Exception:
            return
        cv2.rectangle(img, (x, y), (x + w, y + h), color, thickness, cv2.LINE_AA)

    def _draw_quad(self, img, quad, color, thickness=2):
        if quad is None or np is None:
            return
        try:
            q = np.asarray(quad, dtype=np.int32).reshape(-1, 1, 2)
        except Exception:
            return
        if q.shape[0] < 4:
            return
        cv2.polylines(img, [q], True, color, thickness, cv2.LINE_AA)

    def _draw_dets(self, img, dets: List[Dict[str, Any]]):
        for d in dets:
            bbox = d.get("bbox") or d.get("bbox_px")
            if not bbox:
                continue
            if isinstance(bbox, dict):
                x = int(bbox.get("x", 0))
                y = int(bbox.get("y", 0))
                w = int(bbox.get("w", 0))
                h = int(bbox.get("h", 0))
            else:
                x, y, w, h = map(int, bbox)
            color = (0, 255, 255)  # yellow
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2, cv2.LINE_AA)
            label = str(d.get("label", ""))
            conf = d.get("conf", None)
            if conf is not None:
                try:
                    label = f"{label} {conf:.2f}"
                except Exception:
                    pass
            if label:
                cv2.putText(img, label, (x, max(0, y - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(img, label, (x, max(0, y - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 255, 255), 1, cv2.LINE_AA)

    def _draw_bg_pts(self, img, bg_pts):
        """
        Draw ALL background corners:
          - bg_pts is Nx3 [x, y, flag].
          - flag >= 0.5 => green (inlier & age >= 3).
          - flag <  0.5 => orange (new/young or outlier).
        """
        if bg_pts is None or np is None:
            return
        try:
            arr = np.asarray(bg_pts, dtype=np.float32)
        except Exception:
            return
        if arr.size == 0:
            return

        arr = arr.reshape(-1, arr.shape[-1])
        if arr.shape[1] == 3:
            xy = arr[:, :2]
            fl = arr[:, 2]
        else:
            xy = arr[:, :2]
            fl = np.zeros((arr.shape[0],), dtype=np.float32)

        # Draw single pixels only to reduce CPU
        good_mask = fl >= 0.5
        ints = np.round(xy).astype(np.int32)
        for (xi, yi), good in zip(ints, good_mask):
            color = (0, 255, 0) if good else (0, 165, 255)
            self._draw_plus_pixel(img, int(xi), int(yi), color)

    def _draw_obj_pts(self, img, obj_pts):
        """Draw ALL object-level corners in orange."""
        if obj_pts is None or np is None:
            return
        try:
            arr = np.asarray(obj_pts, dtype=np.float32)
        except Exception:
            return
        if arr.size == 0:
            return
        arr = arr.reshape(-1, arr.shape[-1])
        xy = arr[:, :2]
        for x, y in xy:
            try:
                xi, yi = int(x), int(y)
            except Exception:
                continue
            cv2.circle(img, (xi, yi), 3, (0, 165, 255), -1, cv2.LINE_AA)  # orange

    def _draw_color_legend(self, img, y0: int = 120):
        """
        Simple color legend on the left side.
        """
        entries = [
            ((0, 255, 0),   "Green = stable BG corner (inlier & age>=3)"),
            ((0, 165, 255), "Orange = new/young BG or OBJ corner"),
            ((255, 255, 0), "Cyan = ghost quad (homography)"),
            ((0, 255, 255), "Yellow = search quad / YOLO boxes"),
            ((255,   0, 255), "Magenta = track bbox / fused center"),
            ((255,   0,   0), "Blue = raw homography center"),
        ]
        x0 = 8
        y = y0
        for color, text in entries:
            # little colored square
            cv2.rectangle(img, (x0, y - 10), (x0 + 10, y), color, -1, cv2.LINE_AA)
            cv2.rectangle(img, (x0, y - 10), (x0 + 10, y), (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(img, text, (x0 + 16, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(img, text, (x0 + 16, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 1, cv2.LINE_AA)
            y += 18

    # ------- main entry -------
    def draw(self, frame, viz: Dict[str, Any]):
        if not self.enabled:
            return
        self._frame_idx += 1
        if self.every_n > 1 and (self._frame_idx % self.every_n) != 0:
            return
        if frame is None:
            return
        try:
            img = frame.copy()
        except Exception:
            img = frame

        algo = viz.get("algo", "")
        mode = viz.get("mode", "")
        hud_line   = viz.get("hud_line", "")
        bg_status  = viz.get("bg_status", "")
        obj_status = viz.get("obj_status", "")
        pid_src = viz.get("pid_src", "")
        ts_ms = viz.get("ts_ms", 0)
        seq = viz.get("frame_seq", -1)

        # FPS (if provided by tracker)
        cam_fps  = viz.get("cam_fps", None)
        proc_fps = viz.get("proc_fps", None)

        detections = viz.get("detections") or []
        bg_pts = viz.get("bg_pts", viz.get("pts"))
        obj_pts = viz.get("obj_pts")
        ghost = viz.get("ghost")
        search_quad = viz.get("search_quad")
        search_env = viz.get("search_envelope")
        raw_pt = viz.get("raw_pt")
        filt_pt = viz.get("filt_pt")
        ego_pt = viz.get("ego_pt")
        track_bbox = viz.get("track_bbox")
        track_quad = viz.get("track_quad")
        track_center = viz.get("track_center_px")
        vo_seq = viz.get("vo_seq", -1)
        vo_lag = viz.get("vo_lag", None)
        try:
            n_bg = int(np.asarray(bg_pts).shape[0]) if bg_pts is not None and np is not None else (len(bg_pts) if bg_pts is not None else 0)
        except Exception:
            n_bg = 0
        lag_str = f" lag={vo_lag}" if vo_lag is not None else ""
        vis_line = (
            f"viz: dets={len(detections)} bbox={'Y' if track_bbox else 'N'} raw={'Y' if raw_pt else 'N'} "
            f"filt={'Y' if filt_pt else 'N'} bg_pts={n_bg} vo_seq={vo_seq}{lag_str}"
        )

        # HUD lines
        header = f"ALG={algo} MODE={mode} PID={pid_src} seq={seq} ts={ts_ms}"
        if cam_fps is not None and proc_fps is not None:
            header += f" cam_fps={cam_fps:.1f} proc_fps={proc_fps:.1f}"
        self._draw_text(img, header, y=24, color=(255, 255, 255))

        if hud_line:
            self._draw_text(img, hud_line, y=48, color=(255, 255, 0))
        if bg_status:
            self._draw_text(img, bg_status, y=72, color=(0, 255, 0))
        if obj_status:
            self._draw_text(img, obj_status, y=96, color=(0, 200, 255))
        self._draw_text(img, vis_line, y=120, color=(200, 200, 255))

        # background & object points (ALL)
        self._draw_bg_pts(img, bg_pts)
        self._draw_obj_pts(img, obj_pts)

        # detections
        self._draw_dets(img, detections)

        # track bbox (magenta)
        self._draw_rect(img, track_bbox, (255, 0, 255), 2)
        self._draw_quad(img, track_quad, (255, 0, 255), 2)
        self._draw_cross(img, track_center, (255, 0, 255), size=10, thickness=2)

        # ghost quad (cyan) + search quad (yellow) + search envelope (gray)
        self._draw_quad(img, ghost, (255, 255, 0), 2)         # cyan-ish
        self._draw_quad(img, search_quad, (0, 255, 255), 1)   # yellow
        self._draw_rect(img, search_env, (128, 128, 128), 1)  # gray

        # crosses
        self._draw_cross(img, ego_pt, (0, 255, 255), size=7, thickness=1)  # cyan
        self._draw_cross(img, raw_pt,  (255, 0, 0),   size=9,  thickness=2)  # blue
        self._draw_cross(img, filt_pt, (255, 0, 255), size=11, thickness=2)  # magenta

        # Color legend
        self._draw_color_legend(img, y0=150)

        # Save JPEG
        fname = f"f{seq:08d}_ts{int(ts_ms)}.jpg"
        path = os.path.join(self.out_dir, fname)

        # Prefer GStreamer nvjpegenc when requested and available
        if self.encoder.startswith("nv") and _gst_available:
            if self._ensure_gst_pipeline(img):
                try:
                    buf = Gst.Buffer.new_allocate(None, img.nbytes, None)  # type: ignore[attr-defined]
                    buf.fill(0, img.tobytes())  # type: ignore[attr-defined]
                    buf.offset = self._gst_frame_idx  # type: ignore[attr-defined]
                    self._gst_frame_idx += 1
                    self._gst_appsrc.emit("push-buffer", buf)  # type: ignore[attr-defined]
                    self._frame_idx += 1
                    return
                except Exception as e:
                    print(f"[REC] gst push-buffer failed, falling back to JPEG: {e}", flush=True)

        encoded = None
        if self._nvjpeg is not None:
            try:
                # nvjpeg expects HWC uint8 numpy (BGR is fine for storage)
                encoded = self._nvjpeg.encode(img, quality=int(max(0, min(100, self.jpeg_quality))))
                # Some bindings return (data, size); accept both bytes or tuple
                if isinstance(encoded, tuple) and len(encoded) > 0:
                    encoded = encoded[0]
            except Exception as e:
                if not self._nvjpeg_warned:
                    print(f"[REC] nvjpeg encode failed, falling back to OpenCV JPEG: {e}", flush=True)
                    self._nvjpeg_warned = True
                encoded = None

        if encoded is None:
            try:
                cv2.imwrite(path, img, [int(cv2.IMWRITE_JPEG_QUALITY), int(max(0, min(100, self.jpeg_quality)))])
            except Exception as e:
                print(f"[REC] JPEG dump error: {e}", flush=True)
        else:
            try:
                with open(path, "wb") as f:
                    f.write(encoded)
            except Exception as e:
                print(f"[REC] nvjpeg file write error: {e}", flush=True)

    # -------- GStreamer (nvjpegenc) --------
    def _ensure_gst_pipeline(self, img) -> bool:
        if self._gst_pipeline is not None and self._gst_appsrc is not None:
            return True
        if not _gst_available or Gst is None:
            return False
        try:
            h, w = img.shape[:2]
            loc = os.path.join(self.out_dir, "f%08d.jpg")
            pipe_str = (
                f"appsrc name=src is-live=true format=time do-timestamp=true caps=video/x-raw,format=BGR,width={w},height={h} ! "
                f"videoconvert ! nvjpegenc quality={int(max(0, min(100, self.jpeg_quality)))} ! "
                f"multifilesink location={loc} async=false"
            )
            pipeline = Gst.parse_launch(pipe_str)  # type: ignore[attr-defined]
            appsrc = pipeline.get_by_name("src")  # type: ignore[attr-defined]
            pipeline.set_state(Gst.State.PLAYING)  # type: ignore[attr-defined]
            self._gst_pipeline = pipeline
            self._gst_appsrc = appsrc
            self._gst_frame_idx = 0
            return True
        except Exception as e:
            print(f"[REC] gst pipeline init failed: {e}", flush=True)
            self._gst_pipeline = None
            self._gst_appsrc = None
            return False
