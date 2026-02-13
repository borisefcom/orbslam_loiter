#!/usr/bin/env python3
"""
Standalone entrypoint to run tracker + its worker threads only.
Camera -> FrameBus -> Tracker; OpenCV window for overlay + mouse selection.
JSON events mirror server.py (detect, track_status, calibrate) and are also
printed to stdout for easy scripting.
"""
import argparse
import json
import signal
import sys
import threading
import time
from queue import SimpleQueue
from typing import Any, Dict, Optional, Tuple

import cv2
try:
    cv2.setUseOptimized(True)
    cv2.setNumThreads(6)
except Exception:
    pass

from config import Config
from gstreamer import VideoController
from tracker import Tracker, FrameBus
from tracker_types import FramePacket
from udp import UdpJsonTx, UdpJsonRx, parse_hostport
from recorder import Recorder

# ---------------- Shared JSON TX ----------------
class _FanoutJsonTx:
    """Fan-out JSON sender: optional UDP + stdout + local queue for UI."""
    def __init__(self, udp_tx: Optional[UdpJsonTx], mirror_stdout: bool, sink_q: SimpleQueue):
        self.udp_tx = udp_tx
        self.mirror_stdout = mirror_stdout
        self.q = sink_q

    def send(self, payload: Dict[str, Any]):
        if self.udp_tx is not None:
            try:
                self.udp_tx.send(payload)
            except Exception:
                pass
        if self.mirror_stdout:
            try:
                if payload.get("type") not in ("detect", "track_status", "calibrate"):
                    print(json.dumps(payload, ensure_ascii=False), flush=True)
            except Exception:
                pass
        try:
            self.q.put(payload, block=False)
        except Exception:
            pass

# ---------------- Recorder shim ----------------
class _RecorderShim(Recorder):
    """Wrap Recorder to tap viz frames for the OpenCV HUD without changing tracker."""
    def __init__(self, cfg, fanout_queue: SimpleQueue, base: Recorder):
        # Do not call Recorder.__init__ again; wrap existing recorder
        self.__dict__.update(base.__dict__)
        self.fanout_queue = fanout_queue
        self._base = base

    def draw(self, frame, viz):
        try:
            self.fanout_queue.put(("viz", frame, viz), block=False)
        except Exception:
            pass
        try:
            return self._base.draw(frame, viz)
        except Exception:
            return None

# ---------------- Control handler ----------------
def _apply_control(msg: Dict[str, Any], tracker_obj: Tracker):
    if not isinstance(msg, dict):
        return False
    cmd = str(msg.get("cmd", "")).lower()
    if cmd == "select" and "id" in msg:
        try:
            tracker_obj.target_select(int(msg.get("id")))
            return True
        except Exception:
            return False
    if cmd == "cancel":
        try:
            tracker_obj.target_cancel()
            return True
        except Exception:
            return False
    if cmd == "enable" and "on" in msg:
        try:
            tracker_obj.set_enabled(bool(msg.get("on")))
            return True
        except Exception:
            return False
    if cmd == "track_box" and "box" in msg:
        # box expected [x,y,w,h] pixels
        try:
            box = msg.get("box")
            if isinstance(box, dict):
                x, y, w, h = int(box.get("x", 0)), int(box.get("y", 0)), int(box.get("w", 0)), int(box.get("h", 0))
            else:
                x, y, w, h = map(int, box)
            # synthesize a YOLO-like detection for selection path
            fake_det = {"bbox": (x, y, w, h), "id": int(msg.get("id", 0)), "label": "track", "conf": 1.0}
            tracker_obj.target_select(int(fake_det.get("id", 0)))
            return True
        except Exception:
            return False
    return False

# ---------------- Overlay drawer ----------------
def _draw_overlay(frame, detect_cache, track_state: str, hud_lines, viz=None):
    out = frame.copy()

    def _draw_text(img, text, y, color=(255, 255, 255)):
        if not text:
            return
        cv2.putText(img, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(img, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)

    def _draw_rect(img, rect, color, thickness=2):
        if not rect:
            return
        try:
            x, y, w, h = map(int, rect)
            cv2.rectangle(img, (x, y), (x + w, y + h), color, thickness, cv2.LINE_AA)
        except Exception:
            pass

    def _draw_quad(img, quad, color, thickness=2):
        if quad is None:
            return
        try:
            q = np.asarray(quad, dtype=np.int32).reshape(-1, 1, 2)
            if q.shape[0] >= 4:
                cv2.polylines(img, [q], True, color, thickness, cv2.LINE_AA)
        except Exception:
            pass

    def _draw_cross(img, pt, color, size=8, thickness=2):
        if pt is None:
            return
        try:
            x, y = int(pt[0]), int(pt[1])
            cv2.drawMarker(img, (x, y), color, markerType=cv2.MARKER_CROSS, markerSize=size, thickness=thickness, line_type=cv2.LINE_AA)
        except Exception:
            pass

    # Use viz detections if provided
    if viz and isinstance(viz, dict):
        detect_cache = viz.get("detections", detect_cache) or detect_cache

    # Draw detections
    for d in detect_cache:
        bbox = d.get("bbox_px") or d.get("bbox") or {}
        if isinstance(bbox, dict):
            x, y, w, h = int(bbox.get("x", 0)), int(bbox.get("y", 0)), int(bbox.get("w", 0)), int(bbox.get("h", 0))
        else:
            try:
                x, y, w, h = map(int, bbox)
            except Exception:
                continue
        color = (0, 255, 0) if str(d.get("label")) == "track" else (255, 0, 0)
        _draw_rect(out, (x, y, w, h), color, 2)
        lbl = f"id={d.get('id')} {d.get('label','')}"
        _draw_text(out, lbl, max(12, y - 6), color)

    # Additional overlays from viz
    if viz and isinstance(viz, dict):
        _draw_rect(out, viz.get("track_bbox"), (0, 255, 0), 2)
        _draw_quad(out, viz.get("ghost"), (255, 255, 0), 2)
        _draw_quad(out, viz.get("search_quad"), (0, 255, 255), 1)
        _draw_rect(out, viz.get("search_envelope"), (0, 255, 255), 1)
        _draw_cross(out, viz.get("raw_pt"), (255, 0, 0))
        _draw_cross(out, viz.get("filt_pt"), (128, 0, 128))
        _draw_cross(out, viz.get("ego_pt"), (0, 255, 255))
        # Cross at bbox center if present
        tb = viz.get("track_bbox")
        if tb:
            try:
                x, y, w, h = tb
                _draw_cross(out, (x + w / 2.0, y + h / 2.0), (0, 128, 255), size=10, thickness=2)
            except Exception:
                pass
        # Track center from bbox center if available
        tb = viz.get("track_bbox")
        if tb:
            try:
                x, y, w, h = tb
                _draw_cross(out, (x + w / 2.0, y + h / 2.0), (0, 128, 255), size=10, thickness=2)
            except Exception:
                pass
        # bg_pts / obj_pts
        for pts, color in ((viz.get("bg_pts"), (0, 255, 0)), (viz.get("obj_pts"), (0, 165, 255))):
            try:
                if pts is None:
                    continue
                arr = np.asarray(pts)
                if arr.ndim >= 2:
                    for p in arr:
                        _draw_cross(out, p[:2], color, size=6, thickness=2)
            except Exception:
                pass

    # HUD text
    lines = []
    if viz and isinstance(viz, dict):
        for key in ("hud_line", "bg_status", "obj_status"):
            val = viz.get(key)
            if val:
                lines.append(str(val))
    if track_state:
        lines.append(f"state: {track_state}")
    lines.extend(hud_lines or [])
    # FPS lines
    if viz and isinstance(viz, dict):
        cam_fps = viz.get("cam_fps")
        proc_fps = viz.get("proc_fps")
        if cam_fps is not None or proc_fps is not None:
            lines.append(f"fps cam={cam_fps:.1f} proc={proc_fps:.1f}")
        if "wrms_last" in viz or "wrms_base" in viz:
            lines.append(f"wrms_last={viz.get('wrms_last',0):.3f} wrms_base={viz.get('wrms_base',0):.3f} {viz.get('wrms_units','')}")
        if viz.get("vo_lag") is not None:
            lines.append(f"vo_lag={viz.get('vo_lag')}")
        if viz.get("vo_seq") is not None:
            lines.append(f"vo_seq={viz.get('vo_seq')}")
        if viz.get("track_bbox") is not None:
            lines.append(f"track_bbox={viz.get('track_bbox')}")
        if viz.get("mode") is not None:
            lines.append(f"mode={viz.get('mode')}")
        if viz.get("bg_budget") is not None:
            lines.append(f"vo_budget={viz.get('bg_budget'):.2f}")
    y0 = 20
    for i, line in enumerate(lines):
        _draw_text(out, str(line), y0 + i * 18, color=(0, 255, 255))
    return out

# ---------------- Mouse picker ----------------
def _nearest_det(click_xy: Tuple[int, int], detect_cache):
    if click_xy is None or not detect_cache:
        return None
    cx, cy = click_xy
    best = None
    best_d2 = 1e12
    for d in detect_cache:
        bbox = d.get("bbox_px") or d.get("bbox") or {}
        if isinstance(bbox, dict):
            x, y, w, h = int(bbox.get("x", 0)), int(bbox.get("y", 0)), int(bbox.get("w", 0)), int(bbox.get("h", 0))
        else:
            try:
                x, y, w, h = map(int, bbox)
            except Exception:
                continue
        cdx = x + w / 2.0
        cdy = y + h / 2.0
        d2 = (cdx - cx) ** 2 + (cdy - cy) ** 2
        if d2 < best_d2:
            best_d2 = d2
            best = d
    return best if best is not None else None

# ---------------- Main ----------------
def main():
    parser = argparse.ArgumentParser(description="Local tracker entrypoint (no server)")
    parser.add_argument("--config", default="drone.yaml", help="Path to YAML config")
    parser.add_argument("--control-udp", default=None, help="Override control UDP host:port (default cfg control.json_in)")
    parser.add_argument("--no-stdout-json", action="store_true", help="Disable mirroring tracker JSON to stdout")
    args = parser.parse_args()

    cfg = Config(args.config)
    cfg_dict = cfg.as_dict()

    # Control RX
    ctrl_host, ctrl_port = parse_hostport(args.control_udp or cfg.get("control.json_in", "0.0.0.0:6020"))
    control_rx = UdpJsonRx(ctrl_host, ctrl_port)

    # FrameBus + tracker (shared memory ring)
    fb_w = int(cfg.get("video.width", 1280))
    fb_h = int(cfg.get("video.height", 720))
    framebus = FrameBus(width=fb_w, height=fb_h, channels=3)
    event_q: SimpleQueue = SimpleQueue()
    viz_q: SimpleQueue = SimpleQueue()

    # JSON TX: UDP optional (telemetry json out) + stdout mirror + queue
    tele_host, tele_port = parse_hostport(cfg.get("telemetry.json.udp", "127.0.0.1:6021"))
    udp_tx = UdpJsonTx(tele_host, tele_port) if tele_host and tele_port else None
    fanout_tx = _FanoutJsonTx(udp_tx, mirror_stdout=not args.no_stdout_json, sink_q=event_q)

    # Tracker
    tracker_obj = Tracker(cfg_dict, framebus, fanout_tx, print_fn=print)
    tracker_obj.start()

    # Video
    video = VideoController(cfg)
    video.attach_framebus(framebus)
    ok, detail = video.preflight_camera_check()
    if not ok:
        print(f"[local_main] preflight failed: {detail}", file=sys.stderr, flush=True)
        return 1
    video.start()

    stop_evt = threading.Event()

    # Control loop (UDP)
    def ctrl_loop():
        for msg, addr in control_rx:
            try:
                print(f"[control] recv from {addr}: {msg}", flush=True)
            except Exception:
                pass
            try:
                _applied = _apply_control(msg, tracker_obj)
            except Exception as e:
                print(f"[control] handler error: {e}", flush=True)
            if stop_evt.is_set():
                break
    t_ctrl = threading.Thread(target=ctrl_loop, daemon=True)
    t_ctrl.start()

    # Mouse state
    mouse_click = {"pt": None, "ts": 0.0}
    def _on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            mouse_click["pt"] = (x, y)
            mouse_click["ts"] = time.time()
            print(f"[UI] LMB at {mouse_click['pt']}", flush=True)
        elif event == cv2.EVENT_RBUTTONDOWN:
            try:
                tracker_obj.target_cancel()
                print("[UI] RMB cancel", flush=True)
            except Exception:
                pass
    cv2.namedWindow("Tracker", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Tracker", _on_mouse)

    detect_cache = []
    track_state = ""
    hud_lines = []
    calibrating = False
    latest_viz_frame = None
    latest_viz = None
    last_frame_tuple = None  # (frame, ts_ms, W, H, cam, seq)

    def _drain_events():
        nonlocal detect_cache, track_state, hud_lines, calibrating
        while True:
            try:
                ev = event_q.get_nowait()
            except Exception:
                break
            if not isinstance(ev, dict):
                continue
            et = ev.get("type")
            if et == "detect":
                detect_cache = ev.get("detections", []) or []
                calibrating = False
            elif et == "track_status":
                track_state = str(ev.get("state", ""))
                hud_lines = [f"err=({ev.get('error',{}).get('ex',0):+.3f},{ev.get('error',{}).get('ey',0):+.3f})"]
                calibrating = False
            elif et == "calibrate":
                calibrating = True
                hud_lines = ["calibrating"]
        # If we’ve left calibration but no new status arrived, clear the stale label
        if not calibrating and hud_lines == ["calibrating"]:
            hud_lines = []

    def _drain_viz():
        nonlocal latest_viz_frame, latest_viz
        while True:
            try:
                tag, frame, viz = viz_q.get_nowait()
            except Exception:
                break
            if tag == "viz":
                latest_viz_frame = frame
                latest_viz = viz

    def _shutdown():
        stop_evt.set()
        try:
            tracker_obj.stop()
        except Exception:
            pass
        try:
            video.stop()
        except Exception:
            pass
        try:
            import debug_log
            debug_log.flush("result.txt", append=False)
        except Exception:
            pass
        cv2.destroyAllWindows()

    def _handle_sig(sig, frame):
        _shutdown()
        sys.exit(0)
    signal.signal(signal.SIGINT, _handle_sig)
    signal.signal(signal.SIGTERM, _handle_sig)

    print("[local_main] running. Press 'q' to quit.", flush=True)
    last_click_handled = 0.0
    first_frame_ts = None
    warn_issued = False
    try:
        while not stop_evt.is_set():
            _drain_events()
            _drain_viz()
            latest = framebus.latest()
            # Prefer viz frame if available
            if latest_viz_frame is not None and latest_viz is not None:
                disp_frame = latest_viz_frame.copy()
            elif latest is None:
                if first_frame_ts is None:
                    first_frame_ts = time.time()
                elif not warn_issued and (time.time() - first_frame_ts) > 3.0:
                    print("[local_main] waiting for camera frames...", flush=True)
                    warn_issued = True
                # still poll keyboard so 'q' works even without frames
                key = cv2.waitKey(10) & 0xFF
                if key == ord('q'):
                    break
                continue
            else:
                frame, ts_ms, W, H, cam, seq = latest
                disp_frame = frame.copy()
                last_frame_tuple = latest

            # Handle mouse selection
            if mouse_click["pt"] and mouse_click["ts"] > last_click_handled:
                sel = _nearest_det(mouse_click["pt"], detect_cache)
                if sel is not None:
                    try:
                        # Ensure fresh selection
                        tracker_obj.target_cancel()
                        bbox_used = None
                        det_id = sel.get("id")
                        if det_id is None:
                            try:
                                det_id = detect_cache.index(sel)
                            except Exception:
                                det_id = 0
                        print(f"[UI] target_select id={det_id} bbox={sel.get('bbox_px') or sel.get('bbox')}", flush=True)
                        tracker_obj.target_select(int(det_id))
                        # Also seed bbox directly in case IDs mismatch
                        bbox = sel.get("bbox_px") or sel.get("bbox") or {}
                        try:
                            if isinstance(bbox, dict):
                                x, y, w, h = int(bbox.get("x", 0)), int(bbox.get("y", 0)), int(bbox.get("w", 0)), int(bbox.get("h", 0))
                            else:
                                x, y, w, h = map(int, bbox)
                            if w > 0 and h > 0:
                                setattr(tracker_obj, "_pending_bbox_seed", (x, y, w, h))
                                bbox_used = (x, y, w, h)
                        except Exception:
                            pass
                        # Start tracker immediately on current frame if available
                        if bbox_used and last_frame_tuple is not None:
                            try:
                                lf, lf_ts, lf_w, lf_h, lf_cam, lf_seq = last_frame_tuple
                                gray = cv2.cvtColor(lf, cv2.COLOR_BGR2GRAY) if hasattr(lf, "ndim") and lf.ndim == 3 else lf
                                fp = FramePacket(bgr=lf, gray=gray, ts=float(lf_ts) / 1000.0, W=lf_w, H=lf_h)
                                print(f"[UI] immediate start bbox={bbox_used} seq={lf_seq}", flush=True)
                                tracker_obj.objtrk.start(fp, bbox_used)
                            except Exception:
                                pass
                    except Exception:
                        pass
                last_click_handled = mouse_click["ts"]

            # Prefer tracker-provided viz payload for overlay (includes quads, search, etc.)
            overlay = _draw_overlay(disp_frame, detect_cache, track_state, hud_lines, viz=latest_viz)
            cv2.imshow("Tracker", overlay)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # ESC
                break
            elif key == ord('t'):
                tracker_obj.set_enabled(not bool(getattr(tracker_obj, "enabled", True)))
            elif key == ord('p'):
                try:
                    tracker_obj.yolo.set_enabled(not tracker_obj.yolo.enabled)
                except Exception:
                    pass
    finally:
        _shutdown()
    return 0

if __name__ == "__main__":
    sys.exit(main())
