#!/usr/bin/env python3
# tracker.py - process-isolated tracker using shared FrameBus.

from __future__ import annotations

import json
import time
import threading
import traceback
import signal
import multiprocessing as mp
from typing import Any, Dict, List, Optional, Tuple

try:
    import numpy as np
    import warnings

    warnings.filterwarnings("ignore", message="The value of the smallest subnormal.*")
except Exception:
    np = None  # optional

from imu import FrameBus
from homography import HomographyTracker

try:
    from recorder import Recorder  # overlay JPEG logger
except Exception:
    Recorder = None  # type: ignore[assignment]

__all__ = ["FrameBus", "Tracker"]

# Event kinds sent over the IPC channel
_EVENT_JSON = "json"
_EVENT_PID = "pid"
_EVENT_LOG = "log"
_EVENT_EXIT = "exit"


class TrackerWorker:
    """
    Runs inside a separate process.
    Consumes frames from FrameBus (shared memory) and emits events back to parent.
    """

    def __init__(self, cfg: Dict[str, Any], bus_handles: Dict[str, Any], cmd_rx, evt_tx, print_fn=print):
        self.cfg = cfg or {}
        self.bus = FrameBus.attach(bus_handles)
        self.cmd_rx = cmd_rx
        self.evt_tx = evt_tx
        self.print = print_fn

        d = self.cfg.get("detector", {}) or {}
        t = self.cfg.get("tracker", {}) or {}
        to = (t.get("timeouts") or d.get("timeouts") or {})

        self.enabled = bool(t.get("enable", d.get("enable", True)))
        self.rate_hz = float(t.get("rate_hz", d.get("rate_hz", 30.0)))
        self.lost_s = float(to.get("lost_s", 3.0))
        self.cancel_after_lost_s = float(to.get("cancel_after_lost_s", 6.0))

        mc = (self.cfg.get("mav_control") or {})
        self.pid_axis_map = str(mc.get("axis_map", "xy")).lower()  # "xy" or "yx"
        self.force_center_when_no_raw = bool(mc.get("force_center_when_no_raw", True))

        # Engine: pass FrameBus so it can drain IMU packets if needed
        self.engine = HomographyTracker(self.cfg, framebus=self.bus, print_fn=self.print)

        rec_cfg = self.cfg.get("recorder", {})
        self.rec = Recorder(rec_cfg) if (rec_cfg.get("enable", False) and Recorder is not None) else None

        self._stop = False
        self._seq = 0

        self._sel_lock = threading.Lock()
        self._selected_id: Optional[int] = None
        self._pending_select_id: Optional[int] = None
        self._pending_roi_bbox: Optional[Tuple[int, int, int, int]] = None

        # Monotonic YOLO detection IDs used as "track_id" (UI/PID). ID 0 is reserved for ROI selection.
        # Wrap at signed 32-bit max.
        self._det_id_next: int = 1
        self._det_id_max: int = 2_147_483_647
        self._last_yolo_dets_for_id: List[Dict[str, Any]] = []

        self._last_dets_for_select: List[Dict[str, Any]] = []

        self.pid_measurement_enabled = True

        # When detector is disabled, the GUI may not receive any "detect" messages (with img_size)
        # until tracking starts. Send an empty detect once at startup (and optionally in ACQUIRE)
        # so ROI selection has the correct pixel coordinate basis immediately.
        self._detector_enabled_cfg = bool(d.get("enable", True))
        self._img_size_sent = False
        self._last_idle_detect_t = 0.0

        self._last_ex = 0.0
        self._last_ey = 0.0
        self._last_state_str: Optional[str] = None
        self._last_track_print_t: float = 0.0
        self._track_failed_prev: bool = False
        self._pid_warned: bool = False
        # Perf/debug
        from collections import deque

        self._loop_dt_hist = deque(maxlen=240)
        self._seq_delta_hist = deque(maxlen=240)
        self._step_dt_hist = deque(maxlen=240)
        self._emit_dt_hist = deque(maxlen=240)
        self._age_hist_ms = deque(maxlen=240)
        self._last_loop_t = None
        self._last_perf_print = 0.0
        self._perf_lines: list[str] = []
        self._timing_lines: list[str] = []
        self._last_seq = None
        self._vo_fps: float = 0.0

        self.print(
            f"[TRACK(worker)] init enabled={self.enabled} "
            f"PID axis_map={self.pid_axis_map} "
            f"force_center_no_raw={self.force_center_when_no_raw}",
            flush=True,
        )

        try:
            signal.signal(signal.SIGINT, self._sig_handler)
            signal.signal(signal.SIGTERM, self._sig_handler)
        except Exception:
            pass

    # ----------------- IPC helpers -----------------
    def _send_evt(self, kind: str, payload: Any):
        try:
            self.evt_tx.send({"kind": kind, "payload": payload})
        except Exception:
            pass

    def _handle_cmds(self):
        while self.cmd_rx.poll():
            try:
                msg = self.cmd_rx.recv()
            except EOFError:
                self._stop = True
                return
            if not isinstance(msg, dict):
                continue
            cmd = msg.get("cmd")
            if cmd == "stop":
                self._stop = True
                return
            if cmd == "enable":
                self.enabled = bool(msg.get("value", True))
                continue
            if cmd == "select":
                det_id = msg.get("id")
                if det_id is not None:
                    self.target_select(int(det_id))
                continue
            if cmd == "roi_select":
                bbox = msg.get("bbox", None)
                if bbox is None:
                    bbox = msg.get("roi", None)
                if bbox is not None:
                    self.target_roi_select(bbox)
                continue
            if cmd == "cancel":
                self.target_cancel()
                continue
            if cmd == "set_pid_enabled":
                self.pid_measurement_enabled = bool(msg.get("value", True))
                continue
            if cmd == "set_vo_fps":
                try:
                    self._vo_fps = float(msg.get("fps", 0.0))
                except Exception:
                    self._vo_fps = 0.0
                continue

    # ----------------- selection -----------------
    def _alloc_det_id(self) -> int:
        did = int(self._det_id_next)
        nxt = did + 1
        if nxt > int(self._det_id_max):
            nxt = 1
        self._det_id_next = int(nxt)
        return did

    def _get_selected_id(self) -> Optional[int]:
        with self._sel_lock:
            return self._selected_id

    def _set_selected_id(self, v: Optional[int]):
        with self._sel_lock:
            self._selected_id = v

    def target_select(self, det_id: int):
        with self._sel_lock:
            self._pending_select_id = int(det_id)
            self._pending_roi_bbox = None
        self.print(f"[TRACK(worker)] target_select requested id={det_id}", flush=True)

    def target_roi_select(self, bbox: Any):
        """
        Request tracking start from an explicit ROI bbox.
        Accepted bbox forms:
          - (x, y, w, h) list/tuple
          - {"x":..,"y":..,"w":..,"h":..}
          - {"bbox": {...}} / {"bbox": [x,y,w,h]}
        """
        bx = bbox
        if isinstance(bx, dict) and "bbox" in bx:
            bx = bx.get("bbox")
        x = y = w = h = 0
        try:
            if isinstance(bx, dict):
                x = int(bx.get("x", 0))
                y = int(bx.get("y", 0))
                w = int(bx.get("w", 0))
                h = int(bx.get("h", 0))
            else:
                x, y, w, h = map(int, bx)
        except Exception:
            return
        with self._sel_lock:
            self._pending_roi_bbox = (int(x), int(y), int(w), int(h))
            self._pending_select_id = None
        self.print(f"[TRACK(worker)] roi_select requested bbox=({x},{y},{w},{h})", flush=True)

    def target_cancel(self):
        with self._sel_lock:
            self._pending_select_id = None
            self._pending_roi_bbox = None
            self._selected_id = None
        self.engine.cancel()
        try:
            self.engine.force_detect_next()
        except Exception:
            pass
        self._last_ex = 0.0
        self._last_ey = 0.0
        self.print("[TRACK(worker)] target_cancel", flush=True)

    # ----------------- detection cache (for GUI selection latency) -----------------
    def _det_cache_cfg(self) -> Tuple[int, int]:
        """
        Returns (keep_ms, log_period_ms).

        This exists to make GUI-driven selection robust when the user clicks on
        a detection that disappears from the next YOLO tick.

        YAML (optional):
          detector:
            select_cache_s: 2.0
            select_pending_log_s: 1.0
        """
        try:
            keep_s = float((self.cfg.get("detector") or {}).get("select_cache_s", 2.0) or 0.0)
        except Exception:
            keep_s = 2.0
        try:
            log_s = float((self.cfg.get("detector") or {}).get("select_pending_log_s", 1.0) or 0.0)
        except Exception:
            log_s = 1.0
        keep_ms = int(max(0.0, keep_s) * 1000.0)
        log_ms = int(max(0.1, log_s) * 1000.0)
        return keep_ms, log_ms

    def _det_cache_update(self, ts_ms: int, dets: List[Dict[str, Any]]) -> None:
        """
        Maintain a short mapping {det_id -> (bbox_xywh, ts_ms)} to allow late
        track_select requests to still resolve.
        """
        keep_ms, _ = self._det_cache_cfg()
        if keep_ms <= 0:
            return
        if not hasattr(self, "_det_cache"):
            self._det_cache: Dict[int, Tuple[Tuple[int, int, int, int], int]] = {}
        # Update cache
        try:
            for d in dets or []:
                if not isinstance(d, dict):
                    continue
                try:
                    did = int(d.get("id", 0))
                except Exception:
                    continue
                if did <= 0:
                    continue
                bbox = d.get("bbox_px") or d.get("bbox") or {}
                try:
                    if isinstance(bbox, dict):
                        x = int(bbox.get("x", 0))
                        y = int(bbox.get("y", 0))
                        w = int(bbox.get("w", 0))
                        h = int(bbox.get("h", 0))
                    else:
                        x, y, w, h = map(int, bbox)
                except Exception:
                    continue
                if w <= 0 or h <= 0:
                    continue
                self._det_cache[did] = ((x, y, w, h), int(ts_ms))
        except Exception:
            pass
        # Purge expired
        try:
            expire_before = int(ts_ms) - int(keep_ms)
            if expire_before > 0 and self._det_cache:
                dead = [k for k, (_, t0) in self._det_cache.items() if int(t0) < expire_before]
                for k in dead:
                    self._det_cache.pop(k, None)
        except Exception:
            pass

    # ----------------- coord helpers -----------------
    @staticmethod
    def _bus_scale_point(
        pt_xy: Tuple[float, float],
        frame_shape: Tuple[int, int],
        bus_W: int,
        bus_H: int,
    ) -> Tuple[int, int]:
        if pt_xy is None or pt_xy[0] is None or pt_xy[1] is None:
            return int(bus_W / 2), int(bus_H / 2)
        fy = float(pt_xy[1])
        fx = float(pt_xy[0])
        fH, fW = int(frame_shape[0]), int(frame_shape[1])
        if fW > 0 and fH > 0 and (fW != bus_W or fH != bus_H):
            sx = float(bus_W) / float(fW)
            sy = float(bus_H) / float(fH)
            fx *= sx
            fy *= sy
        fx = max(0.0, min(float(bus_W - 1), fx))
        fy = max(0.0, min(float(bus_H - 1), fy))
        return int(round(fx)), int(round(fy))

    def _apply_axis_map(self, cx_i: int, cy_i: int, W: int, H: int) -> Tuple[int, int]:
        if self.pid_axis_map == "yx":
            return max(0, min(W - 1, cy_i)), max(0, min(H - 1, cx_i))
        return cx_i, cy_i

    # ----------------- emitters -----------------
    def _emit_detect(self, ts_ms: int, W: int, H: int, dets: List[Dict[str, Any]]):
        out = []
        max_det = int(self.cfg.get("detector", {}).get("max_det", 100))
        pad_px = 0
        try:
            pad_px = int((self.cfg.get("detector", {}) or {}).get("bbox_pad_px", 0) or 0)
        except Exception:
            pad_px = 0
        pad_px = max(0, pad_px)

        # Reuse IDs for YOLO detections when bbox is unchanged vs previous YOLO tick.
        prev_ids_by_key: Dict[Tuple[int, int, int, int, int, str], List[int]] = {}
        try:
            for pd in (self._last_yolo_dets_for_id or []):
                if not isinstance(pd, dict):
                    continue
                bx_pd = pd.get("bbox_px") or {}
                if not isinstance(bx_pd, dict):
                    continue
                try:
                    x0 = int(bx_pd.get("x", 0))
                    y0 = int(bx_pd.get("y", 0))
                    w0 = int(bx_pd.get("w", 0))
                    h0 = int(bx_pd.get("h", 0))
                    cls0 = int(pd.get("cls", -1))
                    lbl0 = str(pd.get("label", ""))
                    did0 = int(pd.get("id", 0))
                except Exception:
                    continue
                if did0 <= 0 or w0 <= 0 or h0 <= 0 or cls0 < 0:
                    continue
                key0 = (x0, y0, w0, h0, cls0, lbl0)
                prev_ids_by_key.setdefault(key0, []).append(did0)
        except Exception:
            prev_ids_by_key = {}

        yolo_tick = False
        for i, d in enumerate(dets[:max_det]):
            bx = d.get("bbox") or d.get("bbox_px")
            if not bx:
                continue
            if isinstance(bx, dict):
                x = int(bx.get("x", 0))
                y = int(bx.get("y", 0))
                w = int(bx.get("w", 0))
                h = int(bx.get("h", 0))
            else:
                x, y, w, h = map(int, bx)
            if w <= 0 or h <= 0:
                continue
            c_override = d.get("center_px")
            if isinstance(c_override, (tuple, list)) and len(c_override) == 2:
                cxf, cyf = float(c_override[0]), float(c_override[1])
            elif isinstance(c_override, dict) and "cx" in c_override and "cy" in c_override:
                cxf, cyf = float(c_override["cx"]), float(c_override["cy"])
            else:
                cxf, cyf = x + w / 2.0, y + h / 2.0

            cls_val = int(d.get("cls", -1))
            label = str(d.get("label", d.get("cls", "?")))
            conf = float(d.get("conf", 0.0))

            # GUI padding for YOLO detections only (cls>=0). Track overlays use cls=-1 label="track".
            if pad_px > 0 and cls_val >= 0:
                try:
                    x1 = int(x)
                    y1 = int(y)
                    x2 = int(x + w)
                    y2 = int(y + h)
                    x1 = max(0, min(int(W), x1 - pad_px))
                    y1 = max(0, min(int(H), y1 - pad_px))
                    x2 = max(0, min(int(W), x2 + pad_px))
                    y2 = max(0, min(int(H), y2 + pad_px))
                    if x2 > x1 and y2 > y1:
                        x = int(x1)
                        y = int(y1)
                        w = int(x2 - x1)
                        h = int(y2 - y1)
                except Exception:
                    pass

            # Stable ID assignment:
            # - If upstream provided an id (e.g. "track" overlay), preserve it.
            # - Else if YOLO detection (cls>=0), reuse previous ID if bbox exactly matches; otherwise allocate new.
            # - Else (unknown), fall back to index.
            id_val = d.get("id", None)
            if cls_val >= 0:
                yolo_tick = True
                if id_val is None:
                    key = (int(x), int(y), int(w), int(h), int(cls_val), str(label))
                    prev_list = prev_ids_by_key.get(key)
                    if prev_list:
                        id_val = int(prev_list.pop(0))
                    else:
                        id_val = self._alloc_det_id()
            if id_val is None:
                id_val = i
            try:
                id_val = int(id_val)
            except Exception:
                id_val = int(i)

            out.append(
                {
                    "id": id_val,
                    "cls": cls_val,
                    "label": label,
                    "conf": conf,
                    "bbox_px": {"x": x, "y": y, "w": w, "h": h},
                    "center_px": {"cx": int(round(cxf)), "cy": int(round(cyf))},
                    "center_norm": {
                        "nx": (cxf - W / 2.0) / (W / 2.0) if W > 0 else 0.0,
                        "ny": (cyf - H / 2.0) / (H / 2.0) if H > 0 else 0.0,
                    },
                }
            )

        if yolo_tick:
            try:
                self._last_yolo_dets_for_id = [d for d in out if isinstance(d, dict) and int(d.get("cls", -1)) >= 0]
            except Exception:
                self._last_yolo_dets_for_id = []

        payload = {
            "type": "detect",
            "ts": int(ts_ms),
            "img_size": [int(W), int(H)],
            "seq": int(self._seq),
            "detections": out,
        }

        # Keep detection JSON under a safe UDP datagram size to avoid fragmentation
        # losses on VPN/cellular links (default: 1200 bytes; <=0 disables).
        max_bytes = int(((self.cfg.get("telemetry") or {}).get("json") or {}).get("max_bytes", 1200) or 0)
        if max_bytes > 0 and out:
            try:
                dets_all = out

                def _payload_bytes(det_count: int) -> int:
                    payload["detections"] = dets_all[:det_count]
                    return len(json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8"))

                if _payload_bytes(len(dets_all)) > max_bytes:
                    lo, hi = 1, len(dets_all)
                    best = 1
                    while lo <= hi:
                        mid = (lo + hi) // 2
                        if _payload_bytes(mid) <= max_bytes:
                            best = mid
                            lo = mid + 1
                        else:
                            hi = mid - 1
                    payload["detections"] = dets_all[:best]
            except Exception:
                payload["detections"] = out

        self._last_dets_for_select = payload.get("detections", []) or []
        self._last_detect_ts_ms = int(ts_ms)
        # Cache the full (pre-truncation) detection list so selection can still resolve
        # even if the chosen ID disappears from the next YOLO tick.
        self._det_cache_update(int(ts_ms), out)
        self._send_evt(_EVENT_JSON, payload)

    def _emit_track_status(
        self,
        ts_ms: int,
        state_lc: str,
        W: int,
        H: int,
        pt_for_error_xyi: Tuple[int, int],
        pid_src: str,
    ):
        if pt_for_error_xyi is not None and W > 0 and H > 0:
            ex = (float(pt_for_error_xyi[0]) - (W / 2.0)) / (W / 2.0)
            ey = (float(pt_for_error_xyi[1]) - (H / 2.0)) / (H / 2.0)
            self._last_ex, self._last_ey = ex, ey
        else:
            if state_lc != "idle":
                ex, ey = self._last_ex, self._last_ey
            else:
                ex = ey = 0.0
        self._send_evt(
            _EVENT_JSON,
            {
                "type": "track_status",
                "ts": int(ts_ms),
                "state": state_lc,
                "track_id": self._get_selected_id(),
                "pid_source": pid_src,
                "error": {"ex": float(ex), "ey": float(ey)},
                "cmd": {"yaw": 0.0, "pitch": 0.0},
            },
        )

    def _emit_pid_meas(
        self,
        state_uc: str,
        used_src: str,
        used_xy_bus_int: Tuple[int, int],
        kf_xy_bus_int: Tuple[Optional[int], Optional[int]],
        raw_xy_bus_int: Tuple[Optional[int], Optional[int]],
        W: int,
        H: int,
    ):
        if not self.pid_measurement_enabled:
            return
        payload = {
            "state": state_uc,
            "src": used_src,
            "cx_px": used_xy_bus_int[0],
            "cy_px": used_xy_bus_int[1],
            "kf": {"cx": kf_xy_bus_int[0], "cy": kf_xy_bus_int[1]},
            "raw": {"cx": raw_xy_bus_int[0], "cy": raw_xy_bus_int[1]},
            "W": int(W),
            "H": int(H),
            "track_id": self._get_selected_id(),
        }
        self._send_evt(_EVENT_PID, payload)

    # ----------------- lifecycle -----------------
    def run(self):
        self._raise_priority()
        while not self._stop:
            self._handle_cmds()
            if self._stop:
                break

            item = self.bus.latest()
            if item is None or not self.enabled:
                time.sleep(0.001)
                continue

            if len(item) == 6:
                frame, ts_ms, W_bus, H_bus, cam, seq = item
            else:
                frame, ts_ms, W_bus, H_bus, cam = item
                seq = None

            if frame is None or not hasattr(frame, "shape"):
                time.sleep(0.0005)
                continue

            # Ensure GUI learns the image size even when YOLO is disabled.
            # This fixes the "first roi_select bbox shifted" issue caused by missing img_size context.
            try:
                if (not self._img_size_sent) and int(W_bus) > 0 and int(H_bus) > 0 and int(ts_ms) > 0:
                    self._emit_detect(int(ts_ms), int(W_bus), int(H_bus), [])
                    self._img_size_sent = True
                    self._last_idle_detect_t = time.time()
                    try:
                        self.print(f"[TRACK(worker)] sent idle detect img_size={int(W_bus)}x{int(H_bus)}", flush=True)
                    except Exception:
                        pass
            except Exception:
                pass
            if seq is not None:
                try:
                    if self._last_seq is not None and seq == self._last_seq:
                        time.sleep(0.0005)
                        continue
                except Exception:
                    pass
                prev_seq = self._last_seq
                self._last_seq = seq
                if prev_seq is not None:
                    try:
                        dseq = int(seq) - int(prev_seq)
                        if dseq <= 0:
                            dseq = 1
                        self._seq_delta_hist.append(float(dseq))
                    except Exception:
                        pass

            try:
                now_loop = time.time()
                if self._last_loop_t is not None:
                    self._loop_dt_hist.append(max(0.0, now_loop - self._last_loop_t))
                self._last_loop_t = now_loop
                self._age_hist_ms.append(max(0.0, (now_loop * 1000.0) - float(ts_ms)))

                if hasattr(frame, "shape") and frame.ndim == 3 and frame.shape[2] == 4:
                    frame = frame[:, :, :3]
            except Exception:
                pass

            try:
                t0_step = time.time()
                out = self.engine.step(frame, ts_ms)
                t1_step = time.time()
                step_dt = max(0.0, t1_step - t0_step)
                emit_start = time.time()

                track_failed = bool(out.get("track_failed", False))
                fail_reason = out.get("fail_reason")
                if track_failed and not self._track_failed_prev:
                    msg = fail_reason or "tracker failure detected; reverting to detection mode"
                    lost_track_id = self._get_selected_id()
                    stop_reason = out.get("stop_reason")
                    try:
                        self._send_evt(
                            _EVENT_JSON,
                            {
                                "type": "track_lost",
                                "ts": int(ts_ms),
                                "track_id": lost_track_id,
                                "reason": str(msg),
                                "stop_reason": str(stop_reason) if stop_reason is not None else None,
                                "bbox_area_ratio": out.get("bbox_area_ratio"),
                                "mf": out.get("mf"),
                            },
                        )
                    except Exception:
                        pass
                    # Notify PID about special stop reasons (used for thrust override logic).
                    if stop_reason == "max_bbox_area":
                        try:
                            self._send_evt(
                                _EVENT_PID,
                                {
                                    "event": "max_bbox_area",
                                    "ts": int(ts_ms),
                                    "track_id": lost_track_id,
                                    "bbox_area_ratio": out.get("bbox_area_ratio"),
                                },
                            )
                        except Exception:
                            pass
                    self.print(f"[TRACK(worker)] {msg}", flush=True)
                    try:
                        self.target_cancel()
                    except Exception:
                        pass
                    try:
                        if hasattr(self.engine, "cancel"):
                            self.engine.cancel()
                        if hasattr(self.engine, "force_detect_next"):
                            self.engine.force_detect_next()
                    except Exception:
                        pass
                    # Immediately resume target JSON flow for the GUI by emitting a detect payload,
                    # even before the next YOLO tick provides fresh detections.
                    try:
                        self._emit_detect(int(ts_ms), int(W_bus), int(H_bus), [])
                    except Exception:
                        pass
                self._track_failed_prev = track_failed

                if hasattr(frame, "shape"):
                    fH, fW = frame.shape[0], frame.shape[1]
                else:
                    fH, fW = H_bus, W_bus

                state_uc = out.get("pid_state_uc", "IDLE")
                state_lc = out.get("state_lc", state_uc.lower())

                raw_pt = out.get("raw_pt")
                if raw_pt is not None and raw_pt[0] is not None and raw_pt[1] is not None:
                    chosen_src = "raw"
                    chosen_pt_engine = (float(raw_pt[0]), float(raw_pt[1]))
                else:
                    chosen_src = "center"
                    if self.force_center_when_no_raw:
                        chosen_pt_engine = (fW / 2.0, fH / 2.0)
                    else:
                        filt_pt = out.get("filt_pt")
                        if filt_pt is not None:
                            chosen_pt_engine = (float(filt_pt[0]), float(filt_pt[1]))
                            chosen_src = "kf"
                        else:
                            chosen_pt_engine = (fW / 2.0, fH / 2.0)
                            chosen_src = "center"

                used_xy_bus_int = self._bus_scale_point(
                    chosen_pt_engine, (fH, fW), W_bus, H_bus
                )
                used_xy_bus_int = self._apply_axis_map(
                    used_xy_bus_int[0], used_xy_bus_int[1], W_bus, H_bus
                )

                kf_pt = out.get("filt_pt")
                if kf_pt is not None:
                    kf_xy_bus_int = self._bus_scale_point(
                        (kf_pt[0], kf_pt[1]), (fH, fW), W_bus, H_bus
                    )
                    kf_xy_bus_int = self._apply_axis_map(
                        kf_xy_bus_int[0], kf_xy_bus_int[1], W_bus, H_bus
                    )
                else:
                    kf_xy_bus_int = (None, None)

                if raw_pt is not None:
                    raw_xy_bus_int = self._bus_scale_point(
                        (raw_pt[0], raw_pt[1]), (fH, fW), W_bus, H_bus
                    )
                    raw_xy_bus_int = self._apply_axis_map(
                        raw_xy_bus_int[0], raw_xy_bus_int[1], W_bus, H_bus
                    )
                else:
                    raw_xy_bus_int = (None, None)

                if (time.time() - self._last_perf_print) >= 1.0:
                    try:
                        import numpy as _np

                        def _avg(lst):
                            return float(_np.mean(lst)) if lst else 0.0

                        loop_fps = 0.0
                        if self._loop_dt_hist:
                            dt = _avg(self._loop_dt_hist)
                            loop_fps = 0.0 if dt <= 1e-9 else (1.0 / dt)
                        src_fps = 0.0
                        skip = 0.0
                        if self._loop_dt_hist and self._seq_delta_hist:
                            dt = _avg(self._loop_dt_hist)
                            dseq = _avg(self._seq_delta_hist)
                            if dt > 1e-9:
                                src_fps = float(dseq) / float(dt)
                                skip = max(0.0, float(dseq) - 1.0)
                        step_ms = _avg(self._step_dt_hist) * 1000.0 if self._step_dt_hist else 0.0
                        emit_ms = _avg(self._emit_dt_hist) * 1000.0 if self._emit_dt_hist else 0.0
                        age_ms = _avg(self._age_hist_ms)
                        line = (
                            f"[TRACK(perf)] loop_fps={loop_fps:.1f} src_fps={src_fps:.1f} skip={skip:.1f} "
                            f"step_ms={step_ms:.2f} emit_ms={emit_ms:.2f} age_ms={age_ms:.1f} vo_fps={self._vo_fps:.1f}"
                        )
                        self._send_evt(_EVENT_LOG, line)
                    except Exception:
                        pass
                    self._last_perf_print = time.time()

                just_emitted_detect = False
                if out.get("detect_tick", False):
                    dets_to_emit = out.get("detections")
                    if not isinstance(dets_to_emit, list):
                        dets_to_emit = []
                    self._emit_detect(ts_ms, W_bus, H_bus, dets_to_emit)
                    just_emitted_detect = True
                elif state_uc == "TRACKING":
                    gb = out.get("ghost_bbox", None)
                    if gb is None:
                        g = (out.get("viz") or {}).get("ghost")
                        try:
                            if g is not None:
                                if np is not None:
                                    arr_g = np.asarray(g).reshape(-1, 2)
                                    x1 = int(max(0, np.floor(arr_g[:, 0].min())))
                                    y1 = int(max(0, np.floor(arr_g[:, 1].min())))
                                    x2 = int(min(fW - 1, np.ceil(arr_g[:, 0].max())))
                                    y2 = int(min(fH - 1, np.ceil(arr_g[:, 1].max())))
                                else:
                                    xs = [int(p[0]) for p in g]
                                    ys = [int(p[1]) for p in g]
                                    x1 = max(0, min(xs))
                                    y1 = max(0, min(ys))
                                    x2 = min(fW - 1, max(xs))
                                    y2 = min(fH - 1, max(ys))
                                sx = float(W_bus) / float(fW) if fW > 0 else 1.0
                                sy = float(H_bus) / float(fH) if fH > 0 else 1.0
                                bx = int(round(x1 * sx))
                                by = int(round(y1 * sy))
                                bw = int(round((x2 - x1) * sx))
                                bh = int(round((y2 - y1) * sy))
                                gb = (bx, by, max(0, bw), max(0, bh))
                        except Exception:
                            gb = None

                    if gb is not None and gb[2] > 0 and gb[3] > 0:
                        det = {"bbox": gb, "conf": 1.0, "cls": -1, "label": "track"}
                        sid = self._get_selected_id()
                        if sid is not None:
                            det["id"] = int(sid)
                        det["center_px"] = (used_xy_bus_int[0], used_xy_bus_int[1])
                        self._emit_detect(ts_ms, W_bus, H_bus, [det])
                else:
                    # Detector disabled + not tracking: periodically send empty detect so GUI keeps img_size
                    # and clears any stale overlays without requiring YOLO.
                    if not self._detector_enabled_cfg:
                        try:
                            now = time.time()
                            if (now - float(self._last_idle_detect_t)) >= 1.0 and int(W_bus) > 0 and int(H_bus) > 0:
                                self._emit_detect(int(ts_ms), int(W_bus), int(H_bus), [])
                                self._last_idle_detect_t = now
                        except Exception:
                            pass

                self._apply_pending_selection_if_recent_detect(just_emitted_detect)
                self._apply_pending_roi_if_any(fW=fW, fH=fH)

                if track_failed:
                    self._seq += 1
                    continue

                self._emit_track_status(
                    ts_ms, state_lc, W_bus, H_bus, used_xy_bus_int, chosen_src
                )
                self._emit_pid_meas(
                    state_uc,
                    chosen_src,
                    used_xy_bus_int,
                    kf_xy_bus_int,
                    raw_xy_bus_int,
                    W_bus,
                    H_bus,
                )

                if self.rec is not None and out.get("viz") is not None:
                    try:
                        viz = dict(out["viz"])
                        gb = out.get("ghost_bbox")
                        if gb is not None:
                            viz["track_bbox"] = gb
                        viz["track_center_px"] = (
                            float(used_xy_bus_int[0]),
                            float(used_xy_bus_int[1]),
                        )
                        viz["frame_seq"] = int(seq) if seq is not None else -1
                        viz["ts_ms"] = int(ts_ms)
                        viz["pid_src"] = chosen_src
                        self.rec.draw(frame, viz)
                    except Exception as e:
                        self.print(f"[REC] draw error: {e}")

                self._seq += 1
                emit_end = time.time()
                self._step_dt_hist.append(step_dt)
                self._emit_dt_hist.append(max(0.0, emit_end - emit_start))
                self._timing_lines.append(
                    f"[TRACK(timing)] step_ms={step_dt*1000.0:.2f} emit_ms={max(0.0, emit_end-emit_start)*1000.0:.2f}"
                )

            except Exception as e:
                self.print(f"[TRACK(worker)] error: {e}")
                traceback.print_exc()
                continue

        self._send_evt(_EVENT_EXIT, {"reason": "stop"})

    def _apply_pending_selection_if_recent_detect(self, just_emitted_detect: bool) -> bool:
        if not just_emitted_detect:
            return False
        with self._sel_lock:
            pid = self._pending_select_id
        if pid is None:
            return False
        x = y = w = h = 0
        src = "detect"
        det = next((d for d in self._last_dets_for_select if d.get("id") == pid), None)
        if det:
            bbox = det.get("bbox_px") or det.get("bbox") or {}
            x = int(bbox.get("x", 0) if isinstance(bbox, dict) else bbox[0])
            y = int(bbox.get("y", 0) if isinstance(bbox, dict) else bbox[1])
            w = int(bbox.get("w", 0) if isinstance(bbox, dict) else bbox[2])
            h = int(bbox.get("h", 0) if isinstance(bbox, dict) else bbox[3])
        else:
            # Fall back to a short-lived cache so late UI clicks still work.
            keep_ms, log_ms = self._det_cache_cfg()
            bbox_xywh = None
            age_ms = None
            try:
                cache = getattr(self, "_det_cache", None) or {}
                item = cache.get(int(pid))
                if item is not None:
                    bbox_xywh, t0 = item
                    age_ms = int(getattr(self, "_last_detect_ts_ms", 0)) - int(t0)
                    if keep_ms > 0 and age_ms is not None and age_ms > keep_ms:
                        bbox_xywh = None
            except Exception:
                bbox_xywh = None
                age_ms = None

            if bbox_xywh is None:
                # Rate-limit log spam while pending.
                now_ms = int(time.time() * 1000)
                last_ms = None
                try:
                    last_ms = int(getattr(self, "_pending_select_last_log_ms", 0) or 0)
                except Exception:
                    last_ms = 0
                if last_ms <= 0 or (now_ms - last_ms) >= int(log_ms):
                    self._pending_select_last_log_ms = now_ms
                    self.print(
                        f"[TRACK(worker)] pending select id={pid} not in last detections",
                        flush=True,
                    )
                return False

            src = "cache"
            x, y, w, h = map(int, bbox_xywh)
        if w <= 0 or h <= 0:
            return False
        pad = 0.05
        dx = int(w * pad)
        dy = int(h * pad)
        x = max(0, x - dx)
        y = max(0, y - dy)
        w = w + 2 * dx
        h = h + 2 * dy
        self.engine.start_from_bbox((x, y, w, h))
        with self._sel_lock:
            self._selected_id = pid
            self._pending_select_id = None
        self.print(
            f"[TRACK(worker)] selection applied from {src} id={pid} bbox=({x},{y},{w},{h})",
            flush=True,
        )
        return True

    def _apply_pending_roi_if_any(self, *, fW: int, fH: int) -> bool:
        with self._sel_lock:
            bb = self._pending_roi_bbox
        if bb is None:
            return False
        x, y, w, h = map(int, bb)
        if w <= 0 or h <= 0:
            with self._sel_lock:
                self._pending_roi_bbox = None
            return False
        # Clamp + pad similarly to track_select path.
        x = max(0, min(int(fW - 1), x))
        y = max(0, min(int(fH - 1), y))
        w = max(1, min(int(fW), w))
        h = max(1, min(int(fH), h))
        pad = 0.05
        dx = int(w * pad)
        dy = int(h * pad)
        x1 = max(0, x - dx)
        y1 = max(0, y - dy)
        x2 = min(int(fW), x + w + dx)
        y2 = min(int(fH), y + h + dy)
        if x2 <= x1 or y2 <= y1:
            with self._sel_lock:
                self._pending_roi_bbox = None
            return False
        bbox_xywh = (int(x1), int(y1), int(x2 - x1), int(y2 - y1))
        self.engine.start_from_bbox(bbox_xywh)
        with self._sel_lock:
            self._selected_id = 0
            self._pending_roi_bbox = None
        self.print(f"[TRACK(worker)] roi_select applied track_id=0 bbox={bbox_xywh}", flush=True)
        return True

    def _raise_priority(self):
        try:
            import os

            try:
                os.nice(-5)
            except Exception:
                pass
        except Exception:
            pass

    def _sig_handler(self, signum, frame):
        self._stop = True


def _tracker_process_main(cfg: Dict[str, Any], bus_handles: Dict[str, Any], cmd_rx, evt_tx):
    worker = TrackerWorker(cfg, bus_handles, cmd_rx, evt_tx, print_fn=print)
    try:
        worker.run()
    finally:
        try:
            if hasattr(worker, "engine") and hasattr(worker.engine, "stop"):
                worker.engine.stop()
        except Exception:
            pass
        try:
            worker.bus.close()
        except Exception:
            pass


class Tracker:
    """
    Client-facing tracker. Spawns a worker process and forwards events.
    """

    def __init__(self, cfg: Dict[str, Any], framebus: FrameBus, json_tx, print_fn=print):
        self.cfg = cfg or {}
        self.bus = framebus
        self.tx = json_tx
        self.print = print_fn

        self._ctx = mp.get_context("spawn")
        self._cmd_tx = None
        self._evt_rx = None
        self._proc: Optional[mp.Process] = None
        self._evt_thr: Optional[threading.Thread] = None
        self._stop_evt = threading.Event()
        self._pid_cb = None
        self.enabled = True
        self._perf_lock = threading.Lock()
        self._last_perf: Dict[str, float] = {}
        self._stats_lock = threading.Lock()
        self._stats_last_t: Optional[float] = None
        self._cnt_json_total = 0
        self._cnt_json_detect = 0
        self._cnt_pid = 0

    @staticmethod
    def _parse_track_perf_line(line: str) -> Dict[str, float]:
        # Example: "[TRACK(perf)] loop_fps=55.6 src_fps=59.9 skip=0.0 step_ms=1.7 ..."
        out: Dict[str, float] = {}
        try:
            if not isinstance(line, str) or not line.startswith("[TRACK(perf)]"):
                return out
            parts = line.split()
            for tok in parts[1:]:
                if "=" not in tok:
                    continue
                k, v = tok.split("=", 1)
                try:
                    out[str(k)] = float(v)
                except Exception:
                    continue
        except Exception:
            return {}
        return out

    def snapshot_stats(self) -> Dict[str, Any]:
        """
        Return last known tracker perf + event rates since last snapshot.
        Called from the main thread (e.g., server FPS monitor).
        """
        now = time.time()
        with self._stats_lock:
            dt = None
            if self._stats_last_t is not None:
                dt = max(1e-6, now - float(self._stats_last_t))
            self._stats_last_t = now
            c_json = int(self._cnt_json_total)
            c_det = int(self._cnt_json_detect)
            c_pid = int(self._cnt_pid)
            self._cnt_json_total = 0
            self._cnt_json_detect = 0
            self._cnt_pid = 0
        with self._perf_lock:
            perf = dict(self._last_perf) if self._last_perf else {}
        rates: Dict[str, float] = {"json_hz": 0.0, "detect_hz": 0.0, "pid_hz": 0.0}
        if dt is not None:
            rates["json_hz"] = float(c_json) / float(dt)
            rates["detect_hz"] = float(c_det) / float(dt)
            rates["pid_hz"] = float(c_pid) / float(dt)
        return {"dt_s": dt, "perf": perf, "rates": rates}

    # ----------- API ------------
    def set_pid_callback(self, cb):
        self._pid_cb = cb
        self.print(
            f"[TRACK(client)] set_pid_callback: {'callable' if callable(cb) else 'not callable'}",
            flush=True,
        )

    def target_select(self, det_id: int):
        self._send_cmd({"cmd": "select", "id": int(det_id)})

    def select(self, det_id: int):
        self.target_select(det_id)

    def target_roi_select(self, bbox):
        self._send_cmd({"cmd": "roi_select", "bbox": bbox})

    def roi_select(self, bbox):
        self.target_roi_select(bbox)

    def target_cancel(self):
        self._send_cmd({"cmd": "cancel"})

    def cancel(self):
        self.target_cancel()

    def set_enabled(self, on: bool):
        self.enabled = bool(on)
        self._send_cmd({"cmd": "enable", "value": bool(on)})

    def set_vo_fps(self, fps: float):
        self._send_cmd({"cmd": "set_vo_fps", "fps": float(fps)})

    def start(self):
        if self._proc is not None:
            return
        handles = self.bus.export_handles()
        cmd_tx, cmd_rx = self._ctx.Pipe()
        evt_rx, evt_tx = self._ctx.Pipe()
        self._cmd_tx = cmd_tx
        self._evt_rx = evt_rx
        self._proc = self._ctx.Process(
            target=_tracker_process_main,
            args=(self.cfg, handles, cmd_rx, evt_tx),
            daemon=False,
        )
        self._proc.start()
        self._stop_evt.clear()
        self._evt_thr = threading.Thread(target=self._pump_events, name="TrackerEvt", daemon=True)
        self._evt_thr.start()
        self.print("[TRACK(client)] started worker process", flush=True)

    def stop(self):
        self._stop_evt.set()
        self._send_cmd({"cmd": "stop"})
        try:
            if self._proc:
                self._proc.join(timeout=2.0)
                if self._proc.is_alive():
                    self._proc.terminate()
                    self._proc.join(timeout=1.0)
        except Exception:
            pass
        self._proc = None
        self._cmd_tx = None
        if self._evt_thr:
            self._evt_thr.join(timeout=1.0)
        self._evt_thr = None
        self._evt_rx = None
        self.print("[TRACK(client)] stopped", flush=True)

    # ----------- internals ------------
    def _send_cmd(self, msg: Dict[str, Any]):
        tx = self._cmd_tx
        if tx is None:
            return
        try:
            tx.send(msg)
        except Exception:
            pass

    def _pump_events(self):
        rx = self._evt_rx
        if rx is None:
            return
        while not self._stop_evt.is_set():
            try:
                if not rx.poll(0.1):
                    continue
                evt = rx.recv()
            except (EOFError, OSError):
                break
            if not isinstance(evt, dict):
                continue
            kind = evt.get("kind")
            payload = evt.get("payload")
            if kind == _EVENT_JSON:
                try:
                    with self._stats_lock:
                        self._cnt_json_total += 1
                        if isinstance(payload, dict) and payload.get("type") == "detect":
                            self._cnt_json_detect += 1
                except Exception:
                    pass
                try:
                    if self.tx:
                        self.tx.send(payload)
                except Exception as e:
                    self.print(f"[TRACK(client)] json send error: {e}")
            elif kind == _EVENT_PID:
                try:
                    with self._stats_lock:
                        self._cnt_pid += 1
                except Exception:
                    pass
                cb = self._pid_cb
                if callable(cb):
                    try:
                        cb(payload)
                    except Exception as e:
                        self.print(f"[TRACK(client)] pid cb error: {e}")
            elif kind == _EVENT_LOG:
                s = str(payload)
                if s.startswith("[TRACK(perf)]"):
                    try:
                        perf = self._parse_track_perf_line(s)
                        if perf:
                            with self._perf_lock:
                                self._last_perf = perf
                    except Exception:
                        pass
                self.print(s)
            elif kind == _EVENT_EXIT:
                break
        self.print("[TRACK(client)] event pump exit", flush=True)

    def get_last(self):
        return {
            "enabled": self.enabled,
        }


# Backwards-compat alias
TrackerProcess = Tracker
