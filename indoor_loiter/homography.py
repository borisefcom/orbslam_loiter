#!/usr/bin/env python3
# homography.py - DynMedianFlow + YOLO tracker engine for tracker.py wrapper.

from __future__ import annotations

from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import cv2
import time
from collections import deque

from tracker_types import FramePacket
from dynamic_medianflow import DynMedianFlowTracker, DynMFParams


class HomographyTracker:
    """
    Object tracker engine used by tracker.Tracker.

    API (must match what tracker.py expects):
      - __init__(cfg, framebus, print_fn)
      - step(frame_bgr, ts_ms) -> dict
      - start_from_bbox(bbox_xywh)
      - cancel()
      - force_detect_next()
    """

    STATE_IDLE = "IDLE"
    STATE_ACQUIRE = "ACQUIRE"
    STATE_TRACKING = "TRACKING"

    def __init__(self, cfg: Dict[str, Any], framebus, print_fn=print) -> None:
        self.cfg = cfg or {}
        self.print = print_fn
        self.bus = framebus

        obj_cfg = self.cfg.get("object_tracker", {}) or {}
        self.obj_enabled = bool(obj_cfg.get("enable", True))
        # DynMedianFlowTracker defaults are tuned in dynamic_medianflow.py; use them as-is.
        self.obj_params = DynMFParams()
        self.obj_tracker = DynMedianFlowTracker(self.obj_params)
        self._mf_last_info: Dict[str, Any] = {}

        # DynMF perf / debug (printed from worker process)
        try:
            _cfg_period = float(obj_cfg.get("log_period_s", 1.0))
        except Exception:
            _cfg_period = 1.0
        self._mf_log_period_s = 0.0 if _cfg_period <= 0.0 else float(_cfg_period)
        self._mf_last_log_t: float = 0.0
        self._mf_dt_hist_ms = deque(maxlen=120)

        # State
        self.state = self.STATE_ACQUIRE
        self._prev_fp: Optional[FramePacket] = None
        self._last_kf_pt: Optional[Tuple[float, float]] = None
        self._last_bbox: Optional[Tuple[float, float, float, float]] = None
        self._last_ghost = None
        self._pending_init_bbox: Optional[Tuple[float, float, float, float]] = None

        # YOLO
        self.yolo_thr = None
        self._yolo_enabled = False
        det_cfg = self.cfg.get("detector", {}) or {}
        self._det_enabled_cfg = bool(det_cfg.get("enable", True))
        self._det_model = str(det_cfg.get("model", "yolo11m.pt"))
        self._det_device = str(det_cfg.get("device", "cpu"))
        self._det_period_s = float(det_cfg.get("det_period_s", 0.25))
        self._det_conf = float(det_cfg.get("conf_thres", 0.25))
        self._det_iou = float(det_cfg.get("iou_thres", 0.45))
        self._det_max_det = int(det_cfg.get("max_det", 100))
        self._det_scale = float(det_cfg.get("scale", 1.0))
        self._last_det_ts = 0.0
        self._force_det = False
        self._init_yolo()

        # Track failure logging (e.g., DynMF loss)
        self._track_fail_reason: Optional[str] = None
        self._track_fail_last_log: float = 0.0
        self._track_fail_log_period: float = float(
            (self.cfg.get("tracker") or {}).get("fail_log_period_s", 1.0)
        )

        # Auto-stop tracking if bbox becomes too large (e.g., target fills the frame).
        # Prefer PID_MOVE.auto_exit.max_bbox_area_ratio (profiled), fallback to legacy tracker.max_bbox_area_ratio.
        r = 0.0
        try:
            mc = self.cfg.get("mav_control") or {}
            pm = (mc.get("PID_MOVE") or mc.get("pid_move") or {}) if isinstance(mc, dict) else {}
            ax = (pm.get("auto_exit") or {}) if isinstance(pm, dict) else {}
            if isinstance(ax, dict):
                if ax.get("enabled", True) is False:
                    r = 0.0
                else:
                    v = ax.get("max_bbox_area_ratio", None)
                    if v is not None:
                        r = float(v or 0.0)
        except Exception:
            r = 0.0
        if r <= 0.0:
            try:
                r = float((self.cfg.get("tracker") or {}).get("max_bbox_area_ratio", 0.0) or 0.0)
            except Exception:
                r = 0.0
        # Allow percent-style configs (80 => 0.8)
        if r > 1.0:
            r = r / 100.0
        self._max_bbox_area_ratio = float(max(0.0, min(1.0, r)))

        # Log key tracker settings on load (small, for sanity checks)
        try:
            self.print(
                "[HomographyTracker] initialized "
                f"obj_enabled={self.obj_enabled} "
                f"algo=DynMedianFlow "
                f"dyn_grid={int(bool(self.obj_params.dynamic_grid))} "
                f"grid_min={tuple(self.obj_params.grid_min)} "
                f"grid_max={tuple(self.obj_params.grid_max)} "
                f"lk_win={int(self.obj_params.lk_win_size)} "
                f"lk_levels={int(self.obj_params.lk_max_level)} "
                f"ncc={int(bool(self.obj_params.ncc_enable))}",
                flush=True,
            )
        except Exception:
            pass

    @staticmethod
    def _fmt_mf_counts(info: Dict[str, Any]) -> str:
        # Keep this small: only what we need to understand tracker health.
        try:
            final = int(info.get("final_points", 0) or 0)
        except Exception:
            final = 0
        try:
            cols = int(info.get("grid_cols", 0) or 0)
            rows = int(info.get("grid_rows", 0) or 0)
            total = int(cols * rows)
        except Exception:
            total = 0
        try:
            good = float(info.get("good_frac", 0.0) or 0.0)
        except Exception:
            good = 0.0
        try:
            conf = float(info.get("conf", 0.0) or 0.0)
        except Exception:
            conf = 0.0
        return f"pts={final}/{total} good={good:.2f} conf={conf:.2f}"

    def _maybe_log_mf(self, *, ok: bool, dt_ms: float) -> None:
        if self._mf_log_period_s <= 0:
            return
        now = time.time()
        if (now - self._mf_last_log_t) < self._mf_log_period_s:
            return
        self._mf_last_log_t = now

        info = self._mf_last_info or {}
        cols = int(info.get("grid_cols", 0) or 0)
        rows = int(info.get("grid_rows", 0) or 0)
        stage = str(info.get("reason", "success" if ok else "fail"))
        counts_s = self._fmt_mf_counts(info)

        dt_avg = 0.0
        if self._mf_dt_hist_ms:
            try:
                xs = list(self._mf_dt_hist_ms)
                dt_avg = float(sum(xs) / len(xs)) if xs else 0.0
            except Exception:
                dt_avg = 0.0
        try:
            self.print(
                "[MF(perf)] "
                f"ok={int(bool(ok))} "
                f"backend=DynMF "
                f"grid={int(cols)}x{int(rows)} "
                f"stage={stage} "
                + (f"{counts_s} " if counts_s else "")
                + f"dt_ms={float(dt_ms):.2f} dt_avg_ms={float(dt_avg):.2f}",
                flush=True,
            )
        except Exception:
            pass

    # ------------------------------------------------------------------
    # YOLO support
    # ------------------------------------------------------------------

    def _init_yolo(self) -> None:
        if not self._det_enabled_cfg:
            self._yolo_enabled = False
            self.yolo_thr = None
            self.print("[HomographyTracker] detector disabled", flush=True)
            return
        self.start_yolo_thread()

    def start_yolo_thread(self) -> bool:
        """Start YOLO thread if enabled and not already running."""
        if self._yolo_enabled and self.yolo_thr is not None:
            return True
        try:
            from yolo import YoloThread  # your existing implementation

            self.yolo_thr = YoloThread(
                True,
                self._det_model,
                self._det_period_s,
                self._det_conf,
                self._det_iou,
                self._det_max_det,
                self._det_device,
                self._det_scale,
            )
            self.yolo_thr.start()
            self._yolo_enabled = True
            self.print("[HomographyTracker] YOLO thread started", flush=True)
            return True
        except Exception as e:
            self.print(f"[HomographyTracker] YOLO init failed: {e}", flush=True)
            self.yolo_thr = None
            self._yolo_enabled = False
            return False

    def _maybe_run_yolo(
        self, frame_bgr, W: int, H: int, ts_s: float
    ) -> Tuple[bool, List[Dict[str, Any]]]:
        # Skip detection while actively tracking to save GPU/CPU
        if self.state == self.STATE_TRACKING:
            return False, []
        if not self._yolo_enabled or self.yolo_thr is None:
            return False, []
        t0 = time.time()
        run = (
            self._force_det
            or self._prev_fp is None
            or (ts_s - self._last_det_ts >= self._det_period_s)
        )
        if not run:
            return False, []
        try:
            self.yolo_thr.update(frame_bgr, W, H)
            dets = self.yolo_thr.get() or []
            # Normalize Det namedtuple -> dict for downstream code
            ndets = []
            for d in dets:
                try:
                    if isinstance(d, tuple) and hasattr(d, "_fields"):
                        ndets.append(
                            {
                                "bbox": tuple(d.bbox),
                                "conf": float(d.conf),
                                "cls": int(d.cls),
                                "label": str(d.label),
                            }
                        )
                    else:
                        ndets.append(dict(d))
                except Exception:
                    pass
            dets = ndets
        except Exception as e:
            self.print(f"[HomographyTracker] YOLO error: {e}", flush=True)
            dets = []
        self._last_det_ts = ts_s
        self._force_det = False
        return True, dets

    # ------------------------------------------------------------------
    # Public control API (called from tracker.Tracker)
    # ------------------------------------------------------------------

    def _reset_obj_tracker(self) -> None:
        # DynMedianFlowTracker has no explicit reset API; create a fresh instance.
        self.obj_tracker = DynMedianFlowTracker(self.obj_params)
        self._mf_last_info = {}

    def start_from_bbox(self, bbox_xywh) -> None:
        self._reset_obj_tracker()
        self.state = self.STATE_TRACKING
        self._last_bbox = bbox_xywh
        self._pending_init_bbox = bbox_xywh
        self._clear_track_failure()
        self.print(f"[HomographyTracker] start_from_bbox {bbox_xywh}", flush=True)

    def cancel(self) -> None:
        self.state = self.STATE_ACQUIRE
        self._reset_obj_tracker()
        self._last_bbox = None
        self._last_kf_pt = None
        self._last_ghost = None
        self._pending_init_bbox = None
        self.print("[HomographyTracker] cancel -> ACQUIRE", flush=True)

    def force_detect_next(self) -> None:
        self._force_det = True

    def stop(self) -> None:
        """Best-effort shutdown hook for worker process exit."""
        try:
            if self.yolo_thr is not None:
                try:
                    self.yolo_thr.stop(timeout=1.0)
                except Exception:
                    pass
        finally:
            self.yolo_thr = None
            self._yolo_enabled = False

    def _clear_track_failure(self) -> None:
        self._track_fail_reason = None
        self._track_fail_last_log = 0.0

    def _log_track_failure(self, reason: Optional[str] = None) -> None:
        if reason:
            self._track_fail_reason = reason
        if not self._track_fail_reason:
            return
        now = time.time()
        if (now - self._track_fail_last_log) >= self._track_fail_log_period:
            try:
                self.print(f"[TRACK] failure: {self._track_fail_reason}", flush=True)
            except Exception:
                pass
        self._track_fail_last_log = now

    def _build_acquire_out(
        self,
        *,
        detect_tick: bool,
        dets: List[Dict[str, Any]],
        hud_imu: str,
        reason: Optional[str] = None,
    ) -> Dict[str, Any]:
        bg_status = "VO disabled"
        viz = {
            "algo": "DynMedianFlow",
            "mode": "ACQUIRE",
            "hud_line": hud_imu,
            "bg_status": bg_status,
            "obj_status": "",
            "bg_pts": [],
            "ghost": None,
            "raw_pt": None,
            "filt_pt": None,
            "bg_budget": 0.0,
        }

        out = {
            "pid_state_uc": self.STATE_ACQUIRE,
            "state_lc": "acquire",
            "raw_pt": None,
            "filt_pt": None,
            "detect_tick": detect_tick,
            "detections": dets,
            "ghost_bbox": None,
            "viz": viz,
        }
        if reason:
            out["fail_reason"] = reason
        return out

    # ------------------------------------------------------------------
    # Main step (called by tracker.Tracker)
    # ------------------------------------------------------------------

    def step(self, frame_bgr, ts_ms: int) -> Dict[str, Any]:
        """
        Main engine call. Returns a dict consumed by tracker.Tracker:
          pid_state_uc, state_lc, raw_pt, filt_pt,
          detect_tick, detections, ghost_bbox, viz.
        """
        ts_s = float(ts_ms) / 1000.0

        if frame_bgr is None:
            return {
                "pid_state_uc": self.STATE_IDLE,
                "state_lc": "idle",
                "raw_pt": None,
                "filt_pt": None,
                "detect_tick": False,
                "detections": [],
                "ghost_bbox": None,
                "viz": {},
            }

        if frame_bgr.ndim != 3 or frame_bgr.shape[2] < 3:
            raise ValueError("HomographyTracker.step expects HxWx3 BGR frame")

        H_img, W_img = frame_bgr.shape[:2]
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        fp = FramePacket(bgr=frame_bgr, gray=gray, ts=ts_s, W=W_img, H=H_img)

        hud_imu = ""

        # Background VO removed
        bg_points: List = []

        # If object tracking is disabled, short-circuit: optional YOLO + ACQUIRE output only.
        if not self.obj_enabled:
            detect_tick, dets = self._maybe_run_yolo(frame_bgr, W_img, H_img, ts_s)
            out = self._build_acquire_out(
                detect_tick=detect_tick,
                dets=dets,
                hud_imu=hud_imu,
                reason="object_tracker_disabled",
            )
            self._prev_fp = fp
            return out

        # ----------------- ACQUIRE MODE -----------------
        if self.state != self.STATE_TRACKING:
            detect_tick, dets = self._maybe_run_yolo(frame_bgr, W_img, H_img, ts_s)

            out = self._build_acquire_out(
                detect_tick=detect_tick,
                dets=dets,
                hud_imu=hud_imu,
                reason=self._track_fail_reason,
            )
            self._log_track_failure()
            self._prev_fp = fp
            return out

        # ----------------- TRACKING MODE -----------------
        # Initialize DynMF on first tracking frame if pending.
        if self._pending_init_bbox is not None:
            try:
                ok_init = self.obj_tracker.init(gray, self._pending_init_bbox)
            except Exception as e:
                self.print(f"[HomographyTracker] DynMF init error: {e}", flush=True)
                ok_init = False
            if not ok_init:
                self.state = self.STATE_ACQUIRE
                self._track_fail_reason = "mf_init_failed"
                try:
                    self.print(f"[MF(fail)] {self._track_fail_reason}", flush=True)
                except Exception:
                    pass
                self._reset_obj_tracker()
                self._pending_init_bbox = None
                self._prev_fp = fp
                return self._build_acquire_out(
                    detect_tick=False,
                    dets=[],
                    hud_imu=hud_imu,
                    reason=self._track_fail_reason,
                )
            self._pending_init_bbox = None

        info: Dict[str, Any] = {}
        try:
            t0_mf = time.time()
            ok, bbox, info = self.obj_tracker.update(gray, return_info=True)
            dt_ms = (time.time() - t0_mf) * 1000.0
            self._mf_last_info = info or {}
            try:
                self._mf_dt_hist_ms.append(float(dt_ms))
            except Exception:
                pass
            self._maybe_log_mf(ok=bool(ok), dt_ms=dt_ms)
        except Exception as e:
            self.print(f"[HomographyTracker] DynMF update error: {e}", flush=True)
            ok = False
            bbox = None
            dt_ms = 0.0
            self._mf_last_info = {"reason": "exception"}
            info = self._mf_last_info

        if not ok:
            self.state = self.STATE_ACQUIRE
            reason = str((info or {}).get("reason", "mf_lost"))
            cols = int((info or {}).get("grid_cols", getattr(self.obj_tracker, "_grid_cols", 0)) or 0)
            rows = int((info or {}).get("grid_rows", getattr(self.obj_tracker, "_grid_rows", 0)) or 0)
            counts_s = self._fmt_mf_counts(info or {})

            self._track_fail_reason = (
                f"mf_lost backend=DynMF grid={int(cols)}x{int(rows)} "
                + (f"{counts_s} " if counts_s else "")
                + f"reason='{reason}'"
            )
            mf_diag = {
                "backend": "DynMF",
                "grid": [int(cols), int(rows)],
                "reason": reason,
                "info": {str(k): v for k, v in (info or {}).items()},
            }
            try:
                self.print("[MF(fail)] " + str(self._track_fail_reason) + f" dt_ms={float(dt_ms):.2f}", flush=True)
            except Exception:
                pass
            self._force_det = True
            self._reset_obj_tracker()
            out = self._build_acquire_out(
                detect_tick=False,
                dets=[],
                hud_imu=hud_imu,
                reason=self._track_fail_reason,
            )
            out["track_failed"] = True
            out["mf"] = mf_diag
            self._log_track_failure()
            self._prev_fp = fp
            self._last_kf_pt = None
            self._last_bbox = None
            self._last_ghost = None
            return out

        # Stop tracking if bbox covers too much of the image (area ratio threshold).
        if self._max_bbox_area_ratio > 0.0 and W_img > 0 and H_img > 0:
            try:
                x, y, w, h = bbox
                x1 = max(0.0, float(x))
                y1 = max(0.0, float(y))
                x2 = min(float(W_img), float(x) + max(0.0, float(w)))
                y2 = min(float(H_img), float(y) + max(0.0, float(h)))
                iw = max(0.0, x2 - x1)
                ih = max(0.0, y2 - y1)
                area_ratio = (iw * ih) / (float(W_img) * float(H_img))
            except Exception:
                area_ratio = 0.0

            if area_ratio >= float(self._max_bbox_area_ratio):
                self.state = self.STATE_ACQUIRE
                self._track_fail_reason = (
                    f"max_bbox_area ratio={area_ratio:.3f} thr={float(self._max_bbox_area_ratio):.3f} "
                    f"bbox=({int(x)},{int(y)},{int(w)},{int(h)}) img={int(W_img)}x{int(H_img)}"
                )
                try:
                    self.print(f"[TRACK] stop: {self._track_fail_reason}", flush=True)
                except Exception:
                    pass
                self._reset_obj_tracker()
                self._force_det = True
                out = self._build_acquire_out(
                    detect_tick=False,
                    dets=[],
                    hud_imu=hud_imu,
                    reason=self._track_fail_reason,
                )
                out["track_failed"] = True
                out["stop_reason"] = "max_bbox_area"
                out["bbox_area_ratio"] = float(area_ratio)
                self._log_track_failure()
                self._prev_fp = fp
                self._last_kf_pt = None
                self._last_bbox = None
                self._last_ghost = None
                return out

        x, y, w, h = bbox
        cx = x + w / 2.0
        cy = y + h / 2.0
        raw_pt = (cx, cy)
        kf_pt = raw_pt
        ghost = None
        upd = "DynMF"

        self._prev_fp = fp
        self._last_kf_pt = kf_pt
        self._last_bbox = bbox
        self._last_ghost = ghost
        bg_status = "VO disabled"
        try:
            info = self._mf_last_info or {}
            conf = float(info.get("conf", 0.0) or 0.0)
            good = float(info.get("good_frac", 0.0) or 0.0)
            cols = int(info.get("grid_cols", 0) or 0)
            rows = int(info.get("grid_rows", 0) or 0)
        except Exception:
            conf = 0.0
            good = 0.0
            cols = 0
            rows = 0
        obj_status = (
            f"{upd} bbox=({int(x)},{int(y)},{int(w)},{int(h)}) "
            f"conf={conf:.2f} good={good:.2f} grid={int(cols)}x{int(rows)}"
        )

        viz = {
            "algo": "DynMedianFlow",
            "mode": "TRACK",
            "hud_line": hud_imu,
            "bg_status": bg_status,
            "obj_status": obj_status,
            "bg_pts": bg_points,
            "obj_pts": None,
            "ghost": ghost.tolist() if ghost is not None else None,
            "raw_pt": raw_pt,
            "filt_pt": kf_pt,
            "bg_budget": 0.0,
        }

        out = {
            "pid_state_uc": self.STATE_TRACKING,
            "state_lc": "tracking",
            "raw_pt": raw_pt,
            "filt_pt": kf_pt,
            "detect_tick": False,  # YOLO only in ACQUIRE
            "detections": [],
            "ghost_bbox": bbox,
            "viz": viz,
        }
        return out

    def flush_perf_log(self, path: str = "results.txt") -> None:
        """No-op placeholder; file logging disabled."""
        return
