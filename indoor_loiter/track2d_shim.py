from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

from dynamic_medianflow import DynMFParams, DynMedianFlowTracker

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None


def _now_ms() -> int:
    return int(time.time() * 1000)


def _clamp01(v: object) -> float:
    try:
        x = float(v)
    except Exception:
        return 0.0
    if not math.isfinite(x):
        return 0.0
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return float(x)


def _norm_to_px(x_norm: object, n: int) -> int:
    nn = int(max(1, int(n)))
    return int(round(_clamp01(x_norm) * float(nn - 1)))


def _bbox_from_two_norm_corners(
    *,
    x0_norm: object,
    y0_norm: object,
    x1_norm: object,
    y1_norm: object,
    W: int,
    H: int,
) -> Tuple[float, float, float, float]:
    x0 = _norm_to_px(x0_norm, int(W))
    y0 = _norm_to_px(y0_norm, int(H))
    x1 = _norm_to_px(x1_norm, int(W))
    y1 = _norm_to_px(y1_norm, int(H))
    xa = int(min(x0, x1))
    xb = int(max(x0, x1))
    ya = int(min(y0, y1))
    yb = int(max(y0, y1))
    w = int(max(1, xb - xa))
    h = int(max(1, yb - ya))
    return (float(xa), float(ya), float(w), float(h))


def _gray_u8(frame: Any) -> Optional[np.ndarray]:
    if frame is None or not hasattr(frame, "shape"):
        return None
    try:
        arr = np.asarray(frame)
    except Exception:
        return None
    if arr.ndim == 2:
        g = arr
    elif arr.ndim == 3 and arr.shape[2] == 1:
        g = arr[:, :, 0]
    elif arr.ndim == 3 and arr.shape[2] == 3:
        if cv2 is None:
            return None
        try:
            g = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
        except Exception:
            try:
                g = arr[:, :, 1]
            except Exception:
                return None
    else:
        return None
    if g.dtype != np.uint8:
        try:
            g = g.astype(np.uint8, copy=False)
        except Exception:
            return None
    return g


@dataclass
class Track2dStatus:
    enabled: bool = False
    state: str = "idle"  # idle|tracking|lost
    ts_ms: int = 0
    img_size: Tuple[int, int] = (0, 0)
    bbox_px: Optional[Tuple[float, float, float, float]] = None
    center_px: Optional[Tuple[float, float]] = None
    conf: Optional[float] = None
    reason: Optional[str] = None


class Track2dShim:
    """
    Minimal 2D ROI tracker shim (DynMedianFlow) driven by a simple control protocol:
      - enable/disable
      - rectangle selection (ROI) in normalized coords

    It reads frames from a `frame_source` object with a `.latest()` method.
    """

    def __init__(
        self,
        *,
        frame_source: Any,
        get_intrinsics_fn: Callable[[], Tuple[int, int, float, float, float, float]],
        send_pid_fn: Callable[[str, Optional[float], Optional[float]], None],
        json_event_fn: Optional[Callable[[Dict[str, Any]], None]] = None,
        on_enable_changed: Optional[Callable[[bool], None]] = None,
        print_fn=print,
    ) -> None:
        self.frame_source = frame_source
        self.get_intrinsics = get_intrinsics_fn
        self.send_pid = send_pid_fn
        self.json_event = json_event_fn
        self.on_enable_changed = on_enable_changed
        self.print = print_fn

        self._stop = threading.Event()
        self._th: Optional[threading.Thread] = None

        self._lock = threading.Lock()
        self._enabled: bool = False
        self._pending_bbox: Optional[Tuple[float, float, float, float]] = None
        self._tracking: bool = False
        self._bbox: Optional[Tuple[float, float, float, float]] = None
        self._last_seq: Optional[int] = None
        self._last_wh: Tuple[int, int] = (0, 0)
        self._grid_cols: Optional[int] = None
        self._grid_rows: Optional[int] = None

        self._params = DynMFParams()
        self._trk = DynMedianFlowTracker(self._params)

        self._gray_buf: Optional[np.ndarray] = None
        self._status_last_emit_t: float = 0.0
        self._status_emit_period_s: float = 1.0 / 15.0  # keep UDP load modest
        self._status_last_state: str = ""

    # --------------- lifecycle ---------------
    def start(self) -> None:
        if self._th is not None:
            return
        self._stop.clear()
        th = threading.Thread(target=self._run, name="track2d", daemon=True)
        th.start()
        self._th = th

    def stop(self) -> None:
        try:
            self._stop.set()
        except Exception:
            pass
        th = self._th
        self._th = None
        if th is not None:
            try:
                th.join(timeout=1.0)
            except Exception:
                pass

    # --------------- state ---------------
    def enabled(self) -> bool:
        with self._lock:
            return bool(self._enabled)

    def snapshot_bbox(self) -> Tuple[Optional[Tuple[float, float, float, float]], Tuple[int, int]]:
        """
        Thread-safe snapshot of the current tracked bbox (x,y,w,h in pixels) and the last known image size (W,H).
        Returns (bbox or None, (W,H)).
        """
        with self._lock:
            bb = self._bbox
            wh = self._last_wh
        return (tuple(bb) if bb is not None else None), (int(wh[0]), int(wh[1]))

    def set_enabled(self, on: bool) -> None:
        on_b = bool(on)
        changed = False
        was_enabled = False
        with self._lock:
            was_enabled = bool(self._enabled)
            if bool(was_enabled) != bool(on_b):
                self._enabled = bool(on_b)
                changed = True
            if (not bool(on_b)) and bool(was_enabled):
                self._pending_bbox = None
                self._tracking = False
                self._bbox = None
                self._grid_cols = None
                self._grid_rows = None
        if not changed:
            return
        try:
            if callable(self.on_enable_changed):
                self.on_enable_changed(bool(on_b))
        except Exception:
            pass
        if not bool(on_b):
            try:
                self.send_pid("IDLE", None, None)
            except Exception:
                pass
        self._emit_status(force=True, state="idle", reason=None)

    def cancel(self) -> None:
        with self._lock:
            self._pending_bbox = None
            self._tracking = False
            self._bbox = None
            self._grid_cols = None
            self._grid_rows = None
        try:
            self.send_pid("LOST", None, None)
        except Exception:
            pass
        self._emit_status(force=True, state="idle", reason="cancel")

    def set_rect_norm(
        self,
        *,
        x0_norm: object,
        y0_norm: object,
        x1_norm: object,
        y1_norm: object,
        surface: str = "osd_main",
    ) -> bool:
        if str(surface or "osd_main") != "osd_main":
            return False
        try:
            W, H, fx, fy, cx, cy = self.get_intrinsics()
        except Exception:
            W, H = 0, 0
        if int(W) <= 0 or int(H) <= 0:
            with self._lock:
                W, H = self._last_wh
        if int(W) <= 0 or int(H) <= 0:
            return False

        bbox = _bbox_from_two_norm_corners(
            x0_norm=x0_norm,
            y0_norm=y0_norm,
            x1_norm=x1_norm,
            y1_norm=y1_norm,
            W=int(W),
            H=int(H),
        )
        with self._lock:
            if not bool(self._enabled):
                return False
            self._pending_bbox = bbox
            self._tracking = False
            self._bbox = None
        self._emit_status(force=True, state="idle", reason="roi_pending")
        return True

    def handle_control(self, msg: Dict[str, Any]) -> Dict[str, Any]:
        changed: Dict[str, Any] = {}
        if not isinstance(msg, dict):
            return changed
        def _get_trimmed(key: str, default: Any = None) -> Any:
            """
            Robust lookup for JSON keys that may have accidental surrounding whitespace.
            (We saw clients accidentally emitting `"enable   "` once; accept it.)
            """
            k0 = str(key or "").strip().lower()
            if not k0:
                return default
            try:
                if k0 in msg:
                    return msg.get(k0, default)
            except Exception:
                pass
            try:
                if key in msg:
                    return msg.get(key, default)
            except Exception:
                pass
            try:
                for k, v in msg.items():
                    try:
                        if str(k).strip().lower() == k0:
                            return v
                    except Exception:
                        continue
            except Exception:
                pass
            return default

        t = str(msg.get("type", "") or "").strip().lower()
        if t in ("track2d_enable", "tracker2d_enable", "2d_tracker_enable"):
            en = _get_trimmed("enable", True)
            try:
                en_i = int(en)
                en_b = bool(en_i != 0)
            except Exception:
                en_b = bool(en)
            try:
                already = bool(self.enabled())
            except Exception:
                already = False
            if bool(en_b) == bool(already):
                return changed
            self.set_enabled(bool(en_b))
            changed["track2d_enabled"] = bool(en_b)
            return changed

        if t in ("track2d_cancel", "tracker2d_cancel", "2d_tracker_cancel"):
            self.cancel()
            changed["track2d_cancel"] = True
            return changed

        if t in ("track2d_rect", "track2d_roi", "tracker2d_rect", "tracker2d_roi"):
            surface = str(_get_trimmed("surface", "osd_main") or "osd_main")
            ok = False
            if all((_get_trimmed(k, None) is not None) for k in ("x0_norm", "y0_norm", "x1_norm", "y1_norm")):
                ok = self.set_rect_norm(
                    x0_norm=_get_trimmed("x0_norm", None),
                    y0_norm=_get_trimmed("y0_norm", None),
                    x1_norm=_get_trimmed("x1_norm", None),
                    y1_norm=_get_trimmed("y1_norm", None),
                    surface=surface,
                )
            else:
                bb = _get_trimmed("bbox_norm", None)
                if isinstance(bb, (list, tuple)) and len(bb) >= 4:
                    b = bb
                    x0 = float(b[0])
                    y0 = float(b[1])
                    x1 = float(b[0]) + float(b[2])
                    y1 = float(b[1]) + float(b[3])
                    ok = self.set_rect_norm(x0_norm=x0, y0_norm=y0, x1_norm=x1, y1_norm=y1, surface=surface)
            changed["track2d_roi"] = bool(ok)
            return changed

        return changed

    # --------------- internals ---------------
    def _emit_status(self, *, force: bool, state: str, reason: Optional[str]) -> None:
        jf = self.json_event
        if jf is None:
            return
        now = time.monotonic()
        if not force and (now - float(self._status_last_emit_t)) < float(self._status_emit_period_s):
            return
        self._status_last_emit_t = float(now)
        with self._lock:
            enabled = bool(self._enabled)
            bbox = tuple(self._bbox) if self._bbox is not None else None
            W, H = self._last_wh
            grid_cols = self._grid_cols
            grid_rows = self._grid_rows

        bbox_px = None
        bbox_px_list = None
        bbox_norm = None
        try:
            if bbox is not None:
                x, y, w, h = map(float, bbox)
                bbox_px = {"x": float(x), "y": float(y), "w": float(w), "h": float(h)}
                bbox_px_list = [float(x), float(y), float(w), float(h)]
                if int(W) > 1 and int(H) > 1:
                    x0 = float(x) / float(int(W) - 1)
                    y0 = float(y) / float(int(H) - 1)
                    x1 = float(x + w) / float(int(W) - 1)
                    y1 = float(y + h) / float(int(H) - 1)
                    x0, x1 = (min(x0, x1), max(x0, x1))
                    y0, y1 = (min(y0, y1), max(y0, y1))
                    # Clamp to [0,1].
                    x0 = 0.0 if x0 < 0.0 else (1.0 if x0 > 1.0 else float(x0))
                    y0 = 0.0 if y0 < 0.0 else (1.0 if y0 > 1.0 else float(y0))
                    x1 = 0.0 if x1 < 0.0 else (1.0 if x1 > 1.0 else float(x1))
                    y1 = 0.0 if y1 < 0.0 else (1.0 if y1 > 1.0 else float(y1))
                    bbox_norm = [float(x0), float(y0), float(x1), float(y1)]
        except Exception:
            bbox_px = None
            bbox_px_list = None
            bbox_norm = None
        payload = {
            "type": "track2d",
            "ts": _now_ms(),
            "enabled": bool(enabled),
            "state": str(state),
            "img_size": [int(W), int(H)],
            "bbox_px": bbox_px,
            "bbox_px_list": bbox_px_list,  # back-compat for older clients
            "bbox_norm": bbox_norm,
            "grid_cols": (int(grid_cols) if grid_cols is not None else None),
            "grid_rows": (int(grid_rows) if grid_rows is not None else None),
            "reason": (str(reason) if reason is not None else None),
            "source": "dynmf",
        }
        payload = {k: v for (k, v) in payload.items() if v is not None}
        try:
            jf(payload)
        except Exception:
            pass

    def _run(self) -> None:
        while not self._stop.is_set():
            if not self.enabled():
                time.sleep(0.01)
                continue

            item = None
            try:
                item = self.frame_source.latest()
            except Exception:
                item = None
            if item is None:
                time.sleep(0.005)
                continue

            try:
                if len(item) >= 6:
                    frame, ts_ms, W, H, cam, seq = item[:6]
                else:
                    frame, ts_ms, W, H, cam = item[:5]
                    seq = None
            except Exception:
                time.sleep(0.001)
                continue

            if seq is not None:
                try:
                    seq_i = int(seq)
                except Exception:
                    seq_i = None
                if seq_i is not None:
                    if self._last_seq is not None and int(seq_i) == int(self._last_seq):
                        time.sleep(0.001)
                        continue
                    self._last_seq = int(seq_i)

            try:
                W_i = int(W)
                H_i = int(H)
            except Exception:
                W_i, H_i = 0, 0
            if W_i > 0 and H_i > 0:
                with self._lock:
                    self._last_wh = (int(W_i), int(H_i))

            gray = _gray_u8(frame)
            if gray is None:
                time.sleep(0.001)
                continue

            # Copy into a stable buffer (the source may be a SHM view).
            if self._gray_buf is None or self._gray_buf.shape != gray.shape:
                try:
                    self._gray_buf = np.empty_like(gray)
                except Exception:
                    self._gray_buf = None
                    time.sleep(0.001)
                    continue
            try:
                np.copyto(self._gray_buf, gray)
            except Exception:
                time.sleep(0.001)
                continue

            pending_bbox = None
            with self._lock:
                pending_bbox = self._pending_bbox

            if pending_bbox is not None:
                with self._lock:
                    self._pending_bbox = None
                    self._tracking = False
                    self._bbox = None
                # DynMedianFlowTracker has no explicit reset; create a fresh instance.
                try:
                    self._trk = DynMedianFlowTracker(self._params)
                except Exception:
                    self._trk = DynMedianFlowTracker()
                ok = False
                try:
                    ok = bool(self._trk.init(self._gray_buf, pending_bbox))
                except Exception:
                    ok = False
                with self._lock:
                    self._tracking = bool(ok)
                    self._bbox = tuple(pending_bbox) if bool(ok) else None
                self._emit_status(force=True, state=("tracking" if ok else "lost"), reason=("init" if ok else "init_failed"))
                if not ok:
                    try:
                        self.send_pid("LOST", None, None)
                    except Exception:
                        pass
                continue

            with self._lock:
                tracking = bool(self._tracking)
            if not tracking:
                time.sleep(0.001)
                continue

            ok = False
            bbox = None
            info = None
            try:
                ok, bbox, info = self._trk.update(self._gray_buf, return_info=True)
            except Exception:
                ok = False
                bbox = None
                info = {"reason": "exception"}

            if not ok or bbox is None:
                with self._lock:
                    self._tracking = False
                    self._bbox = None
                    self._grid_cols = None
                    self._grid_rows = None
                reason = None
                try:
                    if isinstance(info, dict):
                        reason = info.get("reason")
                except Exception:
                    reason = None
                self._emit_status(force=True, state="lost", reason=(str(reason) if reason else "lost"))
                try:
                    self.send_pid("LOST", None, None)
                except Exception:
                    pass
                continue

            try:
                x, y, w, h = map(float, bbox)
                cx_px = float(x + 0.5 * w)
                cy_px = float(y + 0.5 * h)
            except Exception:
                cx_px = cy_px = 0.0
                x = y = w = h = 0.0
            with self._lock:
                self._bbox = (float(x), float(y), float(w), float(h))
                try:
                    if isinstance(info, dict):
                        gc = info.get("grid_cols", None)
                        gr = info.get("grid_rows", None)
                        self._grid_cols = (int(gc) if gc is not None else self._grid_cols)
                        self._grid_rows = (int(gr) if gr is not None else self._grid_rows)
                except Exception:
                    pass

            # Convert pixel center -> yaw/pitch errors (radians) using the current intrinsics.
            try:
                intr_W, intr_H, fx, fy, cx0, cy0 = self.get_intrinsics()
            except Exception:
                intr_W, intr_H, fx, fy, cx0, cy0 = (0, 0, 1.0, 1.0, 0.0, 0.0)
            fx = float(fx) if (isinstance(fx, (int, float)) and float(fx) > 1e-6) else 1.0
            fy = float(fy) if (isinstance(fy, (int, float)) and float(fy) > 1e-6) else 1.0
            try:
                yaw_err = math.atan((float(cx_px) - float(cx0)) / float(fx))
                pitch_err = math.atan((float(cy_px) - float(cy0)) / float(fy))
            except Exception:
                yaw_err = 0.0
                pitch_err = 0.0
            try:
                self.send_pid("TRACKING", float(yaw_err), float(pitch_err))
            except Exception:
                pass

            # Best-effort UI telemetry.
            reason = None
            conf = None
            try:
                conf = float(getattr(self._trk, "last_confidence", None))
            except Exception:
                conf = None
            self._emit_status(force=False, state="tracking", reason=reason)
