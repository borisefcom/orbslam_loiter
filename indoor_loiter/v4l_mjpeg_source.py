from __future__ import annotations

import glob
import os
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None


def _now_ms() -> int:
    return int(time.time() * 1000)


def _auto_select_v4l_candidates() -> list[str]:
    """
    Best-effort device auto-detection candidates.

    Prefer a non-RealSense camera (e.g. Arducam OV9281) using stable `/dev/v4l/by-id` symlinks.
    Returns a prioritized list of device paths to try.
    """
    if os.name != "posix":
        return []

    by_id = "/dev/v4l/by-id"
    out: list[str] = []
    rs_nodes: set[str] = set()

    def _append_unique(items: list[str]) -> None:
        for p in items:
            pp = str(p or "").strip()
            if not pp:
                continue
            if pp not in out:
                out.append(pp)

    if os.path.isdir(by_id):
        try:
            entries = sorted(os.listdir(by_id))
        except Exception:
            entries = []

        # Detect RealSense nodes so we can avoid accidentally falling back to them in /dev/video* order.
        for name in entries:
            nl = str(name).lower()
            if ("realsense" not in nl) and ("intel" not in nl):
                continue
            try:
                rp = os.path.realpath(os.path.join(by_id, name))
            except Exception:
                rp = ""
            if rp and rp.startswith("/dev/video"):
                rs_nodes.add(str(rp))

        # Prefer index0 nodes (usually the primary capture node), but also try index1.
        for idx_tag in ("video-index0", "video-index1"):
            cand: list[str] = []
            for name in entries:
                nl = str(name).lower()
                if idx_tag not in nl:
                    continue
                # Exclude RealSense video nodes explicitly.
                if ("realsense" in nl) or ("intel" in nl):
                    continue
                cand.append(os.path.join(by_id, name))
            # Prefer Arducam/OV9281 explicitly if present.
            prefer = [p for p in cand if ("arducam" in p.lower() or "ov9281" in p.lower())]
            rest = [p for p in cand if p not in prefer]
            # For each by-id symlink, also add its realpath (/dev/videoN) as a candidate.
            # Some OpenCV builds are picky about opening the /dev/v4l/by-id symlink directly.
            for p in prefer + rest:
                _append_unique([p])
                try:
                    _append_unique([os.path.realpath(p)])
                except Exception:
                    pass

    # Fallback: try /dev/video* nodes (best-effort). This is intentionally *last* because ordering is unstable
    # and often places RealSense at /dev/video0.
    vids = [v for v in sorted(glob.glob("/dev/video[0-9]*")) if str(v) not in rs_nodes]
    _append_unique(list(map(str, vids)))
    return out


def _auto_select_v4l_device() -> Optional[str]:
    cand = _auto_select_v4l_candidates()
    return str(cand[0]) if cand else None


def _resolve_v4l_device(dev: str) -> str:
    d = str(dev or "").strip()
    if not d or d.lower() in ("auto", "detect", "any"):
        picked = _auto_select_v4l_device()
        return str(picked or "/dev/video0")
    return d


@dataclass
class V4lMjpegCfg:
    device: str = "auto"  # "/dev/videoN" | "/dev/v4l/by-id/..." | "auto"
    width: int = 640
    height: int = 480
    fps: int = 90
    fourcc: str = "MJPG"
    backend: str = "auto"  # "auto" | "gstreamer" | "v4l2"
    v4l2ctl_set_format: bool = True
    v4l2ctl_set_fps: bool = True
    controls: Optional[Dict[str, Any]] = None
    frame_kind: str = "gray"  # "gray" | "bgr"
    name: str = "v4l"


class V4lMjpegSource:
    """
    Lightweight OpenCV VideoCapture source for a V4L2 MJPEG camera.

    Provides a FrameBus-compatible `.latest()` method:
      (frame, ts_ms, W, H, cam_name, seq)

    Notes:
    - Intended for the non-RealSense tracking camera (e.g., Arducam OV9281 @ high FPS).
    - Uses a background capture thread and keeps only the most-recent frame (dropping old frames).
    """

    def __init__(self, cfg: V4lMjpegCfg, *, print_fn=print) -> None:
        if cv2 is None:
            raise RuntimeError("OpenCV (cv2) not available; cannot open V4L camera")
        self.cfg = cfg
        self.print = print_fn

        self._stop = threading.Event()
        self._th: Optional[threading.Thread] = None
        self._cap = None
        self._backend_eff: str = ""

        self._lock = threading.Lock()
        self._latest: Optional[Tuple[Any, int, int, int, str, int]] = None
        self._seq: int = 0
        self._opened: bool = False
        self._device_req: str = str(getattr(cfg, "device", "") or "").strip()
        self._device_eff: str = self._device_req

    def get_device(self) -> str:
        return str(self._device_eff or self._device_req or getattr(self.cfg, "device", "") or "")

    def get_backend(self) -> str:
        return str(self._backend_eff or getattr(self.cfg, "backend", "") or "")

    def start(self) -> None:
        if self._th is not None:
            return
        self._stop.clear()
        self._open()
        th = threading.Thread(target=self._run, name="v4l-capture", daemon=True)
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
        self._close()

    def latest(self) -> Optional[Tuple[Any, int, int, int, str, int]]:
        with self._lock:
            return self._latest

    def get_size(self) -> Tuple[int, int]:
        with self._lock:
            if self._latest is None:
                return int(self.cfg.width), int(self.cfg.height)
            return int(self._latest[2]), int(self._latest[3])

    # -------- internals --------
    def _apply_v4l2ctl(self) -> None:
        # Best-effort: v4l2-ctl is Linux-only.
        if os.name != "posix":
            return
        dev = str(self._device_eff or getattr(self.cfg, "device", "") or "").strip()
        dev_ctl = os.path.realpath(dev)
        if not dev_ctl.startswith("/dev/video"):
            return
        try:
            import shutil

            v4l2ctl = shutil.which("v4l2-ctl")
        except Exception:
            v4l2ctl = None
        if not v4l2ctl:
            return

        w = int(self.cfg.width)
        h = int(self.cfg.height)
        fps = int(self.cfg.fps)
        pix = str(self.cfg.fourcc or "MJPG").strip().upper()

        if bool(self.cfg.v4l2ctl_set_format):
            try:
                args = [v4l2ctl, "-d", dev_ctl, f"--set-fmt-video=width={w},height={h},pixelformat={pix}"]
                subprocess.run(args, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except Exception:
                pass
        if bool(self.cfg.v4l2ctl_set_fps):
            try:
                args = [v4l2ctl, "-d", dev_ctl, f"--set-parm={fps}"]
                subprocess.run(args, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except Exception:
                pass
        ctrls = self.cfg.controls
        if isinstance(ctrls, dict) and ctrls:
            try:
                parts = []
                for k, v in ctrls.items():
                    parts.append(f"{k}={v}")
                args = [v4l2ctl, "-d", dev_ctl, "-c", ",".join(parts)]
                subprocess.run(args, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except Exception:
                pass

    def _open(self) -> None:
        self._device_req = str(getattr(self.cfg, "device", "") or "").strip()
        dev_req = str(self._device_req or "").strip()
        dev_eff = _resolve_v4l_device(dev_req)

        backend_req = str(getattr(self.cfg, "backend", "auto") or "auto").strip().lower()
        if backend_req in ("", "auto", "any", "detect"):
            backend_order = ["gstreamer", "v4l2"]
        elif backend_req in ("gst", "gstreamer"):
            backend_order = ["gstreamer", "v4l2"]
        elif backend_req in ("v4l2", "opencv", "cv2"):
            backend_order = ["v4l2"]
        else:
            backend_order = ["gstreamer", "v4l2"]

        # If the user requested "auto", try a small candidate set (index0 + index1 + fallback /dev/video*)
        # and pick the first that can be opened and yields frames.
        candidates = [dev_eff]
        if not dev_req or dev_req.lower() in ("auto", "detect", "any"):
            try:
                candidates = _auto_select_v4l_candidates()
            except Exception:
                candidates = [dev_eff]

        last_err: Optional[str] = None
        for cand in candidates:
            dev = str(cand or "").strip()
            if not dev:
                continue
            try:
                self._device_eff = dev
                self._backend_eff = ""
                self._apply_v4l2ctl()

                cap = None
                cap_backend = ""

                for be in backend_order:
                    be = str(be or "").strip().lower()
                    if not be:
                        continue

                    if be in ("gst", "gstreamer"):
                        if not hasattr(cv2, "CAP_GSTREAMER"):
                            last_err = f"no_gstreamer_support eff={dev!r}"
                            continue
                        try:
                            w = int(self.cfg.width)
                            h = int(self.cfg.height)
                            fps = int(self.cfg.fps)
                            want_gray = str(self.cfg.frame_kind or "gray").strip().lower() in (
                                "gray",
                                "grey",
                                "mono",
                                "gray8",
                            )
                            fmt = "GRAY8" if want_gray else "BGR"
                            dev_gst = str(dev)
                            try:
                                if os.name == "posix":
                                    dev_gst = os.path.realpath(str(dev_gst))
                            except Exception:
                                dev_gst = str(dev)
                            # v4l2src -> MJPEG -> decode -> appsink (drop old frames; keep latency low)
                            pipeline = (
                                f"v4l2src device={dev_gst} ! "
                                f"image/jpeg,width={w},height={h},framerate={fps}/1 ! "
                                "jpegdec ! "
                                "videoconvert ! "
                                f"video/x-raw,format={fmt} ! "
                                "queue leaky=downstream max-size-buffers=1 ! "
                                "appsink drop=true max-buffers=1 sync=false"
                            )
                            cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
                            cap_backend = "gstreamer"
                        except Exception as e:
                            last_err = f"gst_open_exception eff={dev!r} err={e!r}"
                            try:
                                if cap is not None:
                                    cap.release()
                            except Exception:
                                pass
                            cap = None
                            continue

                    elif be in ("v4l2", "opencv", "cv2"):
                        try:
                            backend = cv2.CAP_V4L2 if hasattr(cv2, "CAP_V4L2") else 0
                            cap = cv2.VideoCapture(dev, backend)
                            cap_backend = "v4l2"
                        except Exception as e:
                            last_err = f"v4l2_open_exception eff={dev!r} err={e!r}"
                            try:
                                if cap is not None:
                                    cap.release()
                            except Exception:
                                pass
                            cap = None
                            continue
                    else:
                        continue

                    if not cap or not cap.isOpened():
                        last_err = f"open_failed backend={cap_backend} eff={dev!r}"
                        try:
                            if cap is not None:
                                cap.release()
                        except Exception:
                            pass
                        cap = None
                        continue

                    if cap_backend == "v4l2":
                        fourcc = str(self.cfg.fourcc or "MJPG").strip().upper()
                        try:
                            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
                        except Exception:
                            pass
                        try:
                            cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(self.cfg.width))
                            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(self.cfg.height))
                        except Exception:
                            pass
                        try:
                            cap.set(cv2.CAP_PROP_FPS, int(self.cfg.fps))
                        except Exception:
                            pass
                        try:
                            if hasattr(cv2, "CAP_PROP_BUFFERSIZE"):
                                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                        except Exception:
                            pass

                    # Validate that frames flow (some devices expose multiple nodes where only one is the actual stream).
                    ok0, fr0 = False, None
                    try:
                        ok0, fr0 = cap.read()
                    except Exception:
                        ok0, fr0 = False, None
                    if not ok0 or fr0 is None:
                        last_err = f"read_failed backend={cap_backend} eff={dev!r}"
                        try:
                            cap.release()
                        except Exception:
                            pass
                        cap = None
                        continue

                    self._backend_eff = str(cap_backend)
                    self._cap = cap
                    self._opened = True
                    try:
                        got_w = int(fr0.shape[1]) if hasattr(fr0, "shape") else int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        got_h = int(fr0.shape[0]) if hasattr(fr0, "shape") else int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        got_fps = float(cap.get(cv2.CAP_PROP_FPS))
                        self.print(
                            f"[track2d] V4L open backend={cap_backend} req={self._device_req!r} eff={dev!r} "
                            f"req={int(self.cfg.width)}x{int(self.cfg.height)}@{int(self.cfg.fps)} "
                            f"got={got_w}x{got_h}@{got_fps:.1f}",
                            flush=True,
                        )
                    except Exception:
                        pass
                    return
            except Exception as e:
                last_err = str(e)
                try:
                    if getattr(self, "_cap", None) is not None:
                        self._close()
                except Exception:
                    pass
                continue

        cand_s = ", ".join([repr(str(c)) for c in candidates[:8]])
        more = "" if len(candidates) <= 8 else f" (+{len(candidates) - 8} more)"
        raise RuntimeError(
            f"Failed to open V4L camera: requested={self._device_req!r} candidates=[{cand_s}]{more} last_err={last_err!r}"
        )

    def _close(self) -> None:
        cap = self._cap
        self._cap = None
        self._opened = False
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass

    def _run(self) -> None:
        cap = self._cap
        if cap is None:
            return
        want_gray = str(self.cfg.frame_kind or "gray").strip().lower() in ("gray", "grey", "mono", "gray8")
        name = str(self.cfg.name or "v4l")
        target_w = int(getattr(self.cfg, "width", 0) or 0)
        target_h = int(getattr(self.cfg, "height", 0) or 0)
        while not self._stop.is_set():
            ok = False
            frame = None
            try:
                ok, frame = cap.read()
            except Exception:
                ok, frame = False, None
            if not ok or frame is None:
                time.sleep(0.005)
                continue
            ts = _now_ms()
            try:
                H, W = int(frame.shape[0]), int(frame.shape[1])
            except Exception:
                H, W = int(self.cfg.height), int(self.cfg.width)

            # Normalize format first (gray/bgr), then enforce output resolution for downstream consumers
            # (the streaming pipeline caps are fixed at startup).
            if want_gray:
                try:
                    if frame.ndim == 3 and frame.shape[2] >= 3:
                        frame = cv2.cvtColor(frame[:, :, :3], cv2.COLOR_BGR2GRAY)
                    elif frame.ndim == 3 and frame.shape[2] == 1:
                        frame = frame[:, :, 0]
                except Exception:
                    pass
            else:
                try:
                    if frame.ndim == 2:
                        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                    elif frame.ndim == 3 and frame.shape[2] > 3:
                        frame = frame[:, :, :3]
                except Exception:
                    pass

            try:
                if int(target_w) > 0 and int(target_h) > 0:
                    h0, w0 = (int(frame.shape[0]), int(frame.shape[1])) if hasattr(frame, "shape") else (int(H), int(W))
                    if int(w0) != int(target_w) or int(h0) != int(target_h):
                        interp = cv2.INTER_AREA if int(w0) > int(target_w) or int(h0) > int(target_h) else cv2.INTER_LINEAR
                        frame = cv2.resize(frame, (int(target_w), int(target_h)), interpolation=interp)
                        W, H = int(target_w), int(target_h)
                    else:
                        W, H = int(w0), int(h0)
            except Exception:
                pass

            with self._lock:
                self._seq += 1
                self._latest = (frame, int(ts), int(W), int(H), name, int(self._seq))
