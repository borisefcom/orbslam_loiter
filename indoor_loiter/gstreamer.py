#!/usr/bin/env python3
import os, threading, time, math
from collections import deque
from typing import Optional, Tuple, Dict, Any

try:
    import gi
    gi.require_version('Gst', '1.0')
    from gi.repository import Gst, GLib
    _HAS_GI = True
except Exception:
    Gst = None
    GLib = None
    _HAS_GI = False

try:
    if _HAS_GI:
        gi.require_version("GstRtspServer", "1.0")
        from gi.repository import GstRtspServer  # type: ignore
        _HAS_RTSP = True
    else:
        GstRtspServer = None  # type: ignore
        _HAS_RTSP = False
except Exception:
    GstRtspServer = None  # type: ignore
    _HAS_RTSP = False

import cv2
try:
    import numpy as np  # required for RealSense capture
except Exception:
    np = None

from tracker import FrameBus  # publish frames + IMU into the tracker

# Optional RealSense
try:
    import pyrealsense2 as rs
    _HAS_RS = True
except Exception:
    rs = None
    _HAS_RS = False


def _kbps_to_bps(kbps:int)->int:
    return int(max(100, int(kbps))) * 1000

def _cfg_get(cfg, dotted: str, default=None):
    """
    Safe getter for both dict and Config objects with dotted keys.
    """
    try:
        return cfg.get(dotted, default)
    except Exception:
        pass
    try:
        cur = cfg
        for part in dotted.split("."):
            if isinstance(cur, dict) and part in cur:
                cur = cur[part]
            else:
                return default
        return cur
    except Exception:
        return default

def _has_prop(elem, name: str) -> bool:
    try:
        for p in elem.list_properties():
            if p.name == name:
                return True
    except Exception:
        pass
    return False

def _apply_imu_map(map_cfg: Dict[str, Any], units: str, x: float, y: float, z: float) -> Tuple[float, float, float]:
    """
    Map/flip IMU axes and convert units (dps->rad/s) into tracker-expected frame.
    map_cfg example: {"x": "z", "y": "-x", "z": "y"}
    units: "rad_s" (default) or "dps"
    """
    axes = {"x": float(x), "y": float(y), "z": float(z)}
    def _one(key: str) -> float:
        raw = map_cfg.get(key, key)
        sign = -1.0 if isinstance(raw, str) and raw.startswith("-") else 1.0
        src = raw[1:] if isinstance(raw, str) and raw.startswith("-") else raw
        src = str(src).lower()
        return sign * axes.get(src, 0.0)
    if not isinstance(map_cfg, dict):
        mapped = (axes["x"], axes["y"], axes["z"])
    else:
        mapped = (_one("x"), _one("y"), _one("z"))
    if str(units).lower() in ("dps", "deg", "deg_s", "deg/s"):
        mapped = tuple(v * math.pi / 180.0 for v in mapped)
    return mapped

class _RealSenseCapture:
    """
    Minimal RealSense color + optional IMU capture that feeds FrameBus.
    Lives in this file to keep VideoController as a single entry-point.
    """
    def __init__(self, cfg: Dict[str, Any], framebus: FrameBus):
        self.cfg = cfg
        self.fb = framebus
        ccfg = _cfg_get(cfg, "camera", {}) or {}
        rscfg = _cfg_get(cfg, "camera.realsense", {}) or {}
        self.req_W = int(rscfg.get("width", ccfg.get("width", 848)))
        self.req_H = int(rscfg.get("height", ccfg.get("height", 480)))
        self.req_FPS = int(rscfg.get("fps", ccfg.get("fps", 30)))
        self.preproc = rscfg.get("preprocess", {}) or {}
        self.sharpen_enable = bool(self.preproc.get("sharpen_enable", True))
        self.sharp_amount   = float(self.preproc.get("amount", 0.30))
        self.sharp_sigma    = float(self.preproc.get("sigma", 1.0))

        imu_cfg = (rscfg.get("imu") or ccfg.get("imu") or {})
        self.imu_map = imu_cfg.get("map", {})
        self.imu_units = imu_cfg.get("units", imu_cfg.get("unit", "rad_s"))
        # RealSense accel reports specific force; invert to treat it as gravity-down for our fusion.
        self.imu_accel_invert = bool(imu_cfg.get("accel_invert", False))

        self.running = False
        self._pipe = None
        self._color_fmt = rs.format.rgb8 if _HAS_RS else None
        self._imu_sensor = None
        self._th_color = None
        self._fps_ema = 0.0
        self._fps_alpha = 0.1
        self._candidates = []
        self._cand_idx = 0

        # Factory intrinsics from active color profile (queried on open)
        self.intr_W: Optional[int] = None
        self.intr_H: Optional[int] = None
        self.intr_fx: Optional[float] = None
        self.intr_fy: Optional[float] = None
        self.intr_cx: Optional[float] = None
        self.intr_cy: Optional[float] = None
        self.intr_model: Optional[str] = None
        self.intr_coeffs: Optional[list] = None

    # ---------- helpers ----------
    @staticmethod
    def _unsharp_luma(bgr, amount, sigma):
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        L,A,B = cv2.split(lab)
        blur = cv2.GaussianBlur(L, (0,0), sigmaX=float(sigma), sigmaY=float(sigma))
        sh   = cv2.addWeighted(L, 1.0+float(amount), blur, -float(amount), 0.0)
        lab2 = cv2.merge([sh,A,B])
        return cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)

    def fps(self) -> float:
        return float(self._fps_ema)

    # ---------- lifecycle ----------
    def open(self):
        if np is None:
            raise RuntimeError("numpy required for RealSense capture")
        if not _HAS_RS:
            raise RuntimeError("pyrealsense2 not available")
        ctx = rs.context()
        devs = ctx.query_devices()
        if devs.size() == 0:
            raise RuntimeError("No RealSense devices found")
        self._rs_dev = devs[0]
        serial = self._rs_dev.get_info(rs.camera_info.serial_number)
        print(f"[RS] using device serial {serial}", flush=True)

        # Start only the configured mode (no fallbacks)
        fmt = rs.format.bgr8
        print(f"[RS] starting color {self.req_W}x{self.req_H}@{self.req_FPS} fmt={fmt}", flush=True)
        try:
            cfg_rs = rs.config()
            cfg_rs.enable_device(serial)
            cfg_rs.enable_stream(rs.stream.color, int(self.req_W), int(self.req_H), fmt, int(self.req_FPS))
            self._pipe = rs.pipeline(ctx)
            self._pipe.start(cfg_rs)
            self._color_fmt = fmt
            try:
                prof = self._pipe.get_active_profile().get_stream(rs.stream.color).as_video_stream_profile()
                print(f"[RS] active color {prof.width()}x{prof.height()}@{prof.fps()}", flush=True)
                try:
                    intr = prof.get_intrinsics()
                    self.intr_W = int(intr.width)
                    self.intr_H = int(intr.height)
                    self.intr_fx = float(intr.fx)
                    self.intr_fy = float(intr.fy)
                    self.intr_cx = float(intr.ppx)
                    self.intr_cy = float(intr.ppy)
                    try:
                        self.intr_model = str(intr.model)
                    except Exception:
                        self.intr_model = None
                    try:
                        self.intr_coeffs = list(intr.coeffs)
                    except Exception:
                        self.intr_coeffs = None
                    print(
                        f"[RS] intrinsics: {self.intr_W}x{self.intr_H} "
                        f"fx={self.intr_fx:.2f} fy={self.intr_fy:.2f} cx={self.intr_cx:.2f} cy={self.intr_cy:.2f}",
                        flush=True,
                    )
                except Exception as e:
                    print(f"[RS] intrinsics query failed: {e}", flush=True)
            except Exception:
                pass
        except Exception as e:
            self._pipe = None
            self._color_fmt = None
            raise RuntimeError(f"RealSense start failed for {self.req_W}x{self.req_H}@{self.req_FPS}: {e}")

        # Try to grab motion module (gyro/accel) similar to working CameraBus
        try:
            self._imu_sensor = None
            mm = None
            for s in self._rs_dev.query_sensors():
                try:
                    name = s.get_info(rs.camera_info.name)
                except Exception:
                    name = ""
                if "motion" in name.lower():
                    mm = s
                    break
            if mm is None:
                print("[RS][IMU] No Motion Module sensor found. IMU OFF.", flush=True)
            else:
                gyro_prof = None
                accel_prof = None
                try:
                    for p in mm.get_stream_profiles():
                        try:
                            vsp = rs.video_stream_profile(p)
                            st  = vsp.stream_type()
                            if st == rs.stream.gyro:
                                if gyro_prof is None or vsp.fps() > rs.video_stream_profile(gyro_prof).fps():
                                    gyro_prof = p
                            elif st == rs.stream.accel:
                                if accel_prof is None or vsp.fps() > rs.video_stream_profile(accel_prof).fps():
                                    accel_prof = p
                        except Exception:
                            continue
                except Exception as e:
                    print(f"[RS][IMU] get_stream_profiles failed: {e}", flush=True)
                if gyro_prof is None:
                    print("[RS][IMU] No gyro profile found. IMU OFF.", flush=True)
                else:
                    try:
                        if accel_prof is not None:
                            mm.open([gyro_prof, accel_prof])
                        else:
                            mm.open(gyro_prof)

                        def _cb(frame: "rs.frame"):
                            if not frame.is_motion_frame():
                                return
                            ts_host = time.monotonic()
                            try:
                                dev_ts = float(frame.get_timestamp()) / 1000.0
                            except Exception:
                                dev_ts = ts_host
                            try:
                                st = frame.get_profile().as_video_stream_profile().stream_type()
                            except Exception:
                                st = (
                                    rs.stream.gyro
                                    if frame.get_profile().stream_type() == rs.stream.gyro
                                    else rs.stream.accel
                                )
                            data = frame.as_motion_frame().get_motion_data()
                            kind = ("gyro" if st == rs.stream.gyro else "accel")
                            mx, my, mz = _apply_imu_map(self.imu_map, self.imu_units, data.x, data.y, data.z)
                            if kind == "accel" and self.imu_accel_invert:
                                mx, my, mz = -mx, -my, -mz
                            self.fb.push_imu(kind, ts_host, dev_ts, mx, my, mz)

                        mm.start(_cb)
                        self._imu_sensor = mm
                        gfps = rs.video_stream_profile(gyro_prof).fps()
                        if accel_prof is not None:
                            afps = rs.video_stream_profile(accel_prof).fps()
                            print(f"[RS][IMU] started gyro~{gfps} Hz accel~{afps} Hz", flush=True)
                        else:
                            print(f"[RS][IMU] started gyro-only @ ~{gfps} Hz", flush=True)
                    except Exception as e:
                        print(f"[RS][IMU] start failed: {e}", flush=True)
                        try:
                            if accel_prof is not None:
                                mm.close([gyro_prof, accel_prof])
                            else:
                                mm.close(gyro_prof)
                        except Exception:
                            pass
                        self._imu_sensor = None
        except Exception as e:
            print(f"[RS][IMU] setup exception: {e}", flush=True)
            self._imu_sensor = None
        self.running = True
        self._th_color = threading.Thread(target=self._color_loop, name="rs-color", daemon=True)
        self._th_color.start()

    def stop(self):
        self.running = False
        try:
            if self._imu_sensor:
                self._imu_sensor.stop()
        except Exception:
            pass
        try:
            if self._th_color:
                self._th_color.join(timeout=1.0)
        except Exception:
            pass
        try:
            if self._pipe:
                self._pipe.stop()
        except Exception:
            pass
        self._pipe = None
        self._imu_sensor = None
        self._th_color = None
        self._last_cfg = None

    # ---------- loops ----------
    def _color_loop(self):
        dbg_first_frame = False
        last_log = time.time()
        while self.running and self._pipe is not None:
            try:
                fs = self._pipe.wait_for_frames()
                c = fs.get_color_frame() if fs else None
                if not c:
                    time.sleep(0.0005); continue
                ts_mono = time.monotonic()
                ts_wall = time.time()
                try:
                    bgr = np.asanyarray(c.get_data()).copy()
                    if self._color_fmt == rs.format.rgb8:
                        bgr = bgr[:, :, ::-1].copy()
                except Exception as e:
                    print(f"[RS] frame convert error: {e}", flush=True)
                    time.sleep(0.0005); continue
                if self.sharpen_enable:
                    bgr = self._unsharp_luma(bgr, self.sharp_amount, self.sharp_sigma)
                # FrameBus timestamps are host wall-clock ms to keep age_ms consistent across backends.
                self.fb.update(bgr, int(ts_wall * 1000), bgr.shape[1], bgr.shape[0], cam="realsense")
                if not dbg_first_frame:
                    print(f"[RS] first frame {bgr.shape[1]}x{bgr.shape[0]} at {ts_mono:.3f}", flush=True)
                    dbg_first_frame = True
                elif (time.time() - last_log) > 2.0:
                    print(f"[RS] frames flowing, fps_est~{self.fps():.1f}", flush=True)
                    last_log = time.time()
                # FPS EMA
                if getattr(self, "_last_ts", None) is not None:
                    dt = ts_mono - self._last_ts
                    if dt > 1e-3:
                        self._fps_ema = (1.0/dt if self._fps_ema == 0 else (self._fps_alpha*(1.0/dt)+(1-self._fps_alpha)*self._fps_ema))
                self._last_ts = ts_mono
                time.sleep(0.0005)
            except Exception as e:
                try:
                    print(f"[RS] color loop error: {e}", flush=True)
                except Exception:
                    pass
                time.sleep(0.01)


class VideoController:
    def __init__(self, cfg):
        self.cfg = cfg
        cam_type = _cfg_get(cfg, "camera.type", _cfg_get(cfg, "video.source", "realsense"))
        self.source = str(cam_type or "realsense").lower().strip()
        if self.source != "realsense":
            raise RuntimeError(f"Only RealSense is supported (camera.type must be 'realsense', got '{self.source}').")

        # RealSense capture uses pyrealsense2; streaming uses GStreamer (appsrc).
        if _HAS_GI:
            Gst.init(None)

        # Camera dims/fps with backend overrides
        common_w = int(_cfg_get(cfg, "camera.width", _cfg_get(cfg, "video.width", 1280)))
        common_h = int(_cfg_get(cfg, "camera.height", _cfg_get(cfg, "video.height", 720)))
        common_fps = int(_cfg_get(cfg, "camera.fps", _cfg_get(cfg, "video.fps", 60)))

        self.width  = int(_cfg_get(cfg, "camera.realsense.width", common_w))
        self.height = int(_cfg_get(cfg, "camera.realsense.height", common_h))
        self.capture_fps = int(_cfg_get(cfg, "camera.realsense.fps", common_fps))

        # Desired stream framerate (encoder / UDP). Defaults to capture FPS.
        self.stream_fps = int(_cfg_get(cfg, "video.fps", self.capture_fps))
        if self.stream_fps < 1:
            self.stream_fps = int(self.capture_fps) if int(self.capture_fps) >= 1 else 30
        # NVENC on Jetson supports up to 60fps for H264/H265; clamp to avoid encoder warnings / fallback.
        if str(_cfg_get(cfg, "video.codec", "h265")).lower() in ("h264", "h265", "hevc") and self.stream_fps > 60:
            print(f"[STREAM] stream_fps={self.stream_fps} > 60 not supported by NVENC; clamping to 60", flush=True)
            self.stream_fps = 60

        # Back-compat: some logs reference self.fps as capture target.
        self.fps = int(self.capture_fps)

        self.codec  = _cfg_get(cfg, "video.codec","h265")
        # Encoder preference:
        #   - "auto" (default): use NVENC where applicable
        #   - "software": force CPU encoder (e.g. x264enc for h264)
        #   - "x264": same as software for h264
        self.encoder = str(_cfg_get(cfg, "video.encoder", "auto") or "auto").strip().lower()
        # Software H.264 (x264enc) tuning (used when video.encoder forces software).
        self.x264_preset = str(_cfg_get(cfg, "video.x264_preset", "ultrafast") or "ultrafast").strip()
        self.x264_tune = str(_cfg_get(cfg, "video.x264_tune", "zerolatency") or "zerolatency").strip()
        self.h264_profile = str(_cfg_get(cfg, "video.h264_profile", "baseline") or "").strip()
        # Set a more realistic default bitrate for 720p
        self.bitrate_kbps = int(_cfg_get(cfg, "video.bitrate_kbps",4000))
        self.rc_mode = _cfg_get(cfg, "video.rc_mode","CBR")
        # Bump iframeinterval default to ~0.5s at 60fps (or overridden by cfg)
        self.iframeinterval = int(_cfg_get(cfg, "video.iframeinterval",30))
        self.idrinterval    = int(_cfg_get(cfg, "video.idrinterval",30))
        self.udp_sink       = _cfg_get(cfg, "video.udp_sink","127.0.0.1:5600")
        # Stream output method: "udp" (legacy) or "rtsp"
        self.stream_method = str(_cfg_get(cfg, "video.stream_method", "udp") or "udp").strip().lower()
        if self.stream_method in ("rtsp_server", "rtspserver"):
            self.stream_method = "rtsp"
        if self.stream_method not in ("udp", "rtsp"):
            self.stream_method = "udp"
        # RTSP server settings
        self.rtsp_port = int(_cfg_get(cfg, "video.rtsp.port", 8554))
        self.rtsp_path = str(_cfg_get(cfg, "video.rtsp.path", "stream") or "stream").strip().lstrip("/")
        self.rtsp_bind = str(_cfg_get(cfg, "video.rtsp.bind", "0.0.0.0") or "0.0.0.0").strip()
        self.rtsp_advertise_host = str(_cfg_get(cfg, "video.rtsp.advertise_host", "") or "").strip()
        # Optional sensors/pose (IMU + positional tracking). If capture.enable_sensors is not set,
        # fall back to imu.enabled to keep YAML simpler for the drone scenario.
        self.enable_sensors = bool(_cfg_get(cfg, "capture.enable_sensors", _cfg_get(cfg, "imu.enabled", False)))
        # Optional decimation for streaming: only enqueue every Nth frame
        self.stream_every_n = int(_cfg_get(cfg, "video.stream_every_n", 1))
        if self.stream_every_n < 1:
            self.stream_every_n = 1
        # Feed streamer from FrameBus instead of capture enqueue (default True)
        self.stream_from_framebus = bool(_cfg_get(cfg, "video.stream_from_framebus", True))
        # If true: do not open RealSense; stream from an external shared-memory frame source.
        self.external_capture = bool(_cfg_get(cfg, "video.external_capture", _cfg_get(cfg, "track3d.enabled", False)))
        # For external capture, control what the server expects to push into appsrc.
        # - "gray8": mono 8-bit (preferred for IR1-only pipelines)
        # - "bgr":   3-channel color
        self.external_raw_format = str(
            _cfg_get(cfg, "video.external_raw_format", "gray8" if self.external_capture else "bgr") or ""
        ).strip().lower()
        # Streaming enabled unless explicitly disabled in config
        self.stream_enable  = bool(_cfg_get(cfg, "video.stream_enable", True) and not _cfg_get(cfg, "video.disable_stream", False))
        # Optional: log GST perf once/sec (defaults to False to reduce log noise)
        self.gst_perf_log = bool(_cfg_get(cfg, "video.gst_perf_log", False))

        self.pipeline = None
        self.appsrc = None
        self.enc = None
        self.sink = None
        # Raw format expected by appsrc for streaming (set when building the pipeline).
        self._appsrc_raw_format = "BGRx"
        self._glib_loop = None
        self._glib_th = None
        self._rtsp_server = None
        self._rtsp_factory = None
        self._rtsp_attach_id = 0

        # Frame bus for detector/tracker
        self._framebus: Optional[FrameBus] = None

        # Intrinsics (queried from camera backend)
        self._intr_W: Optional[int] = None
        self._intr_H: Optional[int] = None
        self._intr_fx: Optional[float] = None
        self._intr_fy: Optional[float] = None
        self._intr_cx: Optional[float] = None
        self._intr_cy: Optional[float] = None

        self.capture_th = None
        self.capturing = False
        self._lock = threading.Lock()
        # Streaming decoupling: bounded queue + streamer thread
        self._stream_q = deque(maxlen=3)
        self._stream_cv = threading.Condition()
        self._stream_th = None

        # Streaming decimation counter
        self._stream_seq = 0
        self._stream_last_fb_seq = 0
        self._stream_last_push_t = 0.0

        # --- camera FPS estimator (EMA) ---
        self._t_prev_frame = None
        self._cam_fps_ema = 0.0
        self._cam_fps_alpha = 0.1
        self._cap_dt_hist = deque(maxlen=240)

        # RealSense backend
        self._rs_cap: Optional[_RealSenseCapture] = None
        self.backend = self.source

        # GStreamer perf tracking
        self._gst_enqueue_dt = deque(maxlen=240)
        self._gst_push_dt = deque(maxlen=240)
        self._gst_last_log = 0.0
        self._gst_q_max = 0

        # FrameBus publish diagnostics (to detect silent drops)
        self._fb_update_errs = 0
        self._fb_update_last_err_t = 0.0

    # ---------- external API for tracker ----------
    def attach_framebus(self, fb: FrameBus):
        self._framebus = fb

    def get_cam_fps(self) -> float:
        if self.backend == "realsense" and self._rs_cap:
            return float(self._rs_cap.fps())
        return float(self._cam_fps_ema)

    def get_intrinsics(self) -> Tuple[int, int, float, float, float, float]:
        """
        Return (W, H, fx, fy, cx, cy) for the currently configured stream.
        For the drone scenario we require factory intrinsics from the active camera backend.
        """
        W = self._intr_W
        H = self._intr_H
        fx = self._intr_fx
        fy = self._intr_fy
        cx = self._intr_cx
        cy = self._intr_cy
        if W is None or H is None or fx is None or fy is None or cx is None or cy is None:
            raise RuntimeError("Camera intrinsics not available (camera backend must provide calibration).")
        return int(W), int(H), float(fx), float(fy), float(cx), float(cy)

    def set_intrinsics(self, W: int, H: int, fx: float, fy: float, cx: float, cy: float) -> None:
        """Set intrinsics from an external camera owner (e.g., 3d_track capture)."""
        self._intr_W = int(W)
        self._intr_H = int(H)
        self._intr_fx = float(fx)
        self._intr_fy = float(fy)
        self._intr_cx = float(cx)
        self._intr_cy = float(cy)

    def get_capture_dt_avg(self) -> float:
        """Average capture delta (s) over recent samples."""
        if not self._cap_dt_hist:
            return 0.0
        try:
            return float(sum(self._cap_dt_hist) / len(self._cap_dt_hist))
        except Exception:
            return 0.0

    # ---------- Preflight (RealSense) ----------
    def preflight_camera_check(self):
        if np is None:
            return False, "numpy not installed (required for RealSense capture)"
        if not _HAS_RS:
            return False, "pyrealsense2 not installed"
        try:
            ctx = rs.context()
            devs = ctx.query_devices()
            if len(devs) == 0:
                return False, "no RealSense devices found"
            return True, f"{len(devs)} device(s) present"
        except Exception as e:
            return False, f"exception: {e}"

    # ---------- Pipeline ----------
    def _codec_chain(self, codec:str):
        want = (codec or "").strip().lower()
        if want == "h264":
            if self.encoder in ("software", "sw", "cpu", "x264", "x264enc"):
                # Prefer x264enc for full software encode; fall back to openh264enc if needed.
                try:
                    if Gst is not None and Gst.ElementFactory.find("x264enc") is not None:
                        # Match known-good RTSP pipeline: appsrc -> queue(leaky) -> videoconvert -> x264enc -> rtph264pay
                        # (no h264parse needed).
                        return ("x264enc", None, "rtph264pay")
                    if Gst is not None and Gst.ElementFactory.find("openh264enc") is not None:
                        return ("openh264enc", None, "rtph264pay")
                except Exception:
                    pass
                raise RuntimeError(
                    "video.encoder=software requested but no software H.264 encoder found (x264enc/openh264enc)."
                )
            return ("nvv4l2h264enc", "h264parse", "rtph264pay")
        if want in ("h265","hevc"):
            return ("nvv4l2h265enc", "h265parse", "rtph265pay")
        if want == "av1":
            if Gst.ElementFactory.find("nvv4l2av1enc"):
                return ("nvv4l2av1enc", None, "rtpav1pay")
            if Gst.ElementFactory.find("av1enc"):
                return ("av1enc", None, "rtpav1pay")
            return ("nvv4l2h265enc", "h265parse", "rtph265pay")
        return ("nvv4l2h265enc", "h265parse", "rtph265pay")

    def _launch_str(self):
        enc_name, parser_name, pay_name = self._codec_chain(self.codec)
        host, port = self.udp_sink.split(":")
        parser = (f"{parser_name} config-interval=1 ! " if parser_name else "")
        pay = (
            f"{pay_name} pt=96 mtu=1200 config-interval=1 ! "
            if pay_name != "rtpav1pay"
            else f"{pay_name} pt=96 mtu=1200 ! "
        )

        sw_enc = enc_name in ("x264enc", "openh264enc")
        want_gray = bool(self.external_capture) and str(self.external_raw_format).lower() in ("gray", "grey", "gray8", "mono", "y8")
        if sw_enc:
            # Software encode (match known-good RTSP pipeline): raw -> videoconvert -> x264enc.
            if want_gray:
                self._appsrc_raw_format = "GRAY8"
                head = (
                    "appsrc name=src is-live=true block=false format=time do-timestamp=true "
                    f"caps=video/x-raw,format=GRAY8,width={self.width},height={self.height},framerate={self.stream_fps}/1 ! "
                    "queue leaky=downstream max-size-buffers=1 ! "
                    "videoconvert ! "
                )
            else:
                self._appsrc_raw_format = "BGR"
                head = (
                    "appsrc name=src is-live=true block=false format=time do-timestamp=true "
                    f"caps=video/x-raw,format=BGR,width={self.width},height={self.height},framerate={self.stream_fps}/1 ! "
                    "queue leaky=downstream max-size-buffers=1 ! "
                    "videoconvert ! "
                )
        else:
            # NVENC path: prefer BGRx into nvvidconv (NVMM NV12).
            # If the external source is mono, do the colorspace expansion in GStreamer (not Python/OpenCV).
            if want_gray:
                self._appsrc_raw_format = "GRAY8"
                head = (
                    "appsrc name=src is-live=true block=false do-timestamp=true format=time "
                    f"caps=video/x-raw,format=GRAY8,width={self.width},height={self.height},framerate={self.stream_fps}/1,interlace-mode=progressive ! "
                    "queue max-size-buffers=2 max-size-bytes=0 max-size-time=0 leaky=downstream ! "
                    "videoconvert ! "
                    f"video/x-raw,format=BGRx,width={self.width},height={self.height},framerate={self.stream_fps}/1,interlace-mode=progressive ! "
                    f"nvvidconv ! video/x-raw(memory:NVMM),format=NV12,width={self.width},height={self.height},framerate={self.stream_fps}/1 ! "
                )
            else:
                self._appsrc_raw_format = "BGRx"
                head = (
                    "appsrc name=src is-live=true block=false do-timestamp=true format=time "
                    f"caps=video/x-raw,format=BGRx,width={self.width},height={self.height},framerate={self.stream_fps}/1,interlace-mode=progressive ! "
                    # Small leaky queue to keep capture non-blocking
                    "queue max-size-buffers=2 max-size-bytes=0 max-size-time=0 leaky=downstream ! "
                    f"nvvidconv ! video/x-raw(memory:NVMM),format=NV12,width={self.width},height={self.height},framerate={self.stream_fps}/1 ! "
                )

        if sw_enc and enc_name == "x264enc":
            profile_caps = f" ! video/x-h264,profile={str(self.h264_profile).strip()}" if str(self.h264_profile).strip() else ""
            enc = (
                f"x264enc name=enc tune={str(self.x264_tune)} speed-preset={str(self.x264_preset)} "
                f"bitrate={int(self.bitrate_kbps)} key-int-max={int(self.iframeinterval)}"
                f"{profile_caps} ! "
            )
        elif sw_enc and enc_name == "openh264enc":
            # openh264enc is also software; keep settings minimal.
            enc = f"openh264enc name=enc bitrate={int(self.bitrate_kbps)} ! "
        else:
            enc = f"{enc_name} name=enc ! "

        pipeline = head + enc + parser
        if not sw_enc:
            # Leaky queue to avoid upstream blocking when network is slow; allow a handful of buffers.
            pipeline += "queue max-size-buffers=2 max-size-bytes=0 max-size-time=0 leaky=downstream ! "
        pipeline += pay + f"udpsink name=sink host={host} port={int(port)} buffer-size=2097152 sync=false async=false"
        return pipeline

    def _rtsp_launch_str(self) -> str:
        enc_name, parser_name, pay_name = self._codec_chain(self.codec)
        parser = (f"{parser_name} config-interval=1 ! " if parser_name else "")
        pay = (
            f"{pay_name} config-interval=1 name=pay0 pt=96 "
            if pay_name != "rtpav1pay"
            else f"{pay_name} name=pay0 pt=96 "
        )

        sw_enc = enc_name in ("x264enc", "openh264enc")
        want_gray = bool(self.external_capture) and str(self.external_raw_format).lower() in ("gray", "grey", "gray8", "mono", "y8")
        if sw_enc:
            if want_gray:
                self._appsrc_raw_format = "GRAY8"
                head = (
                    "appsrc name=src is-live=true block=false format=time do-timestamp=true "
                    f"caps=video/x-raw,format=GRAY8,width={self.width},height={self.height},framerate={self.stream_fps}/1 "
                    "! queue leaky=downstream max-size-buffers=1 "
                    "! videoconvert ! "
                )
            else:
                self._appsrc_raw_format = "BGR"
                head = (
                    "appsrc name=src is-live=true block=false format=time do-timestamp=true "
                    f"caps=video/x-raw,format=BGR,width={self.width},height={self.height},framerate={self.stream_fps}/1 "
                    "! queue leaky=downstream max-size-buffers=1 "
                    "! videoconvert ! "
                )
        else:
            if want_gray:
                self._appsrc_raw_format = "GRAY8"
                head = (
                    "appsrc name=src is-live=true block=false do-timestamp=true format=time "
                    f"caps=video/x-raw,format=GRAY8,width={self.width},height={self.height},framerate={self.stream_fps}/1,interlace-mode=progressive ! "
                    "queue max-size-buffers=2 max-size-bytes=0 max-size-time=0 leaky=downstream ! "
                    "videoconvert ! "
                    f"video/x-raw,format=BGRx,width={self.width},height={self.height},framerate={self.stream_fps}/1,interlace-mode=progressive ! "
                    f"nvvidconv ! video/x-raw(memory:NVMM),format=NV12,width={self.width},height={self.height},framerate={self.stream_fps}/1 ! "
                )
            else:
                self._appsrc_raw_format = "BGRx"
                head = (
                    "appsrc name=src is-live=true block=false do-timestamp=true format=time "
                    f"caps=video/x-raw,format=BGRx,width={self.width},height={self.height},framerate={self.stream_fps}/1,interlace-mode=progressive ! "
                    "queue max-size-buffers=2 max-size-bytes=0 max-size-time=0 leaky=downstream ! "
                    f"nvvidconv ! video/x-raw(memory:NVMM),format=NV12,width={self.width},height={self.height},framerate={self.stream_fps}/1 ! "
                )

        if sw_enc and enc_name == "x264enc":
            profile_caps = f" ! video/x-h264,profile={str(self.h264_profile).strip()}" if str(self.h264_profile).strip() else ""
            enc = (
                f"x264enc name=enc tune={str(self.x264_tune)} speed-preset={str(self.x264_preset)} "
                f"bitrate={int(self.bitrate_kbps)} key-int-max={int(self.iframeinterval)}"
                f"{profile_caps} ! "
            )
        elif sw_enc and enc_name == "openh264enc":
            enc = f"openh264enc name=enc bitrate={int(self.bitrate_kbps)} ! "
        else:
            enc = f"{enc_name} name=enc ! "

        # RTSP server requires a payloader named pay0 as the last element.
        launch = (
            "( " + head + enc + parser + pay + ")"
        )
        return launch

    def _rtsp_start(self) -> None:
        if not _HAS_RTSP or GstRtspServer is None:
            raise RuntimeError("RTSP stream_method selected but GstRtspServer not available")
        # Reset any prior server state
        self._rtsp_stop()

        server = GstRtspServer.RTSPServer()
        # Prefer the explicit set_* APIs (matches known-good implementation); fall back to props for older GI.
        try:
            server.set_service(str(int(self.rtsp_port)))
        except Exception:
            try:
                server.props.service = str(int(self.rtsp_port))
            except Exception:
                pass
        try:
            if self.rtsp_bind:
                server.set_address(str(self.rtsp_bind))
        except Exception:
            try:
                if self.rtsp_bind:
                    server.props.address = str(self.rtsp_bind)
            except Exception:
                pass

        mounts = server.get_mount_points()
        factory = GstRtspServer.RTSPMediaFactory()
        factory.set_shared(True)

        launch = self._rtsp_launch_str()
        factory.set_launch(launch)

        def _on_media_configure(_factory, media):
            try:
                element = media.get_element()
                src = None
                enc = None
                if element is not None:
                    try:
                        src = element.get_child_by_name("src")
                    except Exception:
                        src = element.get_by_name("src")
                    try:
                        enc = element.get_child_by_name("enc")
                    except Exception:
                        enc = element.get_by_name("enc")
                if src is not None:
                    # Mirror odometry.py behavior for robustness.
                    try:
                        src.set_property("format", Gst.Format.TIME)
                    except Exception:
                        pass
                    try:
                        src.set_property("is-live", True)
                    except Exception:
                        pass
                    src.set_property("block", False)
                    try:
                        fmt = str(self._appsrc_raw_format).strip().lower()
                        if fmt in ("gray8", "gray", "grey", "mono", "y8"):
                            bpp = 1
                        elif fmt == "bgr":
                            bpp = 3
                        else:
                            bpp = 4
                        src.set_property("max-bytes", int(self.width) * int(self.height) * int(bpp) * 3)
                    except Exception:
                        pass
                # Best-effort: watch bus for errors
                try:
                    bus = element.get_bus() if element is not None else None
                    if bus is not None:
                        bus.add_signal_watch()
                        bus.connect("message", self._on_bus)
                except Exception:
                    pass
                self.pipeline = element
                self.appsrc = src
                self.enc = enc
                self.sink = None
            except Exception:
                pass

        factory.connect("media-configure", _on_media_configure)

        path = "/" + (self.rtsp_path or "stream").lstrip("/")
        mounts.add_factory(path, factory)
        attach_id = server.attach(None)
        if int(attach_id) == 0:
            raise RuntimeError("RTSP server attach failed (no mainloop?)")

        self._rtsp_server = server
        self._rtsp_factory = factory
        self._rtsp_attach_id = int(attach_id)

        host = self.rtsp_advertise_host or self.rtsp_bind or "127.0.0.1"
        if host == "0.0.0.0":
            host = "127.0.0.1"
        print(
            f"[RTSP] ready src={self.source} {self.codec.upper()} {self.width}x{self.height}@{self.stream_fps} "
            f"{self.bitrate_kbps}kbps -> rtsp://{host}:{int(self.rtsp_port)}{path}",
            flush=True,
        )

    def _rtsp_stop(self) -> None:
        try:
            if self._rtsp_attach_id and GLib is not None:
                try:
                    GLib.source_remove(int(self._rtsp_attach_id))
                except Exception:
                    pass
        except Exception:
            pass
        self._rtsp_attach_id = 0
        self._rtsp_factory = None
        self._rtsp_server = None

    def _apply_encoder_properties(self):
        if not self.enc:
            return
        try:
            factory = self.enc.get_factory().get_name() if self.enc.get_factory() is not None else ""
        except Exception:
            factory = ""
        # Software encoders are fully configured in the pipeline launch string.
        if factory in ("x264enc", "openh264enc"):
            return
        if _has_prop(self.enc, "bitrate"):
            self.enc.set_property("bitrate", _kbps_to_bps(self.bitrate_kbps))
        if _has_prop(self.enc, "control-rate"):
            cr = 1 if str(self.rc_mode).upper() == "CBR" else 2
            self.enc.set_property("control-rate", cr)
        if _has_prop(self.enc, "iframeinterval"):
            self.enc.set_property("iframeinterval", int(self.iframeinterval))
        if _has_prop(self.enc, "idrinterval"):
            self.enc.set_property("idrinterval", int(self.idrinterval))
        if _has_prop(self.enc, "preset-level"):
            self.enc.set_property("preset-level", 1)
        if _has_prop(self.enc, "insert-sps-pps"):
            self.enc.set_property("insert-sps-pps", True)
        if _has_prop(self.enc, "maxperf-enable"):
            self.enc.set_property("maxperf-enable", True)

    def _build_pipeline(self):
        if str(self.stream_method).lower() == "rtsp":
            self._rtsp_start()
            return
        t0 = time.time()
        if self.pipeline:
            try:
                self.pipeline.set_state(Gst.State.NULL)
            except Exception:
                pass
        launch = self._launch_str()
        try:
            self.pipeline = Gst.parse_launch(launch)
        except Exception as e:
            raise RuntimeError(f"GStreamer parse_launch failed: {e}")
        # Fetch common elements
        self.appsrc = self.pipeline.get_by_name("src")
        self.enc    = self.pipeline.get_by_name("enc")
        self.sink   = self.pipeline.get_by_name("sink")
        if not all([self.pipeline, self.enc, self.sink]):
            raise RuntimeError("GStreamer elements missing")
        if self.stream_enable and self.appsrc is None:
            raise RuntimeError("GStreamer appsrc missing")
        if self.appsrc:
            self.appsrc.set_property("block", False)
            try:
                fmt = str(self._appsrc_raw_format).strip().lower()
                if fmt in ("gray8", "gray", "grey", "mono", "y8"):
                    bpp = 1
                elif fmt == "bgr":
                    bpp = 3
                else:
                    bpp = 4
                self.appsrc.set_property("max-bytes", int(self.width) * int(self.height) * int(bpp) * 3)
            except Exception:
                self.appsrc.set_property("max-bytes", self.width * self.height * 4 * 3)

        self._apply_encoder_properties()

        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self._on_bus)

        self.pipeline.set_state(Gst.State.PLAYING)
        t1 = time.time()
        print(f"[UDP] pipeline ready src={self.source} {self.codec.upper()} {self.width}x{self.height}@{self.stream_fps} "
              f"{self.bitrate_kbps}kbps -> udp://{self.udp_sink} (build_ms={(t1 - t0)*1000.0:.2f})")

    def _on_bus(self, bus, msg):
        t = msg.type
        if t == Gst.MessageType.ERROR:
            err, dbg = msg.parse_error()
            print(f"[GST] ERROR: {err} {dbg}")
        elif t == Gst.MessageType.WARNING:
            err, dbg = msg.parse_warning()
            print(f"[GST] WARN: {err} {dbg}")
        elif t == Gst.MessageType.EOS:
            print("[GST] EOS")

    def _log_gst_perf(self) -> None:
        if not self.stream_enable or not self.gst_perf_log:
            return
        now = time.time()
        if (now - self._gst_last_log) < 1.0:
            return
        def avg(lst):
            return (sum(lst) / len(lst)) if lst else 0.0
        with self._stream_cv:
            q_len = len(self._stream_q)
            q_max = max(self._gst_q_max, q_len)
        enqueue_ms = avg(self._gst_enqueue_dt) * 1000.0
        push_ms = avg(self._gst_push_dt) * 1000.0
        cam_fps = self.get_cam_fps()
        print(f"[GST(perf)] enqueue_ms={enqueue_ms:.2f} push_ms={push_ms:.2f} q_len={q_len} q_max={q_max} cam_fps={cam_fps:.1f}", flush=True)
        self._gst_last_log = now
        # reset histories to report per-second averages
        self._gst_enqueue_dt.clear()
        self._gst_push_dt.clear()
        self._gst_q_max = q_len

    def _streamer_loop(self):
        while self.capturing or len(self._stream_q) > 0:
            # Rate-limit encoder input to requested stream_fps (independent from capture fps).
            try:
                if self.stream_fps > 0:
                    now = time.time()
                    min_dt = 1.0 / float(self.stream_fps)
                    if self._stream_last_push_t > 0.0 and (now - self._stream_last_push_t) < min_dt:
                        time.sleep(min(0.01, max(0.0, min_dt - (now - self._stream_last_push_t))))
                        continue
            except Exception:
                pass
            # RTSP: appsrc is created only when a client connects; avoid wasting CPU until then.
            if self.appsrc is None:
                time.sleep(0.02)
                continue
            if self.stream_from_framebus:
                # Pull latest from FrameBus on demand
                fb_latest = self._framebus.latest() if self._framebus else None
                if not fb_latest:
                    time.sleep(0.01)
                    continue
                frame, ts_ms, w, h, cam, seq = fb_latest
                if seq == self._stream_last_fb_seq or (seq % self.stream_every_n) != 0:
                    time.sleep(0.005)
                    continue
                self._stream_last_fb_seq = seq
                frame_out = frame
                try:
                    fmt = str(self._appsrc_raw_format).strip().lower()
                except Exception:
                    fmt = "bgrx"
                want_gray = fmt in ("gray8", "gray", "grey", "mono", "y8")
                want_bgr = fmt == "bgr"
                if want_gray:
                    # Prefer mono end-to-end; only convert if we got a color frame.
                    if frame_out is not None and getattr(frame_out, "ndim", 0) == 3:
                        try:
                            if frame_out.shape[2] >= 4:
                                frame_out = cv2.cvtColor(frame_out[:, :, :4], cv2.COLOR_BGRA2GRAY)
                            else:
                                frame_out = cv2.cvtColor(frame_out[:, :, :3], cv2.COLOR_BGR2GRAY)
                        except Exception:
                            try:
                                frame_out = frame_out[:, :, 0]
                            except Exception:
                                pass
                elif want_bgr:
                    # Software x264 path: appsrc expects BGR.
                    if frame_out is not None and getattr(frame_out, "ndim", 0) == 3 and frame_out.shape[2] > 3:
                        frame_out = frame_out[:, :, :3]
                    elif frame_out is not None and getattr(frame_out, "ndim", 0) == 2:
                        frame_out = cv2.cvtColor(frame_out, cv2.COLOR_GRAY2BGR)
                else:
                    # BGRx/BGRA path.
                    if frame_out is not None and getattr(frame_out, "ndim", 0) == 2:
                        frame_out = cv2.cvtColor(frame_out, cv2.COLOR_GRAY2BGRA)
                    elif frame_out is not None and getattr(frame_out, "ndim", 0) == 3 and frame_out.shape[2] == 3:
                        frame_out = cv2.cvtColor(frame_out, cv2.COLOR_BGR2BGRA)
                try:
                    t_push = time.time()
                    data = frame_out.tobytes()
                    try:
                        buf = Gst.Buffer.new_wrapped(data)
                    except Exception:
                        buf = Gst.Buffer.new_allocate(None, len(data), None)
                        buf.fill(0, data)
                    if self.stream_method != "rtsp":
                        buf.pts = buf.dts = int(ts_ms * 1_000_000)  # nanoseconds
                    ret = self.appsrc.emit("push-buffer", buf)
                    if ret != Gst.FlowReturn.OK:
                        if ret in (Gst.FlowReturn.FLUSHING, Gst.FlowReturn.EOS):
                            time.sleep(0.001)
                        # Drop on other errors; stay non-blocking
                    self._gst_push_dt.append(max(0.0, time.time() - t_push))
                    self._stream_last_push_t = time.time()
                except Exception:
                    time.sleep(0.001)
                self._log_gst_perf()
                continue
            # Legacy path: consume queued frames from capture
            with self._stream_cv:
                if not self._stream_q and self.capturing:
                    self._stream_cv.wait(timeout=0.02)
                if not self._stream_q:
                    continue
                frame_out, ts_ms = self._stream_q.popleft()
            try:
                t_push = time.time()
                try:
                    fmt = str(self._appsrc_raw_format).strip().lower()
                except Exception:
                    fmt = "bgrx"
                want_gray = fmt in ("gray8", "gray", "grey", "mono", "y8")
                want_bgr = fmt == "bgr"
                if want_gray:
                    if frame_out is not None and getattr(frame_out, "ndim", 0) == 3:
                        try:
                            if frame_out.shape[2] >= 4:
                                frame_out = cv2.cvtColor(frame_out[:, :, :4], cv2.COLOR_BGRA2GRAY)
                            else:
                                frame_out = cv2.cvtColor(frame_out[:, :, :3], cv2.COLOR_BGR2GRAY)
                        except Exception:
                            try:
                                frame_out = frame_out[:, :, 0]
                            except Exception:
                                pass
                elif want_bgr:
                    if frame_out is not None and getattr(frame_out, "ndim", 0) == 3 and frame_out.shape[2] == 4:
                        # Convert BGRA -> BGR for software x264 path.
                        try:
                            frame_out = cv2.cvtColor(frame_out, cv2.COLOR_BGRA2BGR)
                        except Exception:
                            frame_out = frame_out[:, :, :3]
                    elif frame_out is not None and getattr(frame_out, "ndim", 0) == 2:
                        frame_out = cv2.cvtColor(frame_out, cv2.COLOR_GRAY2BGR)
                else:
                    if frame_out is not None and getattr(frame_out, "ndim", 0) == 2:
                        frame_out = cv2.cvtColor(frame_out, cv2.COLOR_GRAY2BGRA)
                    elif frame_out is not None and getattr(frame_out, "ndim", 0) == 3 and frame_out.shape[2] == 3:
                        frame_out = cv2.cvtColor(frame_out, cv2.COLOR_BGR2BGRA)
                data = frame_out.tobytes()
                try:
                    buf = Gst.Buffer.new_wrapped(data)
                except Exception:
                    buf = Gst.Buffer.new_allocate(None, len(data), None)
                    buf.fill(0, data)
                if self.stream_method != "rtsp":
                    buf.pts = buf.dts = int(ts_ms * 1_000_000)  # nanoseconds
                ret = self.appsrc.emit("push-buffer", buf)
                if ret != Gst.FlowReturn.OK:
                    if ret in (Gst.FlowReturn.FLUSHING, Gst.FlowReturn.EOS):
                        time.sleep(0.001)
                    # Drop on other errors; stay non-blocking
                self._gst_push_dt.append(max(0.0, time.time() - t_push))
                self._stream_last_push_t = time.time()
            except Exception:
                time.sleep(0.001)
            self._log_gst_perf()

    # ---------- Public API ----------
    def start(self):
        with self._lock:
            if self.source != "realsense":
                raise RuntimeError(
                    f"Only RealSense is supported (camera.type must be 'realsense', got {self.source!r})"
                )
            if self._framebus is None:
                raise RuntimeError("FrameBus not attached")
            # Streaming from RealSense uses GStreamer (appsrc).
            if self.stream_enable and not _HAS_GI:
                raise RuntimeError("GStreamer/gi not available (required for streaming)")
            if self.stream_enable and not self._glib_loop:
                self._glib_loop = GLib.MainLoop()
                self._glib_th = threading.Thread(target=self._glib_loop.run, daemon=True)
                self._glib_th.start()

            self._t_prev_frame = None
            self._cam_fps_ema = 0.0
            self._cap_dt_hist.clear()
            self._stream_last_push_t = 0.0
            self._stream_last_fb_seq = 0

            # External capture mode: the server streams from an external frame source (e.g., 3d_track SHM ring).
            if bool(self.external_capture):
                self.backend = "external"
                self.capturing = True
                if self.stream_enable:
                    self._build_pipeline()
                    if self._stream_th is None or not self._stream_th.is_alive():
                        self._stream_th = threading.Thread(
                            target=self._streamer_loop, name="gst-streamer", daemon=True
                        )
                        self._stream_th.start()
                else:
                    print("[STREAM] disabled (video.stream_enable=false)", flush=True)
                # Require intrinsics to be set externally for PID/debug paths.
                self.get_intrinsics()
                print(f"[STREAM] external source {self.width}x{self.height}@{self.stream_fps}", flush=True)
                return

            if not _HAS_RS:
                raise RuntimeError("RealSense selected but pyrealsense2 not installed")

            self.backend = "realsense"
            self._rs_cap = _RealSenseCapture(self.cfg, self._framebus)
            self._rs_cap.open()

            # Intrinsics MUST come from the RealSense API for angle calculations.
            try:
                if self._rs_cap.intr_W and self._rs_cap.intr_H:
                    self._intr_W = int(self._rs_cap.intr_W)
                    self._intr_H = int(self._rs_cap.intr_H)
                    if self._rs_cap.intr_fx is not None and float(self._rs_cap.intr_fx) > 0.0:
                        self._intr_fx = float(self._rs_cap.intr_fx)
                    if self._rs_cap.intr_fy is not None and float(self._rs_cap.intr_fy) > 0.0:
                        self._intr_fy = float(self._rs_cap.intr_fy)
                    if self._rs_cap.intr_cx is not None:
                        self._intr_cx = float(self._rs_cap.intr_cx)
                    else:
                        self._intr_cx = float(self._intr_W) / 2.0
                    if self._rs_cap.intr_cy is not None:
                        self._intr_cy = float(self._rs_cap.intr_cy)
                    else:
                        self._intr_cy = float(self._intr_H) / 2.0
            except Exception:
                pass

            self.capturing = True
            if self.stream_enable:
                self._build_pipeline()
                if self._stream_th is None or not self._stream_th.is_alive():
                    self._stream_th = threading.Thread(
                        target=self._streamer_loop, name="gst-streamer", daemon=True
                    )
                    self._stream_th.start()
            else:
                print("[STREAM] disabled (video.stream_enable=false)", flush=True)

            print(
                f"[STREAM] RealSense {self._rs_cap.req_W}x{self._rs_cap.req_H}@{self._rs_cap.req_FPS}",
                flush=True,
            )

    def stop(self):
        with self._lock:
            self.capturing = False
            try:
                if self.capture_th and self.capture_th.is_alive():
                    self.capture_th.join(timeout=1.0)
            except Exception:
                pass
            try:
                with self._stream_cv:
                    self._stream_cv.notify_all()
                if self._stream_th and self._stream_th.is_alive():
                    self._stream_th.join(timeout=1.0)
            except Exception:
                pass
            try:
                if self.stream_method == "rtsp":
                    self._rtsp_stop()
            except Exception:
                pass
            try:
                if self.pipeline:
                    self.pipeline.set_state(Gst.State.NULL)
            except Exception:
                pass
            self.pipeline = None
            self.appsrc = None
            self.enc = None
            self.sink = None
            self._stream_th = None
            self._stream_q.clear()
            if self.backend == "realsense":
                try:
                    if self._rs_cap:
                        self._rs_cap.stop()
                except Exception:
                    pass
                self._rs_cap = None
            try:
                if self._glib_loop:
                    self._glib_loop.quit()
                if self._glib_th and self._glib_th.is_alive():
                    self._glib_th.join(timeout=1.0)
            except Exception:
                pass
            self._glib_loop = None
            self._glib_th = None

    # ---------- Control shim (legacy GUI compatibility) ----------
    def apply_control(self, v: dict):
        """
        Legacy hook for GUI control messages.
        Currently only supports updating udp_sink on the fly; others no-op.
        """
        if not isinstance(v, dict):
            return False
        if "udp_sink" in v:
            try:
                self.udp_sink = str(v["udp_sink"])
                if self.sink:
                    host, port = self.udp_sink.split(":")
                    try:
                        self.sink.set_property("host", host)
                        self.sink.set_property("port", int(port))
                        print(f"[video] udp_sink -> {self.udp_sink}", flush=True)
                        return True
                    except Exception:
                        pass
            except Exception:
                pass
        # TODO: add live bitrate/iframeinterval updates if needed
        return False
