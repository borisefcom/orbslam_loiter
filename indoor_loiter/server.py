#!/usr/bin/env python3
import os
import sys


def _maybe_reexec_into_venv() -> None:
    """
    Debian/Ubuntu often ship an "externally managed" system Python that blocks pip installs (PEP 668).
    If a local `.venv` exists, transparently re-exec into it so `python3 server.py` works.
    """
    try:
        if os.environ.get("LOITER_NO_VENV_REEXEC", "").strip() in ("1", "true", "yes"):
            return

        # Already in a venv (or similar)
        base_prefix = getattr(sys, "base_prefix", sys.prefix)
        if sys.prefix != base_prefix:
            return

        root = os.path.dirname(os.path.abspath(__file__))
        candidates = [
            os.path.join(root, ".venv", "bin", "python"),
            os.path.join(root, ".venv", "bin", "python3"),
            os.path.join(root, ".venv", "Scripts", "python.exe"),  # Windows
        ]
        vpy = next((p for p in candidates if os.path.exists(p)), "")
        if not vpy:
            return

        os.execv(vpy, [vpy] + sys.argv)
    except Exception:
        return


_maybe_reexec_into_venv()

import time, signal, threading, math
import subprocess
import copy
import multiprocessing as mp
from typing import Tuple, Dict, Any, Optional, Callable

from config import Config, SingleCsvLogger, diff_mavlink_rates
from udp import UdpJsonTx, UdpJsonRx, parse_hostport
from gstreamer import VideoController
from mavlink_bridge import MavlinkBridge
from telemetry import TelemetryManager

# ---- Tracker imports (FrameBus + Tracker) ----
from tracker import Tracker, FrameBus  # matches new tracker API
from track2d_shim import Track2dShim
from v4l_mjpeg_source import V4lMjpegCfg, V4lMjpegSource
try:
    import cv2
except Exception:
    cv2 = None

BANNER_LINE = "-" * 72

# -------- Thread-safe JSON Event Emitter (shared) --------
_json_tx: Optional[UdpJsonTx] = None
_json_lock = threading.Lock()

def json_event(payload: Dict[str, Any]):
    global _json_tx
    if not isinstance(payload, dict):
        return
    payload.setdefault("type", "event")
    tx = _json_tx
    if tx is None:
        return
    with _json_lock:
        try:
            tx.send(payload)
        except Exception as e:
            print(f"[json_event] send error: {e}", flush=True)

def _deep_merge_dict_inplace(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively merge `src` into `dst` (dict-of-dicts), overwriting non-dict leaves.

    Used for config overrides (e.g., mav_control + 2d_tracker_pid).
    """
    if not isinstance(dst, dict) or not isinstance(src, dict):
        return dst
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_merge_dict_inplace(dst[k], v)  # type: ignore[index]
        else:
            dst[k] = v
    return dst

def _realsense_device_count() -> Optional[int]:
    """
    Best-effort RealSense presence check.

    Used to emit clearer errors in track3d mode, where the server itself does not own the camera.
    """
    try:
        import pyrealsense2 as rs  # type: ignore
    except Exception:
        return None
    try:
        ctx = rs.context()
        devs = ctx.query_devices()
        return int(len(devs))
    except Exception:
        return None

def _check_orbslam_tracker_available(*, python_exe: str, workdir_abs: str) -> Tuple[bool, str]:
    """
    Best-effort dependency check for the external ORB-SLAM3 tracker.

    The goal is to fail fast with a clear message if the tracker interpreter is missing
    key imports (pyrealsense2/cv2/yaml) or the custom ORB-SLAM3 binding.
    """
    py = str(python_exe or "").strip() or sys.executable
    wd = os.path.abspath(str(workdir_abs or os.getcwd()))
    if not os.path.isdir(wd):
        return False, f"track3d.workdir not found: {wd}"

    code = (
        "import os,sys\n"
        "from pathlib import Path\n"
        f"root=Path({wd!r})\n"
        "sys.path.insert(0,str(root))\n"
        "sys.path.insert(0,str(root/'third_party'/'ORB_SLAM3_pybind'))\n"
        "pw=root/'third_party'/'ORB_SLAM3_pybind'/'python_wrapper'\n"
        "try:\n"
        "  if hasattr(os,'add_dll_directory') and pw.exists():\n"
        "    os.add_dll_directory(str(pw))\n"
        "except Exception:\n"
        "  pass\n"
        "import numpy,cv2,yaml\n"
        "import pyrealsense2 as rs\n"
        "from python_wrapper.orb_slam3 import ORB_SLAM3\n"
        "print('ok')\n"
    )

    try:
        cp = subprocess.run(
            [py, "-c", code],
            cwd=wd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=12.0,
        )
        if int(cp.returncode) == 0:
            return True, ""
        msg = (cp.stderr or cp.stdout or f"returncode={int(cp.returncode)}").strip()
        return False, msg
    except FileNotFoundError:
        return False, f"python not found: {py}"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"

# ---------- Helpers for banner / preflight ----------
def _print_banner(cfg: Config, csvlog: SingleCsvLogger, rate_diff: dict,
                  tracker_enabled: bool, track_capable: bool):
    v_source = cfg.get("video.source","realsense")
    v_codec  = cfg.get("video.codec","h265")
    v_size   = f"{int(cfg.get('video.width',1280))}x{int(cfg.get('video.height',720))}"
    v_fps    = int(cfg.get("video.fps",60))
    v_bps    = int(cfg.get("video.bitrate_kbps",1000))
    v_rc     = (cfg.get("video.rc_mode","CBR") or "CBR").upper()
    v_gop    = int(cfg.get("video.iframeinterval", cfg.get("video.gop", 60)))
    v_sink   = cfg.get("video.udp_sink","127.0.0.1:5600")
    cam_w = int(cfg.get("camera.realsense.width", cfg.get("camera.width", cfg.get("video.width", 1280))))
    cam_h = int(cfg.get("camera.realsense.height", cfg.get("camera.height", cfg.get("video.height", 720))))
    cam_fps = int(cfg.get("camera.realsense.fps", cfg.get("camera.fps", cfg.get("video.fps", 60))))

    tele_host, tele_port = parse_hostport(cfg.get("telemetry.json.udp","127.0.0.1:6021"))
    pose_hz = float(cfg.get("telemetry.json.pose_hz", cfg.get("telemetry.pose.rate_hz",10.0)))
    cell_hz = float(cfg.get("telemetry.json.cellular_hz", cfg.get("telemetry.cellular.rate_hz",2.0)))
    csv_path = cfg.get("logging.single_csv.path","./logs/telemetry.csv")
    csv_json_inc = cfg.get("logging.single_csv.include.json",{})
    csv_mav_inc  = cfg.get("logging.single_csv.include.px4",{})

    det_enable = bool(cfg.get("detector.enable", True))
    det_model  = cfg.get("detector.model","yolo11m.pt")
    det_device = cfg.get("detector.device","cpu")

    print(BANNER_LINE)
    print("Drone Server — Active Settings")
    print(f"Video: src={v_source}  {v_codec} {v_size}@{v_fps}  {v_bps} kbps {v_rc}  I:{v_gop} IDR:0 -> {v_sink}")
    print(f"Camera: realsense {cam_w}x{cam_h}@{cam_fps}")
    print(f"JSON Telemetry: udp={tele_host}:{tele_port}  pose={pose_hz} Hz  cellular={cell_hz} Hz")
    print(f"CSV: {'ENABLED' if getattr(csvlog,'enabled',True) else 'DISABLED'}  path={csv_path}")
    if getattr(csvlog,'enabled',True):
        print(f"  include.json={csv_json_inc}")
        print(f"  include.mavlink={csv_mav_inc}")
    if rate_diff:
        print("MAVLink rates (changed vs defaults):")
        for k, (dflt, newv) in sorted(rate_diff.items()):
            print(f"  {k}: {dflt} -> {newv}")
    else:
        print("MAVLink rates: using defaults")
    print(f"Tracker: capable={'YES' if track_capable else 'NO'}  enabled={'YES' if tracker_enabled else 'NO'}")
    print(f"Detector (in-tracker): enable={'YES' if det_enable else 'NO'}  model={det_model}  device={det_device}")
    print(BANNER_LINE)

def _print_preflight_row(name: str, ok: bool, detail: str = ""):
    status = "OK" if ok else "FAIL"
    print(f"[PRE-FLIGHT] {name:<25} {status:<4} {detail}")

def preflight(cfg: Config, video: VideoController, mav: MavlinkBridge) -> Tuple[bool, str]:
    ok_all = True
    try:
        cfg_path = getattr(cfg, "_path", "drone.yaml")
    except Exception:
        cfg_path = "drone.yaml"
    _print_preflight_row("Config", True, f"loaded ({cfg_path})")
    # When using the external 3D tracker pipeline, the server does not own the RealSense device.
    # Camera health is checked implicitly by whether the external tracker starts and emits its init event.
    if bool(cfg.get("track3d.enabled", False)):
        _print_preflight_row("Camera", True, "external (ORB-SLAM3 capture)")
    else:
        cam_ok, cam_detail = video.preflight_camera_check()
        _print_preflight_row("Camera", cam_ok, cam_detail); ok_all &= cam_ok
    ser_ok, ser_detail = mav.try_open_serial_only()
    _print_preflight_row("Serial link", ser_ok, ser_detail); ok_all &= ser_ok
    hb_timeout = float(cfg.get("px4.heartbeat_timeout_s", 5.0))
    hb_ok, hb_detail = mav.wait_heartbeat_once(timeout=hb_timeout)
    _print_preflight_row("PX4 heartbeat", hb_ok, hb_detail); ok_all &= hb_ok
    return ok_all, ("All checks passed" if ok_all else "Preflight failed")

# -------------- PID manager --------------
def _try_start_pid(
    cfg: Config,
    event_cb: Callable[[Dict[str, Any]], None],
    mav: MavlinkBridge,
    tracker_obj: Optional[Any],
    video: VideoController,
    imu_proc: Optional[Any] = None,
):
    try:
        import pid
    except Exception:
        return None
    try:
        ctor = getattr(pid, "PidManager", None)
        if ctor is None:
            print("[PID] module present but no manager/controller class", flush=True)
            return None

        cfg_dict = {}
        try:
            cfg_dict = cfg.get("mav_control", {})
        except Exception:
            cfg_dict = {}

        pid_cfg = pid.config_from_dict(cfg_dict)

        if not bool(getattr(pid_cfg, "enable", True)):
            print("[PID] disabled (mav_control.enable=false)", flush=True)
            return None

        pm = ctor(
            pid_cfg,
            get_rc_norm=getattr(mav, "get_rc_norm", lambda _ch: 0.0),
            get_mode_name=getattr(mav, "mode_name", lambda: ""),
            send_manual_control=getattr(mav, "send_manual_control", lambda *args, **kwargs: False),
            emit_debug_json=event_cb,
        )

        try:
            intr_W, intr_H, fx, fy, cx, cy = video.get_intrinsics()
        except Exception:
            intr_W, intr_H, fx, fy, cx, cy = (0, 0, 1.0, 1.0, 0.0, 0.0)

        # Camera gimbal correction (pitch-only): use the camera IMU's gravity-aligned pitch
        # and treat the startup-calibrated boresight pitch as the 0° reference.
        #
        # This replaces any fixed mount angle correction and does NOT do any FCU/body correction.
        gimbal_pitch_ref: Optional[float] = None
        gimbal_pitch_warned = False
        print(
            f"[PID] angles enabled: intr={intr_W}x{intr_H} gimbal_pitch_correction={'YES' if imu_proc is not None else 'NO'}",
            flush=True,
        )

        def _tracker_to_pid(meas: Dict[str, Any]):
            nonlocal gimbal_pitch_ref, gimbal_pitch_warned
            try:
                # Tracker events (non-pixel measurements)
                if isinstance(meas, dict) and meas.get("event") == "max_bbox_area":
                    try:
                        pm.start_thrust_hold_max(reason="max_bbox_area")
                    except Exception:
                        pass
                    return
                cx_px = meas.get("cx_px")
                cy_px = meas.get("cy_px")
                if cx_px is None or cy_px is None:
                    return
                cx_px = float(cx_px)
                cy_px = float(cy_px)
                x_norm = (cx_px - cx) / fx
                y_norm = (cy_px - cy) / fy
                yaw_err = math.atan(x_norm)
                pitch_cam = math.atan(y_norm)
                pitch_err = pitch_cam

                # Pitch gimbal correction (RealSense IMU): convert camera boresight pitch in world
                # to a delta from the startup reference, then express it in the same sign convention
                # as pitch_cam (positive down in image).
                if imu_proc is not None:
                    bore = None
                    try:
                        bore = imu_proc.boresight_yaw_pitch()  # type: ignore[attr-defined]
                    except Exception:
                        bore = None
                    if bore is not None:
                        cam_pitch = float(bore[1])  # elevation (+up)
                        if gimbal_pitch_ref is None:
                            gimbal_pitch_ref = cam_pitch
                            try:
                                print(
                                    f"[PID] gimbal_pitch_ref_deg={math.degrees(gimbal_pitch_ref):+.2f}",
                                    flush=True,
                                )
                            except Exception:
                                pass
                        if gimbal_pitch_ref is not None:
                            cam_pitch_delta = cam_pitch - float(gimbal_pitch_ref)
                            pitch_err = float(pitch_err) - float(cam_pitch_delta)
                    elif not gimbal_pitch_warned:
                        gimbal_pitch_warned = True
                        print("[PID] gimbal_pitch_correction: waiting for camera IMU pitch", flush=True)

                track_id = meas.get("track_id")
                try:
                    track_id = int(track_id) if track_id is not None else None
                except Exception:
                    track_id = None

                measurement = pid.Measurement(
                    state=str(meas.get("state", "IDLE")).upper(),
                    yaw_err_rad=yaw_err,
                    pitch_err_rad=pitch_err,
                    track_id=track_id,
                    t_ms=int(time.time() * 1000),
                )
            except Exception as err:
                print(f"[PID] measurement convert error: {err}", flush=True)
                return
            try:
                pm.on_measurement(measurement)
            except Exception as err:
                print(f"[PID] on_measurement error: {err}", flush=True)

        if tracker_obj and hasattr(tracker_obj, "set_pid_callback"):
            try:
                tracker_obj.set_pid_callback(_tracker_to_pid)
            except Exception as err:
                print(f"[PID] tracker callback setup error: {err}", flush=True)

        if hasattr(pm, "start"):
            pm.start()
        print("[PID] started", flush=True)
        return pm
    except Exception as e:
        print(f"[PID] init error: {e}", flush=True)
        return None

# -------------- MAIN --------------
def main():
    global _json_tx

    # Set OpenCV thread cap early (no-op if cv2 missing)
    if cv2 is not None:
        try:
            cv2.setNumThreads(4)
            cv2.setUseOptimized(True)
            print("[OpenCV] setNumThreads=4 useOptimized=True")
        except Exception as e:
            print(f"[OpenCV] thread tuning skipped: {e}")

    try:
        cfg = Config("drone.yaml")
    except Exception as e:
        detail = str(e)
        try:
            import yaml as _yaml  # type: ignore

            if isinstance(e, _yaml.YAMLError):
                mark = getattr(e, "problem_mark", None)
                prob = getattr(e, "problem", None) or detail
                if mark is not None:
                    detail = f"{prob} (line {int(mark.line) + 1}, col {int(mark.column) + 1})"
                else:
                    detail = str(prob)
        except Exception:
            pass
        _print_preflight_row("Config", False, detail)
        print("[PRE-FLIGHT] summary: Preflight failed", file=sys.stderr, flush=True)
        return

    if bool(cfg.get("track3d.enabled", False)):
        py_t3 = str(cfg.get("track3d.python", "") or "").strip() or sys.executable
        workdir = str(cfg.get("track3d.workdir", "") or "").strip()
        if not workdir:
            workdir = os.getcwd()
        workdir_abs = os.path.abspath(workdir)
        ok_trk, detail = _check_orbslam_tracker_available(python_exe=py_t3, workdir_abs=workdir_abs)
        if not bool(ok_trk):
            _print_preflight_row("ORB-SLAM3 tracker", False, f"{py_t3}: {detail}")
            print("[PRE-FLIGHT] summary: Preflight failed", file=sys.stderr, flush=True)
            return

    # Telemetry out (JSON)
    tele_host, tele_port = parse_hostport(cfg.get("telemetry.json.udp", "127.0.0.1:6021"))
    _json_tx = UdpJsonTx(tele_host, tele_port)

    # Control in (JSON)
    ctrl_host, ctrl_port = parse_hostport(cfg.get("control.json_in", "0.0.0.0:6020"))
    control_rx = UdpJsonRx(ctrl_host, ctrl_port)

    # Logging
    csvlog = SingleCsvLogger(
        enabled=cfg.get("logging.single_csv.enabled", True),
        path=cfg.get("logging.single_csv.path","./logs/telemetry.csv"),
        max_bytes=int(cfg.get("logging.single_csv.rotate_bytes", 10*1024*1024)),
        include_json=cfg.get("logging.single_csv.include.json", {}),
        include_mavlink=cfg.get("logging.single_csv.include.px4", {}),
    )

    # Optional: external 3D tracker (spawns RealSense capture + polygon tracking).
    track3d = None
    if bool(cfg.get("track3d.enabled", False)):
        try:
            from track3d_bridge import Track3dBridge

            track3d = Track3dBridge(cfg=cfg, print_fn=print)
            track3d.start()

            t0 = time.time()
            timeout_s = float(cfg.get("track3d.init_timeout_s", 10.0))
            while (time.time() - t0) < timeout_s and track3d.init_info() is None:
                for evt in track3d.poll_events(max_n=50):
                    try:
                        track3d.apply_event(evt)
                    except Exception:
                        pass
                try:
                    rc = track3d.returncode()
                    if rc is not None:
                        rs_n = _realsense_device_count()
                        if rs_n == 0:
                            raise RuntimeError(
                                f"RealSense camera not detected (0 devices). ORB-SLAM3 tracker exited early (rc={int(rc)})."
                            )
                        raise RuntimeError(f"ORB-SLAM3 tracker exited early (rc={int(rc)})")
                except Exception:
                    raise
                time.sleep(0.01)
            if track3d.init_info() is None:
                rs_n = _realsense_device_count()
                if rs_n == 0:
                    raise RuntimeError("RealSense camera not detected (0 devices). Check USB/power and retry.")
                raise RuntimeError("Timed out waiting for ORB-SLAM3 tracker init event")
            try:
                init0 = track3d.init_info()
                if init0 is not None:
                    # Match stream resolution to the shared capture ring (IR1 grayscale).
                    cfg.set("video.width", int(init0.out_w))
                    cfg.set("video.height", int(init0.out_h))
                    cfg.set("camera.width", int(init0.out_w))
                    cfg.set("camera.height", int(init0.out_h))
                    cfg.set("video.external_capture", True)
            except Exception:
                pass
        except Exception as e:
            print(f"[track3d] init error: {e}", flush=True)
            try:
                if track3d is not None:
                    tail = []
                    try:
                        tail = track3d.stderr_tail(max_lines=25)
                    except Exception:
                        tail = []
                    if tail:
                        print("[track3d] stderr tail:", flush=True)
                        for ln in tail:
                            print(f"[track3d]   {ln}", flush=True)
                    track3d.stop()
            except Exception:
                pass
            track3d = None
    if bool(cfg.get("track3d.enabled", False)) and track3d is None:
        print("[track3d] enabled but failed to start (refusing to fall back).", file=sys.stderr, flush=True)
        return

    # Video + MAVLink
    video = VideoController(cfg)
    mav = MavlinkBridge(cfg.as_dict() if hasattr(cfg, "as_dict") else cfg,
                        print_fn=print,
                        csvlog=csvlog)

    # FrameBus for legacy tracker (shared memory ring). In 3D tracker mode we defer this.
    fb_w = int(cfg.get("camera.width", cfg.get("video.width", 1280)))
    fb_h = int(cfg.get("camera.height", cfg.get("video.height", 720)))
    framebus = None
    shm_src_rgb = None
    shm_src_ir = None
    shm_src_depth = None
    track3d_intr = None  # (W,H,fx,fy,cx,cy) for external tracker stream
    if track3d is None:
        framebus = FrameBus(width=fb_w, height=fb_h, channels=3)
        if hasattr(video, "attach_framebus") and callable(getattr(video, "attach_framebus")):
            try:
                video.attach_framebus(framebus)
            except Exception as e:
                print(f"[VIDEO] attach_framebus error: {e}", flush=True)
                setattr(video, "framebus", framebus)
        else:
            setattr(video, "framebus", framebus)

    # Tracker
    tracker_obj = None
    tracker_enabled_cfg = bool(cfg.get("tracker.enable", True))  # from tracker section
    track_capable_cfg = bool(Tracker is not None)
    if Tracker is not None and framebus is not None:
        try:
            cfg_dict = cfg.as_dict() if hasattr(cfg, "as_dict") else {}
            tracker_obj = Tracker(cfg_dict, framebus, _json_tx, print_fn=print)
            if hasattr(tracker_obj, "set_enabled"):
                tracker_obj.set_enabled(tracker_enabled_cfg)
        except Exception as e:
            print(f"[TRACK] init error: {e}", flush=True)
            tracker_obj = None
            track_capable_cfg = False

    # Banner + PX4 rates
    defaults = cfg.get_default_mavlink_rates()
    chosen = cfg.get("px4.rates_hz", {}) or {}
    rate_diff = diff_mavlink_rates(defaults, chosen)
    _print_banner(cfg, csvlog, rate_diff,
                  tracker_enabled=bool(getattr(tracker_obj, "enabled", tracker_enabled_cfg)),
                  track_capable=track_capable_cfg)

    # Preflight
    ok, summary = preflight(cfg, video, mav)
    print(f"[PRE-FLIGHT] summary: {summary}")
    if not ok:
        print("[PRE-FLIGHT] FAILED — refusing to start control loops", file=sys.stderr, flush=True)
        return

    # Start video
    if track3d is None:
        video.start()
    else:
        try:
            from shm_ring_source import ShmRingSource, ShmRingSpec

            init0 = track3d.init_info()
            if init0 is None:
                raise RuntimeError("ORB-SLAM3 tracker init missing (cannot attach shared ring)")
            # Build SHM sources from the track3d shared-memory ring:
            # - IR (bw) is the default streamed feed (`gray`) and is used for mouse-driven acquisition.
            # - RGB is available for future debugging/UI (`bgr`).
            # - optional depth visualization (`depth_bgr`) (processed view only; not used for acquisition).
            spec0 = ShmRingSpec.from_dict(dict(init0.ring_spec))
            shm_src_rgb = ShmRingSource(spec=spec0, cam_name="realsense", frame_kind="bgr")
            shm_src_ir = ShmRingSource(spec=spec0, cam_name="realsense", frame_kind="gray")
            shm_src_depth = ShmRingSource(spec=spec0, cam_name="realsense", frame_kind="depth_bgr")
            # Default to IR to match the rest of the loiter stack (and keep GUI overlays consistent).
            video.attach_framebus(shm_src_ir if shm_src_ir is not None else shm_src_rgb)
            if hasattr(video, "set_intrinsics"):
                video.set_intrinsics(int(init0.out_w), int(init0.out_h), float(init0.fx), float(init0.fy), float(init0.cx), float(init0.cy))
            try:
                track3d_intr = (
                    int(init0.out_w),
                    int(init0.out_h),
                    float(init0.fx),
                    float(init0.fy),
                    float(init0.cx),
                    float(init0.cy),
                )
            except Exception:
                track3d_intr = None
            video.start()
        except Exception as e:
            print(f"[STREAM] external source init error: {e}", flush=True)
            raise

    # ---------------- IMU (optional) ----------------
    # This staging tree uses the external ORB-SLAM3 tracker for VO. If `track3d.enabled=true`,
    # the server does not own the RealSense device and cannot drain IMU directly.
    imu_proc = None

    try:
        imu_enabled = bool(cfg.get("imu.enabled", False)) and (track3d is None)
        imu_cal_on_start = bool(cfg.get("imu.calibrate_on_start", True))
        imu_calib_seconds = float(cfg.get("imu.calib_seconds", 5.0)) if imu_cal_on_start else 0.0
        imu_gyro_scale = float(cfg.get("imu.gyro_scale", 1.0))
        imu_accel_scale = float(cfg.get("imu.accel_scale", 1.0))
    except Exception:
        imu_enabled = False
        imu_calib_seconds = 0.0
        imu_gyro_scale = 1.0
        imu_accel_scale = 1.0
    if track3d is not None:
        print("[IMU] disabled (external tracker capture)", flush=True)

    if imu_enabled:
        try:
            from imu_fusion import IMUProcessor
        except Exception as e:
            print(f"[IMU] disabled (import error): {e}", flush=True)
            imu_enabled = False

    if imu_enabled:
        try:
            imu_proc = IMUProcessor(
                imu_drain_fn=framebus.drain_imu,
                vo_queue=None,
                get_mount_yaw_rad=lambda: 0.0,
                imu_calib_seconds=float(imu_calib_seconds),
                gyro_scale=float(imu_gyro_scale),
                accel_scale=float(imu_accel_scale),
                set_vo_fps_cb=(tracker_obj.set_vo_fps if tracker_obj is not None else None),
            )
            imu_proc.start()
        except Exception as e:
            imu_proc = None
            print(f"[IMU] disabled (init error): {e}", flush=True)

    # Start MAVLink worker (full bridge)
    mav.start()

    # Start PID (wired into tracker)
    pidmgr = _try_start_pid(cfg, json_event, mav, tracker_obj, video, imu_proc=imu_proc)

    # Start tracker
    if tracker_obj and hasattr(tracker_obj, "start"):
        tracker_obj.start()
        print("[TRACK] started", flush=True)

    # Telemetry (pose + cellular) + control router
    telem = TelemetryManager(cfg, _json_tx, csvlog, mav, video, tracker_obj)
    telem.start()

    # ---------------- 2D ROI tracker shim (DynMedianFlow) ----------------
    # Only one tracker should drive PID at a time (track3d hole tracker vs. 2D ROI tracker).
    pid_src_lock = threading.Lock()
    pid_src: Dict[str, str] = {"name": ("track3d" if track3d is not None else "tracker")}

    # Active video feed state (used by GUI to gate overlays that only match one camera).
    cam_state_lock = threading.Lock()
    cam_state: Dict[str, Any] = {
        "active": ("rgb" if track3d is not None else "realsense"),
    }

    def _cam_get_active() -> str:
        try:
            with cam_state_lock:
                return str(cam_state.get("active", "") or "")
        except Exception:
            return ""

    def _cam_set_active(name: str) -> None:
        try:
            with cam_state_lock:
                cam_state["active"] = str(name or "")
        except Exception:
            pass

    # Desired 3D hole-acquisition enable state (drives RGB vs IR stream selection when 2D tracker is off).
    hole_state_lock = threading.Lock()
    hole_state: Dict[str, Any] = {"enabled": False}

    def _hole_get_enabled() -> bool:
        try:
            with hole_state_lock:
                return bool(hole_state.get("enabled", False))
        except Exception:
            return False

    def _hole_set_enabled(enabled: bool) -> None:
        try:
            with hole_state_lock:
                hole_state["enabled"] = bool(enabled)
        except Exception:
            pass

    # Desired 3D plane-acquisition enable state (drives RGB vs IR stream selection when 2D tracker is off).
    plane_state_lock = threading.Lock()
    plane_state: Dict[str, Any] = {"enabled": False}

    def _plane_get_enabled() -> bool:
        try:
            with plane_state_lock:
                return bool(plane_state.get("enabled", False))
        except Exception:
            return False

    def _plane_set_enabled(enabled: bool) -> None:
        try:
            with plane_state_lock:
                plane_state["enabled"] = bool(enabled)
        except Exception:
            pass

    def _pid_src_get() -> str:
        try:
            with pid_src_lock:
                return str(pid_src.get("name", "") or "")
        except Exception:
            return ""

    def _pid_src_set(name: str) -> None:
        try:
            with pid_src_lock:
                pid_src["name"] = str(name or "")
        except Exception:
            pass

    pid_mod_global = None
    pid_cfg_base = None
    pid_cfg_track2d = None
    pid_cfg_hole = None
    pid_cfg_plane = None
    pid_dict_base: Optional[dict] = None
    pid_dict_track2d: Optional[dict] = None
    pid_dict_hole: Optional[dict] = None
    pid_dict_plane: Optional[dict] = None
    try:
        import pid as pid_mod_global  # type: ignore

        base_dict = cfg.get("mav_control", {}) or {}
        base_dict = dict(base_dict) if isinstance(base_dict, dict) else {}
        pid_dict_base = base_dict
        pid_cfg_base = pid_mod_global.config_from_dict(base_dict)
        pid_cfg_track2d = pid_cfg_base
        pid_dict_track2d = base_dict
        pid_cfg_hole = pid_cfg_base
        pid_cfg_plane = pid_cfg_base
        pid_dict_hole = base_dict
        pid_dict_plane = base_dict

        ovr = cfg.get("2d_tracker_pid", {}) or {}
        if isinstance(ovr, dict) and ovr:
            merged = copy.deepcopy(base_dict)
            _deep_merge_dict_inplace(merged, dict(ovr))
            pid_cfg_track2d = pid_mod_global.config_from_dict(merged)
            pid_dict_track2d = merged

        ovr_hole = cfg.get("hole_tracker_pid", {}) or {}
        if isinstance(ovr_hole, dict) and ovr_hole:
            merged = copy.deepcopy(base_dict)
            _deep_merge_dict_inplace(merged, dict(ovr_hole))
            pid_cfg_hole = pid_mod_global.config_from_dict(merged)
            pid_dict_hole = merged

        ovr_plane = cfg.get("plane_tracker_pid", {}) or {}
        if isinstance(ovr_plane, dict) and ovr_plane:
            merged = copy.deepcopy(base_dict)
            _deep_merge_dict_inplace(merged, dict(ovr_plane))
            pid_cfg_plane = pid_mod_global.config_from_dict(merged)
            pid_dict_plane = merged
    except Exception:
        pid_mod_global = None
        pid_cfg_base = None
        pid_cfg_track2d = None
        pid_cfg_hole = None
        pid_cfg_plane = None
        pid_dict_base = None
        pid_dict_track2d = None
        pid_dict_hole = None
        pid_dict_plane = None

    v4l_src = None  # created below when track2d is enabled

    video_default_src = shm_src_ir if shm_src_ir is not None else (shm_src_rgb if shm_src_rgb is not None else framebus)

    def _pid_move_max_bbox_area_ratio(mav_control_dict: Optional[dict]) -> float:
        """
        Returns max bbox-area ratio (0..1) for PID_MOVE auto-exit, or 0.0 if disabled/not configured.
        Uses the YAML-style schema: mav_control.PID_MOVE.auto_exit.max_bbox_area_ratio.
        """
        if not isinstance(mav_control_dict, dict):
            return 0.0
        try:
            pm = mav_control_dict.get("PID_MOVE", None)
            if pm is None:
                pm = mav_control_dict.get("pid_move", None)
            pm = dict(pm) if isinstance(pm, dict) else {}
            ax = pm.get("auto_exit", None)
            ax = dict(ax) if isinstance(ax, dict) else {}
            if ax.get("enabled", True) is False:
                return 0.0
            v = ax.get("max_bbox_area_ratio", None)
            if v is None:
                return 0.0
            r = float(v or 0.0)
        except Exception:
            return 0.0
        # Allow percent-style configs (80 => 0.8)
        if r > 1.0:
            r = r / 100.0
        if not math.isfinite(float(r)):
            return 0.0
        return float(max(0.0, min(1.0, float(r))))

    def _track2d_is_active() -> bool:
        try:
            return bool(track2d is not None and bool(track2d.enabled()))
        except Exception:
            return False

    # Track2d safety auto-exit: stop ROI tracking near the ground (rangefinder height too low).
    try:
        track2d_min_rangefinder_m = float(cfg.get("track2d.auto_exit.min_rangefinder_m", 0.25) or 0.25)
    except Exception:
        track2d_min_rangefinder_m = 0.25
    track2d_min_rangefinder_m = float(max(0.0, float(track2d_min_rangefinder_m)))
    try:
        track2d_rangefinder_max_age_s = float(cfg.get("track2d.auto_exit.rangefinder_max_age_s", 0.5) or 0.5)
    except Exception:
        track2d_rangefinder_max_age_s = 0.5
    track2d_rangefinder_max_age_s = float(max(0.0, float(track2d_rangefinder_max_age_s)))

    def _track2d_send_pid(state_uc: str, yaw_err_rad: Optional[float], pitch_err_rad: Optional[float]) -> None:
        if pidmgr is None or pid_mod_global is None:
            return
        if _pid_src_get() != "track2d":
            return
        state0 = str(state_uc or "IDLE").upper()

        # Auto-exit: rangefinder minimum height (meters).
        try:
            if str(state0) == "TRACKING" and float(track2d_min_rangefinder_m) > 0.0 and hasattr(mav, "latest_rangefinder_down_m"):
                h_m = mav.latest_rangefinder_down_m(max_age_s=float(track2d_rangefinder_max_age_s))  # type: ignore[attr-defined]
                if h_m is not None and math.isfinite(float(h_m)) and float(h_m) > 0.0 and float(h_m) <= float(track2d_min_rangefinder_m):
                    try:
                        print(
                            f"[track2d] auto-exit rangefinder_min h={float(h_m):.3f}m thr={float(track2d_min_rangefinder_m):.3f}m",
                            flush=True,
                        )
                    except Exception:
                        pass
                    try:
                        if hasattr(pidmgr, "start_thrust_hold_max"):
                            pidmgr.start_thrust_hold_max(reason="rangefinder_min")
                    except Exception:
                        pass
                    try:
                        pidmgr.on_measurement(
                            pid_mod_global.Measurement(
                                state="LOST",
                                yaw_err_rad=None,
                                pitch_err_rad=None,
                                track_id=None,
                                t_ms=int(time.time() * 1000),
                            )
                        )
                    except Exception:
                        pass
                    try:
                        track2d.cancel()
                    except Exception:
                        pass
                    return
        except Exception:
            pass
        try:
            pidmgr.on_measurement(
                pid_mod_global.Measurement(
                    state=str(state0),
                    yaw_err_rad=(float(yaw_err_rad) if yaw_err_rad is not None else None),
                    pitch_err_rad=(float(pitch_err_rad) if pitch_err_rad is not None else None),
                    track_id=None,
                    t_ms=int(time.time() * 1000),
                )
            )
        except Exception:
            pass
        # PID_MOVE auto-exit (2D ROI tracker):
        # Stop tracking if bbox fills too much of the image, and trigger the existing thrust-hold timer logic.
        try:
            if str(state0) != "TRACKING":
                return
            gate_state = pidmgr.gate_state() if hasattr(pidmgr, "gate_state") else getattr(pidmgr, "_gate_state", None)
            if str(gate_state or "") != "PID_MOVE":
                return
            thr = _pid_move_max_bbox_area_ratio(pid_dict_track2d)
            if not (math.isfinite(float(thr)) and float(thr) > 0.0):
                return
            if track2d is None or not hasattr(track2d, "snapshot_bbox"):
                return
            bb, wh = track2d.snapshot_bbox()
            if bb is None:
                return
            W_i, H_i = int(wh[0]), int(wh[1])
            if W_i <= 0 or H_i <= 0:
                return
            try:
                _x, _y, bw, bh = bb
                area_ratio = (max(0.0, float(bw)) * max(0.0, float(bh))) / float(W_i * H_i)
            except Exception:
                return
            if not (math.isfinite(float(area_ratio)) and float(area_ratio) >= float(thr)):
                return
            try:
                print(
                    f"[track2d] auto-exit max_bbox_area ratio={float(area_ratio):.3f} thr={float(thr):.3f}",
                    flush=True,
                )
            except Exception:
                pass
            try:
                if hasattr(pidmgr, "start_thrust_hold_max"):
                    pidmgr.start_thrust_hold_max(reason="max_bbox_area")
            except Exception:
                pass
            try:
                track2d.cancel()
            except Exception:
                pass
        except Exception:
            pass

    def _track2d_on_enable_changed(enabled: bool) -> None:
        # Switching to 2D ROI tracking should disable the 3D hole acquisition flow (mouse-driven),
        # but the ORB-SLAM3 tracker stays running for VO capture/pose.
        try:
            if bool(enabled):
                _pid_src_set("track2d")
                if pidmgr is not None and pid_cfg_track2d is not None and hasattr(pidmgr, "apply_updates"):
                    pidmgr.apply_updates(pid_cfg_track2d)
                try:
                    if v4l_src is not None and hasattr(v4l_src, "start"):
                        v4l_src.start()
                except Exception as e:
                    print(f"[track2d] V4L start error: {e}", flush=True)
                # Track2d is exclusive: force hole-acquisition off (desired state + external tracker state).
                _hole_set_enabled(False)
                _plane_set_enabled(False)
                if track3d is not None:
                    try:
                        track3d.send_cmd({"cmd": "clear", "ts": int(time.time() * 1000)})
                    except Exception:
                        pass
                    try:
                        track3d.send_cmd({"cmd": "hole_enable", "enable": 0, "ts": int(time.time() * 1000)})
                    except Exception:
                        pass
                    try:
                        track3d.send_cmd({"cmd": "plane_enable", "enable": 0, "ts": int(time.time() * 1000)})
                    except Exception:
                        pass
                    try:
                        msg0 = track3d.make_acq_poly_msg(stage=0, verts_uv=[])
                        if msg0 is not None:
                            json_event(msg0)
                    except Exception:
                        pass
                try:
                    _request_video_switch()
                except Exception:
                    pass
            else:
                # Restore track3d PID settings and allow track3d to drive PID again.
                # Use per-mode PID tuning if hole/plane acquisition is latched on.
                pid_cfg_restore = pid_cfg_base
                try:
                    if bool(_plane_get_enabled()):
                        pid_cfg_restore = pid_cfg_plane if pid_cfg_plane is not None else pid_cfg_base
                    elif bool(_hole_get_enabled()):
                        pid_cfg_restore = pid_cfg_hole if pid_cfg_hole is not None else pid_cfg_base
                except Exception:
                    pid_cfg_restore = pid_cfg_base
                if pidmgr is not None and pid_cfg_restore is not None and hasattr(pidmgr, "apply_updates"):
                    pidmgr.apply_updates(pid_cfg_restore)
                _pid_src_set("track3d" if track3d is not None else "tracker")
                # Apply latched hole_enable state now that 2D tracking is off (hole_enable may have been suppressed).
                if track3d is not None:
                    try:
                        track3d.send_cmd({"cmd": "hole_enable", "enable": int(1 if _hole_get_enabled() else 0), "ts": int(time.time() * 1000)})
                    except Exception:
                        pass
                    try:
                        track3d.send_cmd({"cmd": "plane_enable", "enable": int(1 if _plane_get_enabled() else 0), "ts": int(time.time() * 1000)})
                    except Exception:
                        pass
                try:
                    if v4l_src is not None and hasattr(v4l_src, "stop"):
                        v4l_src.stop()
                except Exception:
                    pass
                try:
                    _request_video_switch()
                except Exception:
                    pass
        except Exception:
            pass

    def _track2d_get_intrinsics() -> Tuple[int, int, float, float, float, float]:
        """
        Intrinsics for the non-RealSense 2D tracking camera (V4L MJPEG /dev/video*).

        Uses `camera.v4l.*` in `drone.yaml`. If intrinsics are missing, assumes an ideal pinhole model.
        If intrinsics were calibrated at a different resolution, scales them to current stream size when possible.
        """
        # Prefer live capture size if available; else configured size.
        W = int(cfg.get("camera.v4l.width", 640))
        H = int(cfg.get("camera.v4l.height", 480))
        try:
            if v4l_src is not None and hasattr(v4l_src, "get_size"):
                W0, H0 = v4l_src.get_size()
                if int(W0) > 0 and int(H0) > 0:
                    W, H = int(W0), int(H0)
        except Exception:
            pass

        intr = cfg.get("camera.v4l.intrinsics", None)
        fx = fy = cx = cy = None
        calib_w = calib_h = None
        try:
            if isinstance(intr, dict):
                fx = intr.get("fx", fx)
                fy = intr.get("fy", fy)
                cx = intr.get("cx", cx)
                cy = intr.get("cy", cy)
                calib_w = intr.get("width", intr.get("W", calib_w))
                calib_h = intr.get("height", intr.get("H", calib_h))
        except Exception:
            pass
        # Back-compat flat keys.
        fx = cfg.get("camera.v4l.fx", fx)
        fy = cfg.get("camera.v4l.fy", fy)
        cx = cfg.get("camera.v4l.cx", cx)
        cy = cfg.get("camera.v4l.cy", cy)

        if fx is None or fy is None or cx is None or cy is None:
            fx = float(W) / 2.0
            fy = float(H) / 2.0
            cx = float(W) / 2.0
            cy = float(H) / 2.0
            return int(W), int(H), float(fx), float(fy), float(cx), float(cy)

        try:
            fx_f = float(fx)
            fy_f = float(fy)
            cx_f = float(cx)
            cy_f = float(cy)
        except Exception:
            fx_f = float(W) / 2.0
            fy_f = float(H) / 2.0
            cx_f = float(W) / 2.0
            cy_f = float(H) / 2.0
            return int(W), int(H), float(fx_f), float(fy_f), float(cx_f), float(cy_f)

        # Scale intrinsics if calibration resolution is provided and differs.
        try:
            if calib_w is not None and calib_h is not None:
                cw = int(calib_w)
                ch = int(calib_h)
                if cw > 0 and ch > 0 and (cw != int(W) or ch != int(H)):
                    sx = float(W) / float(cw)
                    sy = float(H) / float(ch)
                    fx_f *= sx
                    cx_f *= sx
                    fy_f *= sy
                    cy_f *= sy
        except Exception:
            pass

        return int(W), int(H), float(fx_f), float(fy_f), float(cx_f), float(cy_f)

    def _emit_camera_settings() -> None:
        active = _cam_get_active().strip().lower()
        ts = int(time.time() * 1000)

        W = H = None
        fx = fy = cx = cy = None
        device = None
        fmt = None
        src_fps = None
        backend = None

        if active == "v4l":
            device = cfg.get("camera.v4l.device", "auto")
            try:
                if v4l_src is not None and hasattr(v4l_src, "get_device") and callable(getattr(v4l_src, "get_device")):
                    dev_eff = str(v4l_src.get_device() or "").strip()
                    if dev_eff:
                        device = dev_eff
            except Exception:
                pass
            backend = cfg.get("camera.v4l.backend", "auto")
            try:
                if v4l_src is not None and hasattr(v4l_src, "get_backend") and callable(getattr(v4l_src, "get_backend")):
                    be_eff = str(v4l_src.get_backend() or "").strip()
                    if be_eff:
                        backend = be_eff
            except Exception:
                pass
            fmt = cfg.get("camera.v4l.frame_kind", "gray")
            src_fps = cfg.get("camera.v4l.fps", None)
            try:
                W0, H0, fx0, fy0, cx0, cy0 = _track2d_get_intrinsics()
                W, H, fx, fy, cx, cy = int(W0), int(H0), float(fx0), float(fy0), float(cx0), float(cy0)
            except Exception:
                pass
        else:
            device = "track3d_shm" if track3d is not None else "realsense"
            fmt = cfg.get("video.external_raw_format", None) if track3d is not None else cfg.get("video.raw_format", None)
            src_fps = cfg.get("video.fps", None)
            if track3d_intr is not None:
                try:
                    W, H, fx, fy, cx, cy = track3d_intr
                except Exception:
                    W = H = fx = fy = cx = cy = None
            if W is None or H is None or fx is None or fy is None or cx is None or cy is None:
                try:
                    W0, H0, fx0, fy0, cx0, cy0 = video.get_intrinsics()
                    W, H, fx, fy, cx, cy = int(W0), int(H0), float(fx0), float(fy0), float(cx0), float(cy0)
                except Exception:
                    pass

        payload: Dict[str, Any] = {
            "type": "camera",
            "ts": int(ts),
            "active": str(active or ""),
            "source": ("v4l" if active == "v4l" else ("track3d" if track3d is not None else "realsense")),
            "device": device,
            "backend": backend,
            "img_size": ([int(W), int(H)] if W is not None and H is not None else None),
            "fps": (float(src_fps) if isinstance(src_fps, (int, float)) else None),
            "format": (str(fmt) if fmt is not None else None),
            "intrinsics": (
                {"fx": float(fx), "fy": float(fy), "cx": float(cx), "cy": float(cy)}
                if fx is not None and fy is not None and cx is not None and cy is not None
                else None
            ),
        }
        # Trim Nones to keep UDP payload small.
        payload = {k: v for (k, v) in payload.items() if v is not None}
        json_event(payload)

    def _update_video_feed_from_modes() -> None:
        """
        Automatically select streamed video based on the selected tracker mode:
          - Tracking off (none)        -> IR (bw) from track3d SHM ring (`gray`)
          - 3D hole/plane selected    -> IR (bw) from track3d SHM ring (`gray`)
          - 2D ROI tracker selected   -> V4L camera (`camera.v4l.*`)
        """
        try:
            prev_active = _cam_get_active().strip().lower()
            want_active = ""
            want_src = None
            want_intr = None

            if _track2d_is_active():
                want_active = "v4l"
                want_src = v4l_src
                try:
                    want_intr = _track2d_get_intrinsics()
                except Exception:
                    want_intr = None
            else:
                # Default feed when ORB-SLAM3 capture is active: keep the "regular IR" stream.
                if track3d is not None and shm_src_ir is not None:
                    want_active = "ir"
                    want_src = shm_src_ir
                else:
                    if track3d is not None and shm_src_rgb is not None:
                        want_active = "rgb"
                        want_src = shm_src_rgb
                    else:
                        want_active = "realsense"
                        want_src = video_default_src
                want_intr = track3d_intr

            same_active = str(prev_active or "") == str(want_active or "")

            # Always re-attach the desired source (it's just a pointer swap) so we can recover from
            # transient start-order issues (e.g. V4L open delay) even if `camera.active` didn't change.
            if want_src is not None and hasattr(video, "attach_framebus") and callable(getattr(video, "attach_framebus")):
                try:
                    video.attach_framebus(want_src)
                except Exception as e:
                    try:
                        print(f"[VIDEO] attach_framebus error: {e}", flush=True)
                    except Exception:
                        pass
            if (
                want_intr is not None
                and hasattr(video, "set_intrinsics")
                and callable(getattr(video, "set_intrinsics"))
            ):
                try:
                    W0, H0, fx0, fy0, cx0, cy0 = want_intr
                    video.set_intrinsics(int(W0), int(H0), float(fx0), float(fy0), float(cx0), float(cy0))
                except Exception:
                    pass

            if not bool(same_active):
                _cam_set_active(str(want_active or ""))
            _emit_camera_settings()
            try:
                if not bool(same_active):
                    print(f"[VIDEO] active feed -> {want_active}", flush=True)
            except Exception:
                pass
        except Exception as e:
            try:
                print(f"[VIDEO] stream switch error: {e}", flush=True)
            except Exception:
                pass

    # Coalesce rapid mode-control packets (GUI sends 3 enable toggles back-to-back).
    _video_switch_req_lock = threading.Lock()
    _video_switch_lock = threading.Lock()
    _video_switch_evt = threading.Event()
    _video_switch_last_req: Dict[str, float] = {"t": 0.0}
    try:
        _video_switch_debounce_s = float(cfg.get("video.auto_switch.debounce_s", 0.12) or 0.12)
    except Exception:
        _video_switch_debounce_s = 0.12
    _video_switch_debounce_s = float(max(0.0, min(1.0, float(_video_switch_debounce_s))))

    def _request_video_switch() -> None:
        try:
            with _video_switch_req_lock:
                _video_switch_last_req["t"] = float(time.monotonic())
            _video_switch_evt.set()
        except Exception:
            pass

    track2d = None
    if bool(cfg.get("track2d.enabled", True)):
        try:
            v4l_src = V4lMjpegSource(
                V4lMjpegCfg(
                    device=str(cfg.get("camera.v4l.device", "auto")),
                    width=int(cfg.get("camera.v4l.width", 640)),
                    height=int(cfg.get("camera.v4l.height", 480)),
                    fps=int(cfg.get("camera.v4l.fps", 90)),
                    fourcc=str(cfg.get("camera.v4l.fourcc", "MJPG")),
                    backend=str(cfg.get("camera.v4l.backend", "auto")),
                    v4l2ctl_set_format=bool(cfg.get("camera.v4l.v4l2ctl_set_format", True)),
                    v4l2ctl_set_fps=bool(cfg.get("camera.v4l.v4l2ctl_set_fps", True)),
                    controls=(cfg.get("camera.v4l.controls", None) if isinstance(cfg.get("camera.v4l.controls", None), dict) else None),
                    frame_kind=str(cfg.get("camera.v4l.frame_kind", "gray")),
                    name=str(cfg.get("camera.v4l.name", "v4l")),
                ),
                print_fn=print,
            )
            track2d = Track2dShim(
                frame_source=v4l_src,
                get_intrinsics_fn=_track2d_get_intrinsics,
                send_pid_fn=_track2d_send_pid,
                json_event_fn=json_event,
                on_enable_changed=_track2d_on_enable_changed,
                print_fn=print,
            )
            track2d.start()
            print("[track2d] ready (enable via control JSON, uses camera.v4l)", flush=True)
        except Exception as e:
            track2d = None
            v4l_src = None
            print(f"[track2d] init error: {e}", flush=True)

    # Emit initial camera settings snapshot (client may miss; resent after telemetry_out retarget control).
    try:
        _update_video_feed_from_modes()
    except Exception:
        pass

    # ---------------- Control loop (delegates to TelemetryManager) ----------------
    stop = threading.Event()

    # 2D tracker debug: print downward rangefinder height once/sec while 2D tracking mode is enabled.
    t_track2d_rf = None
    if track2d is not None and mav is not None and hasattr(mav, "latest_rangefinder_down_m"):
        def _track2d_rangefinder_debug_loop() -> None:
            last_print_t = 0.0
            while not stop.is_set():
                try:
                    if not _track2d_is_active():
                        time.sleep(0.05)
                        continue
                    now = float(time.monotonic())
                    if (now - float(last_print_t)) < 1.0:
                        time.sleep(0.05)
                        continue
                    last_print_t = float(now)
                    h_m = mav.latest_rangefinder_down_m(max_age_s=float(track2d_rangefinder_max_age_s))  # type: ignore[attr-defined]
                    if h_m is None:
                        print(
                            f"[track2d] rangefinder_down_m=None max_age_s={float(track2d_rangefinder_max_age_s):.3f} thr={float(track2d_min_rangefinder_m):.3f}",
                            flush=True,
                        )
                    else:
                        print(
                            f"[track2d] rangefinder_down_m={float(h_m):.3f} max_age_s={float(track2d_rangefinder_max_age_s):.3f} thr={float(track2d_min_rangefinder_m):.3f}",
                            flush=True,
                        )
                except Exception:
                    pass
                time.sleep(0.05)

        t_track2d_rf = threading.Thread(
            target=_track2d_rangefinder_debug_loop,
            daemon=True,
            name="track2d-rangefinder",
        )
        t_track2d_rf.start()

    def _video_switch_loop() -> None:
        """
        Debounced video switching to avoid transient stream flips during mode transitions.
        """
        while not stop.is_set():
            try:
                if not _video_switch_evt.wait(timeout=0.5):
                    continue
            except Exception:
                continue
            if stop.is_set():
                break

            # Wait for quiet period (coalesce rapid back-to-back enable packets).
            try:
                while not stop.is_set():
                    with _video_switch_req_lock:
                        last_t = float(_video_switch_last_req.get("t", 0.0))
                    now = float(time.monotonic())
                    dt = now - last_t
                    if dt >= float(_video_switch_debounce_s):
                        break
                    time.sleep(max(0.0, float(_video_switch_debounce_s) - float(dt)))
            except Exception:
                pass
            if stop.is_set():
                break

            try:
                _video_switch_evt.clear()
            except Exception:
                pass
            try:
                with _video_switch_lock:
                    _update_video_feed_from_modes()
            except Exception:
                pass

    t_video_switch = threading.Thread(target=_video_switch_loop, daemon=True, name="video-switch")
    t_video_switch.start()

    # MAVLink external-vision TX counters (used by FPS logger).
    vpe_stats_lock = threading.Lock()
    vpe_stats: Dict[str, int] = {
        "sent": 0,
        "tx_err": 0,
        "skip_no_pose": 0,
        "skip_stale": 0,
        "skip_src": 0,
        "skip_gate": 0,
        "skip_no_pos": 0,
    }

    # PX4 external-vision pose injection (MAVLink VISION_POSITION_ESTIMATE, msg 102).
    # Driven from the same VO pose snapshot used for GUI `type=pose` JSON.
    t_vpe = None
    t_vse = None
    if (
        mav is not None
        and bool(getattr(mav, "enabled", True))
        and bool(cfg.get("px4.enabled", True))
        and bool(cfg.get("px4.vision_position_estimate.enabled", False))
    ):
        # Emit one VPE per new VO pose update (no resampling).
        try:
            vpe_stale_timeout_s = float(cfg.get("px4.vision_position_estimate.stale_timeout_s", 0.2))
        except Exception:
            vpe_stale_timeout_s = 0.2
        vpe_stale_timeout_s = float(max(0.0, float(vpe_stale_timeout_s)))
        vpe_require_ok = bool(cfg.get("px4.vision_position_estimate.require_ok", True))
        vpe_require_stable = bool(cfg.get("px4.vision_position_estimate.require_stable", False))
        vpe_source = str(cfg.get("px4.vision_position_estimate.source", "track3d") or "").strip().lower()

        try:
            axis_sign_cfg = dict(cfg.get("px4.vision_position_estimate.axis_sign", {}) or {})
        except Exception:
            axis_sign_cfg = {}
        try:
            axis_map_cfg = dict(cfg.get("px4.vision_position_estimate.axis_map", {}) or {})
        except Exception:
            axis_map_cfg = {}
        map_x = str(axis_map_cfg.get("x", "x") or "").strip().lower()
        map_y = str(axis_map_cfg.get("y", "y") or "").strip().lower()
        map_z = str(axis_map_cfg.get("z", "z") or "").strip().lower()
        if map_x not in ("x", "y", "z"):
            map_x = "x"
        if map_y not in ("x", "y", "z"):
            map_y = "y"
        if map_z not in ("x", "y", "z"):
            map_z = "z"
        sign_x = float(axis_sign_cfg.get("x", 1.0))
        sign_y = float(axis_sign_cfg.get("y", 1.0))
        sign_z = float(axis_sign_cfg.get("z", 1.0))
        try:
            yaw_sign = float(cfg.get("px4.vision_position_estimate.yaw_sign", 1.0))
        except Exception:
            yaw_sign = 1.0
        try:
            yaw_offset_deg = float(cfg.get("px4.vision_position_estimate.yaw_offset_deg", 0.0))
        except Exception:
            yaw_offset_deg = 0.0
        try:
            vpe_pos_scale = float(cfg.get("px4.vision_position_estimate.pos_scale", 1.0))
        except Exception:
            vpe_pos_scale = 1.0
        if (not math.isfinite(float(vpe_pos_scale))) or float(vpe_pos_scale) <= 0.0:
            vpe_pos_scale = 1.0

        cov_unknown = bool(cfg.get("px4.vision_position_estimate.covariance_unknown", True))
        try:
            reset_counter = int(cfg.get("px4.vision_position_estimate.reset_counter", 0))
        except Exception:
            reset_counter = 0

        def _vpe_loop() -> None:
            use_ext = bool(cov_unknown)
            cov21 = ([float("nan")] + [0.0] * 20) if bool(use_ext) else None
            reset_counter_send = int(reset_counter) if bool(use_ext) else None
            last_seq = -1
            while not stop.is_set():
                # Block until a new VO pose update is available (or timeout to check stop).
                if hasattr(telem, "wait_vo_pose_snapshot"):
                    try:
                        seq, vo_pose, vo_wall_s = telem.wait_vo_pose_snapshot(  # type: ignore[attr-defined]
                            last_seq=int(last_seq),
                            timeout_s=0.5,
                        )
                    except Exception:
                        seq, vo_pose, vo_wall_s = last_seq, None, 0.0
                    if int(seq) == int(last_seq):
                        continue
                    last_seq = int(seq)
                elif hasattr(telem, "get_vo_pose_snapshot"):
                    vo_pose, vo_wall_s = telem.get_vo_pose_snapshot()  # type: ignore[attr-defined]
                    time.sleep(0.01)
                else:
                    vo_pose, vo_wall_s = None, 0.0
                if not isinstance(vo_pose, dict) or float(vo_wall_s) <= 0.0:
                    # No VO pose yet.
                    try:
                        with vpe_stats_lock:
                            vpe_stats["skip_no_pose"] = int(vpe_stats.get("skip_no_pose", 0)) + 1
                    except Exception:
                        pass
                else:
                    age_s = float(time.time()) - float(vo_wall_s)
                    if float(vpe_stale_timeout_s) > 0.0 and float(age_s) > float(vpe_stale_timeout_s):
                        try:
                            with vpe_stats_lock:
                                vpe_stats["skip_stale"] = int(vpe_stats.get("skip_stale", 0)) + 1
                        except Exception:
                            pass
                    else:
                        if vpe_source:
                            src0 = str(vo_pose.get("source", "") or "").strip().lower()
                            if src0 != vpe_source:
                                try:
                                    with vpe_stats_lock:
                                        vpe_stats["skip_src"] = int(vpe_stats.get("skip_src", 0)) + 1
                                except Exception:
                                    pass
                            else:
                                ok_gate = (not bool(vpe_require_ok)) or bool(vo_pose.get("ok", True))
                                stable_gate = (not bool(vpe_require_stable)) or bool(vo_pose.get("stable", True))
                                if ok_gate and stable_gate:
                                    pos = vo_pose.get("position_m", None)
                                    if isinstance(pos, dict):
                                        x = float(pos.get(map_x, 0.0)) * float(sign_x) * float(vpe_pos_scale)
                                        y = float(pos.get(map_y, 0.0)) * float(sign_y) * float(vpe_pos_scale)
                                        z = float(pos.get(map_z, 0.0)) * float(sign_z) * float(vpe_pos_scale)

                                        # Prefer full orientation (roll/pitch/yaw) when available, otherwise send yaw only.
                                        ang0 = vo_pose.get("orientation_deg", None)
                                        have_ang = isinstance(ang0, dict)
                                        roll_deg = float(ang0.get("roll", 0.0)) if have_ang else 0.0
                                        pitch_deg = float(ang0.get("pitch", 0.0)) if have_ang else 0.0
                                        yaw_deg0 = (ang0.get("yaw", None) if have_ang else None)
                                        if not isinstance(yaw_deg0, (int, float)):
                                            yaw_deg0 = vo_pose.get("yaw_deg", None)
                                        yaw_deg = float(yaw_deg0) if isinstance(yaw_deg0, (int, float)) else 0.0
                                        yaw_deg = float(yaw_sign) * float(yaw_deg + float(yaw_offset_deg))

                                        try:
                                            pose_send: Dict[str, Any] = {"position_m": {"x": x, "y": y, "z": z}}
                                            if have_ang:
                                                pose_send["orientation_deg"] = {
                                                    "roll": float(roll_deg),
                                                    "pitch": float(pitch_deg),
                                                    "yaw": float(yaw_deg),
                                                }
                                            else:
                                                pose_send["yaw_deg"] = float(yaw_deg)
                                            ok_sent = mav.send_external_vision_pose(
                                                pose_send,
                                                covariance=cov21,
                                                reset_counter=reset_counter_send,
                                            )
                                            try:
                                                with vpe_stats_lock:
                                                    if bool(ok_sent):
                                                        vpe_stats["sent"] = int(vpe_stats.get("sent", 0)) + 1
                                                    else:
                                                        vpe_stats["tx_err"] = int(vpe_stats.get("tx_err", 0)) + 1
                                            except Exception:
                                                pass
                                        except Exception:
                                            try:
                                                with vpe_stats_lock:
                                                    vpe_stats["tx_err"] = int(vpe_stats.get("tx_err", 0)) + 1
                                            except Exception:
                                                pass
                                    else:
                                        try:
                                            with vpe_stats_lock:
                                                vpe_stats["skip_no_pos"] = int(vpe_stats.get("skip_no_pos", 0)) + 1
                                        except Exception:
                                            pass
                                else:
                                    try:
                                        with vpe_stats_lock:
                                            vpe_stats["skip_gate"] = int(vpe_stats.get("skip_gate", 0)) + 1
                                    except Exception:
                                        pass

        t_vpe = threading.Thread(target=_vpe_loop, daemon=True, name="vpe")
        t_vpe.start()
        try:
            print(
                f"[MAV] VISION_POSITION_ESTIMATE enabled mode=on_update "
                f"stale_timeout_s={float(vpe_stale_timeout_s):.3f} src={vpe_source or '-'} ext={int(bool(cov_unknown))}",
                flush=True,
            )
        except Exception:
            pass

    # PX4 external-vision speed injection (MAVLink VISION_SPEED_ESTIMATE, msg 103).
    # Derived from the VO position via finite differences (in the same mapped frame used for VPE).
    if (
        mav is not None
        and bool(getattr(mav, "enabled", True))
        and bool(cfg.get("px4.enabled", True))
        and bool(cfg.get("px4.vision_speed_estimate.enabled", False))
    ):
        try:
            vse_rate_hz = float(cfg.get("px4.vision_speed_estimate.rate_hz", 30.0))
        except Exception:
            vse_rate_hz = 30.0
        vse_rate_hz = float(max(0.0, float(vse_rate_hz)))
        try:
            vse_stale_timeout_s = float(cfg.get("px4.vision_speed_estimate.stale_timeout_s", 0.2))
        except Exception:
            vse_stale_timeout_s = 0.2
        vse_stale_timeout_s = float(max(0.0, float(vse_stale_timeout_s)))
        vse_require_ok = bool(cfg.get("px4.vision_speed_estimate.require_ok", True))
        vse_require_stable = bool(cfg.get("px4.vision_speed_estimate.require_stable", False))
        vse_source = str(cfg.get("px4.vision_speed_estimate.source", "track3d") or "").strip().lower()

        # Defaults follow vision_position_estimate (when configured), otherwise identity.
        try:
            axis_map_base_cfg = dict(cfg.get("px4.vision_position_estimate.axis_map", {}) or {})
        except Exception:
            axis_map_base_cfg = {}
        try:
            axis_sign_base_cfg = dict(cfg.get("px4.vision_position_estimate.axis_sign", {}) or {})
        except Exception:
            axis_sign_base_cfg = {}
        base_map_x = str(axis_map_base_cfg.get("x", "x") or "").strip().lower()
        base_map_y = str(axis_map_base_cfg.get("y", "y") or "").strip().lower()
        base_map_z = str(axis_map_base_cfg.get("z", "z") or "").strip().lower()
        if base_map_x not in ("x", "y", "z"):
            base_map_x = "x"
        if base_map_y not in ("x", "y", "z"):
            base_map_y = "y"
        if base_map_z not in ("x", "y", "z"):
            base_map_z = "z"
        base_sign_x = float(axis_sign_base_cfg.get("x", 1.0))
        base_sign_y = float(axis_sign_base_cfg.get("y", 1.0))
        base_sign_z = float(axis_sign_base_cfg.get("z", 1.0))
        try:
            base_pos_scale = float(cfg.get("px4.vision_position_estimate.pos_scale", 1.0))
        except Exception:
            base_pos_scale = 1.0
        if (not math.isfinite(float(base_pos_scale))) or float(base_pos_scale) <= 0.0:
            base_pos_scale = 1.0
        try:
            vse_pos_scale = float(cfg.get("px4.vision_speed_estimate.pos_scale", base_pos_scale))
        except Exception:
            vse_pos_scale = float(base_pos_scale)
        if (not math.isfinite(float(vse_pos_scale))) or float(vse_pos_scale) <= 0.0:
            vse_pos_scale = float(base_pos_scale)

        try:
            axis_sign_cfg_v = dict(cfg.get("px4.vision_speed_estimate.axis_sign", {}) or {})
        except Exception:
            axis_sign_cfg_v = {}
        try:
            axis_map_cfg_v = dict(cfg.get("px4.vision_speed_estimate.axis_map", {}) or {})
        except Exception:
            axis_map_cfg_v = {}
        v_map_x = str(axis_map_cfg_v.get("x", base_map_x) or "").strip().lower()
        v_map_y = str(axis_map_cfg_v.get("y", base_map_y) or "").strip().lower()
        v_map_z = str(axis_map_cfg_v.get("z", base_map_z) or "").strip().lower()
        if v_map_x not in ("x", "y", "z"):
            v_map_x = base_map_x
        if v_map_y not in ("x", "y", "z"):
            v_map_y = base_map_y
        if v_map_z not in ("x", "y", "z"):
            v_map_z = base_map_z
        v_sign_x = float(axis_sign_cfg_v.get("x", base_sign_x))
        v_sign_y = float(axis_sign_cfg_v.get("y", base_sign_y))
        v_sign_z = float(axis_sign_cfg_v.get("z", base_sign_z))

        use_src_ts = bool(cfg.get("px4.vision_speed_estimate.use_source_ts", True))
        try:
            min_dt_s = float(cfg.get("px4.vision_speed_estimate.min_dt_s", 0.005))
        except Exception:
            min_dt_s = 0.005
        try:
            max_dt_s = float(cfg.get("px4.vision_speed_estimate.max_dt_s", 0.25))
        except Exception:
            max_dt_s = 0.25
        min_dt_s = float(max(0.0, float(min_dt_s)))
        max_dt_s = float(max(0.0, float(max_dt_s)))
        if max_dt_s > 0.0 and max_dt_s < min_dt_s:
            max_dt_s = min_dt_s
        try:
            ema_alpha = float(cfg.get("px4.vision_speed_estimate.ema_alpha", 0.0))
        except Exception:
            ema_alpha = 0.0
        ema_alpha = float(max(0.0, min(1.0, float(ema_alpha))))

        def _vse_loop() -> None:
            if vse_rate_hz <= 0.0:
                return
            period = 1.0 / max(1e-6, float(vse_rate_hz))
            next_t = time.time()

            prev_p = None  # type: Optional[Tuple[float, float, float]]
            prev_wall = None  # type: Optional[float]
            prev_src_ts = None  # type: Optional[float]
            v_ema = None  # type: Optional[Tuple[float, float, float]]

            while not stop.is_set():
                next_t += period
                if hasattr(telem, "get_vo_pose_snapshot"):
                    vo_pose, vo_wall_s = telem.get_vo_pose_snapshot()  # type: ignore[attr-defined]
                else:
                    vo_pose, vo_wall_s = None, 0.0
                if not isinstance(vo_pose, dict) or float(vo_wall_s) <= 0.0:
                    pass
                else:
                    age_s = float(time.time()) - float(vo_wall_s)
                    if float(vse_stale_timeout_s) > 0.0 and float(age_s) > float(vse_stale_timeout_s):
                        pass
                    else:
                        if vse_source:
                            src0 = str(vo_pose.get("source", "") or "").strip().lower()
                            if src0 != vse_source:
                                pass
                            else:
                                ok_gate = (not bool(vse_require_ok)) or bool(vo_pose.get("ok", True))
                                stable_gate = (not bool(vse_require_stable)) or bool(vo_pose.get("stable", True))
                                if ok_gate and stable_gate:
                                    pos = vo_pose.get("position_m", None)
                                    if isinstance(pos, dict):
                                        try:
                                            x = float(pos.get(v_map_x, 0.0)) * float(v_sign_x) * float(vse_pos_scale)
                                            y = float(pos.get(v_map_y, 0.0)) * float(v_sign_y) * float(vse_pos_scale)
                                            z = float(pos.get(v_map_z, 0.0)) * float(v_sign_z) * float(vse_pos_scale)
                                        except Exception:
                                            x, y, z = 0.0, 0.0, 0.0

                                        cur_p = (float(x), float(y), float(z))
                                        cur_wall = float(vo_wall_s)
                                        cur_src_ts = None
                                        try:
                                            ts0 = vo_pose.get("ts_s", None)
                                            if isinstance(ts0, (int, float)):
                                                cur_src_ts = float(ts0)
                                        except Exception:
                                            cur_src_ts = None

                                        dt = None
                                        if bool(use_src_ts) and cur_src_ts is not None and prev_src_ts is not None:
                                            dts = float(cur_src_ts) - float(prev_src_ts)
                                            if math.isfinite(dts) and dts > 0.0:
                                                dt = float(dts)
                                        if dt is None and prev_wall is not None:
                                            dts = float(cur_wall) - float(prev_wall)
                                            if math.isfinite(dts) and dts > 0.0:
                                                dt = float(dts)

                                        if dt is not None:
                                            if float(min_dt_s) > 0.0 and float(dt) < float(min_dt_s):
                                                dt = None
                                            if dt is not None and float(max_dt_s) > 0.0 and float(dt) > float(max_dt_s):
                                                dt = None

                                        if dt is not None and prev_p is not None:
                                            vx = (float(cur_p[0]) - float(prev_p[0])) / float(dt)
                                            vy = (float(cur_p[1]) - float(prev_p[1])) / float(dt)
                                            vz = (float(cur_p[2]) - float(prev_p[2])) / float(dt)

                                            if float(ema_alpha) > 0.0 and v_ema is not None:
                                                a = float(ema_alpha)
                                                vx = a * float(vx) + (1.0 - a) * float(v_ema[0])
                                                vy = a * float(vy) + (1.0 - a) * float(v_ema[1])
                                                vz = a * float(vz) + (1.0 - a) * float(v_ema[2])
                                            v_ema = (float(vx), float(vy), float(vz))

                                        if v_ema is not None:
                                            try:
                                                mav.send_external_vision_speed(
                                                    {"velocity_mps": {"x": float(v_ema[0]), "y": float(v_ema[1]), "z": float(v_ema[2])}}
                                                )
                                            except Exception:
                                                pass

                                        prev_p = cur_p
                                        prev_wall = cur_wall
                                        prev_src_ts = cur_src_ts if cur_src_ts is not None else prev_src_ts

                delay = float(next_t) - float(time.time())
                if float(delay) > 0.0:
                    time.sleep(delay)
                else:
                    next_t = time.time()

        t_vse = threading.Thread(target=_vse_loop, daemon=True, name="vse")
        t_vse.start()
        try:
            print(
                f"[MAV] VISION_SPEED_ESTIMATE enabled rate={float(vse_rate_hz):.1f}Hz "
                f"stale_timeout_s={float(vse_stale_timeout_s):.3f} src={vse_source or '-'}",
                flush=True,
            )
        except Exception:
            pass

    # Track3d-mode perf counters (updated by the track3d pump thread; read by FPS logger).
    t3_lock = threading.Lock()
    t3_evt_total = 0
    t3_evt_preview = 0
    t3_evt_telem = 0
    t3_evt_features = 0
    t3_evt_pid = 0
    t3_last_evt_wall = 0.0
    # Latest track3d state counters (from telemetry payloads).
    t3_last_telem_wall = 0.0
    t3_last_lk_wall = 0.0
    t3_last_lk_fi = -1
    t3_last_lk_ts = 0.0
    t3_last_lk_n_tracks = 0
    t3_last_lk_n_corr = 0
    t3_last_lk_of_ms = 0.0
    t3_last_lk_ms = 0.0
    t3_last_lk_reseed_ms = 0.0
    t3_last_lk_depth_ms = 0.0
    t3_last_lk_corr_ms = 0.0
    t3_last_lk_err_pre_n = 0
    t3_last_lk_err_post_n = 0
    t3_last_lk_err_rej_n = 0
    t3_last_lk_err_med = float("nan")
    t3_last_lk_err_mad = float("nan")
    t3_last_lk_err_sigma = float("nan")
    t3_last_lk_err_thr = float("nan")
    t3_last_state_fi = -1
    t3_last_state_ts = 0.0
    t3_last_state_corr = 0
    t3_last_state_inliers = 0
    t3_last_state_rmse_m = 0.0
    t3_last_state_status = 0
    t3_last_state_est_code = 0
    t3_last_state_est = ""
    t3_last_state_ms_est = 0.0
    t3_last_state_ms_total = 0.0
    t3_last_state_ms_prefilter = 0.0
    t3_last_state_ms_pg = 0.0
    t3_last_state_ms_jitter = 0.0
    t3_last_state_ms_imu = 0.0
    t3_last_state_ms_weights = 0.0
    t3_last_state_ms_gate = 0.0
    t3_hole_fail_total = 0
    t3_hole_fail_sector_low = 0
    t3_hole_fail_leak = 0
    t3_hole_fail_other = 0
    mouse_total = 0
    mouse_move = 0
    mouse_stop = 0
    mouse_click = 0
    mouse_last_wall = 0.0

    # 3D tracker event pump: forward polygons to GUI and wire dhv_deg -> existing PID->MAVLink path.
    t_track3d = None
    if track3d is not None:
        def _track3d_loop():
            nonlocal t3_evt_total, t3_evt_preview, t3_evt_telem, t3_evt_features, t3_evt_pid
            nonlocal t3_last_evt_wall, t3_last_telem_wall
            nonlocal t3_last_lk_wall, t3_last_lk_fi, t3_last_lk_ts, t3_last_lk_n_tracks, t3_last_lk_n_corr, t3_last_lk_of_ms, t3_last_lk_ms, t3_last_lk_reseed_ms, t3_last_lk_depth_ms, t3_last_lk_corr_ms
            nonlocal t3_last_lk_err_pre_n, t3_last_lk_err_post_n, t3_last_lk_err_rej_n, t3_last_lk_err_med, t3_last_lk_err_mad, t3_last_lk_err_sigma, t3_last_lk_err_thr
            nonlocal t3_last_state_fi, t3_last_state_ts, t3_last_state_corr, t3_last_state_inliers, t3_last_state_rmse_m, t3_last_state_status
            nonlocal t3_last_state_est_code, t3_last_state_est
            nonlocal t3_last_state_ms_est, t3_last_state_ms_total, t3_last_state_ms_prefilter, t3_last_state_ms_pg
            nonlocal t3_last_state_ms_jitter, t3_last_state_ms_imu, t3_last_state_ms_weights, t3_last_state_ms_gate
            nonlocal t3_hole_fail_total, t3_hole_fail_sector_low, t3_hole_fail_leak, t3_hole_fail_other
            pid_mod = None
            try:
                import pid as pid_mod  # type: ignore
            except Exception:
                pid_mod = None

            pid_active = False
            pid_move_auto_exit_latched = False
            poly_was_active = False
            last_hole_reason = ""
            last_plane_reason = ""

            hole_log_fh = None
            try:
                hole_log_path = str(cfg.get("debug.track3d_hole_log_path", "./logs/track3d_hole.log") or "").strip()
                if hole_log_path:
                    os.makedirs(os.path.dirname(hole_log_path) or ".", exist_ok=True)
                    hole_log_fh = open(hole_log_path, "a", encoding="utf-8")
                    print(f"[track3d] hole_log={hole_log_path}", flush=True)
            except Exception:
                hole_log_fh = None

            plane_log_fh = None
            try:
                plane_log_path = str(cfg.get("debug.track3d_plane_log_path", "./logs/track3d_plane.log") or "").strip()
                if plane_log_path:
                    os.makedirs(os.path.dirname(plane_log_path) or ".", exist_ok=True)
                    plane_log_fh = open(plane_log_path, "a", encoding="utf-8")
                    print(f"[track3d] plane_log={plane_log_path}", flush=True)
            except Exception:
                plane_log_fh = None

            # VO feature overlays (for GUI): cache latest points and emit at a steady 30 Hz.
            feat_last_pts = None  # List[Tuple[int,int,int]] (u,v,group)
            feat_last_fi = None
            feat_last_update_t = 0.0
            feat_last_send_t = 0.0
            feat_period_s = 1.0 / 30.0
            feat_stale_s = 1.0
            # Feature overlay payload sizing (UDP JSON).
            # - max_pts: cap total points shown (<=0 = send all)
            # - chunk_pts: if >0, split the outgoing `vo_features` into multiple packets
            #   so each UDP datagram stays small (avoids fragmentation loss on WiFi).
            try:
                feat_max_pts = int(cfg.get("track3d.features.max_pts", 80))
            except Exception:
                feat_max_pts = 80
            try:
                feat_chunk_pts = int(cfg.get("track3d.features.chunk_pts", 60))
            except Exception:
                feat_chunk_pts = 60
            if feat_chunk_pts < 0:
                feat_chunk_pts = 0

            # Safety: stop tracking if the polygon grows to (almost) full-frame.
            # This usually indicates selection drift / bad lock and can lead to unstable PID.
            try:
                poly_stop_extent_frac = float(cfg.get("track3d.poly_stop_extent_frac", 0.95))
            except Exception:
                poly_stop_extent_frac = 0.95
            poly_stop_extent_frac = float(max(0.0, min(1.0, float(poly_stop_extent_frac))))

            # Track3d plane auto-exit: stop when we are close enough to the selected plane (meters).
            try:
                track3d_plane_min_range_m = float(cfg.get("track3d.auto_exit.plane_min_range_m", 2.0) or 2.0)
            except Exception:
                track3d_plane_min_range_m = 2.0
            track3d_plane_min_range_m = float(max(0.0, float(track3d_plane_min_range_m)))

            try:
                while not stop.is_set():
                    evts = track3d.poll_events(max_n=200)
                    for evt in (evts or []):
                        try:
                            track3d.apply_event(evt)
                        except Exception:
                            pass
                        et = str(evt.get("type", "") or "").strip().lower()
                        payload = evt.get("payload", None)
                        try:
                            with t3_lock:
                                t3_evt_total += 1
                                t3_last_evt_wall = float(time.time())
                                if et == "preview":
                                    t3_evt_preview += 1
                                elif et == "telemetry":
                                    t3_evt_telem += 1
                                elif et == "features":
                                    t3_evt_features += 1
                                elif et == "pid":
                                    t3_evt_pid += 1
                        except Exception:
                            pass

                        # Stage 0: hover preview polygon.
                        if et == "preview" and isinstance(payload, dict):
                            try:
                                sel_kind = payload.get("sel_kind", None)
                                verts = payload.get("verts_uv", None)
                                verts_uv = []
                                if isinstance(verts, list):
                                    for p in verts:
                                        if isinstance(p, (list, tuple)) and len(p) >= 2:
                                            verts_uv.append((float(p[0]), float(p[1])))
                                msg = track3d.make_acq_poly_msg(stage=0, verts_uv=verts_uv, sel_kind=sel_kind)
                                if msg is not None:
                                    json_event(msg)
                            except Exception:
                                pass
                            continue

                        # Stage 1/2: active polygon state (rate-limited by tracker telemetry).
                        if et == "telemetry" and isinstance(payload, dict):
                            try:
                                # VO pose (track3d) -> GUI "pose" JSON (telemetry.py publishes at pose_hz).
                                try:
                                    # Track3d state cadence diagnostics (used by the server FPS logger).
                                    try:
                                        st0 = payload.get("state", None)
                                        st_fi0 = int(st0.get("fi", -1)) if isinstance(st0, dict) else -1
                                        st_ts0 = float(st0.get("ts", 0.0)) if isinstance(st0, dict) else 0.0
                                        st_corr0 = int(st0.get("corr", 0)) if isinstance(st0, dict) else 0
                                        st_in0 = int(st0.get("inliers", 0)) if isinstance(st0, dict) else 0
                                        st_rmse0 = float(st0.get("rmse_m", 0.0)) if isinstance(st0, dict) else 0.0
                                        st_status0 = int(st0.get("status_code", 0)) if isinstance(st0, dict) else 0
                                        st_est_code0 = int(st0.get("est_code", 0)) if isinstance(st0, dict) else 0
                                        st_est0 = str(st0.get("est", "") or "") if isinstance(st0, dict) else ""
                                        st_ms_est0 = float(st0.get("ms_est", 0.0)) if isinstance(st0, dict) else 0.0
                                        st_ms_total0 = float(st0.get("ms_total", 0.0)) if isinstance(st0, dict) else 0.0
                                        st_ms_prefilter0 = float(st0.get("ms_prefilter", 0.0)) if isinstance(st0, dict) else 0.0
                                        st_ms_pg0 = float(st0.get("ms_pg", 0.0)) if isinstance(st0, dict) else 0.0
                                        st_ms_jitter0 = float(st0.get("ms_jitter", 0.0)) if isinstance(st0, dict) else 0.0
                                        st_ms_imu0 = float(st0.get("ms_imu", 0.0)) if isinstance(st0, dict) else 0.0
                                        st_ms_weights0 = float(st0.get("ms_weights", 0.0)) if isinstance(st0, dict) else 0.0
                                        st_ms_gate0 = float(st0.get("ms_gate", 0.0)) if isinstance(st0, dict) else 0.0
                                    except Exception:
                                        st_fi0 = -1
                                        st_ts0 = 0.0
                                        st_corr0 = 0
                                        st_in0 = 0
                                        st_rmse0 = 0.0
                                        st_status0 = 0
                                        st_est_code0 = 0
                                        st_est0 = ""
                                        st_ms_est0 = 0.0
                                        st_ms_total0 = 0.0
                                        st_ms_prefilter0 = 0.0
                                        st_ms_pg0 = 0.0
                                        st_ms_jitter0 = 0.0
                                        st_ms_imu0 = 0.0
                                        st_ms_weights0 = 0.0
                                        st_ms_gate0 = 0.0
                                    try:
                                        lk0 = payload.get("lk", None)
                                        lk_fi0 = int(lk0.get("fi", -1)) if isinstance(lk0, dict) else -1
                                        lk_ts0 = float(lk0.get("cur_ts", 0.0)) if isinstance(lk0, dict) else 0.0
                                        lk_nt0 = int(lk0.get("n_tracks", 0)) if isinstance(lk0, dict) else 0
                                        lk_nc0 = int(lk0.get("n_corr", 0)) if isinstance(lk0, dict) else 0
                                        lk_of_ms0 = float(lk0.get("lk_of_ms", 0.0)) if isinstance(lk0, dict) else 0.0
                                        lk_ms0 = float(lk0.get("lk_ms", 0.0)) if isinstance(lk0, dict) else 0.0
                                        lk_reseed_ms0 = float(lk0.get("lk_reseed_ms", 0.0)) if isinstance(lk0, dict) else 0.0
                                        lk_depth_ms0 = float(lk0.get("lk_depth_ms", 0.0)) if isinstance(lk0, dict) else 0.0
                                        lk_corr_ms0 = float(lk0.get("lk_corr_ms", 0.0)) if isinstance(lk0, dict) else 0.0
                                        lk_err_pre_n0 = int(lk0.get("err_pre_n", 0)) if isinstance(lk0, dict) else 0
                                        lk_err_post_n0 = int(lk0.get("err_post_n", 0)) if isinstance(lk0, dict) else 0
                                        lk_err_rej_n0 = int(lk0.get("err_rej_n", 0)) if isinstance(lk0, dict) else 0
                                        try:
                                            lk_err_med0 = float(lk0.get("err_med")) if isinstance(lk0, dict) and lk0.get("err_med", None) is not None else float("nan")
                                        except Exception:
                                            lk_err_med0 = float("nan")
                                        try:
                                            lk_err_mad0 = float(lk0.get("err_mad")) if isinstance(lk0, dict) and lk0.get("err_mad", None) is not None else float("nan")
                                        except Exception:
                                            lk_err_mad0 = float("nan")
                                        try:
                                            lk_err_sigma0 = float(lk0.get("err_sigma")) if isinstance(lk0, dict) and lk0.get("err_sigma", None) is not None else float("nan")
                                        except Exception:
                                            lk_err_sigma0 = float("nan")
                                        try:
                                            lk_err_thr0 = float(lk0.get("err_thr")) if isinstance(lk0, dict) and lk0.get("err_thr", None) is not None else float("nan")
                                        except Exception:
                                            lk_err_thr0 = float("nan")
                                    except Exception:
                                        lk_fi0 = -1
                                        lk_ts0 = 0.0
                                        lk_nt0 = 0
                                        lk_nc0 = 0
                                        lk_of_ms0 = 0.0
                                        lk_ms0 = 0.0
                                        lk_reseed_ms0 = 0.0
                                        lk_depth_ms0 = 0.0
                                        lk_corr_ms0 = 0.0
                                        lk_err_pre_n0 = 0
                                        lk_err_post_n0 = 0
                                        lk_err_rej_n0 = 0
                                        lk_err_med0 = float("nan")
                                        lk_err_mad0 = float("nan")
                                        lk_err_sigma0 = float("nan")
                                        lk_err_thr0 = float("nan")
                                    try:
                                        with t3_lock:
                                            t3_last_state_fi = int(st_fi0)
                                            t3_last_state_ts = float(st_ts0)
                                            t3_last_state_corr = int(st_corr0)
                                            t3_last_state_inliers = int(st_in0)
                                            t3_last_state_rmse_m = float(st_rmse0)
                                            t3_last_state_status = int(st_status0)
                                            t3_last_state_est_code = int(st_est_code0)
                                            t3_last_state_est = str(st_est0)
                                            t3_last_state_ms_est = float(st_ms_est0)
                                            t3_last_state_ms_total = float(st_ms_total0)
                                            t3_last_state_ms_prefilter = float(st_ms_prefilter0)
                                            t3_last_state_ms_pg = float(st_ms_pg0)
                                            t3_last_state_ms_jitter = float(st_ms_jitter0)
                                            t3_last_state_ms_imu = float(st_ms_imu0)
                                            t3_last_state_ms_weights = float(st_ms_weights0)
                                            t3_last_state_ms_gate = float(st_ms_gate0)
                                            t3_last_telem_wall = float(time.time())
                                            if int(lk_fi0) >= 0:
                                                t3_last_lk_fi = int(lk_fi0)
                                                t3_last_lk_ts = float(lk_ts0)
                                                t3_last_lk_n_tracks = int(lk_nt0)
                                                t3_last_lk_n_corr = int(lk_nc0)
                                                t3_last_lk_of_ms = float(lk_of_ms0)
                                                t3_last_lk_ms = float(lk_ms0)
                                                t3_last_lk_reseed_ms = float(lk_reseed_ms0)
                                                t3_last_lk_depth_ms = float(lk_depth_ms0)
                                                t3_last_lk_corr_ms = float(lk_corr_ms0)
                                                t3_last_lk_err_pre_n = int(lk_err_pre_n0)
                                                t3_last_lk_err_post_n = int(lk_err_post_n0)
                                                t3_last_lk_err_rej_n = int(lk_err_rej_n0)
                                                t3_last_lk_err_med = float(lk_err_med0)
                                                t3_last_lk_err_mad = float(lk_err_mad0)
                                                t3_last_lk_err_sigma = float(lk_err_sigma0)
                                                t3_last_lk_err_thr = float(lk_err_thr0)
                                                t3_last_lk_wall = float(time.time())
                                    except Exception:
                                        pass

                                    if telem is not None:
                                        try:
                                            st = payload.get("state", None)
                                            ps = payload.get("pose", None)
                                            ps_map = payload.get("pose_map", None)
                                            ok = bool(st.get("ok", False)) if isinstance(st, dict) else False
                                            stable = bool(st.get("stable", False)) if isinstance(st, dict) else False
                                            status_code = int(st.get("status_code", 0)) if isinstance(st, dict) else 0
                                            inliers = int(st.get("inliers", 0)) if isinstance(st, dict) else 0
                                            rmse_m = float(st.get("rmse_m", 0.0)) if isinstance(st, dict) else 0.0

                                            ts_s = None
                                            fi = None
                                            try:
                                                if isinstance(st, dict):
                                                    ts0 = st.get("ts", None)
                                                    if isinstance(ts0, (int, float)):
                                                        ts_s = float(ts0)
                                                    fi0 = st.get("fi", None)
                                                    if isinstance(fi0, (int, float)):
                                                        fi = int(fi0)
                                            except Exception:
                                                ts_s = None
                                                fi = None

                                            pos = None
                                            yaw_deg = None
                                            roll_deg = None
                                            pitch_deg = None
                                            drift_m = None
                                            if isinstance(ps, dict):
                                                pos = ps.get("pos_rel_w_m", None) or ps.get("pos_w_m", None)
                                                yaw_deg = ps.get("yaw_deg", None)
                                                roll_deg = ps.get("roll_deg", None)
                                                pitch_deg = ps.get("pitch_deg", None)
                                                drift_m = ps.get("drift_m", None)

                                            pos_map = None
                                            yaw_deg_map = None
                                            roll_deg_map = None
                                            pitch_deg_map = None
                                            if isinstance(ps_map, dict):
                                                pos_map = ps_map.get("pos_w_m", None)
                                                yaw_deg_map = ps_map.get("yaw_deg", None)
                                                roll_deg_map = ps_map.get("roll_deg", None)
                                                pitch_deg_map = ps_map.get("pitch_deg", None)

                                            if isinstance(pos, (list, tuple)) and len(pos) >= 3:
                                                x, y, z = float(pos[0]), float(pos[1]), float(pos[2])
                                                pose_out: Dict[str, Any] = {
                                                    "ok": bool(ok),
                                                    "stable": bool(stable),
                                                    "ts_s": (float(ts_s) if isinstance(ts_s, (int, float)) else None),
                                                    "fi": (int(fi) if isinstance(fi, int) else None),
                                                    "position_m": {"x": float(x), "y": float(y), "z": float(z)},
                                                    "yaw_deg": (float(yaw_deg) if yaw_deg is not None else None),
                                                    "drift_m": (float(drift_m) if drift_m is not None else None),
                                                    "inliers": int(inliers),
                                                    "rmse_m": float(rmse_m),
                                                    "status_code": int(status_code),
                                                    "source": "track3d",
                                                }
                                                # Optional full orientation (deg) for MAVLink VISION_POSITION_ESTIMATE.
                                                # Older telemetry had yaw only; keep yaw_deg for back-compat.
                                                if isinstance(roll_deg, (int, float)) or isinstance(pitch_deg, (int, float)):
                                                    try:
                                                        yaw0 = float(yaw_deg) if isinstance(yaw_deg, (int, float)) else 0.0
                                                        pose_out["orientation_deg"] = {
                                                            "roll": float(roll_deg) if isinstance(roll_deg, (int, float)) else 0.0,
                                                            "pitch": float(pitch_deg) if isinstance(pitch_deg, (int, float)) else 0.0,
                                                            "yaw": float(yaw0),
                                                        }
                                                    except Exception:
                                                        pass

                                                # Optional ORB map/world pose (jumpy after BA / loop closure).
                                                if isinstance(pos_map, (list, tuple)) and len(pos_map) >= 3:
                                                    try:
                                                        pose_out["map_position_m"] = {
                                                            "x": float(pos_map[0]),
                                                            "y": float(pos_map[1]),
                                                            "z": float(pos_map[2]),
                                                        }
                                                    except Exception:
                                                        pass
                                                if (
                                                    isinstance(roll_deg_map, (int, float))
                                                    or isinstance(pitch_deg_map, (int, float))
                                                    or isinstance(yaw_deg_map, (int, float))
                                                ):
                                                    try:
                                                        pose_out["map_orientation_deg"] = {
                                                            "roll": float(roll_deg_map) if isinstance(roll_deg_map, (int, float)) else 0.0,
                                                            "pitch": float(pitch_deg_map) if isinstance(pitch_deg_map, (int, float)) else 0.0,
                                                            "yaw": float(yaw_deg_map) if isinstance(yaw_deg_map, (int, float)) else 0.0,
                                                        }
                                                    except Exception:
                                                        pass
                                                telem.set_vo_pose(pose_out)
                                            else:
                                                telem.set_vo_pose(None)
                                        except Exception:
                                            try:
                                                telem.set_vo_pose(None)
                                            except Exception:
                                                pass
                                except Exception:
                                    pass
                                
                                poly = payload.get("poly", None)
                                if not isinstance(poly, dict):
                                    continue
                                sel_kind0 = str(poly.get("sel_kind", "") or "").strip().lower()
                                poly_active_now = bool(poly.get("active", False)) and str(sel_kind0) in ("hole", "plane")
                                if not bool(poly_active_now):
                                    # Hole-detector failures are often only visible in the tracker logs. Since you run
                                    # server.py as the primary entrypoint, keep a lightweight server-side log and counters.
                                    try:
                                        reason = str(poly.get("reason", "") or "").strip()
                                        if reason:
                                            if str(sel_kind0) == "hole" and str(reason) != str(last_hole_reason):
                                                last_hole_reason = str(reason)
                                                hs = poly.get("hole_stats", None)
                                                herr = str(hs.get("err", "") or "") if isinstance(hs, dict) else ""
                                                herr = herr.strip()
                                                kind = (herr.split(" ", 1)[0].strip() if herr else "")
                                                if reason.startswith("hole_") and reason not in ("hole_done",):
                                                    try:
                                                        with t3_lock:
                                                            t3_hole_fail_total += 1
                                                            if kind == "plane_sector_low":
                                                                t3_hole_fail_sector_low += 1
                                                            elif kind.startswith("hole_leak_"):
                                                                t3_hole_fail_leak += 1
                                                            else:
                                                                t3_hole_fail_other += 1
                                                    except Exception:
                                                        pass
                                                if hole_log_fh is not None:
                                                    try:
                                                        r_m = hs.get("r_m", None) if isinstance(hs, dict) else None
                                                        fill = hs.get("fill", None) if isinstance(hs, dict) else None
                                                        ok_pid = hs.get("ok_pid", None) if isinstance(hs, dict) else None
                                                        hole_log_fh.write(
                                                            f"[HOLE] ts_wall={time.time():.3f} fi={int(poly.get('fi', -1))} reason={reason} kind={kind or '-'} err={herr or '-'} r_m={float(r_m) if isinstance(r_m,(int,float)) else float('nan'):.3f} fill={float(fill) if isinstance(fill,(int,float)) else float('nan'):.3f} ok_pid={int(bool(ok_pid)) if ok_pid is not None else -1}\n"
                                                        )
                                                        hole_log_fh.flush()
                                                    except Exception:
                                                        pass

                                            if str(sel_kind0) == "plane" and str(reason) != str(last_plane_reason):
                                                last_plane_reason = str(reason)
                                                if plane_log_fh is not None:
                                                    try:
                                                        ps = poly.get("plane_stats", None)
                                                        perr = str(ps.get("err", "") or "") if isinstance(ps, dict) else ""
                                                        perr = perr.strip()
                                                        r_m = ps.get("r_m", None) if isinstance(ps, dict) else None
                                                        ok_pid = ps.get("ok_pid", None) if isinstance(ps, dict) else None
                                                        pin = ps.get("plane_inliers", None) if isinstance(ps, dict) else None
                                                        prms = ps.get("plane_rms_m", None) if isinstance(ps, dict) else None
                                                        pcov = ps.get("plane_cov", None) if isinstance(ps, dict) else None
                                                        prange = ps.get("plane_center_range_m", None) if isinstance(ps, dict) else None
                                                        plane_log_fh.write(
                                                            f"[PLANE] ts_wall={time.time():.3f} fi={int(poly.get('fi', -1))} reason={reason} err={perr or '-'} r_m={float(r_m) if isinstance(r_m,(int,float)) else float('nan'):.3f} ok_pid={int(bool(ok_pid)) if ok_pid is not None else -1} inliers={int(pin) if isinstance(pin,(int,float)) else -1} rms_m={float(prms) if isinstance(prms,(int,float)) else float('nan'):.4f} cov={float(pcov) if isinstance(pcov,(int,float)) else float('nan'):.3f} range_m={float(prange) if isinstance(prange,(int,float)) else float('nan'):.3f}\n"
                                                        )
                                                        plane_log_fh.flush()
                                                    except Exception:
                                                        pass
                                    except Exception:
                                        pass
                                    # Do not continuously clear stage-0 preview overlays. Only clear when
                                    # transitioning from an active selection (stage 1/2) back to idle.
                                    if bool(poly_was_active):
                                        msg = track3d.make_acq_poly_msg(stage=0, verts_uv=[])
                                        if msg is not None:
                                            json_event(msg)
                                        poly_was_active = False
                                    if pid_active and pidmgr is not None and pid_mod is not None:
                                        try:
                                            pidmgr.on_measurement(
                                                pid_mod.Measurement(
                                                    state="IDLE",
                                                    yaw_err_rad=None,
                                                    pitch_err_rad=None,
                                                    track_id=None,
                                                    t_ms=int(time.time() * 1000),
                                                )
                                            )
                                        except Exception:
                                            pass
                                        pid_active = False
                                    pid_move_auto_exit_latched = False
                                    continue
                                poly_was_active = True
                                stage = 2 if bool(poly.get("pid_confirmed", False)) else 1
                                verts = poly.get("verts_uv", None)
                                verts_uv = []
                                if isinstance(verts, list):
                                    for p in verts:
                                        if isinstance(p, (list, tuple)) and len(p) >= 2:
                                            verts_uv.append((float(p[0]), float(p[1])))

                                # PID_MOVE auto-exit (hole/plane):
                                # Stop tracking if the polygon bbox fills too much of the image, and trigger the
                                # existing thrust-hold timer logic (same as the 2D ROI tracker).
                                try:
                                    if (
                                        int(stage) == 2
                                        and (not bool(pid_move_auto_exit_latched))
                                        and pidmgr is not None
                                        and pid_mod is not None
                                        and _pid_src_get() == "track3d"
                                    ):
                                        gate_state = (
                                            pidmgr.gate_state()
                                            if hasattr(pidmgr, "gate_state")
                                            else getattr(pidmgr, "_gate_state", None)
                                        )
                                        if str(gate_state or "") == "PID_MOVE":
                                            # Per-mode PID tuning: allow different auto-exit thresholds per tracker kind.
                                            thr_src = pid_dict_base
                                            try:
                                                if str(sel_kind0 or "") == "plane":
                                                    thr_src = pid_dict_plane if pid_dict_plane is not None else pid_dict_base
                                                elif str(sel_kind0 or "") == "hole":
                                                    thr_src = pid_dict_hole if pid_dict_hole is not None else pid_dict_base
                                            except Exception:
                                                thr_src = pid_dict_base
                                            thr = _pid_move_max_bbox_area_ratio(thr_src)
                                            if math.isfinite(float(thr)) and float(thr) > 0.0:
                                                bb = poly.get("bbox", None)
                                                if isinstance(bb, (list, tuple)) and int(len(bb)) == 4:
                                                    try:
                                                        bx0, by0, bx1, by1 = (
                                                            float(bb[0]),
                                                            float(bb[1]),
                                                            float(bb[2]),
                                                            float(bb[3]),
                                                        )
                                                    except Exception:
                                                        bx0 = by0 = bx1 = by1 = None
                                                    if bx0 is not None:
                                                        init0 = track3d.init_info()
                                                        out_w = int(getattr(init0, "out_w", 0)) if init0 is not None else 0
                                                        out_h = int(getattr(init0, "out_h", 0)) if init0 is not None else 0
                                                        if out_w > 0 and out_h > 0:
                                                            bw = abs(float(bx1) - float(bx0))
                                                            bh = abs(float(by1) - float(by0))
                                                            area_ratio = (max(0.0, float(bw)) * max(0.0, float(bh))) / float(
                                                                int(out_w) * int(out_h)
                                                            )
                                                            if math.isfinite(float(area_ratio)) and float(area_ratio) >= float(thr):
                                                                pid_move_auto_exit_latched = True
                                                                try:
                                                                    print(
                                                                        f"[track3d] auto-exit max_bbox_area kind={sel_kind0 or '-'} ratio={float(area_ratio):.3f} thr={float(thr):.3f}",
                                                                        flush=True,
                                                                    )
                                                                except Exception:
                                                                    pass
                                                                try:
                                                                    if hasattr(pidmgr, "start_thrust_hold_max"):
                                                                        pidmgr.start_thrust_hold_max(reason="max_bbox_area")
                                                                except Exception:
                                                                    pass
                                                                try:
                                                                    pidmgr.on_measurement(
                                                                        pid_mod.Measurement(
                                                                            state="LOST",
                                                                            yaw_err_rad=None,
                                                                            pitch_err_rad=None,
                                                                            track_id=None,
                                                                            t_ms=int(time.time() * 1000),
                                                                        )
                                                                    )
                                                                except Exception:
                                                                    pass
                                                                pid_active = False
                                                                try:
                                                                    track3d.send_cmd({"cmd": "clear"})
                                                                except Exception:
                                                                    pass
                                                                msg0 = track3d.make_acq_poly_msg(stage=0, verts_uv=[], sel_kind=sel_kind0)
                                                                if msg0 is not None:
                                                                    json_event(msg0)
                                                                poly_was_active = False
                                                                continue

                                            # Plane selection: stop when close to the plane.
                                            if str(sel_kind0) == "plane" and float(track3d_plane_min_range_m) > 0.0:
                                                try:
                                                    d_m = float(poly.get("plane_plane_center_range_m", float("nan")))
                                                except Exception:
                                                    d_m = float("nan")
                                                if math.isfinite(float(d_m)) and float(d_m) > 0.0 and float(d_m) <= float(track3d_plane_min_range_m):
                                                    pid_move_auto_exit_latched = True
                                                    try:
                                                        print(
                                                            f"[track3d] auto-exit plane_min_range d={float(d_m):.3f}m thr={float(track3d_plane_min_range_m):.3f}m",
                                                            flush=True,
                                                        )
                                                    except Exception:
                                                        pass
                                                    try:
                                                        if hasattr(pidmgr, "start_thrust_hold_max"):
                                                            pidmgr.start_thrust_hold_max(reason="plane_min_range")
                                                    except Exception:
                                                        pass
                                                    try:
                                                        pidmgr.on_measurement(
                                                            pid_mod.Measurement(
                                                                state="LOST",
                                                                yaw_err_rad=None,
                                                                pitch_err_rad=None,
                                                                track_id=None,
                                                                t_ms=int(time.time() * 1000),
                                                            )
                                                        )
                                                    except Exception:
                                                        pass
                                                    pid_active = False
                                                    try:
                                                        track3d.send_cmd({"cmd": "clear"})
                                                    except Exception:
                                                        pass
                                                    msg0 = track3d.make_acq_poly_msg(stage=0, verts_uv=[], sel_kind=sel_kind0)
                                                    if msg0 is not None:
                                                        json_event(msg0)
                                                    poly_was_active = False
                                                    continue
                                except Exception:
                                    pass

                                # Auto-stop if the polygon spans too much of the image.
                                if verts_uv and poly_stop_extent_frac > 0.0:
                                    try:
                                        init0 = track3d.init_info()
                                        out_w = int(getattr(init0, "out_w", 0)) if init0 is not None else 0
                                        out_h = int(getattr(init0, "out_h", 0)) if init0 is not None else 0
                                    except Exception:
                                        out_w = 0
                                        out_h = 0

                                    if out_w > 0 and out_h > 0:
                                        try:
                                            xs = [float(p[0]) for p in verts_uv]
                                            ys = [float(p[1]) for p in verts_uv]
                                            span_x = float(max(xs) - min(xs)) if xs else 0.0
                                            span_y = float(max(ys) - min(ys)) if ys else 0.0
                                            if (span_x >= float(poly_stop_extent_frac) * float(out_w)) or (
                                                span_y >= float(poly_stop_extent_frac) * float(out_h)
                                            ):
                                                print(
                                                    f"[track3d] stop: poly_extent span=({span_x:.1f},{span_y:.1f}) img=({out_w},{out_h}) frac={poly_stop_extent_frac:.2f}",
                                                    flush=True,
                                                )
                                                pid_move_auto_exit_latched = True
                                                try:
                                                    if pidmgr is not None and hasattr(pidmgr, "start_thrust_hold_max"):
                                                        pidmgr.start_thrust_hold_max(reason="poly_extent")
                                                except Exception:
                                                    pass
                                                try:
                                                    track3d.send_cmd({"cmd": "clear"})
                                                except Exception:
                                                    pass
                                                msg0 = track3d.make_acq_poly_msg(stage=0, verts_uv=[], sel_kind=sel_kind0)
                                                if msg0 is not None:
                                                    json_event(msg0)
                                                poly_was_active = False
                                                if pid_active and pidmgr is not None and pid_mod is not None:
                                                    try:
                                                        pidmgr.on_measurement(
                                                            pid_mod.Measurement(
                                                                state="IDLE",
                                                                yaw_err_rad=None,
                                                                pitch_err_rad=None,
                                                                track_id=None,
                                                                t_ms=int(time.time() * 1000),
                                                            )
                                                        )
                                                    except Exception:
                                                        pass
                                                    pid_active = False
                                                continue
                                        except Exception:
                                            pass
                                msg = track3d.make_acq_poly_msg(stage=int(stage), verts_uv=verts_uv, sel_kind=sel_kind0)
                                if msg is not None:
                                    json_event(msg)
                            except Exception:
                                pass
                            continue

                        # Feature stream: draw tracked points in the GUI.
                        if et == "features" and isinstance(payload, dict):
                            try:
                                tracks = payload.get("tracks", None)
                                pts = []
                                if isinstance(tracks, list):
                                    for tr in tracks:
                                        u = v = None
                                        g = 0
                                        if isinstance(tr, dict):
                                            # Full payload: {"uv_px":[u,v], "group":g, ...}
                                            # Compact payload variant: {"uvg":[u,v,g]}
                                            uvg = tr.get("uvg", None)
                                            if isinstance(uvg, (list, tuple)) and len(uvg) >= 3:
                                                try:
                                                    u = int(round(float(uvg[0])))
                                                    v = int(round(float(uvg[1])))
                                                    g = int(uvg[2])
                                                except Exception:
                                                    u = v = None
                                            else:
                                                uv = tr.get("uv_px", None)
                                                if isinstance(uv, (list, tuple)) and len(uv) >= 2:
                                                    try:
                                                        u = int(round(float(uv[0])))
                                                        v = int(round(float(uv[1])))
                                                    except Exception:
                                                        u = v = None
                                                try:
                                                    g = int(tr.get("group", 0))
                                                except Exception:
                                                    g = 0
                                        elif isinstance(tr, (list, tuple)) and len(tr) >= 3:
                                            # Compact payload: [u,v,g]
                                            try:
                                                u = int(round(float(tr[0])))
                                                v = int(round(float(tr[1])))
                                                g = int(tr[2])
                                            except Exception:
                                                u = v = None
                                        if u is None or v is None:
                                            continue
                                        pts.append((int(u), int(v), int(g)))
                                if pts:
                                    if int(feat_max_pts) > 0 and len(pts) > int(feat_max_pts):
                                        step = max(1, len(pts) // int(feat_max_pts))
                                        pts = pts[::step][: int(feat_max_pts)]
                                    feat_last_pts = pts
                                    feat_last_update_t = float(time.time())
                                    try:
                                        feat_last_fi = int(payload.get("state_fi", payload.get("ring", {}).get("fi")))
                                    except Exception:
                                        feat_last_fi = None
                            except Exception:
                                pass
                            continue

                        # PID stream: feed stage 1+2 (selected/confirmed). Only one tracker should drive PID.
                        if et == "pid" and isinstance(payload, dict) and pidmgr is not None and pid_mod is not None:
                            try:
                                if _pid_src_get() != "track3d":
                                    continue
                                try:
                                    stage_i = int(payload.get("stage", 0) or 0)
                                except Exception:
                                    stage_i = 0
                                if int(stage_i) < 1:
                                    continue
                                dhv = payload.get("dhv_deg", None)
                                if not (isinstance(dhv, (list, tuple)) and len(dhv) >= 2):
                                    continue
                                yaw_err = math.radians(float(dhv[0]))
                                pitch_err = math.radians(float(dhv[1]))
                                pidmgr.on_measurement(
                                    pid_mod.Measurement(
                                        state="TRACKING",
                                        yaw_err_rad=float(yaw_err),
                                        pitch_err_rad=float(pitch_err),
                                        track_id=None,
                                        t_ms=int(time.time() * 1000),
                                    )
                                )
                                pid_active = True
                            except Exception:
                                pass
                            continue

                    # Emit VO features at a steady 30 Hz (re-sends last set if track3d runs features slower).
                    now = float(time.time())
                    if feat_last_pts is not None and (now - float(feat_last_update_t)) <= float(feat_stale_s):
                        if (now - float(feat_last_send_t)) >= float(feat_period_s):
                            try:
                                init0 = track3d.init_info()
                                if init0 is not None and int(init0.out_w) > 0 and int(init0.out_h) > 0:
                                    pts_send = list(feat_last_pts or [])
                                    if int(feat_max_pts) > 0 and len(pts_send) > int(feat_max_pts):
                                        step = max(1, len(pts_send) // int(feat_max_pts))
                                        pts_send = pts_send[::step][: int(feat_max_pts)]

                                    emit_ts_ms = int(now * 1000.0)
                                    base = {
                                        "type": "vo_features",
                                        "ts": int(emit_ts_ms),
                                        "img_size": [int(init0.out_w), int(init0.out_h)],
                                        "source": "track3d",
                                    }
                                    if feat_last_fi is not None:
                                        base["fi"] = int(feat_last_fi)

                                    # Chunk large point sets to avoid UDP fragmentation loss.
                                    chunk_n = 1
                                    try:
                                        if int(feat_chunk_pts) > 0 and len(pts_send) > int(feat_chunk_pts):
                                            chunk_n = int(
                                                (len(pts_send) + int(feat_chunk_pts) - 1)
                                                // int(feat_chunk_pts)
                                            )
                                    except Exception:
                                        chunk_n = 1

                                    if int(chunk_n) <= 1:
                                        msg = dict(base)
                                        msg["pts"] = [[int(u), int(v), int(g)] for (u, v, g) in (pts_send or [])]
                                        json_event(msg)
                                    else:
                                        for ci in range(int(chunk_n)):
                                            a = int(ci) * int(feat_chunk_pts)
                                            b = min(len(pts_send), a + int(feat_chunk_pts))
                                            chunk = pts_send[a:b]
                                            msg = dict(base)
                                            msg["chunk_i"] = int(ci)
                                            msg["chunk_n"] = int(chunk_n)
                                            msg["pts"] = [[int(u), int(v), int(g)] for (u, v, g) in (chunk or [])]
                                            json_event(msg)
                                    feat_last_send_t = float(now)
                            except Exception:
                                pass

                    time.sleep(0.01)
            finally:
                try:
                    if hole_log_fh is not None:
                        hole_log_fh.close()
                except Exception:
                    pass
                try:
                    if plane_log_fh is not None:
                        plane_log_fh.close()
                except Exception:
                    pass

        t_track3d = threading.Thread(target=_track3d_loop, daemon=True, name="track3d")
        t_track3d.start()

    # ---------------- FPS monitor (once/sec consolidated line) ----------------
    fps_log = bool(cfg.get("debug.fps_log", True))
    t_fps = None
    if fps_log:
        def _fps_loop():
            last_fb_count = 0
            last_t = time.time()
            last_vpe_sent = 0
            try:
                if framebus is not None and hasattr(framebus, "get_update_count"):
                    last_fb_count = int(framebus.get_update_count())
            except Exception:
                last_fb_count = 0

            # Track3d capture (SHM ring) stats
            last_t3_fi = None
            last_t3_ts_s = None
            last_t3_seen_fi = None
            last_t3_seen_wall = time.time()
            last_t3_total = last_t3_preview = last_t3_telem = last_t3_features = last_t3_pid = 0
            last_t3_state_fi = None
            last_t3_state_seen_fi = None
            last_t3_state_seen_wall = time.time()
            last_t3_lk_fi = None
            last_t3_lk_seen_fi = None
            last_t3_lk_seen_wall = time.time()
            last_mouse_total = last_mouse_move = last_mouse_stop = last_mouse_click = 0
            last_hole_fail_total = 0
            last_hole_fail_sector_low = 0
            last_hole_fail_leak = 0
            last_hole_fail_other = 0

            def _hz_eq_from_ms(ms: float) -> float:
                try:
                    ms_f = float(ms)
                    if not math.isfinite(ms_f) or ms_f <= 1e-6:
                        return 0.0
                    return 1000.0 / ms_f
                except Exception:
                    return 0.0

            fps_detail_fh = None
            try:
                fps_detail_path = str(cfg.get("debug.fps_detail_log_path", "./logs/fps_detail.log") or "").strip()
                if fps_detail_path:
                    try:
                        os.makedirs(os.path.dirname(fps_detail_path) or ".", exist_ok=True)
                        fps_detail_fh = open(fps_detail_path, "a", encoding="utf-8")
                    except Exception:
                        fps_detail_fh = None

                while not stop.is_set():
                    time.sleep(1.0)
                    now = time.time()
                    dt = max(1e-6, now - last_t)
                    last_t = now

                    # Track3d mode: FPS comes from the shared capture ring + track3d event rates.
                    shm_src_any = shm_src_ir if shm_src_ir is not None else shm_src_rgb
                    if track3d is not None and shm_src_any is not None and hasattr(shm_src_any, "read_latest"):
                        cap_fps = 0.0
                        fi = None
                        ring_ts_s = None
                        stale_ms = 0.0
                        try:
                            info = shm_src_any.read_latest()
                            if info is not None:
                                _slot, fi0, ts_s = info
                                fi = int(fi0)
                                ts_s = float(ts_s)
                                ring_ts_s = float(ts_s)
                                if last_t3_fi is not None and last_t3_ts_s is not None:
                                    dfi = int(fi) - int(last_t3_fi)
                                    dts = float(ts_s) - float(last_t3_ts_s)
                                    if dfi >= 0 and dts > 1e-6:
                                        cap_fps = float(dfi) / float(dts)
                                    elif dfi >= 0:
                                        cap_fps = float(dfi) / float(dt)
                                last_t3_fi = int(fi)
                                last_t3_ts_s = float(ts_s)
                                if last_t3_seen_fi is None or int(fi) != int(last_t3_seen_fi):
                                    last_t3_seen_fi = int(fi)
                                    last_t3_seen_wall = float(now)
                        except Exception:
                            cap_fps = 0.0
                        stale_ms = float(max(0.0, float(now) - float(last_t3_seen_wall))) * 1000.0

                        poly_stage = 0
                        poly_verts = 0
                        try:
                            if hasattr(track3d, "snapshot_poly"):
                                ps = track3d.snapshot_poly()
                                active = bool(ps.get("active", False))
                                pid_conf = bool(ps.get("pid_confirmed", False))
                                poly_verts = int(ps.get("n_verts", 0))
                                poly_stage = 0 if not active else (2 if pid_conf else 1)
                        except Exception:
                            poly_stage = 0
                            poly_verts = 0

                        evt_hz = prev_hz = telem_hz = feat_hz = pid_hz = 0.0
                        mouse_hz = mouse_move_hz = mouse_stop_hz = mouse_click_hz = 0.0
                        mouse_age_ms = 0.0
                        state_fps = 0.0
                        state_fi_txt = "--"
                        state_lag = None
                        state_lag_ms = None
                        t3_telem_age_ms = 0.0
                        t3_evt_age_ms = None
                        state_stale_ms = 0.0
                        lk_fps = 0.0
                        lk_fi_txt = "--"
                        lk_lag = None
                        lk_lag_ms = None
                        lk_stale_ms = 0.0
                        lk_n_tracks = 0
                        lk_n_corr = 0
                        lk_of_ms = 0.0
                        lk_ms = 0.0
                        lk_reseed_ms = 0.0
                        lk_depth_ms = 0.0
                        lk_corr_ms = 0.0
                        lk_err_pre_n = 0
                        lk_err_post_n = 0
                        lk_err_rej_n = 0
                        lk_err_med = float("nan")
                        lk_err_mad = float("nan")
                        lk_err_sigma = float("nan")
                        lk_err_thr = float("nan")
                        odom_n_corr = 0
                        odom_n_in = 0
                        odom_rmse_m = 0.0
                        odom_status = 0
                        odom_ms_est = 0.0
                        odom_ms_total = 0.0
                        odom_ms_prefilter = 0.0
                        odom_ms_pg = 0.0
                        odom_ms_jitter = 0.0
                        odom_ms_imu = 0.0
                        odom_ms_weights = 0.0
                        odom_ms_gate = 0.0
                        odom_est_code = 0
                        odom_est = ""
                        hole_fail_hz = 0.0
                        hole_fail_sector_low_hz = 0.0
                        hole_fail_leak_hz = 0.0
                        hole_fail_other_hz = 0.0
                        try:
                            with t3_lock:
                                cur_total = int(t3_evt_total)
                                cur_preview = int(t3_evt_preview)
                                cur_telem = int(t3_evt_telem)
                                cur_feat = int(t3_evt_features)
                                cur_pid = int(t3_evt_pid)
                                cur_evt_wall = float(t3_last_evt_wall)
                                cur_mouse_total = int(mouse_total)
                                cur_mouse_move = int(mouse_move)
                                cur_mouse_stop = int(mouse_stop)
                                cur_mouse_click = int(mouse_click)
                                cur_mouse_last_wall = float(mouse_last_wall)
                                cur_lk_fi = int(t3_last_lk_fi)
                                cur_lk_ts_s = float(t3_last_lk_ts)
                                cur_lk_n_tracks = int(t3_last_lk_n_tracks)
                                cur_lk_n_corr = int(t3_last_lk_n_corr)
                                cur_lk_of_ms = float(t3_last_lk_of_ms)
                                cur_lk_ms = float(t3_last_lk_ms)
                                cur_lk_reseed_ms = float(t3_last_lk_reseed_ms)
                                cur_lk_depth_ms = float(t3_last_lk_depth_ms)
                                cur_lk_corr_ms = float(t3_last_lk_corr_ms)
                                cur_lk_err_pre_n = int(t3_last_lk_err_pre_n)
                                cur_lk_err_post_n = int(t3_last_lk_err_post_n)
                                cur_lk_err_rej_n = int(t3_last_lk_err_rej_n)
                                cur_lk_err_med = float(t3_last_lk_err_med)
                                cur_lk_err_mad = float(t3_last_lk_err_mad)
                                cur_lk_err_sigma = float(t3_last_lk_err_sigma)
                                cur_lk_err_thr = float(t3_last_lk_err_thr)
                                cur_state_fi = int(t3_last_state_fi)
                                cur_telem_wall = float(t3_last_telem_wall)
                                cur_state_ts_s = float(t3_last_state_ts)
                                cur_state_corr = int(t3_last_state_corr)
                                cur_state_in = int(t3_last_state_inliers)
                                cur_state_rmse = float(t3_last_state_rmse_m)
                                cur_state_status = int(t3_last_state_status)
                                cur_state_est_code = int(t3_last_state_est_code)
                                cur_state_est = str(t3_last_state_est)
                                cur_state_ms_est = float(t3_last_state_ms_est)
                                cur_state_ms_total = float(t3_last_state_ms_total)
                                cur_state_ms_prefilter = float(t3_last_state_ms_prefilter)
                                cur_state_ms_pg = float(t3_last_state_ms_pg)
                                cur_state_ms_jitter = float(t3_last_state_ms_jitter)
                                cur_state_ms_imu = float(t3_last_state_ms_imu)
                                cur_state_ms_weights = float(t3_last_state_ms_weights)
                                cur_state_ms_gate = float(t3_last_state_ms_gate)
                                cur_hole_fail_total = int(t3_hole_fail_total)
                                cur_hole_fail_sector_low = int(t3_hole_fail_sector_low)
                                cur_hole_fail_leak = int(t3_hole_fail_leak)
                                cur_hole_fail_other = int(t3_hole_fail_other)
                            evt_hz = float(cur_total - last_t3_total) / float(dt)
                            prev_hz = float(cur_preview - last_t3_preview) / float(dt)
                            telem_hz = float(cur_telem - last_t3_telem) / float(dt)
                            feat_hz = float(cur_feat - last_t3_features) / float(dt)
                            pid_hz = float(cur_pid - last_t3_pid) / float(dt)
                            last_t3_total = int(cur_total)
                            last_t3_preview = int(cur_preview)
                            last_t3_telem = int(cur_telem)
                            last_t3_features = int(cur_feat)
                            last_t3_pid = int(cur_pid)

                            # External vision (VISION_POSITION_ESTIMATE) TX cadence diagnostics.
                            vpe_hz = 0.0
                            try:
                                with vpe_stats_lock:
                                    cur_vpe_sent = int(vpe_stats.get("sent", 0))
                                d_vpe = int(cur_vpe_sent) - int(last_vpe_sent)
                                if int(d_vpe) >= 0:
                                    vpe_hz = float(d_vpe) / float(dt)
                                last_vpe_sent = int(cur_vpe_sent)
                            except Exception:
                                vpe_hz = 0.0

                            # Track3d processing rate (state fi cadence).
                            if int(cur_state_fi) >= 0:
                                state_fi_txt = str(int(cur_state_fi))
                                if last_t3_state_fi is not None:
                                    dfi_state = int(cur_state_fi) - int(last_t3_state_fi)
                                    # `fi` can reset on tracker restart / ring reinit. Clamp to avoid negative "Hz".
                                    if int(dfi_state) >= 0:
                                        state_fps = float(dfi_state) / float(dt)
                                    else:
                                        state_fps = 0.0
                                last_t3_state_fi = int(cur_state_fi)
                                if last_t3_state_seen_fi is None or int(cur_state_fi) != int(last_t3_state_seen_fi):
                                    last_t3_state_seen_fi = int(cur_state_fi)
                                    last_t3_state_seen_wall = float(now)
                                state_stale_ms = float(max(0.0, float(now) - float(last_t3_state_seen_wall))) * 1000.0
                                if fi is not None:
                                    try:
                                        state_lag = int(int(fi) - int(cur_state_fi))
                                    except Exception:
                                        state_lag = None
                                if ring_ts_s is not None and float(cur_state_ts_s) > 1e-6:
                                    try:
                                        state_lag_ms = float(float(ring_ts_s) - float(cur_state_ts_s)) * 1000.0
                                    except Exception:
                                        state_lag_ms = None
                                odom_n_corr = int(cur_state_corr)
                                odom_n_in = int(cur_state_in)
                                odom_rmse_m = float(cur_state_rmse)
                                odom_status = int(cur_state_status)
                                odom_est_code = int(cur_state_est_code)
                                odom_est = str(cur_state_est)
                                odom_ms_est = float(cur_state_ms_est)
                                odom_ms_total = float(cur_state_ms_total)
                                odom_ms_prefilter = float(cur_state_ms_prefilter)
                                odom_ms_pg = float(cur_state_ms_pg)
                                odom_ms_jitter = float(cur_state_ms_jitter)
                                odom_ms_imu = float(cur_state_ms_imu)
                                odom_ms_weights = float(cur_state_ms_weights)
                                odom_ms_gate = float(cur_state_ms_gate)

                            # LK (correspondence builder) cadence (from telemetry payload).
                            if int(cur_lk_fi) >= 0:
                                lk_fi_txt = str(int(cur_lk_fi))
                                if last_t3_lk_fi is not None:
                                    lk_fps = float(int(cur_lk_fi) - int(last_t3_lk_fi)) / float(dt)
                                last_t3_lk_fi = int(cur_lk_fi)
                                if last_t3_lk_seen_fi is None or int(cur_lk_fi) != int(last_t3_lk_seen_fi):
                                    last_t3_lk_seen_fi = int(cur_lk_fi)
                                    last_t3_lk_seen_wall = float(now)
                                lk_stale_ms = float(max(0.0, float(now) - float(last_t3_lk_seen_wall))) * 1000.0
                                if fi is not None:
                                    try:
                                        lk_lag = int(int(fi) - int(cur_lk_fi))
                                    except Exception:
                                        lk_lag = None
                                if ring_ts_s is not None and float(cur_lk_ts_s) > 1e-6:
                                    try:
                                        lk_lag_ms = float(float(ring_ts_s) - float(cur_lk_ts_s)) * 1000.0
                                    except Exception:
                                        lk_lag_ms = None
                                lk_n_tracks = int(cur_lk_n_tracks)
                                lk_n_corr = int(cur_lk_n_corr)
                                lk_of_ms = float(cur_lk_of_ms)
                                lk_ms = float(cur_lk_ms)
                                lk_reseed_ms = float(cur_lk_reseed_ms)
                                lk_depth_ms = float(cur_lk_depth_ms)
                                lk_corr_ms = float(cur_lk_corr_ms)
                                lk_err_pre_n = int(cur_lk_err_pre_n)
                                lk_err_post_n = int(cur_lk_err_post_n)
                                lk_err_rej_n = int(cur_lk_err_rej_n)
                                lk_err_med = float(cur_lk_err_med)
                                lk_err_mad = float(cur_lk_err_mad)
                                lk_err_sigma = float(cur_lk_err_sigma)
                                lk_err_thr = float(cur_lk_err_thr)
                            if float(cur_telem_wall) > 1e-6:
                                t3_telem_age_ms = float(max(0.0, float(now) - float(cur_telem_wall))) * 1000.0
                            if float(cur_evt_wall) > 1e-6:
                                t3_evt_age_ms = float(max(0.0, float(now) - float(cur_evt_wall))) * 1000.0

                            mouse_hz = float(cur_mouse_total - last_mouse_total) / float(dt)
                            mouse_move_hz = float(cur_mouse_move - last_mouse_move) / float(dt)
                            mouse_stop_hz = float(cur_mouse_stop - last_mouse_stop) / float(dt)
                            mouse_click_hz = float(cur_mouse_click - last_mouse_click) / float(dt)
                            last_mouse_total = int(cur_mouse_total)
                            last_mouse_move = int(cur_mouse_move)
                            last_mouse_stop = int(cur_mouse_stop)
                            last_mouse_click = int(cur_mouse_click)
                            if cur_mouse_last_wall > 1e-6:
                                mouse_age_ms = float(max(0.0, float(now) - float(cur_mouse_last_wall))) * 1000.0

                            hole_fail_hz = float(cur_hole_fail_total - last_hole_fail_total) / float(dt)
                            hole_fail_sector_low_hz = float(cur_hole_fail_sector_low - last_hole_fail_sector_low) / float(dt)
                            hole_fail_leak_hz = float(cur_hole_fail_leak - last_hole_fail_leak) / float(dt)
                            hole_fail_other_hz = float(cur_hole_fail_other - last_hole_fail_other) / float(dt)
                            last_hole_fail_total = int(cur_hole_fail_total)
                            last_hole_fail_sector_low = int(cur_hole_fail_sector_low)
                            last_hole_fail_leak = int(cur_hole_fail_leak)
                            last_hole_fail_other = int(cur_hole_fail_other)
                        except Exception:
                            pass

                    fi_txt = "--" if fi is None else str(int(fi))
                    try:
                        odom_other_ms = float(odom_ms_total) - (
                            float(odom_ms_est)
                            + float(odom_ms_prefilter)
                            + float(odom_ms_pg)
                            + float(odom_ms_jitter)
                            + float(odom_ms_imu)
                            + float(odom_ms_weights)
                            + float(odom_ms_gate)
                        )
                        odom_other_ms = float(max(0.0, float(odom_other_ms)))
                    except Exception:
                        odom_other_ms = 0.0

                    odom_est_tag = str(odom_est or "").strip()
                    if not odom_est_tag:
                        if int(odom_est_code) == 1:
                            odom_est_tag = "prosac_jit"
                        elif int(odom_est_code) == 2:
                            odom_est_tag = "ransac_numpy"
                        elif int(odom_est_code) == 4:
                            odom_est_tag = "trans_ransac"
                        else:
                            odom_est_tag = "unknown"

                    # Keep the consolidated line Hz-only: convert ms timings to Hz-equivalent throughput.
                    lk_tracks_hz = float(max(0, int(lk_n_tracks))) * float(lk_fps)
                    lk_corr_hz = float(max(0, int(lk_n_corr))) * float(lk_fps)
                    lk_err_pre_hz = float(max(0, int(lk_err_pre_n))) * float(lk_fps)
                    lk_err_post_hz = float(max(0, int(lk_err_post_n))) * float(lk_fps)
                    lk_err_rej_hz = float(max(0, int(lk_err_rej_n))) * float(lk_fps)
                    lk_step_hz_eq = _hz_eq_from_ms(lk_ms)
                    lk_of_hz_eq = _hz_eq_from_ms(lk_of_ms)
                    lk_reseed_hz_eq = _hz_eq_from_ms(lk_reseed_ms)
                    lk_depth_hz_eq = _hz_eq_from_ms(lk_depth_ms)
                    lk_corr_build_hz_eq = _hz_eq_from_ms(lk_corr_ms)

                    odom_corr_hz = float(max(0, int(odom_n_corr))) * float(state_fps)
                    odom_in_hz = float(max(0, int(odom_n_in))) * float(state_fps)
                    odom_step_hz_eq = _hz_eq_from_ms(odom_ms_total)
                    odom_est_hz_eq = _hz_eq_from_ms(odom_ms_est)
                    odom_pre_hz_eq = _hz_eq_from_ms(odom_ms_prefilter)
                    odom_pg_hz_eq = _hz_eq_from_ms(odom_ms_pg)
                    odom_jitter_hz_eq = _hz_eq_from_ms(odom_ms_jitter)
                    odom_imu_hz_eq = _hz_eq_from_ms(odom_ms_imu)
                    odom_weights_hz_eq = _hz_eq_from_ms(odom_ms_weights)
                    odom_gate_hz_eq = _hz_eq_from_ms(odom_ms_gate)
                    odom_other_hz_eq = _hz_eq_from_ms(odom_other_ms)

                    # Preserve full detail (including non-Hz fields) in a text log file for debugging.
                    if fps_detail_fh is not None:
                        try:
                            fps_detail_fh.write(
                                f"[FPS] "
                                f"src=track3d "
                                f"cam_ring_hz={cap_fps:.1f} cam_ring_fi={fi_txt} cam_ring_stale_ms={stale_ms:.0f} "
                                f"vo_hz={state_fps:.1f} vo_fi={state_fi_txt} vo_lag_fi={state_lag if state_lag is not None else '--'} "
                                f"vo_lag_ms={(f'{state_lag_ms:.0f}' if state_lag_ms is not None else '--')} vo_stale_ms={state_stale_ms:.0f} "
                                f"lk_hz={lk_fps:.1f} lk_fi={lk_fi_txt} lk_lag_fi={lk_lag if lk_lag is not None else '--'} "
                                f"lk_lag_ms={(f'{lk_lag_ms:.0f}' if lk_lag_ms is not None else '--')} lk_stale_ms={lk_stale_ms:.0f} "
                                f"lk_tracks={int(lk_n_tracks)} lk_corr={int(lk_n_corr)} lk_of_ms={float(lk_of_ms):.2f} lk_ms={float(lk_ms):.2f} "
                                f"lk_rs_ms={float(lk_reseed_ms):.2f} lk_d_ms={float(lk_depth_ms):.2f} lk_corr_ms={float(lk_corr_ms):.2f} "
                                f"lk_err_pre_n={int(lk_err_pre_n)} lk_err_post_n={int(lk_err_post_n)} lk_err_rej_n={int(lk_err_rej_n)} "
                                f"lk_err_med={(f'{float(lk_err_med):.2f}' if math.isfinite(float(lk_err_med)) else '--')} "
                                f"lk_err_mad={(f'{float(lk_err_mad):.2f}' if math.isfinite(float(lk_err_mad)) else '--')} "
                                f"lk_err_sigma={(f'{float(lk_err_sigma):.2f}' if math.isfinite(float(lk_err_sigma)) else '--')} "
                                f"lk_err_thr={(f'{float(lk_err_thr):.2f}' if math.isfinite(float(lk_err_thr)) else '--')} "
                                f"odom_corr={int(odom_n_corr)} odom_in={int(odom_n_in)} odom_rmse_mm={1000.0*float(odom_rmse_m):.1f} odom_status={int(odom_status)} "
                                f"odom_est={str(odom_est_tag)} odom_est_code={int(odom_est_code)} "
                                f"odom_est_ms={float(odom_ms_est):.2f} odom_pre_ms={float(odom_ms_prefilter):.2f} odom_pg_ms={float(odom_ms_pg):.2f} "
                                f"odom_j_ms={float(odom_ms_jitter):.2f} odom_imu_ms={float(odom_ms_imu):.2f} odom_w_ms={float(odom_ms_weights):.2f} "
                                f"odom_g_ms={float(odom_ms_gate):.2f} odom_other_ms={float(odom_other_ms):.2f} odom_ms={float(odom_ms_total):.2f} "
                                f"t3_telem_age_ms={t3_telem_age_ms:.0f} t3_evt_age_ms={t3_evt_age_ms if t3_evt_age_ms is not None else '--'} "
                                f"t3_jsonl_hz={evt_hz:.1f} t3_preview_poly_hz={prev_hz:.1f} t3_poly_telem_hz={telem_hz:.1f} t3_features_hz={feat_hz:.1f} t3_pid_hz={pid_hz:.1f} "
                                f"hole_fail_hz={hole_fail_hz:.2f} hole_sector_low_hz={hole_fail_sector_low_hz:.2f} hole_leak_hz={hole_fail_leak_hz:.2f} hole_other_hz={hole_fail_other_hz:.2f} "
                                f"mouse_in_hz={mouse_hz:.1f} mouse_move_hz={mouse_move_hz:.1f} mouse_stop_hz={mouse_stop_hz:.1f} mouse_click_hz={mouse_click_hz:.1f} "
                                f"mouse_age_ms={mouse_age_ms:.0f} poly_stage={int(poly_stage)} poly_verts={int(poly_verts)}\n"
                            )
                            fps_detail_fh.flush()
                        except Exception:
                            pass
                    # Console: keep this minimal. Full detail goes to fps_detail.log.
                    print(
                        f"[FPS] "
                        f"Capture={cap_fps:.1f}Hz "
                        f"VO={state_fps:.1f}Hz "
                        f"VPE={vpe_hz:.1f}Hz "
                        f"Interface={evt_hz:.1f}Hz",
                        flush=True,
                    )
                    # Periodic camera settings keepalive (camera.active/img_size/intrinsics) for the GUI.
                    try:
                        _emit_camera_settings()
                    except Exception:
                        pass
                    continue

                cam_fps = 0.0
                fb_fps = 0.0
                fb_hz = 0.0
                fb_count = last_fb_count
                try:
                    cam_fps = float(video.get_cam_fps()) if hasattr(video, "get_cam_fps") else 0.0
                except Exception:
                    cam_fps = 0.0
                try:
                    fb_fps = float(framebus.get_capture_fps()) if framebus is not None else 0.0
                except Exception:
                    fb_fps = 0.0
                try:
                    fb_count = int(framebus.get_update_count()) if framebus is not None else last_fb_count
                    fb_hz = float(fb_count - last_fb_count) / dt
                    last_fb_count = fb_count
                except Exception:
                    fb_hz = 0.0

                trk_loop = trk_src = trk_skip = trk_step_ms = trk_age_ms = trk_vo = 0.0
                trk_step_fps = 0.0
                det_hz = pid_hz = 0.0
                try:
                    if tracker_obj is not None and hasattr(tracker_obj, "snapshot_stats"):
                        snap = tracker_obj.snapshot_stats()
                        perf = (snap.get("perf") or {}) if isinstance(snap, dict) else {}
                        rates = (snap.get("rates") or {}) if isinstance(snap, dict) else {}
                        trk_loop = float(perf.get("loop_fps", 0.0))
                        trk_src = float(perf.get("src_fps", 0.0))
                        trk_skip = float(perf.get("skip", 0.0))
                        trk_step_ms = float(perf.get("step_ms", 0.0))
                        trk_age_ms = float(perf.get("age_ms", 0.0))
                        trk_vo = float(perf.get("vo_fps", 0.0))
                        det_hz = float(rates.get("detect_hz", 0.0))
                        pid_hz = float(rates.get("pid_hz", 0.0))
                except Exception:
                    pass
                try:
                    trk_step_fps = 0.0 if trk_step_ms <= 1e-6 else (1000.0 / trk_step_ms)
                except Exception:
                    trk_step_fps = 0.0

                # Console: keep this minimal. The detailed perf counters are track3d-only (fps_detail.log).
                print(
                    f"[FPS] "
                    f"Capture={cam_fps:.1f}Hz "
                    f"VO={trk_vo:.1f}Hz "
                    f"Interface={fb_hz:.1f}Hz",
                    flush=True,
                )
                # Periodic camera settings keepalive (camera.active/img_size/intrinsics) for the GUI.
                try:
                    _emit_camera_settings()
                except Exception:
                    pass
            finally:
                try:
                    if fps_detail_fh is not None:
                        fps_detail_fh.close()
                except Exception:
                    pass

        t_fps = threading.Thread(target=_fps_loop, daemon=True, name="fps")
        t_fps.start()

    def ctrl_loop():
        nonlocal mouse_total, mouse_move, mouse_stop, mouse_click, mouse_last_wall
        print(f"[control] listening on {ctrl_host}:{ctrl_port}")
        for msg, addr in control_rx:
            # Avoid log spam from high-rate mouse move packets and mode keepalives.
            try:
                if isinstance(msg, dict) and str(msg.get("type", "")).strip().lower() in (
                    "mouse_move",
                    "mouse_stop",
                    "hole_enable",
                    "plane_enable",
                    "track2d_enable",
                ):
                    pass
                else:
                    print(f"[control] recv from {addr} : {msg}", flush=True)
            except Exception:
                print(f"[control] recv from {addr}", flush=True)
            try:
                mtype = str(msg.get("type", "")).strip().lower() if isinstance(msg, dict) else ""

                applied2d: Dict[str, Any] = {}
                if track2d is not None and isinstance(msg, dict):
                    try:
                        applied2d = track2d.handle_control(msg)
                    except Exception:
                        applied2d = {}

                # Only one tracker should own the mouse-driven acquisition flow.
                # If 2D ROI tracking is enabled, suppress track3d mouse/hole commands.
                if track3d is not None and isinstance(msg, dict):
                    try:
                        if not (
                            track2d is not None
                            and bool(track2d.enabled())
                            and mtype in ("mouse_move", "mouse_stop", "mouse_click", "hole_enable", "plane_enable")
                        ):
                            track3d.handle_mouse(msg)
                    except Exception:
                        pass

                applied = telem.handle_control(msg, sender_addr=addr)
                # If the client retargets JSON telemetry, re-emit camera settings so the GUI can
                # immediately update overlay gating even if it missed the startup snapshot.
                try:
                    if isinstance(applied, dict) and "telemetry_out" in applied:
                        _emit_camera_settings()
                except Exception:
                    pass
                if applied2d:
                    try:
                        if applied:
                            applied = dict(applied)
                            applied.update(dict(applied2d))
                        else:
                            applied = dict(applied2d)
                    except Exception:
                        pass
                try:
                    # If the client toggles hole/plane acquisition, proactively clear any stale stage-0 preview overlay.
                    if mtype in ("hole_enable", "plane_enable"):
                        try:
                            en = msg.get("enable", True) if isinstance(msg, dict) else True
                            try:
                                en_i = int(en)
                                en_b = bool(en_i != 0)
                            except Exception:
                                en_b = bool(en)
                            prev_hole = bool(_hole_get_enabled())
                            prev_plane = bool(_plane_get_enabled())
                            if mtype == "hole_enable":
                                _hole_set_enabled(bool(en_b))
                                if bool(en_b):
                                    _plane_set_enabled(False)
                            else:
                                _plane_set_enabled(bool(en_b))
                                if bool(en_b):
                                    _hole_set_enabled(False)
                            new_hole = bool(_hole_get_enabled())
                            new_plane = bool(_plane_get_enabled())
                            mode_changed = (bool(new_hole) != bool(prev_hole)) or (bool(new_plane) != bool(prev_plane))
                            if bool(mode_changed):
                                _request_video_switch()
                            # Per-mode PID tuning for track3d (hole vs plane).
                            # Do not touch PID while the 2D tracker owns the PID source.
                            if bool(mode_changed) and (not bool(_track2d_is_active())):
                                try:
                                    _pid_src_set("track3d" if track3d is not None else "tracker")
                                except Exception:
                                    pass
                                pid_cfg_apply = pid_cfg_base
                                try:
                                    if bool(new_plane):
                                        pid_cfg_apply = pid_cfg_plane if pid_cfg_plane is not None else pid_cfg_base
                                    elif bool(new_hole):
                                        pid_cfg_apply = pid_cfg_hole if pid_cfg_hole is not None else pid_cfg_base
                                    else:
                                        pid_cfg_apply = pid_cfg_base
                                except Exception:
                                    pid_cfg_apply = pid_cfg_base
                                try:
                                    if pidmgr is not None and pid_cfg_apply is not None and hasattr(pidmgr, "apply_updates"):
                                        pidmgr.apply_updates(pid_cfg_apply)
                                except Exception:
                                    pass
                            if bool(mode_changed):
                                # Mark as applied for logs/UI (even though TelemetryManager ignores `type=...` controls).
                                try:
                                    a = dict(applied) if isinstance(applied, dict) else {}
                                    a["hole_enable"] = int(1 if bool(new_hole) else 0)
                                    a["plane_enable"] = int(1 if bool(new_plane) else 0)
                                    # Desired camera for logs (the actual switch is debounced).
                                    cam_want = None
                                    try:
                                        if _track2d_is_active():
                                            cam_want = "v4l"
                                        else:
                                            if track3d is not None and shm_src_ir is not None:
                                                cam_want = "ir"
                                            else:
                                                if track3d is not None and shm_src_rgb is not None:
                                                    cam_want = "rgb"
                                                else:
                                                    cam_want = "realsense"
                                    except Exception:
                                        cam_want = None
                                    a["camera_active"] = str(cam_want or _cam_get_active() or "").strip().lower()
                                    applied = a
                                except Exception:
                                    pass

                                # Clear stage-0 preview only when BOTH hole+plane acquisition are disabled.
                                if track3d is not None and (not bool(new_hole)) and (not bool(new_plane)):
                                    try:
                                        msg0 = track3d.make_acq_poly_msg(stage=0, verts_uv=[])
                                        if msg0 is not None:
                                            json_event(msg0)
                                    except Exception:
                                        pass
                        except Exception:
                            pass

                    if mtype in ("mouse_move", "mouse_stop", "mouse_click"):
                        with t3_lock:
                            mouse_total += 1
                            if mtype == "mouse_move":
                                mouse_move += 1
                            elif mtype == "mouse_stop":
                                mouse_stop += 1
                            else:
                                mouse_click += 1
                            mouse_last_wall = float(time.time())
                except Exception:
                    pass
                if mtype in ("mouse_move", "mouse_stop"):
                    # High-rate, best-effort; don't spam logs.
                    pass
                elif applied:
                    print(f"[control] applied: {applied} from {addr}", flush=True)
                else:
                    # Best-effort UDP keepalives (no-op most of the time); don't spam logs.
                    if mtype in ("hole_enable", "plane_enable", "track2d_enable", "mouse_click"):
                        pass
                    else:
                        print(f"[control] ignored: {msg}", flush=True)
            except Exception as e:
                print(f"[control] handler_exception: {e}", flush=True)
            if stop.is_set():
                break

    t_ctrl = threading.Thread(target=ctrl_loop, daemon=True)
    t_ctrl.start()

    # Graceful shutdown
    _shutting_down = threading.Event()
    _stop_main = threading.Event()

    def _request_shutdown(sig, frame):
        if _shutting_down.is_set():
            return
        _shutting_down.set()
        print(f"\n[server] shutting down (signal={sig}) ...", flush=True)
        # Nudge the external tracker to stop immediately (best-effort), so RealSense is released fast
        # even if the main shutdown path is cut short (service manager timeouts, etc).
        try:
            if track3d is not None and hasattr(track3d, "send_cmd"):
                track3d.send_cmd({"cmd": "shutdown", "ts": int(time.time() * 1000)})
        except Exception:
            pass
        # NOTE: Do not stop/terminate the subprocess from a daemon thread here.
        # If the interpreter exits, daemon threads are killed immediately, which can leave the tracker
        # orphaned (and the RealSense device stuck busy). The main `finally:` cleanup calls `track3d.stop()`
        # synchronously.
        stop.set()
        _stop_main.set()

    signal.signal(signal.SIGINT, _request_shutdown)
    signal.signal(signal.SIGTERM, _request_shutdown)
    print("[server] running. Press Ctrl+C to stop.")
    try:
        while not _stop_main.is_set():
            time.sleep(0.5)
    except KeyboardInterrupt:
        _request_shutdown(signal.SIGINT, None)
    finally:
        stop.set()
        try:
            if t_fps is not None and t_fps.is_alive():
                t_fps.join(timeout=0.5)
        except Exception:
            pass
        try:
            if imu_proc is not None:
                imu_proc.stop()
                imu_proc.join(timeout=1.0)
        except Exception:
            pass
        try:
            if t_ctrl.is_alive():
                t_ctrl.join(timeout=0.5)
        except Exception:
            pass
        try:
            if t_track3d is not None and t_track3d.is_alive():
                t_track3d.join(timeout=0.5)
        except Exception:
            pass
        try:
            if t_vpe is not None and t_vpe.is_alive():
                t_vpe.join(timeout=0.5)
        except Exception:
            pass
        try:
            if t_vse is not None and t_vse.is_alive():
                t_vse.join(timeout=0.5)
        except Exception:
            pass
        try:
            telem.stop()
        except Exception:
            pass
        # Notify GUI to clear detections/track overlays
        try:
            if _json_tx is not None:
                _json_tx.send(
                    {
                        "type": "detect",
                        "ts": int(time.time() * 1000),
                        "img_size": [0, 0],
                        "seq": -1,
                        "detections": [],
                    }
                )
        except Exception:
            pass
        try:
            if pidmgr and hasattr(pidmgr, "stop"):
                pidmgr.stop()
        except Exception:
            pass
        try:
            if tracker_obj and hasattr(tracker_obj, "stop"):
                tracker_obj.stop()
        except Exception:
            pass
        try:
            if track2d is not None and hasattr(track2d, "stop"):
                track2d.stop()
        except Exception:
            pass
        try:
            if v4l_src is not None and hasattr(v4l_src, "stop"):
                v4l_src.stop()
        except Exception:
            pass
        try:
            if track3d is not None:
                track3d.stop()
        except Exception:
            pass
        try:
            video.stop()
        except Exception:
            pass
        try:
            mav.stop()
        except Exception:
            pass
        try:
            for src in (shm_src_rgb, shm_src_ir, shm_src_depth):
                if src is not None and hasattr(src, "close"):
                    src.close()
        except Exception:
            pass
        try:
            if framebus and hasattr(framebus, "close"):
                framebus.close()
        except Exception:
            pass
        try:
            csvlog.close()
        except Exception:
            pass
        print("[server] stopped.", flush=True)

if __name__ == "__main__":
    main()
