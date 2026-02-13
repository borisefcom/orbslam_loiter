from __future__ import annotations

"""
Minimal RealSense RGB-D-Inertial -> ORB-SLAM3 (pybind) example.

This script is intentionally small but follows the stability rules we learned while integrating ORB-SLAM3 from Python:

1) Use a ring buffer for the numpy arrays passed into ORB-SLAM3.
   The binding converts numpy -> cv::Mat by pointer (zero-copy), and some ORB-SLAM3 builds keep cv::Mat references
   beyond a single TrackRGBD() call. Reusing/overwriting buffers can cause native crashes later.

2) Batch IMU between frames and ensure valid timestamps.
   vImuMeas is Nx7 float64: [ax,ay,az,gx,gy,gz,t_s] with strictly increasing t_s.
   Include a measurement at/after the image timestamp (real if available, otherwise a tiny synthetic "post" sample).

3) Use undistorted frames and set distortion=0 in ORB-SLAM3 YAML.
   This avoids issues with RealSense distortion models (e.g. inverse_brown_conrady).

For a deeper guide, see:
  - `orbslam/README_REALSENSE_PYTHON_GUIDE.md`

Note:
  - The production app `realsence_3d.py` runs ORB-SLAM3 in a separate process and publishes an explicit
    ORB startup/tracking state machine to the main GUI (LOADING_VOCAB / WAIT_IMU / WAIT_MOTION / TRACKING / LOST ...).
"""

import argparse
import json
import math
import multiprocessing as mp
import os
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from orbslam_app.imu_sync import ImuSynchronizer
from orbslam_app.paths import (
    DEFAULT_SETTINGS_RGBD_INERTIAL,
    DEFAULT_VOCAB_TAR_GZ,
    DEFAULT_VOCAB_TXT,
    THIRD_PARTY_ORB,
)
from orbslam_app.track3d_ipc import (
    IMU_KIND_ACCEL,
    IMU_KIND_GYRO,
    ShmImuRingWriter,
    ShmRingWriter,
    parse_shm_slots,
    sanitize_shm_prefix,
)
from orbslam_app.vocab import ensure_orb_vocab
from orbslam_app.headless_json_io import HeadlessIoSettings, HeadlessJsonIo

from orbslam_track3d.frames import OrbslamFrameResolver
from orbslam_track3d.file_io import write_json_atomic
from orbslam_track3d.ipc_vo import IpcVoState, InterProcessVoCommunicator
from orbslam_track3d.px4_vision import Px4VisionState, Px4VisionPublisher
from orbslam_track3d.realsense_utils import (
    build_undistort_maps,
    compute_T_b_c1_from_realsense,
    configure_realsense_depth_sensor,
    import_realsense,
    write_orbslam3_settings_from_realsense,
)
from orbslam_track3d.yaml_config import load_yaml_dict

from rpi_position_hold.processes.poly_vo_lk_track import _poly_vo_lk_track_process_main


def _linux_set_pdeathsig_best_effort(sig: int = signal.SIGTERM) -> None:
    """
    Linux-only: ask the kernel to deliver `sig` to this process if its parent dies.

    This prevents orphaned RealSense readers when the parent (server) is killed abruptly
    (e.g. SIGKILL from a service manager or `timeout` after SIGTERM).
    """
    try:
        if not str(sys.platform or "").lower().startswith("linux"):
            return
    except Exception:
        return
    try:
        import ctypes
        import ctypes.util

        libc_path = ctypes.util.find_library("c")
        if not libc_path:
            return
        libc = ctypes.CDLL(libc_path, use_errno=True)
        PR_SET_PDEATHSIG = 1
        try:
            libc.prctl(PR_SET_PDEATHSIG, int(sig), 0, 0, 0)
        except Exception:
            return
        # If we're already orphaned, terminate immediately (best-effort).
        try:
            if int(os.getppid()) == 1:
                os.kill(int(os.getpid()), int(sig))
        except Exception:
            pass
    except Exception:
        return


def _wait_for_first_frames(*, q, timeout_s: float, require_motion: bool) -> tuple[bool, bool]:
    """
    Return (got_frameset, got_motion).

    We need framesets for RGB/IR+Depth, and motion frames for IMU_RGBD tracking.
    """
    end = time.monotonic() + float(max(0.05, timeout_s))
    got_fs = False
    got_motion = False
    while time.monotonic() < end:
        try:
            f = q.wait_for_frame(200)
        except Exception:
            continue
        if f is None:
            continue
        try:
            if bool(f.is_motion_frame()):
                got_motion = True
            elif bool(f.is_frameset()):
                got_fs = True
        except Exception:
            pass
        if bool(got_fs) and (not bool(require_motion) or bool(got_motion)):
            break
    return bool(got_fs), bool(got_motion)


def _hardware_reset_realsense_best_effort(*, rs, dev, serial: str, print_fn=print, timeout_s: float = 8.0) -> bool:
    serial_s = str(serial or "").strip()

    def _wait_back() -> bool:
        end = time.monotonic() + float(max(1.0, timeout_s))
        while time.monotonic() < end:
            try:
                devs2 = rs.context().query_devices()
                n = int(devs2.size())  # type: ignore[attr-defined]
                if n <= 0:
                    time.sleep(0.1)
                    continue
                if not serial_s:
                    return True
                for i in range(n):
                    try:
                        d2 = devs2[i]
                        if str(d2.get_info(rs.camera_info.serial_number)) == serial_s:
                            return True
                    except Exception:
                        continue
            except Exception:
                pass
            time.sleep(0.1)
        return False

    try:
        msg = f"[min] WARN: RealSense no frames"
        if serial_s:
            msg += f"; hardware_reset serial={serial_s}"
        msg += " ..."
        print_fn(msg, flush=True)
    except Exception:
        pass

    # Prefer resetting the already-open device handle (works even if querying device info fails).
    if dev is not None:
        try:
            dev.hardware_reset()
            return bool(_wait_back())
        except Exception:
            pass

    # Fallback: reset by scanning the context.
    try:
        devs = rs.context().query_devices()
        n = int(devs.size())  # type: ignore[attr-defined]
    except Exception:
        devs = None
        n = 0
    if devs is None or n <= 0:
        return False

    d0 = None
    if serial_s:
        try:
            for i in range(n):
                try:
                    d = devs[i]
                    if str(d.get_info(rs.camera_info.serial_number)) == serial_s:
                        d0 = d
                        break
                except Exception:
                    continue
        except Exception:
            d0 = None
    if d0 is None:
        try:
            d0 = devs[0]
        except Exception:
            d0 = None
    if d0 is None:
        return False
    try:
        d0.hardware_reset()
    except Exception:
        return False
    return bool(_wait_back())


def _import_realsense():
    try:
        import pyrealsense2 as rs  # type: ignore
    except Exception as exc:
        raise RuntimeError(f"pyrealsense2 is required: {exc}") from exc
    return rs


def _build_undistort_maps(rs, intr) -> tuple[bool, Optional[np.ndarray], Optional[np.ndarray], str]:
    """
    Return (enabled, map1, map2, msg).
    Maps are for cv2.remap: dst(u,v) = src(map1(u,v), map2(u,v)).
    """
    model = intr.model
    if model == rs.distortion.none:
        return False, None, None, "none"

    if model in (rs.distortion.brown_conrady, rs.distortion.modified_brown_conrady):
        K = np.array(
            [
                [float(intr.fx), 0.0, float(intr.ppx)],
                [0.0, float(intr.fy), float(intr.ppy)],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        dist = np.array(intr.coeffs[:5], dtype=np.float32)
        map1, map2 = cv2.initUndistortRectifyMap(
            K, dist, None, K, (int(intr.width), int(intr.height)), cv2.CV_32FC1
        )
        return True, map1, map2, "brown_conrady"

    if model == rs.distortion.inverse_brown_conrady:
        H, W = int(intr.height), int(intr.width)
        map1 = np.empty((H, W), dtype=np.float32)
        map2 = np.empty((H, W), dtype=np.float32)
        xs = (np.arange(W, dtype=np.float32) - float(intr.ppx)) / float(intr.fx)
        for v in range(H):
            y = (float(v) - float(intr.ppy)) / float(intr.fy)
            for u in range(W):
                pt = [float(xs[u]), float(y), 1.0]
                pix = rs.rs2_project_point_to_pixel(intr, pt)
                map1[v, u] = pix[0]
                map2[v, u] = pix[1]
        return True, map1, map2, "inverse_brown_conrady"

    return False, None, None, f"unsupported({int(model)})"


def _write_orbslam3_settings_from_realsense(
    *,
    intr,
    template_path: Path,
    out_path: Path,
    camera_rgb: int,
    depth_map_factor: float,
    camera_fps: Optional[int],
    imu_frequency_hz: Optional[int],
    imu_T_b_c1: Optional[np.ndarray],
) -> Path:
    """
    Patch a template ORB-SLAM3 YAML with:
    - Camera intrinsics
    - Optional IMU frequency
    - Optional IMU.T_b_c1 (IMU/body -> camera)

    The templates in this repo already use distortion=0, so the caller should feed undistorted frames.
    Writes atomically (tmp + replace) to avoid corrupted YAML reads.
    """
    text = template_path.read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()
    out: list[str] = []

    def _fmt(v: float) -> str:
        return f"{float(v):.6f}"

    imu_block = False
    imu_data_line: Optional[str] = None
    imu_skip_multiline_data = False
    if imu_T_b_c1 is not None:
        T = np.asarray(imu_T_b_c1, dtype=np.float64).reshape(4, 4)
        flat = T.reshape(-1).tolist()
        flat_s = ",".join([f"{float(v):.6f}" for v in flat])
        imu_data_line = f"data: [{flat_s}]"

    for line in lines:
        if imu_skip_multiline_data:
            if "]" in line:
                imu_skip_multiline_data = False
            continue

        s = line.strip()
        if s.startswith("Camera1.fx:"):
            out.append(f"Camera1.fx: {_fmt(intr.fx)}")
            continue
        if s.startswith("Camera1.fy:"):
            out.append(f"Camera1.fy: {_fmt(intr.fy)}")
            continue
        if s.startswith("Camera1.cx:"):
            out.append(f"Camera1.cx: {_fmt(intr.ppx)}")
            continue
        if s.startswith("Camera1.cy:"):
            out.append(f"Camera1.cy: {_fmt(intr.ppy)}")
            continue
        if s.startswith("Camera.width:"):
            out.append(f"Camera.width: {int(intr.width)}")
            continue
        if s.startswith("Camera.height:"):
            out.append(f"Camera.height: {int(intr.height)}")
            continue
        if camera_fps is not None and s.startswith("Camera.fps:"):
            out.append(f"Camera.fps: {int(camera_fps)}")
            continue
        if s.startswith("Camera.RGB:"):
            out.append(f"Camera.RGB: {int(camera_rgb)}")
            continue
        if s.startswith("RGBD.DepthMapFactor:"):
            out.append(f"RGBD.DepthMapFactor: {float(depth_map_factor):.6f}")
            continue
        if imu_frequency_hz is not None and s.startswith("IMU.Frequency:"):
            out.append(f"IMU.Frequency: {float(int(imu_frequency_hz)):.1f}")
            continue
        if s.startswith("IMU.T_b_c1:"):
            imu_block = True
            out.append(line)
            continue
        if imu_block and s.startswith("data:"):
            if imu_data_line is not None:
                prefix = line[: len(line) - len(line.lstrip())]
                out.append(prefix + imu_data_line)
            else:
                out.append(line)
            if "]" not in line:
                imu_skip_multiline_data = True
            imu_block = False
            continue
        out.append(line)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp.write_text("\n".join(out) + "\n", encoding="utf-8")
    tmp.replace(out_path)
    return out_path


def _compute_T_b_c1_from_realsense(rs, profile, video_stream) -> Optional[np.ndarray]:
    """
    RealSense gives extrinsics from video stream -> gyro stream (cam -> imu).
    ORB-SLAM3 expects IMU.T_b_c1 = (IMU/body -> camera), so we invert.
    """
    try:
        gyro_sp = profile.get_stream(rs.stream.gyro).as_motion_stream_profile()
    except Exception:
        return None
    try:
        ext = video_stream.get_extrinsics_to(gyro_sp)  # cam(video) -> gyro(IMU)
        R = np.array(ext.rotation, dtype=np.float64).reshape(3, 3)
        t = np.array(ext.translation, dtype=np.float64).reshape(3)
        T_cam_gyro = np.eye(4, dtype=np.float64)
        T_cam_gyro[:3, :3] = R
        T_cam_gyro[:3, 3] = t
        return np.linalg.inv(T_cam_gyro)
    except Exception:
        return None


def _configure_realsense_depth_sensor(rs, depth_sensor) -> tuple[str, str]:
    """
    Configure RealSense depth sensor options for stable SLAM capture.

    - Set visual preset to High Accuracy (when supported).
    - Turn off IR emitter / illuminator (when supported).

    Returns (preset_msg, emitter_msg) for logging.
    """

    preset_msg = "n/a"
    emitter_msg = "n/a"

    # 1) Preset (can override other depth options).
    try:
        if depth_sensor.supports(rs.option.visual_preset):
            wanted_val: Optional[float] = None
            try:
                r = depth_sensor.get_option_range(rs.option.visual_preset)
                v_min = int(round(float(getattr(r, "min", 0.0))))
                v_max = int(round(float(getattr(r, "max", 0.0))))
                for v in range(v_min, v_max + 1):
                    try:
                        desc = str(depth_sensor.get_option_value_description(rs.option.visual_preset, v) or "")
                    except Exception:
                        continue
                    if "high accuracy" in desc.lower():
                        wanted_val = float(v)
                        break
            except Exception:
                wanted_val = None

            if wanted_val is None:
                # Common for D4xx: 3 == High Accuracy.
                wanted_val = 3.0

            depth_sensor.set_option(rs.option.visual_preset, float(wanted_val))

            try:
                cur = int(round(float(depth_sensor.get_option(rs.option.visual_preset))))
                desc = str(depth_sensor.get_option_value_description(rs.option.visual_preset, cur) or "")
                preset_msg = f"{desc}({cur})" if desc else str(cur)
            except Exception:
                preset_msg = f"set({wanted_val:g})"
    except Exception as exc:
        preset_msg = f"err({exc})"

    # 2) IR emitter / illuminator OFF (do after preset).
    try:
        if hasattr(rs.option, "emitter_enabled") and depth_sensor.supports(rs.option.emitter_enabled):
            depth_sensor.set_option(rs.option.emitter_enabled, 0.0)
            emitter_msg = "off"
        elif hasattr(rs.option, "laser_power") and depth_sensor.supports(rs.option.laser_power):
            depth_sensor.set_option(rs.option.laser_power, 0.0)
            emitter_msg = "laser_power=0"
    except Exception as exc:
        emitter_msg = f"err({exc})"

    return preset_msg, emitter_msg


def _parse_scalar(value: str) -> object:
    text = str(value).strip()
    if not text:
        return ""
    low = text.lower()
    if low in ("true", "yes", "on"):
        return True
    if low in ("false", "no", "off"):
        return False
    if low in ("null", "none", "~"):
        return None
    try:
        if "." in text or "e" in low:
            return float(text)
        return int(text)
    except Exception:
        return text.strip("\"'")


def _load_px4_yaml(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    try:
        import yaml  # type: ignore

        try:
            with path.open("r", encoding="utf-8") as f:
                data = yaml.safe_load(f)  # type: ignore[attr-defined]
        except UnicodeDecodeError:
            with path.open("r", encoding="utf-8-sig") as f:
                data = yaml.safe_load(f)  # type: ignore[attr-defined]
        return data if isinstance(data, dict) else {}
    except Exception:
        pass

    data: dict[str, object] = {}
    stack: list[tuple[int, dict[str, object]]] = [(0, data)]
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception:
        return {}
    for raw in lines:
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        key, sep, rest = line.strip().partition(":")
        if not sep:
            continue
        while stack and indent < stack[-1][0]:
            stack.pop()
        if not stack:
            stack = [(0, data)]
        parent = stack[-1][1]
        if rest.strip():
            parent[key] = _parse_scalar(rest)
        else:
            child: dict[str, object] = {}
            parent[key] = child
            stack.append((indent + 1, child))
    return data


def _rpy_deg_from_R(R: np.ndarray) -> tuple[float, float, float]:
    try:
        R = np.asarray(R, dtype=np.float64).reshape(3, 3)
        pitch = math.asin(max(-1.0, min(1.0, -float(R[2, 0]))))
        roll = math.atan2(float(R[2, 1]), float(R[2, 2]))
        yaw = math.atan2(float(R[1, 0]), float(R[0, 0]))
        return math.degrees(roll), math.degrees(pitch), math.degrees(yaw)
    except Exception:
        return 0.0, 0.0, 0.0


class _Px4VisionState:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._pose: Optional[dict[str, Any]] = None
        self._vel: Optional[dict[str, Any]] = None
        self._ts: Optional[float] = None

    def update(self, *, ts: float, pose: dict[str, Any], vel: Optional[dict[str, Any]]) -> None:
        with self._lock:
            self._ts = float(ts)
            self._pose = pose
            self._vel = vel

    def snapshot(self) -> tuple[Optional[dict[str, Any]], Optional[dict[str, Any]], Optional[float]]:
        with self._lock:
            return self._pose, self._vel, self._ts


class _Px4VisionPublisher:
    def __init__(self, *, cfg_path: Path, state: _Px4VisionState, print_fn=print) -> None:
        self.cfg_path = Path(cfg_path)
        self.state = state
        self.print = print_fn
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True, name="px4-vision")
        self._mav_bridge = None

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        try:
            if self._mav_bridge is not None:
                self._mav_bridge.stop()
        except Exception:
            pass
        try:
            if self._thread.is_alive():
                self._thread.join(timeout=1.0)
        except Exception:
            pass

    def _run(self) -> None:
        cfg = _load_px4_yaml(self.cfg_path)
        px4 = cfg.get("px4", {}) if isinstance(cfg, dict) else {}
        if not isinstance(px4, dict):
            return
        px4_enabled = bool(px4.get("enabled", False))
        odom = px4.get("odom", {})
        odom_enabled = bool(odom.get("enabled", True)) if isinstance(odom, dict) else True
        if not (px4_enabled and odom_enabled):
            try:
                if self.cfg_path.exists():
                    self.print("[px4] vision tx disabled (px4.enabled / px4.odom.enabled).", flush=True)
            except Exception:
                pass
            return

        serial = str(px4.get("serial", "")).strip()
        if not serial:
            try:
                self.print("[px4] vision tx disabled (px4.serial missing).", flush=True)
            except Exception:
                pass
            return

        baud = int(px4.get("baud", 115200) or 115200)
        dialect = str(px4.get("dialect", "common") or "common").strip() or "common"
        mavlink2 = bool(px4.get("mavlink2", True))
        rate_hz = float(odom.get("rate_hz", 30.0) or 30.0) if isinstance(odom, dict) else 30.0
        if not math.isfinite(rate_hz) or rate_hz <= 0.0:
            try:
                self.print("[px4] vision tx disabled (px4.odom.rate_hz <= 0).", flush=True)
            except Exception:
                pass
            return

        hb_hz = float(px4.get("heartbeat_hz", 1.0) or 1.0)
        status_hz = float(px4.get("status_hz", 0.0) or 0.0)

        os.environ["MAVLINK_DIALECT"] = dialect
        os.environ["MAVLINK20"] = "1" if mavlink2 else "0"

        try:
            from orbslam_app.mavlink_bridge import MavlinkBridge  # type: ignore
        except Exception as exc:
            try:
                self.print(f"[px4] vision tx disabled (pymavlink import error: {exc}).", flush=True)
            except Exception:
                pass
            return

        px4_cfg: dict[str, object] = {
            "serial": serial,
            "baud": int(baud),
            "dialect": dialect,
            "mavlink2": bool(mavlink2),
            "heartbeat": {"enabled": bool(hb_hz > 0.0), "rate_hz": float(hb_hz)},
            "status": {"enabled": bool(status_hz > 0.0), "rate_hz": float(status_hz)},
            "rates_hz": px4.get("rates_hz", {}) if isinstance(px4.get("rates_hz", {}), dict) else {},
        }
        if "source_system" in px4:
            try:
                px4_cfg["source_system"] = int(px4.get("source_system", 1) or 1)
            except Exception:
                pass
        if "source_component" in px4:
            try:
                px4_cfg["source_component"] = int(px4.get("source_component", 197) or 197)
            except Exception:
                pass

        try:
            self._mav_bridge = MavlinkBridge({"px4": px4_cfg}, print_fn=self.print)
            self._mav_bridge.start()
            if not self._mav_bridge.is_connected():
                err = self._mav_bridge.open_error()
                self.print(f"[px4] connect failed: {serial}@{int(baud)} ({err or 'unknown'})", flush=True)
                try:
                    self._mav_bridge.stop()
                except Exception:
                    pass
                self._mav_bridge = None
                return
            try:
                self._mav_bridge.print_status_once()
            except Exception:
                pass
            self.print(
                f"[px4] vision tx enabled -> {serial}@{int(baud)} "
                f"rate={rate_hz:.1f}Hz dialect={dialect} mavlink2={bool(mavlink2)}",
                flush=True,
            )
        except Exception as exc:
            try:
                self.print(f"[px4] init error: {exc}", flush=True)
            except Exception:
                pass
            self._mav_bridge = None
            return

        period_s = 1.0 / float(rate_hz)
        while not self._stop.is_set():
            try:
                pose, vel, _ts = self.state.snapshot()
                if pose is not None and self._mav_bridge is not None:
                    self._mav_bridge.send_external_vision_pose(pose)
                    if vel is not None:
                        self._mav_bridge.send_external_vision_speed(vel)
            except Exception:
                pass
            self._stop.wait(timeout=float(period_s))


def _write_json_atomic(path: Path, data: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=True, separators=(",", ":")) + "\n", encoding="utf-8")
    os.replace(str(tmp), str(path))


def _print_json_event(*, event: str, payload: dict) -> None:
    try:
        line = json.dumps(
            {"type": str(event), "ts_wall": float(time.time()), "payload": dict(payload)},
            ensure_ascii=True,
            separators=(",", ":"),
        )
    except Exception:
        return
    try:
        sys.__stdout__.write(line + "\n")
        sys.__stdout__.flush()
    except Exception:
        try:
            print(line, flush=True)
        except Exception:
            pass


class _IpcVoState:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._updated = threading.Event()
        self._frame_idx: int = -1
        self._timestamp: float = 0.0
        self._Twc: Optional[np.ndarray] = None
        self._ok: bool = False

    def update(self, *, frame_idx: int, timestamp: float, Twc: Optional[np.ndarray], ok: bool) -> None:
        with self._lock:
            self._frame_idx = int(frame_idx)
            self._timestamp = float(timestamp)
            self._ok = bool(ok)
            if Twc is None:
                self._Twc = None
            else:
                self._Twc = np.asarray(Twc, dtype=np.float64).reshape(4, 4).copy()
        self._updated.set()

    def wait_updated(self, timeout_s: float) -> bool:
        return bool(self._updated.wait(timeout=float(timeout_s)))

    def snapshot(self) -> tuple[int, float, bool, Optional[np.ndarray]]:
        with self._lock:
            fi = int(self._frame_idx)
            ts = float(self._timestamp)
            ok = bool(self._ok)
            Twc = self._Twc
            if Twc is not None:
                Twc = np.asarray(Twc, dtype=np.float64).copy()
        try:
            self._updated.clear()
        except Exception:
            pass
        return fi, ts, ok, Twc


class _InterProcessVoCommunicator:
    """
    Publish latest VO pose to shared memory using the layout in:
      D:\\DroneServer\\Drone_client\\New folder\\ipc\\shm_state.py
    """

    def __init__(
        self,
        *,
        state: _IpcVoState,
        cfg_path: Optional[Path] = None,
        out_spec_path: Optional[Path] = None,
        print_fn=print,
    ) -> None:
        self.state = state
        self.cfg_path = Path(cfg_path) if cfg_path is not None else None
        self.out_spec_path = Path(out_spec_path) if out_spec_path is not None else None
        self.print = print_fn
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True, name="ipc-vo")

        self._shm_meta = None
        self._shm_R = None
        self._shm_t = None
        self._shm_Twc = None
        self._meta = None
        self._R = None
        self._t = None
        self._Twc = None
        self._spec: Optional[dict[str, str]] = None
        self._last_Twc_w: Optional[np.ndarray] = None

    def start(self) -> None:
        self._thread.start()

    def spec_dict(self) -> Optional[dict[str, str]]:
        """Return the shared-memory state spec (once initialized)."""
        try:
            return dict(self._spec) if isinstance(self._spec, dict) else None
        except Exception:
            return None

    def wait_spec(self, *, timeout_s: float = 2.0) -> Optional[dict[str, str]]:
        """Block until `spec_dict()` becomes available or timeout."""
        deadline = float(time.time()) + float(max(0.0, float(timeout_s)))
        while float(time.time()) < float(deadline) and (not self._stop.is_set()):
            spec = self.spec_dict()
            if isinstance(spec, dict) and spec:
                return spec
            try:
                time.sleep(0.01)
            except Exception:
                break
        return self.spec_dict()

    def stop(self) -> None:
        self._stop.set()
        try:
            if self._thread.is_alive():
                self._thread.join(timeout=1.0)
        except Exception:
            pass
        try:
            for shm in (self._shm_meta, self._shm_R, self._shm_t, self._shm_Twc):
                if shm is None:
                    continue
                try:
                    shm.close()
                except Exception:
                    pass
        except Exception:
            pass
        try:
            unlink_on_exit = True
            try:
                cfg_path = self.cfg_path or (PROJECT_ROOT / "apps" / "px4_mavlink.yaml")
                cfg = _load_px4_yaml(cfg_path) if cfg_path is not None else {}
                ipc_cfg = cfg.get("ipc_vo", {}) if isinstance(cfg, dict) else {}
                if isinstance(ipc_cfg, dict) and ("unlink_on_exit" in ipc_cfg):
                    unlink_on_exit = bool(ipc_cfg.get("unlink_on_exit", True))
            except Exception:
                unlink_on_exit = True
            if "ORB_VO_IPC_UNLINK_ON_EXIT" in os.environ:
                env = str(os.environ.get("ORB_VO_IPC_UNLINK_ON_EXIT", "1") or "1").strip().lower()
                unlink_on_exit = env not in ("0", "false", "no", "off")

            if bool(unlink_on_exit):
                for shm in (self._shm_meta, self._shm_R, self._shm_t, self._shm_Twc):
                    if shm is None:
                        continue
                    try:
                        shm.unlink()
                    except Exception:
                        pass
        except Exception:
            pass

    def _attach_shm(self, name: str):
        from multiprocessing.shared_memory import SharedMemory

        try:
            return SharedMemory(name=str(name), create=False, track=False)  # type: ignore[call-arg]
        except TypeError:
            return SharedMemory(name=str(name), create=False)

    def _create_shm_state(self, name_prefix: str) -> None:
        from multiprocessing.shared_memory import SharedMemory

        meta_dtype = np.dtype(
            [
                ("seq", np.uint32),  # even=stable, odd=writer in progress
                ("frame_idx", np.int64),
                ("timestamp", np.float64),
                ("ok", np.uint8),
                ("odom_stable", np.uint8),
                ("n_corr", np.int32),
                ("n_inliers", np.int32),
                ("rmse_m", np.float32),
                ("rot_deg", np.float32),
                ("status_code", np.int32),
                ("est_code", np.int32),
                ("ms_est", np.float32),
                ("ms_total", np.float32),
                ("ms_prefilter", np.float32),
                ("ms_pg", np.float32),
                ("ms_jitter", np.float32),
                ("ms_imu", np.float32),
                ("ms_weights", np.float32),
                ("ms_gate", np.float32),
            ]
        )

        meta_nbytes = int(meta_dtype.itemsize)
        R_nbytes = int(3 * 3 * np.dtype(np.float32).itemsize)
        t_nbytes = int(3 * np.dtype(np.float32).itemsize)
        Twc_nbytes = int(4 * 4 * np.dtype(np.float32).itemsize)

        def _try_create() -> tuple[object, object, object, object]:
            shm_meta = SharedMemory(create=True, size=meta_nbytes, name=f"{name_prefix}_meta")
            shm_R = SharedMemory(create=True, size=R_nbytes, name=f"{name_prefix}_R")
            shm_t = SharedMemory(create=True, size=t_nbytes, name=f"{name_prefix}_t")
            shm_Twc = SharedMemory(create=True, size=Twc_nbytes, name=f"{name_prefix}_Twc")
            return shm_meta, shm_R, shm_t, shm_Twc

        try:
            shm_meta, shm_R, shm_t, shm_Twc = _try_create()
        except FileExistsError:
            # Best-effort cleanup of stale segments.
            for n in (f"{name_prefix}_meta", f"{name_prefix}_R", f"{name_prefix}_t", f"{name_prefix}_Twc"):
                try:
                    old = self._attach_shm(n)
                    try:
                        old.close()
                    except Exception:
                        pass
                    try:
                        old.unlink()
                    except Exception:
                        pass
                except Exception:
                    pass
            shm_meta, shm_R, shm_t, shm_Twc = _try_create()

        self._shm_meta = shm_meta
        self._shm_R = shm_R
        self._shm_t = shm_t
        self._shm_Twc = shm_Twc

        self._meta = np.ndarray((1,), dtype=meta_dtype, buffer=shm_meta.buf)
        self._R = np.ndarray((3, 3), dtype=np.float32, buffer=shm_R.buf)
        self._t = np.ndarray((3,), dtype=np.float32, buffer=shm_t.buf)
        self._Twc = np.ndarray((4, 4), dtype=np.float32, buffer=shm_Twc.buf)

        self._meta["seq"][0] = np.uint32(0)
        self._meta["frame_idx"][0] = np.int64(-1)
        self._meta["timestamp"][0] = float(0.0)
        self._meta["ok"][0] = np.uint8(0)
        self._meta["odom_stable"][0] = np.uint8(0)
        self._meta["n_corr"][0] = np.int32(0)
        self._meta["n_inliers"][0] = np.int32(0)
        self._meta["rmse_m"][0] = np.float32(0.0)
        self._meta["rot_deg"][0] = np.float32(0.0)
        self._meta["status_code"][0] = np.int32(0)
        self._meta["est_code"][0] = np.int32(0)
        self._meta["ms_est"][0] = np.float32(0.0)
        self._meta["ms_total"][0] = np.float32(0.0)
        self._meta["ms_prefilter"][0] = np.float32(0.0)
        self._meta["ms_pg"][0] = np.float32(0.0)
        self._meta["ms_jitter"][0] = np.float32(0.0)
        self._meta["ms_imu"][0] = np.float32(0.0)
        self._meta["ms_weights"][0] = np.float32(0.0)
        self._meta["ms_gate"][0] = np.float32(0.0)
        self._R[:] = np.eye(3, dtype=np.float32)
        self._t[:] = 0.0
        self._Twc[:] = np.eye(4, dtype=np.float32)

        self._spec = {
            "name_prefix": str(name_prefix),
            "shm_meta": str(getattr(shm_meta, "name", f"{name_prefix}_meta")),
            "shm_R": str(getattr(shm_R, "name", f"{name_prefix}_R")),
            "shm_t": str(getattr(shm_t, "name", f"{name_prefix}_t")),
            "shm_Twc": str(getattr(shm_Twc, "name", f"{name_prefix}_Twc")),
        }

    def _write_state(self, *, frame_idx: int, timestamp: float, ok: bool, Twc: Optional[np.ndarray]) -> None:
        if self._meta is None or self._R is None or self._t is None or self._Twc is None:
            return

        seq0 = int(self._meta["seq"][0])
        self._meta["seq"][0] = np.uint32(seq0 + 1)

        self._meta["frame_idx"][0] = np.int64(int(frame_idx))
        self._meta["timestamp"][0] = float(timestamp)
        self._meta["ok"][0] = np.uint8(1 if bool(ok) else 0)
        self._meta["odom_stable"][0] = np.uint8(1 if bool(ok) else 0)
        self._meta["n_corr"][0] = np.int32(0)
        self._meta["n_inliers"][0] = np.int32(0)
        self._meta["rmse_m"][0] = np.float32(0.0)
        rot_deg = 0.0
        self._meta["status_code"][0] = np.int32(0)
        self._meta["est_code"][0] = np.int32(0)

        if Twc is not None:
            T = np.asarray(Twc, dtype=np.float32).reshape(4, 4)
            # R/t are the per-step delta in the VO pipeline; Twc is the integrated pose.
            try:
                if self._last_Twc_w is not None:
                    Tprev = np.asarray(self._last_Twc_w, dtype=np.float32).reshape(4, 4)
                    Rprev = Tprev[:3, :3]
                    Rcur = T[:3, :3]
                    Rstep = (Rprev.T @ Rcur).astype(np.float32, copy=False)
                    tstep = (T[:3, 3] - Tprev[:3, 3]).astype(np.float32, copy=False)
                    self._R[:, :] = Rstep
                    self._t[:] = tstep
                    tr = float(Rstep[0, 0] + Rstep[1, 1] + Rstep[2, 2])
                    c = max(-1.0, min(1.0, (tr - 1.0) * 0.5))
                    rot_deg = float(math.degrees(math.acos(c)))
                else:
                    self._R[:, :] = np.eye(3, dtype=np.float32)
                    self._t[:] = 0.0
            except Exception:
                self._R[:, :] = np.eye(3, dtype=np.float32)
                self._t[:] = 0.0
                rot_deg = 0.0

            self._Twc[:, :] = T
            self._last_Twc_w = T.copy()
        else:
            # Keep last Twc if available; clear step delta.
            self._R[:, :] = np.eye(3, dtype=np.float32)
            self._t[:] = 0.0
            if self._last_Twc_w is None:
                self._Twc[:, :] = np.eye(4, dtype=np.float32)
            else:
                try:
                    self._Twc[:, :] = np.asarray(self._last_Twc_w, dtype=np.float32).reshape(4, 4)
                except Exception:
                    self._Twc[:, :] = np.eye(4, dtype=np.float32)
            rot_deg = 0.0

        self._meta["rot_deg"][0] = np.float32(float(rot_deg))

        self._meta["seq"][0] = np.uint32(seq0 + 2)

    def _resolve_name_prefix(self) -> str:
        env_prefix = str(os.environ.get("ORB_VO_IPC_PREFIX", "") or "").strip()
        if env_prefix:
            return env_prefix

        cfg_path = self.cfg_path or (PROJECT_ROOT / "apps" / "px4_mavlink.yaml")
        cfg = _load_px4_yaml(cfg_path) if cfg_path is not None else {}
        ipc_cfg = cfg.get("ipc_vo", {}) if isinstance(cfg, dict) else {}
        if isinstance(ipc_cfg, dict):
            cfg_prefix = str(ipc_cfg.get("name_prefix", "") or ipc_cfg.get("prefix", "") or "").strip()
            if cfg_prefix:
                return cfg_prefix
        cap = cfg.get("capture", {}) if isinstance(cfg, dict) else {}
        if isinstance(cap, dict):
            base = str(cap.get("shm_name_prefix", "") or "").strip()
            if base:
                return f"{base}_orb_state"
        return f"orb_state_{os.getpid()}"

    def _run(self) -> None:
        enabled = True
        cfg = {}
        ipc_cfg: dict[str, object] = {}
        try:
            cfg_path = self.cfg_path or (PROJECT_ROOT / "apps" / "px4_mavlink.yaml")
            cfg = _load_px4_yaml(cfg_path) if cfg_path is not None else {}
            ipc_raw = cfg.get("ipc_vo", {}) if isinstance(cfg, dict) else {}
            if isinstance(ipc_raw, dict):
                ipc_cfg = dict(ipc_raw)
        except Exception:
            cfg = {}
            ipc_cfg = {}

        if "enabled" in ipc_cfg:
            enabled = bool(ipc_cfg.get("enabled", True))
        if "ORB_VO_IPC_ENABLED" in os.environ:
            enabled_env = str(os.environ.get("ORB_VO_IPC_ENABLED", "1") or "1").strip().lower()
            enabled = enabled_env not in ("0", "false", "no", "off")
        if not bool(enabled):
            return

        if self.out_spec_path is None:
            try:
                out_cfg = str(ipc_cfg.get("out_spec_path", "") or "").strip()
                if out_cfg:
                    p = Path(out_cfg)
                    self.out_spec_path = (p if p.is_absolute() else (PROJECT_ROOT / p))
            except Exception:
                pass

        try:
            name_prefix = self._resolve_name_prefix()
            self._create_shm_state(name_prefix)
        except Exception as exc:
            try:
                self.print(f"[ipc_vo] init error: {exc}", flush=True)
            except Exception:
                pass
            return

        try:
            spec = dict(self._spec or {})
            out_path = self.out_spec_path or (PROJECT_ROOT / ".tmp" / "orb_vo_ipc_state_spec.json")
            _write_json_atomic(Path(out_path), spec)
            self.print(f"[ipc_vo] shm_state spec -> {out_path}", flush=True)
        except Exception:
            pass

        last_fi = -1
        while not self._stop.is_set():
            try:
                _ = self.state.wait_updated(timeout_s=0.2)
                fi, ts, ok, Twc = self.state.snapshot()
                if fi < 0:
                    continue
                if int(fi) == int(last_fi):
                    continue
                last_fi = int(fi)
                self._write_state(frame_idx=int(fi), timestamp=float(ts), ok=bool(ok), Twc=Twc if ok else None)
            except Exception:
                pass


def main() -> int:
    _linux_set_pdeathsig_best_effort(signal.SIGTERM)
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--video",
        choices=["ir_left", "color"],
        default="ir_left",
        help="Video stream to use for tracking (default: ir_left for global shutter).",
    )
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--imu-fps", type=int, default=200)
    ap.add_argument("--auto-exit-s", type=float, default=0.0)
    ap.add_argument("--no-vis", action="store_true", help="Disable OpenCV visualization.")
    ap.add_argument(
        "--track3d-config",
        "--config",
        dest="track3d_config",
        type=str,
        default=str(PROJECT_ROOT / "apps" / "px4_mavlink.yaml"),
        help="3d_track-style YAML config (used for SHM ring/IMU + headless init/telemetry). Alias: --config.",
    )
    ap.add_argument(
        "--headless",
        action="store_true",
        help="3d_track-compatible headless mode (implies --no-vis; JSON I/O is controlled by the YAML config).",
    )
    args = ap.parse_args()
    if bool(getattr(args, "headless", False)):
        args.no_vis = True

    track3d_cfg_path = None
    track3d_cfg: dict[str, object] = {}
    try:
        cfg_s = str(getattr(args, "track3d_config", "") or "").strip()
        if cfg_s:
            track3d_cfg_path = Path(cfg_s)
            track3d_cfg = load_yaml_dict(track3d_cfg_path)
    except Exception:
        track3d_cfg_path = None
        track3d_cfg = {}

    cap_cfg = track3d_cfg.get("capture", {}) if isinstance(track3d_cfg, dict) else {}
    if not isinstance(cap_cfg, dict):
        cap_cfg = {}
    app_cfg = track3d_cfg.get("app", {}) if isinstance(track3d_cfg, dict) else {}
    if not isinstance(app_cfg, dict):
        app_cfg = {}
    headless_io_cfg = app_cfg.get("headless_io", {}) if isinstance(app_cfg, dict) else {}
    if not isinstance(headless_io_cfg, dict):
        headless_io_cfg = {}
    headless_io_enabled = bool(headless_io_cfg.get("enabled", False))
    headless_transport = str(headless_io_cfg.get("transport", "stdio_jsonl") or "stdio_jsonl").strip().lower()
    headless_stdin_enabled = bool(headless_io_cfg.get("stdin_enabled", True))
    headless_stdout_enabled = bool(headless_io_cfg.get("stdout_enabled", True))
    try:
        headless_max_in = int(headless_io_cfg.get("max_in_per_tick", 50) or 50)
    except Exception:
        headless_max_in = 50
    try:
        telemetry_rate_hz = float(headless_io_cfg.get("telemetry_rate_hz", 0.0) or 0.0)
    except Exception:
        telemetry_rate_hz = 0.0
    try:
        features_rate_hz = float(headless_io_cfg.get("features_rate_hz", 0.0) or 0.0)
    except Exception:
        features_rate_hz = 0.0
    features_compact = bool(headless_io_cfg.get("features_compact", True))
    try:
        features_max_tracks = int(headless_io_cfg.get("max_tracks", 120) or 120)
    except Exception:
        features_max_tracks = 120
    telemetry_period_s = (1.0 / telemetry_rate_hz) if telemetry_rate_hz > 0.0 else 0.0
    features_period_s = (1.0 / features_rate_hz) if features_rate_hz > 0.0 else 0.0

    headless_settings = HeadlessIoSettings(
        enabled=bool(headless_io_enabled),
        transport=str(headless_transport),
        stdin_enabled=bool(headless_stdin_enabled),
        stdout_enabled=bool(headless_stdout_enabled),
        max_in_per_tick=int(max(1, headless_max_in)),
        eof_is_shutdown=True,
    )
    headless_io = HeadlessJsonIo(settings=headless_settings)

    # ORB-SLAM3 steady-odom continuity thresholds (detect map/BA jumps; keep odom continuous).
    orb_frames_cfg = app_cfg.get("orbslam_frames", {}) if isinstance(app_cfg, dict) else {}
    if not isinstance(orb_frames_cfg, dict):
        orb_frames_cfg = {}
    try:
        odom_jump_trans_m = float(orb_frames_cfg.get("jump_trans_m", 0.10) or 0.10)
    except Exception:
        odom_jump_trans_m = 0.10
    try:
        odom_jump_rot_deg = float(orb_frames_cfg.get("jump_rot_deg", 7.0) or 7.0)
    except Exception:
        odom_jump_rot_deg = 7.0
    if not math.isfinite(odom_jump_trans_m) or odom_jump_trans_m <= 0.0:
        odom_jump_trans_m = 0.10
    if not math.isfinite(odom_jump_rot_deg) or odom_jump_rot_deg <= 0.0:
        odom_jump_rot_deg = 7.0

    poly_proc_cfg: dict[str, object] = {}
    poly_vo_lk_cfg: dict[str, object] = {}
    try:
        if isinstance(app_cfg, dict):
            poly_proc_cfg = dict(app_cfg.get("poly_process") or {})
            poly_vo_lk_cfg = dict(app_cfg.get("poly_vo_lk_track") or {})
    except Exception:
        poly_proc_cfg = {}
        poly_vo_lk_cfg = {}
    poly_use_process = (
        bool(getattr(args, "headless", False))
        and bool(headless_io_enabled)
        and bool(poly_proc_cfg.get("enabled", True))
        and bool(poly_vo_lk_cfg.get("enabled", False))
    )

    # Optional: pull capture defaults from the 3d_track-style YAML config.
    try:
        if "out_width" in cap_cfg:
            args.width = int(cap_cfg.get("out_width", int(args.width)) or int(args.width))
        if "out_height" in cap_cfg:
            args.height = int(cap_cfg.get("out_height", int(args.height)) or int(args.height))
        if "fps" in cap_cfg:
            args.fps = int(cap_cfg.get("fps", int(args.fps)) or int(args.fps))
        rs_color_stream = str(cap_cfg.get("rs_color_stream", "") or "").strip().lower()
        if rs_color_stream:
            if rs_color_stream in ("color", "rgb"):
                args.video = "color"
            elif rs_color_stream in ("infra1", "ir1", "infrared1", "left", "ir_left", "ir"):
                args.video = "ir_left"
    except Exception:
        pass

    # RealSense
    rs = import_realsense()

    # Import ORB-SLAM3 wrapper.
    sys.path.insert(0, str(THIRD_PARTY_ORB))
    orb_pw = THIRD_PARTY_ORB / "python_wrapper"
    if hasattr(os, "add_dll_directory") and orb_pw.exists():
        os.add_dll_directory(str(orb_pw))
    from python_wrapper.orb_slam3 import ORB_SLAM3  # type: ignore

    vocab_txt = ensure_orb_vocab(vocab_txt=DEFAULT_VOCAB_TXT, vocab_tar_gz=DEFAULT_VOCAB_TAR_GZ)

    pipe = rs.pipeline()
    cfg = rs.config()

    use_color = str(args.video).lower() == "color"
    # Optional extra RGB stream for the GUI/video pipeline.
    # Enables Off-mode RGB streaming while SLAM stays on IR (global shutter).
    extra_color_stream = bool(cap_cfg.get("rs_extra_color_stream", True)) and (not bool(use_color))
    if use_color or extra_color_stream:
        cfg.enable_stream(rs.stream.color, int(args.width), int(args.height), rs.format.bgr8, int(args.fps))
    if not use_color:
        cfg.enable_stream(rs.stream.infrared, 1, int(args.width), int(args.height), rs.format.y8, int(args.fps))
    cfg.enable_stream(rs.stream.depth, int(args.width), int(args.height), rs.format.z16, int(args.fps))
    cfg.enable_stream(rs.stream.gyro, rs.format.motion_xyz32f, int(args.imu_fps))
    cfg.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, int(args.imu_fps))

    q = rs.frame_queue(4096)
    profile = None
    start_exc: Optional[BaseException] = None
    try:
        profile = pipe.start(cfg, q)
    except Exception as exc:
        start_exc = exc
        if bool(extra_color_stream):
            # Some setups cannot stream IR+depth+IMU+RGB simultaneously (USB bandwidth / firmware constraints).
            # Retry without the extra RGB stream (keeps the original IR-only behavior).
            try:
                print("[min] RealSense start failed with extra RGB stream; retrying without RGB...", flush=True)
            except Exception:
                pass
            try:
                cfg2 = rs.config()
                cfg2.enable_stream(rs.stream.infrared, 1, int(args.width), int(args.height), rs.format.y8, int(args.fps))
                cfg2.enable_stream(rs.stream.depth, int(args.width), int(args.height), rs.format.z16, int(args.fps))
                cfg2.enable_stream(rs.stream.gyro, rs.format.motion_xyz32f, int(args.imu_fps))
                cfg2.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, int(args.imu_fps))
                profile = pipe.start(cfg2, q)
                cfg = cfg2
                extra_color_stream = False
                start_exc = None
            except Exception as exc2:
                start_exc = exc2

    if start_exc is not None or profile is None:
        exc = start_exc if start_exc is not None else RuntimeError("RealSense start failed")
        # RealSense isn't visible/connected in some environments; fail with an actionable message.
        try:
            n_devs = None
            try:
                devs = rs.context().query_devices()
                n_devs = int(devs.size())  # type: ignore[attr-defined]
            except Exception:
                n_devs = None
            msg = f"[min] RealSense start failed: {exc}"
            if n_devs is not None:
                msg += f" (ctx.query_devices={int(n_devs)})"
            print(msg, flush=True)
        except Exception:
            pass
        try:
            headless_io.close()
        except Exception:
            pass
        return 1

    dev = profile.get_device()
    serial = ""
    usb = ""
    try:
        serial = dev.get_info(rs.camera_info.serial_number)
        usb = dev.get_info(rs.camera_info.usb_type_descriptor)
    except Exception:
        pass

    # If the device is present but hung (we've observed this after abrupt SIGTERM / orphaned readers),
    # the pipeline can "start" successfully but produce no frames (all waits time out).
    # Detect that early and attempt a single hardware_reset + restart.
    try:
        frame_watchdog_s = float(cap_cfg.get("rs_start_frame_timeout_s", 2.0) or 2.0)
    except Exception:
        frame_watchdog_s = 2.0
    require_imu = True
    try:
        require_imu = bool(cap_cfg.get("rs_start_require_imu", True))
    except Exception:
        require_imu = True
    got_fs, got_motion = _wait_for_first_frames(q=q, timeout_s=frame_watchdog_s, require_motion=bool(require_imu))
    if (not bool(got_fs)) or (bool(require_imu) and (not bool(got_motion))):
        try:
            pipe.stop()
        except Exception:
            pass
        ok_reset = _hardware_reset_realsense_best_effort(rs=rs, dev=dev, serial=str(serial), print_fn=print, timeout_s=8.0)
        if not ok_reset:
            try:
                if bool(require_imu) and bool(got_fs) and (not bool(got_motion)):
                    print(f"[min] RealSense has video but no IMU frames; hardware_reset failed (serial={serial or '-'}).", flush=True)
                else:
                    print(f"[min] RealSense appears stuck and hardware_reset failed (serial={serial or '-'}).", flush=True)
            except Exception:
                pass
            try:
                headless_io.close()
            except Exception:
                pass
            return 1

        # Rebuild the pipeline after reset.
        pipe = rs.pipeline()
        cfg = rs.config()
        if use_color or extra_color_stream:
            cfg.enable_stream(rs.stream.color, int(args.width), int(args.height), rs.format.bgr8, int(args.fps))
        if not use_color:
            cfg.enable_stream(rs.stream.infrared, 1, int(args.width), int(args.height), rs.format.y8, int(args.fps))
        cfg.enable_stream(rs.stream.depth, int(args.width), int(args.height), rs.format.z16, int(args.fps))
        cfg.enable_stream(rs.stream.gyro, rs.format.motion_xyz32f, int(args.imu_fps))
        cfg.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, int(args.imu_fps))
        q = rs.frame_queue(4096)
        try:
            profile = pipe.start(cfg, q)
        except Exception as exc:
            try:
                print(f"[min] RealSense restart after hardware_reset failed: {exc}", flush=True)
            except Exception:
                pass
            try:
                headless_io.close()
            except Exception:
                pass
            return 1
        dev = profile.get_device()
        try:
            serial = dev.get_info(rs.camera_info.serial_number)
            usb = dev.get_info(rs.camera_info.usb_type_descriptor)
        except Exception:
            pass

    def _parse_bool_auto(v: object) -> Optional[bool]:
        if v is None:
            return None
        if isinstance(v, str):
            s = str(v).strip().lower()
            if s in ("", "auto", "default", "none", "na", "n/a"):
                return None
            return bool(s in ("1", "true", "yes", "on", "enabled"))
        try:
            return bool(v)
        except Exception:
            return None

    # Optional: apply device timestamp domain settings (3d_track capture compatibility).
    try:
        rs_global_time_enabled = _parse_bool_auto(cap_cfg.get("rs_global_time_enabled", None))
        if rs_global_time_enabled is not None:
            for s in dev.query_sensors():
                try:
                    if s.supports(rs.option.global_time_enabled):
                        s.set_option(rs.option.global_time_enabled, 1.0 if bool(rs_global_time_enabled) else 0.0)
                except Exception:
                    pass
            try:
                print(f"[min] rs_global_time_enabled={int(bool(rs_global_time_enabled))}", flush=True)
            except Exception:
                pass
    except Exception:
        pass

    try:
        rs_enable_motion_correction = _parse_bool_auto(cap_cfg.get("rs_enable_motion_correction", None))
        if rs_enable_motion_correction is not None:
            for s in dev.query_sensors():
                try:
                    if s.supports(rs.option.enable_motion_correction):
                        s.set_option(
                            rs.option.enable_motion_correction, 1.0 if bool(rs_enable_motion_correction) else 0.0
                        )
                except Exception:
                    pass
            try:
                print(f"[min] rs_enable_motion_correction={int(bool(rs_enable_motion_correction))}", flush=True)
            except Exception:
                pass
    except Exception:
        pass

    depth_sensor = dev.first_depth_sensor()
    preset_msg, emitter_msg = configure_realsense_depth_sensor(rs, depth_sensor)
    depth_scale = float(depth_sensor.get_depth_scale())

    align = rs.align(rs.stream.color if use_color else rs.stream.infrared)

    # Intrinsics for the chosen video stream (used for undistortion + YAML patching).
    if use_color:
        vsp = profile.get_stream(rs.stream.color).as_video_stream_profile()
    else:
        vsp = profile.get_stream(rs.stream.infrared, 1).as_video_stream_profile()
    intr = vsp.get_intrinsics()

    # Undistort maps (if needed).
    undistort_enabled, map1, map2, undistort_msg = build_undistort_maps(rs, intr)

    # Factory IMU->cam extrinsics.
    imu_T_b_c1 = compute_T_b_c1_from_realsense(rs, profile, vsp)

    # ORB settings YAML (patch the template with factory intrinsics and extrinsics).
    runtime_yaml = PROJECT_ROOT / ".tmp" / "orbslam3_realsense_runtime.yaml"
    settings = write_orbslam3_settings_from_realsense(
        intr=intr,
        template_path=Path(DEFAULT_SETTINGS_RGBD_INERTIAL),
        out_path=runtime_yaml,
        camera_rgb=0,            # BGR (we request rs.format.bgr8 for color)
        depth_map_factor=1.0,    # we feed meters float32
        camera_fps=int(args.fps),
        imu_frequency_hz=int(args.imu_fps),
        imu_T_b_c1=imu_T_b_c1,
    )

    slam = ORB_SLAM3(str(vocab_txt), str(settings), "IMU_RGBD", False)
    frame_resolver = OrbslamFrameResolver(jump_trans_m=odom_jump_trans_m, jump_rot_deg=odom_jump_rot_deg)

    # Ring buffers for ORB input (avoid buffer reuse crashes).
    H = int(intr.height)
    W = int(intr.width)
    ring_n = 64
    img_ring = np.empty((ring_n, H, W, 3), dtype=np.uint8)  # for color (or IR expanded to BGR)
    pub_bgr_ring = np.empty((ring_n, H, W, 3), dtype=np.uint8) if bool(extra_color_stream) and (not bool(use_color)) else None
    depth_ring = np.empty((ring_n, H, W), dtype=np.float32)
    gray_ring = np.empty((ring_n, H, W), dtype=np.uint8)
    depth_raw_ring = np.empty((ring_n, H, W), dtype=np.uint16)
    intr_dict = {
        "w": int(W),
        "h": int(H),
        "fx": float(intr.fx),
        "fy": float(intr.fy),
        "cx": float(intr.ppx),
        "cy": float(intr.ppy),
    }
    # OpenCV remap() requires a contiguous destination; a single-channel view like `img_in[:, :, 0]`
    # is not a valid cv::Mat layout for `dst`. Keep a dedicated contiguous IR buffer per ring slot.
    ir_ring = np.empty((ring_n, H, W), dtype=np.uint8)
    ring_i = 0

    depth_m_buf = np.empty((H, W), dtype=np.float32)
    img_u8 = None

    # Optional depth visualization for the external consumer stream (stored into the SHM ring `depth_bgr` slot).
    # We keep this as an 8-bit grayscale image replicated to BGR so consumers can choose BGR or gray8 streaming.
    depth_vis_enabled = bool(cap_cfg.get("depth_vis_enabled", True))
    try:
        depth_vis_max_m = float(cap_cfg.get("depth_vis_max_m", 4.0) or 4.0)
    except Exception:
        depth_vis_max_m = 4.0
    depth_vis_max_m = float(max(0.05, float(depth_vis_max_m)))
    depth_vis_invert = bool(cap_cfg.get("depth_vis_invert", True))
    depth_vis_f = np.empty((H, W), dtype=np.float32) if depth_vis_enabled else None
    depth_vis_u8 = np.empty((H, W), dtype=np.uint8) if depth_vis_enabled else None
    depth_vis_bgr = np.empty((H, W, 3), dtype=np.uint8) if depth_vis_enabled else None
    depth_vis_mask0 = np.empty((H, W), dtype=np.bool_) if depth_vis_enabled else None

    imu = ImuSynchronizer(max_gyro=20000, max_accel=6000)
    last_img_ts: Optional[float] = None
    ts_base_raw: Optional[float] = None

    # IMU time offset (align IMU to camera time); can be tuned by calibration.
    try:
        imu_time_offset_s = float(os.environ.get("RS_IMU_TIME_OFFSET_S", "0.013") or 0.013)
    except Exception:
        imu_time_offset_s = 0.013

    # Optional shared-memory publish of frames + IMU (3d_track compatible).
    shm_ring = None
    shm_imu_ring = None
    shm_init_payload = None
    headless_init_sent = False
    try:
        shm_enabled = bool(cap_cfg.get("shm_enabled", True))
        if "ORB_TRACK3D_SHM_RING_ENABLED" in os.environ:
            enabled_env = str(os.environ.get("ORB_TRACK3D_SHM_RING_ENABLED", "1") or "1").strip().lower()
            shm_enabled = enabled_env not in ("0", "false", "no", "off")
        prefix = sanitize_shm_prefix(os.environ.get("ORB_TRACK3D_SHM_PREFIX", None))
        if prefix is None:
            prefix = sanitize_shm_prefix(cap_cfg.get("shm_name_prefix", None))
        if shm_enabled and prefix:
            force_unlink = bool(cap_cfg.get("shm_force_unlink", True))
            if "ORB_TRACK3D_SHM_FORCE_UNLINK_ON_START" in os.environ:
                force_unlink_env = str(os.environ.get("ORB_TRACK3D_SHM_FORCE_UNLINK_ON_START", "1") or "1").strip().lower()
                force_unlink = force_unlink_env not in ("0", "false", "no", "off")
            slots_v = os.environ.get("ORB_TRACK3D_SHM_SLOTS", None)
            if slots_v is None:
                slots_v = cap_cfg.get("shm_slots", "auto")
            shm_slots = int(max(3, parse_shm_slots(slots_v, fps=int(args.fps), min_slots=3)))
            shm_ring = ShmRingWriter.create(
                h=int(H),
                w=int(W),
                slots=int(shm_slots),
                name_prefix=str(prefix),
                force_unlink=bool(force_unlink),
            )

            imu_enabled = bool(cap_cfg.get("enable_imu", True))
            if "ORB_TRACK3D_SHM_IMU_ENABLED" in os.environ:
                imu_enabled_env = str(os.environ.get("ORB_TRACK3D_SHM_IMU_ENABLED", "1") or "1").strip().lower()
                imu_enabled = imu_enabled_env not in ("0", "false", "no", "off")
            if imu_enabled:
                imu_prefix = sanitize_shm_prefix(os.environ.get("ORB_TRACK3D_SHM_IMU_PREFIX", None))
                if imu_prefix is None:
                    imu_prefix = sanitize_shm_prefix(cap_cfg.get("shm_imu_name_prefix", None))
                if not imu_prefix:
                    imu_prefix = f"{prefix}_imu"
                imu_slots_v = os.environ.get("ORB_TRACK3D_SHM_IMU_SLOTS", None)
                if imu_slots_v is None:
                    imu_slots_v = cap_cfg.get("imu_slots", 8192)
                imu_slots = int(max(128, int(imu_slots_v)))
                shm_imu_ring = ShmImuRingWriter.create(
                    slots=int(imu_slots),
                    name_prefix=str(imu_prefix),
                    force_unlink=bool(force_unlink),
                )

            shm_init_payload = {
                "ring_spec": dict(shm_ring.spec.to_dict()) if shm_ring is not None else None,
                "imu_ring_spec": dict(shm_imu_ring.spec.to_dict()) if shm_imu_ring is not None else None,
                "depth_units": float(depth_scale),
                "out_wh": [int(W), int(H)],
                "intr": {
                    "w": int(W),
                    "h": int(H),
                    "fx": float(intr.fx),
                    "fy": float(intr.fy),
                    "cx": float(intr.ppx),
                    "cy": float(intr.ppy),
                },
            }
            try:
                write_json_atomic(PROJECT_ROOT / ".tmp" / "orb_track3d_init.json", dict(shm_init_payload))
            except Exception:
                pass
            if bool(headless_io_enabled):
                headless_io.send_event(event="init", payload=dict(shm_init_payload))
                headless_init_sent = True
    except Exception:
        shm_ring = None
        shm_imu_ring = None
        shm_init_payload = None

    # Always emit a headless init event (even if SHM is disabled) so external controllers can learn intrinsics.
    if bool(headless_io_enabled) and (not bool(headless_init_sent)):
        try:
            payload = (
                dict(shm_init_payload)
                if isinstance(shm_init_payload, dict)
                else {
                    "ring_spec": dict(shm_ring.spec.to_dict()) if shm_ring is not None else None,
                    "imu_ring_spec": dict(shm_imu_ring.spec.to_dict()) if shm_imu_ring is not None else None,
                    "depth_units": float(depth_scale),
                    "out_wh": [int(W), int(H)],
                    "intr": {
                        "w": int(W),
                        "h": int(H),
                        "fx": float(intr.fx),
                        "fy": float(intr.fy),
                        "cx": float(intr.ppx),
                        "cy": float(intr.ppy),
                    },
                }
            )
            headless_io.send_event(event="init", payload=dict(payload))
            headless_init_sent = True
        except Exception:
            pass

    print(
        f"[min] serial={serial} usb={usb} video={args.video} extra_rgb={int(bool(extra_color_stream))} {W}x{H}@{args.fps} "
        f"imu={args.imu_fps}Hz preset={preset_msg} emitter={emitter_msg} "
        f"depth_scale={depth_scale:.6f} undistort={undistort_msg} "
        f"imu_time_offset_s={imu_time_offset_s:+.6f}",
        flush=True,
    )
    if imu_T_b_c1 is not None:
        t = np.asarray(imu_T_b_c1[:3, 3], dtype=np.float64).reshape(3)
        print(f"[min] IMU.T_b_c1 t=[{t[0]:+.4f},{t[1]:+.4f},{t[2]:+.4f}]", flush=True)

    win = "ORB-SLAM3 minimal (q=quit, r=reset)"
    if not bool(args.no_vis):
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    t_start = time.monotonic()
    deadline = float(t_start + float(args.auto_exit_s)) if float(args.auto_exit_s) > 0 else None

    # Optional PX4 external-vision publisher (VISION_POSITION_ESTIMATE / VISION_SPEED_ESTIMATE).
    px4_state = Px4VisionState()
    px4_pub = Px4VisionPublisher(
        cfg_path=(track3d_cfg_path or (PROJECT_ROOT / "apps" / "px4_mavlink.yaml")),
        state=px4_state,
        print_fn=print,
    )
    px4_pub.start()
    prev_px4_ts: Optional[float] = None
    prev_px4_p: Optional[np.ndarray] = None

    # Optional inter-process VO communicator (shared-memory state).
    ipc_state = IpcVoState()
    ipc_vo = InterProcessVoCommunicator(state=ipc_state, cfg_path=track3d_cfg_path, print_fn=print)
    ipc_vo.start()

    # Optional polygon tracking subprocess (proven `poly_vo_lk_track` pipeline).
    try:
        mp.set_start_method("spawn", force=True)
    except Exception:
        pass
    ctx = mp.get_context("spawn")

    poly_proc = None
    poly_cmd_q = None
    poly_out_q = None
    poly_stop_evt = ctx.Event()

    poly_active = False
    poly_proc_last_fi = -1
    poly_proc_reason = ""
    poly_proc_sel_kind = "none"
    poly_pid_confirmed = False
    hole_center_uv = None  # type: Optional[tuple[float, float]]
    hole_r_px = float("nan")
    hole_r_m = float("nan")
    hole_fill = float("nan")
    hole_ok_pid = False
    hole_plane_inliers = 0
    hole_plane_rms_m = float("nan")
    hole_plane_cov = float("nan")
    hole_plane_center_range_m = float("nan")
    hole_err = ""
    plane_center_uv = None  # type: Optional[tuple[float, float]]
    plane_r_px = float("nan")
    plane_r_m = float("nan")
    plane_ok_pid = False
    plane_plane_inliers = 0
    plane_plane_rms_m = float("nan")
    plane_plane_cov = float("nan")
    plane_plane_center_range_m = float("nan")
    plane_err = ""
    poly_uv = None  # type: Optional[np.ndarray]
    poly_bbox = None  # type: Optional[tuple[int, int, int, int]]
    # LK lock points used by the polygon tracker (published under legacy "orb_*" field names).
    poly_lock_locked = False
    poly_lock_used_uv = None  # type: Optional[np.ndarray]
    poly_lock_reason = ""

    hio_last_preview_fi = -1
    hio_last_pid_fi = -1

    if bool(poly_use_process) and shm_ring is not None:
        state_spec_dict = ipc_vo.wait_spec(timeout_s=2.0)
        if not (isinstance(state_spec_dict, dict) and state_spec_dict):
            poly_use_process = False
            print("[track3d] poly_process disabled (ipc_vo spec missing)", flush=True)
        else:
            try:
                cmd_q_max = int(max(1, int(poly_proc_cfg.get("cmd_q_max", 8) or 8)))
            except Exception:
                cmd_q_max = 8
            try:
                out_q_max = int(max(1, int(poly_proc_cfg.get("out_q_max", 2) or 2)))
            except Exception:
                out_q_max = 2
            poly_cmd_q = ctx.Queue(maxsize=int(cmd_q_max))
            poly_out_q = ctx.Queue(maxsize=int(out_q_max))
            poly_proc = ctx.Process(
                target=_poly_vo_lk_track_process_main,
                kwargs={
                    "stop_event": poly_stop_evt,
                    "ring_spec_dict": dict(shm_ring.spec.to_dict()),
                    "state_spec_dict": dict(state_spec_dict),
                    "cfg": dict(track3d_cfg),
                    "intr": dict(intr_dict),
                    "depth_units": float(depth_scale),
                    "cmd_q": poly_cmd_q,
                    "out_q": poly_out_q,
                },
                name="Poly",
                daemon=True,
            )
            poly_proc.start()
            try:
                print("[track3d] poly_process enabled=1 impl=vo_lk", flush=True)
            except Exception:
                pass
    fi = -1
    last_telemetry_mono: float = 0.0
    last_features_mono: float = 0.0
    prev_frame_ts_raw_s: Optional[float] = None

    # Pose summary state (headless telemetry parity).
    start_pos_w: Optional[np.ndarray] = None
    prev_pos_w: Optional[np.ndarray] = None
    path_len_m: float = 0.0
    last_track_ms: float = 0.0
    last_kp_n: int = 0
    last_gyro_xyz: Optional[tuple[float, float, float]] = None
    last_accel_xyz: Optional[tuple[float, float, float]] = None
    last_imu_synth_log_mono: float = 0.0
    imu_synth_count: int = 0

    # Handle SIGTERM/SIGINT in the tracker process so RealSense + ORB-SLAM3 clean up properly.
    shutdown_sig = threading.Event()

    def _sig_handler(sig, frame):
        try:
            shutdown_sig.set()
        except Exception:
            pass

    try:
        signal.signal(signal.SIGINT, _sig_handler)
        signal.signal(signal.SIGTERM, _sig_handler)
    except Exception:
        # Some environments (non-main threads, restricted runtimes) may not allow signal handlers.
        pass

    try:
        while True:
            if shutdown_sig.is_set():
                break
            if deadline is not None and time.monotonic() >= deadline:
                break

            # Keep this timeout modest so SIGTERM / stdin shutdown responds quickly even if frames stop arriving.
            try:
                f = q.wait_for_frame(500)
            except Exception:
                continue
            if f is None:
                continue

            if f.is_motion_frame():
                mf = f.as_motion_frame()
                st = mf.get_profile().stream_type()
                data = mf.get_motion_data()
                try:
                    if st == rs.stream.gyro:
                        last_gyro_xyz = (float(data.x), float(data.y), float(data.z))
                    elif st == rs.stream.accel:
                        last_accel_xyz = (float(data.x), float(data.y), float(data.z))
                except Exception:
                    pass
                t_s_raw_no_off = float(mf.get_timestamp()) * 1e-3
                if shm_imu_ring is not None:
                    try:
                        if st == rs.stream.gyro:
                            shm_imu_ring.write(
                                timestamp=float(t_s_raw_no_off),
                                kind=int(IMU_KIND_GYRO),
                                gyro_xyz_rad_s=(float(data.x), float(data.y), float(data.z)),
                            )
                        elif st == rs.stream.accel:
                            shm_imu_ring.write(
                                timestamp=float(t_s_raw_no_off),
                                kind=int(IMU_KIND_ACCEL),
                                accel_xyz_m_s2=(float(data.x), float(data.y), float(data.z)),
                            )
                    except Exception:
                        pass

                t_s_raw = float(t_s_raw_no_off) + float(imu_time_offset_s)
                # The time base is anchored on the first image frame; until then keep raw times.
                if ts_base_raw is not None:
                    t_s = float(t_s_raw - float(ts_base_raw))
                else:
                    t_s = float(t_s_raw)
                if st == rs.stream.gyro:
                    imu.add_gyro(t_s=t_s, gx=float(data.x), gy=float(data.y), gz=float(data.z))
                elif st == rs.stream.accel:
                    imu.add_accel(t_s=t_s, ax=float(data.x), ay=float(data.y), az=float(data.z))
                continue

            if not f.is_frameset():
                continue

            fs = f.as_frameset()
            color_frame = None
            if bool(extra_color_stream) and (not bool(use_color)):
                try:
                    color_frame = fs.get_color_frame()
                except Exception:
                    color_frame = None
            aligned = align.process(fs)
            depth_frame = aligned.get_depth_frame()
            if not depth_frame:
                continue

            if use_color:
                vid_frame = aligned.get_color_frame()
            else:
                try:
                    vid_frame = aligned.get_infrared_frame(1)
                except TypeError:
                    vid_frame = aligned.get_infrared_frame()
            if not vid_frame:
                continue

            # Image timestamp base for ORB (keep values near 0).
            t_cam_raw_s = float(vid_frame.get_timestamp()) * 1e-3
            if ts_base_raw is None:
                ts_base_raw = float(t_cam_raw_s)
                last_img_ts = None  # restart IMU batching
            ts = float(t_cam_raw_s - float(ts_base_raw))
            # ORB-SLAM3 requires strictly increasing frame timestamps.
            # If the RealSense timestamp domain jitters/steps backwards (global time / NTP / driver quirks),
            # clamp so we never feed a non-monotonic timestamp into TrackRGBD().
            try:
                if last_img_ts is not None and float(ts) <= float(last_img_ts):
                    ts = float(last_img_ts) + 1e-6
            except Exception:
                pass

            # Convert depth to meters float32.
            depth_raw = np.asanyarray(depth_frame.get_data())
            np.multiply(depth_raw, depth_scale, out=depth_m_buf, casting="unsafe")

            # Build ORB input buffers (ring slot).
            slot = int(ring_i)
            ring_i = int((ring_i + 1) % ring_n)
            depth_in = depth_ring[slot]
            img_in = img_ring[slot]
            gray_in = gray_ring[slot]
            depth_raw_in = depth_raw_ring[slot]

            # Video frame -> BGR uint8 in `img_in`
            if use_color:
                img_u8 = np.asanyarray(vid_frame.get_data())
                if img_u8.ndim != 3:
                    continue
                if undistort_enabled and map1 is not None and map2 is not None:
                    cv2.remap(img_u8, map1, map2, interpolation=cv2.INTER_LINEAR, dst=img_in, borderMode=cv2.BORDER_CONSTANT)
                    cv2.remap(depth_m_buf, map1, map2, interpolation=cv2.INTER_NEAREST, dst=depth_in, borderMode=cv2.BORDER_CONSTANT)
                    cv2.remap(depth_raw, map1, map2, interpolation=cv2.INTER_NEAREST, dst=depth_raw_in, borderMode=cv2.BORDER_CONSTANT)
                else:
                    np.copyto(img_in, img_u8)
                    np.copyto(depth_in, depth_m_buf)
                    np.copyto(depth_raw_in, depth_raw)
                try:
                    cv2.cvtColor(img_in, cv2.COLOR_BGR2GRAY, dst=gray_in)
                except Exception:
                    try:
                        gray_in[:] = img_in[:, :, 1]
                    except Exception:
                        pass
            else:
                ir_u8 = np.asanyarray(vid_frame.get_data())
                if ir_u8.ndim != 2:
                    continue
                if undistort_enabled and map1 is not None and map2 is not None:
                    ir_dst = ir_ring[slot]
                    cv2.remap(ir_u8, map1, map2, interpolation=cv2.INTER_LINEAR, dst=ir_dst, borderMode=cv2.BORDER_CONSTANT)
                    img_in[:, :, 0] = ir_dst
                    img_in[:, :, 1] = ir_dst
                    img_in[:, :, 2] = ir_dst
                    cv2.remap(depth_m_buf, map1, map2, interpolation=cv2.INTER_NEAREST, dst=depth_in, borderMode=cv2.BORDER_CONSTANT)
                    cv2.remap(depth_raw, map1, map2, interpolation=cv2.INTER_NEAREST, dst=depth_raw_in, borderMode=cv2.BORDER_CONSTANT)
                    np.copyto(gray_in, ir_dst)
                else:
                    img_in[:, :, 0] = ir_u8
                    img_in[:, :, 1] = ir_u8
                    img_in[:, :, 2] = ir_u8
                    np.copyto(depth_in, depth_m_buf)
                    np.copyto(depth_raw_in, depth_raw)
                    np.copyto(gray_in, ir_u8)

            fi += 1
            depth_bgr_out = None
            bgr_pub = img_in
            if pub_bgr_ring is not None and color_frame is not None:
                try:
                    color_u8 = np.asanyarray(color_frame.get_data())
                    if color_u8.ndim == 3 and int(color_u8.shape[0]) == int(H) and int(color_u8.shape[1]) == int(W):
                        pub_in = pub_bgr_ring[slot]
                        np.copyto(pub_in, color_u8)
                        bgr_pub = pub_in
                except Exception:
                    bgr_pub = img_in
            if (
                depth_vis_enabled
                and depth_vis_f is not None
                and depth_vis_u8 is not None
                and depth_vis_bgr is not None
                and depth_vis_mask0 is not None
            ):
                try:
                    # Map depth meters -> u8 (0..255), optionally inverted so near objects are bright.
                    np.clip(depth_in, 0.0, float(depth_vis_max_m), out=depth_vis_f)
                    np.multiply(depth_vis_f, float(255.0 / float(depth_vis_max_m)), out=depth_vis_f)
                    np.copyto(depth_vis_u8, depth_vis_f, casting="unsafe")
                    if bool(depth_vis_invert):
                        np.bitwise_xor(depth_vis_u8, np.uint8(255), out=depth_vis_u8)
                    # Keep invalid depth (0) as black.
                    np.equal(depth_raw_in, 0, out=depth_vis_mask0)
                    depth_vis_u8[depth_vis_mask0] = 0
                    # Replicate to BGR.
                    depth_vis_bgr[:, :, 0] = depth_vis_u8
                    depth_vis_bgr[:, :, 1] = depth_vis_u8
                    depth_vis_bgr[:, :, 2] = depth_vis_u8
                    depth_bgr_out = depth_vis_bgr
                except Exception:
                    depth_bgr_out = None
            if shm_ring is not None and shm_init_payload is not None:
                try:
                    shm_ring.write(
                        frame_idx=int(fi),
                        timestamp=float(t_cam_raw_s),
                        bgr=bgr_pub,
                        gray=gray_in,
                        depth_raw=depth_raw_in,
                        depth_bgr=depth_bgr_out,
                        fill_bgr_from_gray=False,
                    )
                except Exception:
                    pass

            # IMU batching between frames.
            if last_img_ts is None:
                last_img_ts = float(ts)
                continue

            imu_batch = imu.pop_batch(t0_s=float(last_img_ts), t1_s=float(ts), min_samples=2)
            if imu_batch.size == 0:
                # If IMU timestamps are misaligned/lagging, don't stall the whole tracker.
                # Synthesize a tiny batch from the latest accel/gyro so ORB-SLAM3 can advance and
                # the headless protocol keeps emitting telemetry/features.
                imu_synth_count = int(imu_synth_count) + 1
                try:
                    ax, ay, az = last_accel_xyz if last_accel_xyz is not None else (0.0, 0.0, 0.0)
                    gx, gy, gz = last_gyro_xyz if last_gyro_xyz is not None else (0.0, 0.0, 0.0)
                    t0 = float(ts) - 0.001
                    try:
                        t0 = max(float(t0), float(last_img_ts) + 1e-4)
                    except Exception:
                        pass
                    t1 = float(ts) + 0.001
                    if float(t1) <= float(t0):
                        t1 = float(t0) + 1e-4
                    imu_batch = np.asarray(
                        [
                            [float(ax), float(ay), float(az), float(gx), float(gy), float(gz), float(t0)],
                            [float(ax), float(ay), float(az), float(gx), float(gy), float(gz), float(t1)],
                        ],
                        dtype=np.float64,
                    )
                except Exception:
                    imu_batch = np.empty((0, 7), dtype=np.float64)
                if imu_batch.size == 0:
                    continue
                try:
                    now_m = float(time.monotonic())
                    if (now_m - float(last_imu_synth_log_mono)) >= 1.0:
                        last_imu_synth_log_mono = float(now_m)
                        print(f"[min] WARN: IMU batch empty; using synthesized IMU (count={int(imu_synth_count)})", flush=True)
                except Exception:
                    pass
            # Ensure we have a post-frame sample (ORB likes at least one t>=ts).
            try:
                if float(imu_batch[-1, 6]) <= float(ts):
                    row = np.asarray(imu_batch[-1, :], dtype=np.float64).copy()
                    row[6] = float(ts) + 0.001
                    imu_batch = np.vstack((imu_batch, row.reshape(1, 7)))
            except Exception:
                pass

            # Final safeguard: enforce strictly increasing timestamps in the IMU batch.
            # The pybind wrapper validates this strictly and raises, which would otherwise kill the tracker.
            try:
                if imu_batch.size > 0:
                    tcol = np.asarray(imu_batch[:, 6], dtype=np.float64).reshape(-1)
                    for k in range(1, int(tcol.shape[0])):
                        if float(tcol[k]) <= float(tcol[k - 1]):
                            tcol[k] = float(tcol[k - 1]) + 1e-6
                    imu_batch = np.asarray(imu_batch, dtype=np.float64, order="C")
                    imu_batch[:, 6] = tcol
            except Exception:
                pass

            last_img_ts = float(ts)

            # ORB tracking.
            t_tr0 = time.perf_counter()
            Tcw = slam.TrackRGBD(img_in, depth_in, float(ts), imu_batch)
            t_tr1 = time.perf_counter()
            try:
                last_track_ms = float(max(0.0, (float(t_tr1) - float(t_tr0)) * 1000.0))
            except Exception:
                last_track_ms = 0.0
            Tcw = np.asarray(Tcw, dtype=np.float64).reshape(4, 4)
            Twc_map = None
            if np.all(np.isfinite(Tcw)):
                try:
                    Twc_map = np.linalg.inv(Tcw)
                except Exception:
                    Twc_map = None

            ok_map = bool(Twc_map is not None and np.all(np.isfinite(np.asarray(Twc_map, dtype=np.float64))))
            tracking_ok = bool(ok_map)
            tracking_stable = bool(ok_map)
            trk_state = None
            try:
                if callable(getattr(slam, "GetTrackingState", None)):
                    trk_state = int(slam.GetTrackingState())
                    # ORB-SLAM3 tracking states:
                    #   2=OK, 3=RECENTLY_LOST, 4=LOST, 5=OK_KLT
                    # Treat RECENTLY_LOST as "ok" for continuity, but not "stable".
                    tracking_ok = int(trk_state) in (2, 3, 5)
                    tracking_stable = int(trk_state) in (2, 5)
                elif callable(getattr(slam, "isLost", None)):
                    tracking_ok = not bool(slam.isLost())
                    tracking_stable = bool(tracking_ok)
            except Exception:
                tracking_ok = bool(ok_map)
                tracking_stable = bool(ok_map)
                trk_state = None

            try:
                trk_state_code = int(trk_state) if trk_state is not None else int(0 if bool(tracking_ok) else 1)
            except Exception:
                trk_state_code = int(0 if bool(tracking_ok) else 1)

            # Maintain a continuous local odom frame even when ORB-SLAM3 applies map corrections (BA / loop closure).
            try:
                frame_resolver.observe_slam(slam=slam)
            except Exception:
                pass
            pose_frames = frame_resolver.resolve(Twc_map=Twc_map, tracking_ok=bool(tracking_ok), map_id=None)
            Twc = pose_frames.Twc_odom
            Twc_map = pose_frames.Twc_map

            if pose_frames.ok and Twc is not None:
                try:
                    p = np.asarray(Twc[:3, 3], dtype=np.float64).reshape(3)
                    if np.all(np.isfinite(p)):
                        roll, pitch, yaw = _rpy_deg_from_R(Twc[:3, :3])
                        pose = {
                            "position_m": {"x": float(p[0]), "y": float(p[1]), "z": float(p[2])},
                            "orientation_deg": {"roll": float(roll), "pitch": float(pitch), "yaw": float(yaw)},
                        }
                        vel = None
                        if prev_px4_ts is not None and prev_px4_p is not None:
                            dt = float(t_cam_raw_s) - float(prev_px4_ts)
                            if math.isfinite(dt) and 1e-3 < dt < 0.5:
                                v = (p - prev_px4_p) / dt
                                if np.all(np.isfinite(v)):
                                    vel = {"x": float(v[0]), "y": float(v[1]), "z": float(v[2])}
                        px4_state.update(ts=float(t_cam_raw_s), pose=pose, vel=vel)
                        prev_px4_ts = float(t_cam_raw_s)
                        prev_px4_p = p.copy()
                except Exception:
                    pass
            stable_ipc = False
            try:
                ok_ipc = bool(pose_frames.ok and Twc is not None and np.all(np.isfinite(np.asarray(Twc, dtype=np.float64))))
                stable_ipc = bool(bool(tracking_stable) and bool(ok_ipc))
                ipc_state.update(frame_idx=int(fi), timestamp=float(t_cam_raw_s), Twc=Twc if ok_ipc else None, ok=ok_ipc)
            except Exception:
                ok_ipc = False
                stable_ipc = False

            # Headless JSON protocol parity (stdin commands + stdout events).
            if bool(headless_io_enabled):
                shutdown_req = False

                # Drain latest polygon process state (stage1 refine / stage2 PID lock).
                if bool(poly_use_process) and poly_out_q is not None:
                    try:
                        while True:
                            msg = poly_out_q.get_nowait()
                            if not isinstance(msg, dict):
                                continue
                            poly_active = bool(msg.get("active", False))
                            try:
                                poly_proc_last_fi = int(msg.get("fi", poly_proc_last_fi))
                            except Exception:
                                pass
                            try:
                                poly_proc_reason = str(msg.get("reason", poly_proc_reason))
                            except Exception:
                                pass
                            try:
                                poly_proc_sel_kind = str(msg.get("sel_kind", poly_proc_sel_kind) or "none")
                            except Exception:
                                pass
                            try:
                                poly_pid_confirmed = bool(msg.get("pid_confirmed", poly_pid_confirmed))
                            except Exception:
                                pass
                            try:
                                hc = msg.get("hole_center_uv", None)
                                if hc is None:
                                    hole_center_uv = None
                                else:
                                    hc = tuple(hc) if isinstance(hc, (list, tuple)) else None
                                    if hc is not None and int(len(hc)) >= 2:
                                        hole_center_uv = (float(hc[0]), float(hc[1]))
                            except Exception:
                                hole_center_uv = None
                            try:
                                hole_r_px = float(msg.get("hole_r_px", hole_r_px))
                                hole_r_m = float(msg.get("hole_r_m", hole_r_m))
                                hole_fill = float(msg.get("hole_fill", hole_fill))
                                hole_ok_pid = bool(msg.get("hole_ok_pid", hole_ok_pid))
                                hole_plane_inliers = int(msg.get("hole_plane_inliers", hole_plane_inliers))
                                hole_plane_rms_m = float(msg.get("hole_plane_rms_m", hole_plane_rms_m))
                                hole_plane_cov = float(msg.get("hole_plane_cov", hole_plane_cov))
                                hole_plane_center_range_m = float(msg.get("hole_plane_center_range_m", hole_plane_center_range_m))
                                hole_err = str(msg.get("hole_err", hole_err) or "")
                            except Exception:
                                pass
                            try:
                                pc = msg.get("plane_center_uv", None)
                                if pc is None:
                                    plane_center_uv = None
                                else:
                                    pc = tuple(pc) if isinstance(pc, (list, tuple)) else None
                                    if pc is not None and int(len(pc)) >= 2:
                                        plane_center_uv = (float(pc[0]), float(pc[1]))
                            except Exception:
                                plane_center_uv = None
                            try:
                                plane_r_px = float(msg.get("plane_r_px", plane_r_px))
                                plane_r_m = float(msg.get("plane_r_m", plane_r_m))
                                plane_ok_pid = bool(msg.get("plane_ok_pid", plane_ok_pid))
                                plane_plane_inliers = int(msg.get("plane_plane_inliers", plane_plane_inliers))
                                plane_plane_rms_m = float(msg.get("plane_plane_rms_m", plane_plane_rms_m))
                                plane_plane_cov = float(msg.get("plane_plane_cov", plane_plane_cov))
                                plane_plane_center_range_m = float(
                                    msg.get("plane_plane_center_range_m", plane_plane_center_range_m)
                                )
                                plane_err = str(msg.get("plane_err", plane_err) or "")
                            except Exception:
                                pass
                            uv = msg.get("poly_uv", None)
                            if uv is None:
                                poly_uv = None
                            else:
                                try:
                                    poly_uv = np.asarray(uv, dtype=np.float32).reshape(-1, 2)
                                except Exception:
                                    poly_uv = None
                            bb = msg.get("bbox", None)
                            poly_bbox = (
                                tuple(int(v) for v in bb) if isinstance(bb, (list, tuple)) and int(len(bb)) == 4 else poly_bbox
                            )

                            # LK lock points used to hold the polygon in place (legacy "orb_*" fields).
                            try:
                                poly_lock_locked = bool(msg.get("orb_locked", poly_lock_locked))
                            except Exception:
                                pass
                            try:
                                ou = msg.get("orb_used_uv", None)
                                if ou is None:
                                    poly_lock_used_uv = None
                                else:
                                    poly_lock_used_uv = np.asarray(ou, dtype=np.float32).reshape(-1, 2)
                            except Exception:
                                poly_lock_used_uv = None
                            try:
                                poly_lock_reason = str(msg.get("orb_reason", poly_lock_reason) or "")
                            except Exception:
                                pass
                            if not bool(poly_active):
                                poly_lock_used_uv = None

                            # Stage-0 preview: emit preview polygons coming from the polygon process.
                            prev = msg.get("preview", None)
                            if isinstance(prev, dict):
                                try:
                                    pfi = int(prev.get("fi", -1))
                                except Exception:
                                    pfi = -1
                                if int(pfi) >= 0 and int(pfi) != int(hio_last_preview_fi):
                                    hio_last_preview_fi = int(pfi)
                                    try:
                                        pv = prev.get("poly_uv", None)
                                        verts = np.asarray(pv, dtype=np.float32).reshape(-1, 2) if pv is not None else None
                                    except Exception:
                                        verts = None
                                    verts_uv = []
                                    if verts is not None and int(verts.shape[0]) >= 3:
                                        try:
                                            verts_uv = [
                                                (int(round(float(u))), int(round(float(v))))
                                                for u, v in np.asarray(verts, dtype=np.float32).reshape(-1, 2).tolist()
                                            ]
                                        except Exception:
                                            verts_uv = []
                                    c_u = c_v = None
                                    try:
                                        cuv = prev.get("center_uv", None)
                                        if cuv is not None and isinstance(cuv, (list, tuple)) and int(len(cuv)) >= 2:
                                            c_u = float(cuv[0])
                                            c_v = float(cuv[1])
                                    except Exception:
                                        c_u = c_v = None
                                    if c_u is None or c_v is None:
                                        try:
                                            if verts is not None:
                                                c_u = float(np.mean(np.asarray(verts)[:, 0]))
                                                c_v = float(np.mean(np.asarray(verts)[:, 1]))
                                        except Exception:
                                            c_u = float(intr_dict.get("cx", 0.0))
                                            c_v = float(intr_dict.get("cy", 0.0))

                                    try:
                                        ok_pid = int(bool(prev.get("ok_pid", False)))
                                    except Exception:
                                        ok_pid = 0

                                    try:
                                        headless_io.send_event(
                                            event="preview",
                                            payload={
                                                "fi": int(pfi),
                                                "sel_kind": (str(prev.get("sel_kind", "")) if prev.get("sel_kind", None) is not None else None),
                                                "verts_uv": verts_uv,
                                                "center_uv": (float(c_u), float(c_v)),
                                                "ok_pid": int(ok_pid),
                                                "hole_stats": (
                                                    dict(prev.get("hole_stats", {}))
                                                    if isinstance(prev.get("hole_stats", None), dict)
                                                    else None
                                                ),
                                                "plane_stats": (
                                                    dict(prev.get("plane_stats", {}))
                                                    if isinstance(prev.get("plane_stats", None), dict)
                                                    else None
                                                ),
                                            },
                                        )
                                    except Exception:
                                        pass
                    except Exception:
                        pass

                # Apply incoming JSON commands.
                try:
                    for in_msg in headless_io.poll_in():
                        cmd = str(in_msg.get("cmd", "") or "").strip().lower()
                        if not cmd:
                            continue
                        if cmd in ("shutdown", "quit", "exit", "stop"):
                            shutdown_req = True
                            headless_io.send_event(event="debug", payload={"msg": "shutdown_requested", "cmd": str(cmd)})
                            break
                        if cmd in ("hole_enable", "hole_detector_enable"):
                            try:
                                en = in_msg.get("enable", True)
                                try:
                                    en_i = int(en)
                                    en_b = bool(en_i != 0)
                                except Exception:
                                    en_b = bool(en)
                            except Exception:
                                en_b = True

                            if bool(poly_use_process) and poly_cmd_q is not None:
                                try:
                                    poly_cmd_q.put_nowait({"cmd": "hole_enable", "enable": int(bool(en_b))})
                                except Exception:
                                    # Best-effort: drop an old command and retry.
                                    try:
                                        _ = poly_cmd_q.get_nowait()
                                    except Exception:
                                        pass
                                    try:
                                        poly_cmd_q.put_nowait({"cmd": "hole_enable", "enable": int(bool(en_b))})
                                    except Exception:
                                        pass
                            continue
                        if cmd in ("plane_enable", "plane_detector_enable"):
                            try:
                                en = in_msg.get("enable", True)
                                try:
                                    en_i = int(en)
                                    en_b = bool(en_i != 0)
                                except Exception:
                                    en_b = bool(en)
                            except Exception:
                                en_b = True

                            if bool(poly_use_process) and poly_cmd_q is not None:
                                try:
                                    poly_cmd_q.put_nowait({"cmd": "plane_enable", "enable": int(bool(en_b))})
                                except Exception:
                                    # Best-effort: drop an old command and retry.
                                    try:
                                        _ = poly_cmd_q.get_nowait()
                                    except Exception:
                                        pass
                                    try:
                                        poly_cmd_q.put_nowait({"cmd": "plane_enable", "enable": int(bool(en_b))})
                                    except Exception:
                                        pass
                            continue
                        if cmd in ("hover", "mouse_move", "pointer"):
                            try:
                                hx = int(in_msg.get("x", -1))
                                hy = int(in_msg.get("y", -1))
                            except Exception:
                                hx = -1
                                hy = -1

                            # Forward hover to the polygon process (owns the expensive depth-based hover preview).
                            if bool(poly_use_process) and poly_cmd_q is not None and (not bool(poly_active)):
                                try:
                                    poly_cmd_q.put_nowait({"cmd": "hover", "x": int(hx), "y": int(hy)})
                                except Exception:
                                    # Keep only newest hover update.
                                    try:
                                        _ = poly_cmd_q.get_nowait()
                                    except Exception:
                                        pass
                                    try:
                                        poly_cmd_q.put_nowait({"cmd": "hover", "x": int(hx), "y": int(hy)})
                                    except Exception:
                                        pass
                            continue

                        # Forward selection/control commands to the polygon process.
                        if cmd in ("clear", "select_hole", "confirm_hole", "select_plane", "confirm_plane", "select_bbox"):
                            if (not bool(poly_use_process)) or poly_cmd_q is None:
                                headless_io.send_event(event="cmd_err", payload={"cmd": str(cmd), "err": "poly_process_disabled"})
                                continue
                            out_cmd: dict[str, Any] = {"cmd": str(cmd)}
                            if str(cmd) != "clear":
                                try:
                                    out_cmd["fi"] = int(in_msg.get("fi", fi))
                                except Exception:
                                    out_cmd["fi"] = int(fi)
                                try:
                                    out_cmd["ts"] = float(in_msg.get("ts", t_cam_raw_s))
                                except Exception:
                                    out_cmd["ts"] = float(t_cam_raw_s)
                            if str(cmd) == "select_bbox":
                                bb = in_msg.get("bbox", None)
                                if isinstance(bb, (list, tuple)) and int(len(bb)) == 4:
                                    out_cmd["bbox"] = tuple(int(x) for x in bb)
                                else:
                                    headless_io.send_event(event="cmd_err", payload={"cmd": str(cmd), "err": "bbox_invalid"})
                                    continue
                            else:
                                try:
                                    out_cmd["x"] = int(in_msg.get("x", 0))
                                    out_cmd["y"] = int(in_msg.get("y", 0))
                                except Exception:
                                    out_cmd["x"] = 0
                                    out_cmd["y"] = 0
                            try:
                                poly_cmd_q.put_nowait(dict(out_cmd))
                            except Exception:
                                headless_io.send_event(event="cmd_err", payload={"cmd": str(cmd), "err": "cmd_q_full"})
                            continue

                        headless_io.send_event(event="cmd_err", payload={"cmd": str(cmd), "err": "unknown_cmd"})
                except Exception:
                    pass

                if bool(shutdown_req):
                    break

                pid_out = None
                try:
                    # - Stage 1 (pre-confirm): use detector/seed center when available.
                    # - Stage 2 (confirmed): use polygon center.
                    poly_center_uv = None
                    if poly_uv is not None and int(np.asarray(poly_uv).shape[0]) >= 3:
                        try:
                            cont = np.asarray(poly_uv, dtype=np.float32).reshape(-1, 1, 2)
                            m = cv2.moments(cont)
                            if abs(float(m.get("m00", 0.0))) > 1e-6:
                                poly_center_uv = (float(m["m10"]) / float(m["m00"]), float(m["m01"]) / float(m["m00"]))
                            else:
                                poly_center_uv = (
                                    float(np.mean(np.asarray(poly_uv, dtype=np.float64)[:, 0])),
                                    float(np.mean(np.asarray(poly_uv, dtype=np.float64)[:, 1])),
                                )
                        except Exception:
                            poly_center_uv = None

                    sk = str(poly_proc_sel_kind or "none")
                    center_uv = None
                    if bool(poly_pid_confirmed) and poly_center_uv is not None:
                        center_uv = (float(poly_center_uv[0]), float(poly_center_uv[1]))
                    elif sk == "hole" and hole_center_uv is not None:
                        center_uv = (float(hole_center_uv[0]), float(hole_center_uv[1]))
                    elif sk == "plane" and plane_center_uv is not None:
                        center_uv = (float(plane_center_uv[0]), float(plane_center_uv[1]))
                    elif poly_center_uv is not None:
                        center_uv = (float(poly_center_uv[0]), float(poly_center_uv[1]))
                    else:
                        center_uv = (float(intr_dict.get("cx", 0.0)), float(intr_dict.get("cy", 0.0)))

                    u0, v0 = float(center_uv[0]), float(center_uv[1])
                    fx = float(intr_dict.get("fx", 0.0))
                    fy = float(intr_dict.get("fy", 0.0))
                    cx = float(intr_dict.get("cx", 0.0))
                    cy = float(intr_dict.get("cy", 0.0))
                    h_deg = float(np.degrees(np.arctan2(float(u0) - float(cx), float(fx)))) if fx > 0.0 else 0.0
                    v_deg = float(np.degrees(np.arctan2(float(v0) - float(cy), float(fy)))) if fy > 0.0 else 0.0

                    pid_out = {
                        "sel_kind": str(sk),
                        "stage": int(2 if bool(poly_pid_confirmed) else 1),
                        "fi": int(fi),
                        "center_uv": (float(u0), float(v0)),
                        "dhv_deg": (float(h_deg), float(v_deg)),
                    }
                    if sk == "hole":
                        pid_out.update(
                            {
                                "r_m": float(hole_r_m),
                                "fill": float(hole_fill),
                                "ok_pid": int(bool(hole_ok_pid)),
                                "plane": {
                                    "inliers": int(hole_plane_inliers),
                                    "rms_m": float(hole_plane_rms_m),
                                    "cov": float(hole_plane_cov),
                                    "range_m": float(hole_plane_center_range_m),
                                },
                                "err": str(hole_err or ""),
                            }
                        )
                    elif sk == "plane":
                        pid_out.update(
                            {
                                "r_m": float(plane_r_m),
                                "ok_pid": int(bool(plane_ok_pid)),
                                "plane": {
                                    "inliers": int(plane_plane_inliers),
                                    "rms_m": float(plane_plane_rms_m),
                                    "cov": float(plane_plane_cov),
                                    "range_m": float(plane_plane_center_range_m),
                                },
                                "err": str(plane_err or ""),
                            }
                        )
                except Exception:
                    pid_out = None

                if pid_out is not None:
                    try:
                        pid_fi = int(poly_proc_last_fi) if int(poly_proc_last_fi) >= 0 else int(fi)
                    except Exception:
                        pid_fi = int(fi)
                    if int(pid_fi) != int(hio_last_pid_fi):
                        hio_last_pid_fi = int(pid_fi)
                        try:
                            headless_io.send_event(event="pid", payload=dict(pid_out))
                        except Exception:
                            pass

                # Telemetry (rate-limited).
                if telemetry_period_s > 0.0:
                    try:
                        now_m = float(time.monotonic())
                        if (now_m - float(last_telemetry_mono)) >= float(telemetry_period_s):
                            last_telemetry_mono = float(now_m)

                            ok = bool(ok_ipc)
                            stable = bool(stable_ipc)
                            have_pose = bool(Twc is not None and np.all(np.isfinite(np.asarray(Twc, dtype=np.float64))))
                            pos = None
                            Rwc_now = None
                            if have_pose and Twc is not None:
                                try:
                                    pos = np.asarray(Twc[:3, 3], dtype=np.float64).reshape(3)
                                    Rwc_now = np.asarray(Twc[:3, :3], dtype=np.float64).reshape(3, 3)
                                    if not np.isfinite(pos).all():
                                        pos = None
                                        Rwc_now = None
                                except Exception:
                                    pos = None
                                    Rwc_now = None

                            if stable and pos is not None and start_pos_w is None:
                                start_pos_w = np.asarray(pos, dtype=np.float64).copy()
                            if stable and pos is not None and prev_pos_w is None:
                                prev_pos_w = np.asarray(pos, dtype=np.float64).copy()

                            t_step = np.zeros(3, dtype=np.float64)
                            if stable and pos is not None and prev_pos_w is not None:
                                try:
                                    t_step = np.asarray(pos - prev_pos_w, dtype=np.float64).reshape(3)
                                    path_len_m = float(path_len_m) + float(np.linalg.norm(t_step))
                                    prev_pos_w = np.asarray(pos, dtype=np.float64).copy()
                                except Exception:
                                    t_step = np.zeros(3, dtype=np.float64)

                            pos0 = np.asarray(start_pos_w, dtype=np.float64).reshape(3) if start_pos_w is not None else None
                            rel = (np.asarray(pos - pos0, dtype=np.float64).reshape(3) if (pos is not None and pos0 is not None) else None)
                            drift_m = float(np.linalg.norm(rel)) if rel is not None else None

                            yaw_deg = roll_deg = pitch_deg = None
                            fwd_w = None
                            if Rwc_now is not None:
                                try:
                                    fwd = Rwc_now @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
                                    yaw_deg = float(np.degrees(np.arctan2(float(fwd[0]), float(fwd[2]))))
                                    fwd_w = [float(fwd[0]), float(fwd[1]), float(fwd[2])]
                                    r02 = float(Rwc_now[0, 2])
                                    r12 = float(Rwc_now[1, 2])
                                    r22 = float(Rwc_now[2, 2])
                                    pitch_rad = float(np.arctan2(-float(r12), float(np.sqrt(float(r02 * r02 + r22 * r22)))))
                                    pitch_deg = float(np.degrees(float(pitch_rad)))
                                    roll_rad = float(np.arctan2(float(Rwc_now[1, 0]), float(Rwc_now[1, 1])))
                                    roll_deg = float(np.degrees(float(roll_rad)))
                                except Exception:
                                    yaw_deg = roll_deg = pitch_deg = None
                                    fwd_w = None

                            # ORB-SLAM3 map/world pose (can jump after BA / loop closure).
                            pos_map = None
                            yaw_deg_map = roll_deg_map = pitch_deg_map = None
                            if Twc_map is not None:
                                try:
                                    pos_map = np.asarray(Twc_map[:3, 3], dtype=np.float64).reshape(3)
                                    Rwc_map = np.asarray(Twc_map[:3, :3], dtype=np.float64).reshape(3, 3)
                                    if not np.isfinite(pos_map).all():
                                        pos_map = None
                                        Rwc_map = None
                                    if Rwc_map is not None:
                                        fwd = Rwc_map @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
                                        yaw_deg_map = float(np.degrees(np.arctan2(float(fwd[0]), float(fwd[2]))))
                                        r02 = float(Rwc_map[0, 2])
                                        r12 = float(Rwc_map[1, 2])
                                        r22 = float(Rwc_map[2, 2])
                                        pitch_rad = float(np.arctan2(-float(r12), float(np.sqrt(float(r02 * r02 + r22 * r22)))))
                                        pitch_deg_map = float(np.degrees(float(pitch_rad)))
                                        roll_rad = float(np.arctan2(float(Rwc_map[1, 0]), float(Rwc_map[1, 1])))
                                        roll_deg_map = float(np.degrees(float(roll_rad)))
                                except Exception:
                                    pos_map = None
                                    yaw_deg_map = roll_deg_map = pitch_deg_map = None

                            poly_out = {
                                "active": int(bool(poly_active)),
                                "sel_kind": str(poly_proc_sel_kind),
                                "pid_confirmed": int(bool(poly_pid_confirmed)),
                                "plane_plane_center_range_m": float(plane_plane_center_range_m),
                                "bbox": (tuple(int(v) for v in poly_bbox) if poly_bbox is not None else None),
                                "verts_uv": (np.asarray(poly_uv, dtype=np.float32).reshape(-1, 2).tolist() if poly_uv is not None else None),
                                "reason": str(poly_proc_reason),
                                "fi": int(poly_proc_last_fi),
                                "hole_stats": {
                                    "center_uv": (tuple(float(x) for x in hole_center_uv) if hole_center_uv is not None else None),
                                    "r_m": float(hole_r_m),
                                    "fill": float(hole_fill),
                                    "ok_pid": int(bool(hole_ok_pid)),
                                    "plane_inliers": int(hole_plane_inliers),
                                    "plane_rms_m": float(hole_plane_rms_m),
                                    "plane_cov": float(hole_plane_cov),
                                    "err": str(hole_err or ""),
                                },
                                "plane_stats": {
                                    "center_uv": (tuple(float(x) for x in plane_center_uv) if plane_center_uv is not None else None),
                                    "r_m": float(plane_r_m),
                                    "ok_pid": int(bool(plane_ok_pid)),
                                    "plane_inliers": int(plane_plane_inliers),
                                    "plane_rms_m": float(plane_plane_rms_m),
                                    "plane_cov": float(plane_plane_cov),
                                    "plane_center_range_m": float(plane_plane_center_range_m),
                                    "err": str(plane_err or ""),
                                },
                            }
                            hole_out = dict(poly_out.get("hole_stats", {})) if isinstance(poly_out.get("hole_stats"), dict) else {}
                            payload = {
                                "ring": {"fi": int(fi), "ts": float(t_cam_raw_s)},
                                "lk": {
                                    "fi": int(fi),
                                    "prev_ts": (float(prev_frame_ts_raw_s) if prev_frame_ts_raw_s is not None else 0.0),
                                    "cur_ts": float(t_cam_raw_s),
                                    "n_tracks": int(last_kp_n),
                                    "n_corr": int(last_kp_n),
                                    "lk_of_ms": 0.0,
                                    "lk_ms": 0.0,
                                    "lk_reseed_ms": 0.0,
                                    "lk_depth_ms": 0.0,
                                    "lk_corr_ms": 0.0,
                                    "err_pre_n": 0,
                                    "err_post_n": 0,
                                    "err_rej_n": 0,
                                    "err_med": None,
                                    "err_mad": None,
                                    "err_sigma": None,
                                    "err_thr": None,
                                },
                                "state": {
                                    "fi": int(fi),
                                    "ts": float(t_cam_raw_s),
                                    "ok": int(bool(ok)),
                                    "stable": int(bool(stable)),
                                    "status_code": int(trk_state_code),
                                    "est_code": int(0),
                                    "est": str("orbslam3"),
                                    "corr": int(0),
                                    "inliers": int(0),
                                    "rmse_m": float(0.0),
                                    "rot_deg": float(0.0),
                                    "ms_est": float(last_track_ms),
                                    "ms_total": float(last_track_ms),
                                    "ms_prefilter": 0.0,
                                    "ms_pg": 0.0,
                                    "ms_jitter": 0.0,
                                    "ms_imu": 0.0,
                                    "ms_weights": 0.0,
                                    "ms_gate": 0.0,
                                },
                                "pose": {
                                    "pos_w_m": (pos.tolist() if pos is not None else None),
                                    "pos0_w_m": (pos0.tolist() if pos0 is not None else None),
                                    "pos_rel_w_m": (rel.tolist() if rel is not None else None),
                                    "fwd_w": (list(fwd_w) if fwd_w is not None else None),
                                    "t_step_m": (t_step.tolist() if t_step is not None else [0.0, 0.0, 0.0]),
                                    "roll_deg": (float(roll_deg) if roll_deg is not None else None),
                                    "pitch_deg": (float(pitch_deg) if pitch_deg is not None else None),
                                    "yaw_deg": (float(yaw_deg) if yaw_deg is not None else None),
                                    "path_len_m": float(path_len_m),
                                    "drift_m": (float(drift_m) if drift_m is not None else None),
                                },
                                "pose_map": {
                                    "pos_w_m": (pos_map.tolist() if pos_map is not None else None),
                                    "roll_deg": (float(roll_deg_map) if roll_deg_map is not None else None),
                                    "pitch_deg": (float(pitch_deg_map) if pitch_deg_map is not None else None),
                                    "yaw_deg": (float(yaw_deg_map) if yaw_deg_map is not None else None),
                                },
                                "imu": {"idx": None},
                                "poly": poly_out,
                                "hole": hole_out,
                            }
                            headless_io.send_event(event="telemetry", payload=payload)
                    except Exception:
                        pass

                # Features (rate-limited).
                if features_period_s > 0.0:
                    try:
                        now_m = float(time.monotonic())
                        if (now_m - float(last_features_mono)) >= float(features_period_s):
                            last_features_mono = float(now_m)
                            tracks_out: list[Any] = []
                            try:
                                tracks_lock: list[Any] = []
                                # Polygon-lock points (LK). These are the points that actually hold the polygon in place.
                                try:
                                    if poly_lock_used_uv is not None:
                                        uv_lock = np.asarray(poly_lock_used_uv, dtype=np.float32).reshape(-1, 2)
                                        if int(uv_lock.shape[0]) > 0:
                                            lock_group = 2 if bool(poly_lock_locked) else 3  # yellow when locked; red when not
                                            if int(features_max_tracks) > 0 and int(uv_lock.shape[0]) > int(features_max_tracks):
                                                step = max(1, int(uv_lock.shape[0]) // int(features_max_tracks))
                                                uv_lock = uv_lock[::step][: int(features_max_tracks)]
                                            for u0, v0 in uv_lock.tolist():
                                                tracks_lock.append((int(round(float(u0))), int(round(float(v0))), int(lock_group)))
                                except Exception:
                                    tracks_lock = []

                                tracks_orb: list[Any] = []
                                try:
                                    kps = slam.GetTrackedKeyPointsUn()
                                    if kps is not None:
                                        uv = np.asarray(kps, dtype=np.float32).reshape(-1, 2)
                                        if int(uv.shape[0]) > 0:
                                            last_kp_n = int(uv.shape[0])
                                            # Keep some budget for lock points.
                                            if int(features_max_tracks) > 0:
                                                budget = max(0, int(features_max_tracks) - int(len(tracks_lock)))
                                            else:
                                                budget = int(features_max_tracks)
                                            if int(budget) > 0 and int(uv.shape[0]) > int(budget):
                                                step = max(1, int(uv.shape[0]) // int(budget))
                                                uv = uv[::step][: int(budget)]
                                            for u0, v0 in uv.tolist():
                                                tracks_orb.append((int(round(float(u0))), int(round(float(v0))), 1))
                                except Exception:
                                    tracks_orb = []

                                # Prioritize lock points in the outgoing overlay.
                                tracks_out = list(tracks_lock) + list(tracks_orb)
                            except Exception:
                                tracks_out = []

                            if bool(tracks_out):
                                headless_io.send_event(
                                    event="features",
                                    payload={
                                        "ring": {"fi": int(fi), "ts": float(t_cam_raw_s)},
                                        "state_fi": int(fi),
                                        "jitter_med": {"yaw_deg": 0.0, "pitch_deg": 0.0, "depth_cm": 0.0},
                                        "tracks": (tracks_out if bool(features_compact) else [{"uvg": t} for t in tracks_out]),
                                    },
                                )
                    except Exception:
                        pass

            prev_frame_ts_raw_s = float(t_cam_raw_s)

            if bool(args.no_vis):
                continue

            # Visualization: draw tracked keypoints (safe API).
            vis = img_in.copy()
            try:
                kps = slam.GetTrackedKeyPointsUn()
                if kps is not None:
                    uv = np.asarray(kps, dtype=np.float32).reshape(-1, 2)
                    u = uv[:, 0].astype(np.int32, copy=False)
                    v = uv[:, 1].astype(np.int32, copy=False)
                    ok = (u >= 0) & (u < W) & (v >= 0) & (v < H)
                    u = u[ok]
                    v = v[ok]
                    vis[v, u] = (0, 255, 0)
            except Exception:
                pass

            msg = f"t={ts:6.2f}s imu_n={int(imu_batch.shape[0])}"
            if Twc is not None:
                p = Twc[:3, 3].reshape(3)
                msg += f"  p=({p[0]:+.2f},{p[1]:+.2f},{p[2]:+.2f})"
            cv2.putText(vis, msg, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(vis, msg, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 1, cv2.LINE_AA)

            cv2.imshow(win, vis)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord("r"):
                try:
                    slam.ResetActiveMap()
                except Exception:
                    try:
                        slam.Reset()
                    except Exception:
                        pass
                frame_resolver = OrbslamFrameResolver(jump_trans_m=odom_jump_trans_m, jump_rot_deg=odom_jump_rot_deg)
                prev_px4_ts = None
                prev_px4_p = None
                start_pos_w = None
                prev_pos_w = None
                path_len_m = 0.0
                imu.reset()
                last_img_ts = None
                ts_base_raw = None

    finally:
        try:
            if poly_proc is not None:
                try:
                    poly_stop_evt.set()
                except Exception:
                    pass
                try:
                    poly_proc.join(timeout=1.0)
                except Exception:
                    pass
                try:
                    if poly_proc.is_alive():
                        poly_proc.terminate()
                        poly_proc.join(timeout=1.0)
                except Exception:
                    pass
        except Exception:
            pass
        try:
            ipc_vo.stop()
        except Exception:
            pass
        try:
            px4_pub.stop()
        except Exception:
            pass
        try:
            pipe.stop()
        except Exception:
            pass
        try:
            headless_io.close()
        except Exception:
            pass
        try:
            if shm_ring is not None:
                shm_ring.close()
        except Exception:
            pass
        try:
            if shm_imu_ring is not None:
                shm_imu_ring.close()
        except Exception:
            pass
        try:
            unlink_on_exit = bool(cap_cfg.get("shm_unlink_on_exit", True))
            if "ORB_TRACK3D_SHM_UNLINK_ON_EXIT" in os.environ:
                unlink_env = str(os.environ.get("ORB_TRACK3D_SHM_UNLINK_ON_EXIT", "1") or "1").strip().lower()
                unlink_on_exit = unlink_env not in ("0", "false", "no", "off")
            if bool(unlink_on_exit):
                if shm_ring is not None:
                    shm_ring.unlink()
                if shm_imu_ring is not None:
                    shm_imu_ring.unlink()
        except Exception:
            pass
        try:
            slam.Shutdown()
        except Exception:
            pass
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
