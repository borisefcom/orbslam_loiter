from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
import time
from collections import deque
from dataclasses import replace
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np

from orbslam_app.framebus import (
    FrameBusConfig,
    FrameBusError,
    FrameBusReader,
    FLAG_DEPTH,
    FLAG_IR_LEFT,
    FLAG_IR_RIGHT,
    FLAG_RGB,
)
from orbslam_app.imu_sync import ImuSynchronizer
from orbslam_app.modes import MODE_BY_LABEL, MODE_SPECS, resolve_mode, mode_id as mode_id_for
from orbslam_app.mavlink_bridge import MavlinkBridge
from orbslam_app.paths import DEFAULT_VOCAB_TAR_GZ, DEFAULT_VOCAB_TXT, THIRD_PARTY_ORB
from orbslam_app.top_view import layout_bottom_right_buttons, layout_mode_buttons, render_top_view, tracking_state_name
from orbslam_app.vocab import ensure_orb_vocab


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


def _load_px4_yaml(path: Path) -> dict:
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


def _apply_px4_yaml_defaults(parser: argparse.ArgumentParser, path: Path) -> None:
    cfg = _load_px4_yaml(path)
    if not cfg:
        return
    px4 = cfg.get("px4", {})
    if not isinstance(px4, dict):
        return
    defaults: dict[str, object] = {}
    if "enabled" in px4:
        defaults["px4_enabled"] = bool(px4.get("enabled"))
    if "serial" in px4:
        defaults["px4_serial"] = str(px4.get("serial", "")).strip()
    if "baud" in px4:
        defaults["px4_baud"] = int(px4.get("baud", 0) or 0)
    if "dialect" in px4:
        defaults["px4_dialect"] = str(px4.get("dialect", "")).strip() or "common"
    if "mavlink2" in px4:
        defaults["px4_mavlink2"] = bool(px4.get("mavlink2"))
    if "heartbeat_hz" in px4:
        defaults["px4_heartbeat_hz"] = float(px4.get("heartbeat_hz", 0.0) or 0.0)
    if "status_hz" in px4:
        defaults["px4_status_hz"] = float(px4.get("status_hz", 0.0) or 0.0)
    rates = px4.get("rates_hz", {})
    if isinstance(rates, dict):
        if "RAW_IMU" in rates:
            defaults["px4_raw_imu_hz"] = float(rates.get("RAW_IMU", 0.0) or 0.0)
        if "HIGHRES_IMU" in rates:
            defaults["px4_highres_imu_hz"] = float(rates.get("HIGHRES_IMU", 0.0) or 0.0)
    calib = px4.get("calib", {})
    if isinstance(calib, dict):
        if "window_s" in calib:
            defaults["px4_calib_window_s"] = float(calib.get("window_s", 0.0) or 0.0)
        if "timeout_s" in calib:
            defaults["px4_calib_timeout_s"] = float(calib.get("timeout_s", 0.0) or 0.0)
    odom = px4.get("odom", {})
    if isinstance(odom, dict):
        if "enabled" in odom:
            defaults["px4_odom_enabled"] = bool(odom.get("enabled"))
        if "rate_hz" in odom:
            defaults["px4_odom_hz"] = float(odom.get("rate_hz", 0.0) or 0.0)
    logging = cfg.get("logging", {})
    if isinstance(logging, dict):
        odom_log = logging.get("odom_log", {})
        if isinstance(odom_log, dict):
            if "enabled" in odom_log:
                defaults["odom_log_enabled"] = bool(odom_log.get("enabled"))
            if "path" in odom_log:
                defaults["odom_log_path"] = str(odom_log.get("path", "")).strip()
            if "rate_hz" in odom_log:
                defaults["odom_log_hz"] = float(odom_log.get("rate_hz", 0.0) or 0.0)
    if defaults:
        parser.set_defaults(**defaults)


def _import_orbslam3() -> object:
    sys.path.insert(0, str(THIRD_PARTY_ORB))
    try:
        from python_wrapper.orb_slam3 import ORB_SLAM3  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "Failed to import ORB-SLAM3 python wrapper.\n"
            "Build `third_party/ORB_SLAM3_pybind` first (CMake + pybind11), then ensure the built extension is importable.\n"
            f"Import error: {exc}"
        ) from exc
    return ORB_SLAM3


def _required_flags_for_mode(mode) -> int:
    flags = 0
    if mode.uses_rgbd():
        flags |= FLAG_RGB | FLAG_DEPTH
    elif mode.uses_stereo():
        flags |= FLAG_IR_LEFT | FLAG_IR_RIGHT
    elif mode.uses_mono():
        flags |= FLAG_IR_LEFT
    return int(flags)


def _colorize_depth(depth_raw: np.ndarray, *, max_depth_m: float) -> np.ndarray:
    max_mm = max(1.0, float(max_depth_m) * 1000.0)
    scale = 255.0 / max_mm
    depth_clipped = np.minimum(depth_raw, int(max_mm)).astype(np.float32, copy=False)
    depth_u8 = (depth_clipped * scale).astype(np.uint8)
    depth_u8[depth_raw == 0] = 0
    return cv2.applyColorMap(depth_u8, cv2.COLORMAP_TURBO)


def _unit_vec(v: np.ndarray) -> Optional[np.ndarray]:
    n = float(np.linalg.norm(v))
    if not math.isfinite(n) or n <= 1e-6:
        return None
    return (v / n).astype(np.float64, copy=False)


def _mean_direction(samples: list[tuple[float, float, float]], *, invert: bool = False) -> tuple[Optional[np.ndarray], float]:
    if not samples:
        return None, float("inf")
    arr = np.asarray(samples, dtype=np.float64).reshape(-1, 3)
    if bool(invert):
        arr = -arr
    norms = np.linalg.norm(arr, axis=1)
    keep = norms > 1e-6
    if not np.any(keep):
        return None, float("inf")
    arr = arr[keep] / norms[keep, None]
    mean = np.mean(arr, axis=0)
    mean_u = _unit_vec(mean)
    if mean_u is None:
        return None, float("inf")
    dots = np.clip(arr @ mean_u, -1.0, 1.0)
    ang = np.degrees(np.arccos(dots))
    med = float(np.median(ang)) if ang.size > 0 else float("inf")
    return mean_u, med


def _rotation_from_to(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    au = _unit_vec(a)
    bu = _unit_vec(b)
    if au is None or bu is None:
        return np.eye(3, dtype=np.float64)
    v = np.cross(au, bu)
    s = float(np.linalg.norm(v))
    c = float(np.dot(au, bu))
    if s <= 1e-6:
        if c >= 0.0:
            return np.eye(3, dtype=np.float64)
        axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(float(au[0])) > 0.9:
            axis = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        axis = np.cross(au, axis)
        axis_u = _unit_vec(axis)
        if axis_u is None:
            return np.eye(3, dtype=np.float64)
        vx, vy, vz = axis_u
        K = np.array([[0.0, -vz, vy], [vz, 0.0, -vx], [-vy, vx, 0.0]], dtype=np.float64)
        return np.eye(3, dtype=np.float64) + 2.0 * (K @ K)
    vx, vy, vz = v / s
    K = np.array([[0.0, -vz, vy], [vz, 0.0, -vx], [-vy, vx, 0.0]], dtype=np.float64)
    R = np.eye(3, dtype=np.float64) + K + (K @ K) * ((1.0 - c) / (s * s))
    return R


def _ned_from_acc_mag(acc_down: np.ndarray, mag: np.ndarray) -> Optional[np.ndarray]:
    down = _unit_vec(acc_down)
    mag_u = _unit_vec(mag)
    if down is None or mag_u is None:
        return None
    east = _unit_vec(np.cross(mag_u, down))
    if east is None:
        return None
    north = _unit_vec(np.cross(down, east))
    if north is None:
        return None
    return np.stack([north, east, down], axis=0)


def _rpy_from_ned(R: np.ndarray) -> Optional[tuple[float, float, float]]:
    if R is None or R.shape != (3, 3):
        return None
    try:
        yaw = math.degrees(math.atan2(float(R[0, 1]), float(R[0, 0])))
        pitch = math.degrees(math.asin(max(-1.0, min(1.0, float(-R[0, 2])))))
        roll = math.degrees(math.atan2(float(R[1, 2]), float(R[2, 2])))
        return float(roll), float(pitch), float(yaw)
    except Exception:
        return None


def _quat_from_R(R: np.ndarray) -> tuple[float, float, float, float]:
    m = np.asarray(R, dtype=np.float64).reshape(3, 3)
    trace = float(m[0, 0] + m[1, 1] + m[2, 2])
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (m[2, 1] - m[1, 2]) / s
        qy = (m[0, 2] - m[2, 0]) / s
        qz = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        qw = (m[2, 1] - m[1, 2]) / s
        qx = 0.25 * s
        qy = (m[0, 1] + m[1, 0]) / s
        qz = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        qw = (m[0, 2] - m[2, 0]) / s
        qx = (m[0, 1] + m[1, 0]) / s
        qy = 0.25 * s
        qz = (m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        qw = (m[1, 0] - m[0, 1]) / s
        qx = (m[0, 2] + m[2, 0]) / s
        qy = (m[1, 2] + m[2, 1]) / s
        qz = 0.25 * s
    return float(qw), float(qx), float(qy), float(qz)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", type=str, default=str(DEFAULT_VOCAB_TXT), help="Path to ORBvoc.txt")
    ap.add_argument(
        "--vocab-tar",
        type=str,
        default=str(DEFAULT_VOCAB_TAR_GZ),
        help="Path to ORBvoc.txt.tar.gz (used if ORBvoc.txt missing)",
    )
    ap.add_argument(
        "--mode",
        type=str,
        default="rgbd+imu",
        help="Initial mode: rgbd, rgbd+imu, stereo, stereo+imu, mono, mono+imu",
    )
    ap.add_argument(
        "--mode-switch-restart",
        action=argparse.BooleanOptionalAction,
        default=False,
        dest="mode_switch_restart",
        help="Restart the process on mode switch (slower; avoids in-process reinit)",
    )
    ap.add_argument("--settings", type=str, default="", help="Override settings YAML for initial mode only")
    ap.add_argument("--viewer", action="store_true", help="Enable ORB-SLAM3 Pangolin viewer (if built)")
    ap.add_argument(
        "--slam-shutdown",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Call ORB-SLAM3 Shutdown() on mode switch and exit (disable if it crashes)",
    )
    ap.add_argument("--map-size", type=int, default=700, help="Top-view window size (pixels)")
    ap.add_argument("--max-map-points", type=int, default=50000, help="Max accumulated map points to keep")
    ap.add_argument(
        "--map-points",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Show map points overlay (default: off)",
    )
    ap.add_argument(
        "--map-points-hz",
        type=float,
        default=1.0,
        help="Map point query rate when enabled (Hz)",
    )
    ap.add_argument("--depth-max-m", type=float, default=5.0, help="Max depth (m) for depth display")
    ap.add_argument("--debug", action="store_true", help="Print extra debug info")
    # TEST ONLY: temporary debug options for mode-switch freeze investigation.
    ap.add_argument(
        "--test-switch-interval-s",
        type=float,
        default=0.0,
        help="(TEST) Auto-switch modes every N seconds (0 disables)",
    )
    ap.add_argument(
        "--test-switch-count",
        type=int,
        default=0,
        help="(TEST) Stop auto-switch after N switches (0 disables)",
    )
    ap.add_argument(
        "--test-switch-order",
        type=str,
        default="",
        help="(TEST) Comma-separated mode list for auto-switch",
    )
    ap.add_argument(
        "--test-log-stall-s",
        type=float,
        default=2.0,
        help="(TEST) Log if tracking stalls for N seconds (0 disables)",
    )
    ap.add_argument(
        "--test-switch-log",
        action="store_true",
        help="(TEST) Log timing details during mode switches",
    )
    ap.add_argument(
        "--test-log-fps",
        action="store_true",
        help="(TEST) Log camera/SLAM FPS once per second",
    )
    ap.add_argument(
        "--auto-exit-s",
        type=float,
        default=0.0,
        help="Auto-exit after N seconds (0 disables)",
    )
    ap.add_argument("--framebus-name", type=str, default=FrameBusConfig().name, help="Shared memory name prefix")
    ap.add_argument(
        "--no-framebus-auto-start",
        action="store_false",
        dest="framebus_auto_start",
        help="Do not auto-start the RealSense framebus process",
    )
    ap.add_argument(
        "--no-framebus-auto-stop",
        action="store_false",
        dest="framebus_auto_stop",
        help="Do not auto-stop the RealSense framebus process on exit",
    )
    ap.add_argument(
        "--framebus-start-wait-s",
        type=float,
        default=6.0,
        help="Seconds to wait for framebus shared memory after auto-start",
    )
    ap.add_argument(
        "--framebus-force",
        action="store_true",
        help="Start framebus with --force (recreate shared memory)",
    )
    ap.add_argument(
        "--framebus-fixed-streams",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep framebus streams fixed for fast mode switching (default: enabled)",
    )
    ap.add_argument(
        "--framebus-owned",
        action="store_true",
        help="(internal) Treat the framebus as owned by this process for shutdown",
    )
    ap.add_argument("--px4", dest="px4_enabled", action="store_true", default=False, help="Enable PX4 MAVLink link")
    ap.add_argument("--px4-serial", type=str, default="", help="PX4 serial port (e.g. COM5 or /dev/ttyUSB0)")
    ap.add_argument("--px4-baud", type=int, default=115200, help="PX4 serial baud rate")
    ap.add_argument("--px4-dialect", type=str, default="common", help="MAVLink dialect (default: common)")
    ap.add_argument(
        "--px4-mavlink2",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use MAVLink2 extensions (default: enabled)",
    )
    ap.add_argument("--px4-heartbeat-hz", type=float, default=1.0, help="PX4 heartbeat rate (Hz)")
    ap.add_argument("--px4-status-hz", type=float, default=1.0, help="PX4 status log rate (Hz)")
    ap.add_argument("--px4-raw-imu-hz", type=float, default=100.0, help="Request RAW_IMU rate (Hz)")
    ap.add_argument("--px4-highres-imu-hz", type=float, default=0.0, help="Request HIGHRES_IMU rate (Hz)")
    ap.add_argument("--px4-calib-window-s", type=float, default=1.5, help="IMU calibration window (s)")
    ap.add_argument("--px4-calib-timeout-s", type=float, default=10.0, help="IMU calibration timeout (s)")
    ap.add_argument(
        "--px4-odom",
        action=argparse.BooleanOptionalAction,
        default=False,
        dest="px4_odom_enabled",
        help="Send MAVLink ODOMETRY messages",
    )
    ap.add_argument("--px4-odom-hz", type=float, default=30.0, help="ODOMETRY send rate (Hz)")
    ap.add_argument(
        "--debug-odom",
        action="store_true",
        help="Print detailed odom debug snapshots (noisy)",
    )
    ap.add_argument("--debug-odom-hz", type=float, default=2.0, help="Odom debug print rate (Hz)")
    ap.add_argument(
        "--odom-log",
        action=argparse.BooleanOptionalAction,
        default=False,
        dest="odom_log_enabled",
        help="Write odom debug log to file (see YAML logging.odom_log)",
    )
    ap.add_argument(
        "--odom-log-path",
        type=str,
        default="logs/odom_debug.log",
        help="Path to odom debug log (cleared on start)",
    )
    ap.add_argument("--odom-log-hz", type=float, default=10.0, help="Odom log rate (Hz)")
    _apply_px4_yaml_defaults(ap, PROJECT_ROOT / "apps" / "px4_mavlink.yaml")
    args = ap.parse_args()
    debug = bool(args.debug)
    debug_odom = bool(args.debug_odom)
    debug_odom_hz = max(0.0, float(args.debug_odom_hz))
    odom_log_enabled = bool(args.odom_log_enabled)
    odom_log_hz = max(0.0, float(args.odom_log_hz))
    odom_log_path = str(args.odom_log_path).strip()
    map_size = int(max(200, int(args.map_size)))
    depth_max_m = max(0.1, float(args.depth_max_m))
    map_points_enabled = bool(args.map_points)
    map_points_hz = max(0.0, float(args.map_points_hz))
    map_points_period = (1.0 / map_points_hz) if map_points_hz > 0 else None

    px4_enabled = bool(args.px4_enabled) or bool(str(args.px4_serial).strip())
    if px4_enabled and not str(args.px4_serial).strip():
        print("[orbslam] PX4 enabled but no --px4-serial provided; disabling PX4 link.", flush=True)
        px4_enabled = False
    px4_odom_enabled = bool(args.px4_odom_enabled)
    px4_odom_hz = max(0.0, float(args.px4_odom_hz))
    if px4_odom_enabled and not px4_enabled:
        print("[orbslam] PX4 odom enabled but PX4 link disabled; disabling odom.", flush=True)
        px4_odom_enabled = False

    vocab_txt = ensure_orb_vocab(vocab_txt=Path(args.vocab), vocab_tar_gz=Path(args.vocab_tar))
    ORB_SLAM3 = _import_orbslam3()
    current_mode = resolve_mode(str(args.mode))
    settings_override = Path(args.settings) if str(args.settings).strip() else None
    if settings_override is not None:
        if not settings_override.exists():
            raise FileNotFoundError(f"Missing settings file: {settings_override}")
        current_mode = replace(current_mode, settings=settings_override)
    current_mode_id = mode_id_for(current_mode)

    test_switch_interval_s = max(0.0, float(args.test_switch_interval_s))
    test_switch_count = max(0, int(args.test_switch_count))
    test_log_stall_s = max(0.0, float(args.test_log_stall_s))
    test_switch_log = bool(args.test_switch_log)
    test_switch_modes: list[object] = []
    if str(args.test_switch_order).strip():
        for token in str(args.test_switch_order).split(","):
            token = token.strip()
            if token:
                test_switch_modes.append(resolve_mode(token))
    if not test_switch_modes:
        test_switch_modes = list(MODE_SPECS)
    test_switch_index = 0
    if test_switch_interval_s > 0 and test_switch_modes:
        for idx, mode in enumerate(test_switch_modes):
            if mode.key == current_mode.key:
                test_switch_index = int(idx + 1)
                break
    test_switch_done = 0
    next_test_switch_t: float | None = None

    def make_slam(mode) -> object:
        if not mode.settings.exists():
            raise FileNotFoundError(f"Missing settings file for {mode.label}: {mode.settings}")
        return ORB_SLAM3(str(vocab_txt), str(mode.settings), mode.sensor, bool(args.viewer))

    if test_switch_interval_s > 0:
        next_test_switch_t = float(time.time() + test_switch_interval_s)

    odom_log_fp = None
    odom_log_period = (1.0 / odom_log_hz) if odom_log_enabled and odom_log_hz > 0 else None
    last_odom_log_perf = 0.0
    if odom_log_enabled:
        if not odom_log_path:
            print("[orbslam] Odom log disabled (empty path).", flush=True)
        else:
            log_path = Path(odom_log_path)
            if not log_path.is_absolute():
                log_path = PROJECT_ROOT / log_path
            try:
                log_path.parent.mkdir(parents=True, exist_ok=True)
                odom_log_fp = log_path.open("w", encoding="utf-8", buffering=1)
                print(f"[orbslam] Odom log -> {log_path}", flush=True)
            except Exception as exc:
                print(f"[orbslam] Odom log disabled (open error: {exc}).", flush=True)
                odom_log_fp = None
                odom_log_period = None

    framebus_cfg = FrameBusConfig(name=str(args.framebus_name))
    framebus_proc: Optional[subprocess.Popen] = None
    framebus_owned = bool(args.framebus_owned)
    handoff_restart = False

    def _next_test_mode() -> object:
        nonlocal test_switch_index
        if not test_switch_modes:
            return current_mode
        for _ in range(len(test_switch_modes)):
            mode = test_switch_modes[int(test_switch_index % len(test_switch_modes))]
            test_switch_index = int(test_switch_index + 1)
            if mode.key != current_mode.key:
                return mode
        return test_switch_modes[0]

    def _start_framebus(*, force: bool = False) -> Optional[subprocess.Popen]:
        cmd = [sys.executable, str(PROJECT_ROOT / "apps" / "realsense_framebus.py")]
        cmd += ["--mode", str(args.mode)]
        cmd += ["--framebus-name", str(framebus_cfg.name)]
        if force or bool(args.framebus_force):
            cmd.append("--force")
        if bool(args.framebus_fixed_streams):
            cmd.append("--fixed-streams")
        else:
            cmd.append("--no-fixed-streams")
        creationflags = 0
        if hasattr(subprocess, "CREATE_NEW_PROCESS_GROUP"):
            creationflags = subprocess.CREATE_NEW_PROCESS_GROUP
        try:
            return subprocess.Popen(cmd, cwd=str(PROJECT_ROOT), creationflags=creationflags)
        except Exception as exc:
            print(f"[orbslam] Failed to start framebus: {exc}", flush=True)
            return None

    def _wait_for_framebus_activity(reader: FrameBusReader, timeout_s: float) -> bool:
        if float(timeout_s) <= 0:
            return True
        try:
            status = reader.status()
            last_seq = int(status["frame_seq"])
        except Exception:
            return False
        deadline = float(time.time() + float(timeout_s))
        while time.time() < deadline:
            time.sleep(0.05)
            try:
                status = reader.status()
            except Exception:
                return False
            if int(status["frame_seq"]) != last_seq:
                return True
        return False

    def _connect_framebus() -> FrameBusReader:
        nonlocal framebus_proc, framebus_owned
        probe_s = min(1.0, max(0.1, float(args.framebus_start_wait_s)))

        def _start_and_connect(*, force: bool) -> FrameBusReader:
            nonlocal framebus_proc, framebus_owned
            framebus_proc = _start_framebus(force=force)
            if framebus_proc is None:
                raise FrameBusError("Failed to start framebus process.")
            framebus_owned = True
            deadline = time.time() + max(0.1, float(args.framebus_start_wait_s))
            reader = None
            while time.time() < deadline:
                try:
                    reader = FrameBusReader(framebus_cfg)
                    break
                except FrameBusError:
                    time.sleep(0.1)
            if reader is None:
                raise FrameBusError("Timed out waiting for framebus shared memory.")
            if not _wait_for_framebus_activity(reader, probe_s):
                print("[orbslam] Framebus shared memory attached but no frames yet.", flush=True)
            return reader

        try:
            reader = FrameBusReader(framebus_cfg)
        except FrameBusError as exc:
            if not bool(args.framebus_auto_start):
                raise
            print(f"[orbslam] {exc}", flush=True)
            print("[orbslam] Auto-starting framebus...", flush=True)
            return _start_and_connect(force=bool(args.framebus_force))

        if bool(args.framebus_force) and bool(args.framebus_auto_start) and not framebus_owned and framebus_proc is None:
            print("[orbslam] --framebus-force set; restarting framebus...", flush=True)
            try:
                reader.request_stop()
            except Exception:
                pass
            try:
                reader.close()
            except Exception:
                pass
            return _start_and_connect(force=True)

        if _wait_for_framebus_activity(reader, probe_s):
            return reader
        if not bool(args.framebus_auto_start):
            print("[orbslam] Framebus shared memory is idle; waiting for frames.", flush=True)
            return reader
        print("[orbslam] Framebus shared memory is idle; restarting...", flush=True)
        try:
            reader.request_stop()
        except Exception:
            pass
        try:
            reader.close()
        except Exception:
            pass
        return _start_and_connect(force=True)

    try:
        framebus = _connect_framebus()
    except FrameBusError as exc:
        print(f"[orbslam] {exc}", flush=True)
        print("[orbslam] Start the framebus with: python apps/realsense_framebus.py", flush=True)
        return 1
    framebus.request_mode(int(current_mode_id))

    mav_bridge: Optional[MavlinkBridge] = None
    if px4_enabled:
        os.environ["MAVLINK_DIALECT"] = str(args.px4_dialect)
        os.environ["MAVLINK20"] = "1" if bool(args.px4_mavlink2) else "0"
        px4_cfg: dict[str, object] = {
            "serial": str(args.px4_serial),
            "baud": int(args.px4_baud),
            "mavlink2": bool(args.px4_mavlink2),
            "dialect": str(args.px4_dialect),
            "heartbeat": {"enabled": True, "rate_hz": float(args.px4_heartbeat_hz)},
            "status": {"enabled": True, "rate_hz": float(args.px4_status_hz)},
            "rates_hz": {},
        }
        rates = px4_cfg["rates_hz"]
        if isinstance(rates, dict):
            if float(args.px4_raw_imu_hz) > 0:
                rates["RAW_IMU"] = float(args.px4_raw_imu_hz)
            if float(args.px4_highres_imu_hz) > 0:
                rates["HIGHRES_IMU"] = float(args.px4_highres_imu_hz)
        try:
            mav_bridge = MavlinkBridge({"px4": px4_cfg}, print_fn=print)
            mav_bridge.start()
            if mav_bridge.is_connected():
                mav_bridge.print_status_once()
                print("[orbslam] PX4 link enabled for calibration.", flush=True)
            else:
                err = mav_bridge.open_error()
                detail = f" ({err})" if err else ""
                print(f"[orbslam] PX4 link unavailable{detail}; disabling alignment/odom.", flush=True)
                mav_bridge = None
                px4_enabled = False
                px4_odom_enabled = False
        except Exception as exc:
            print(f"[orbslam] PX4 init error: {exc}", flush=True)
            mav_bridge = None
            px4_enabled = False
            px4_odom_enabled = False

    def _calibrate_alignment() -> tuple[np.ndarray, Optional[np.ndarray], str]:
        if mav_bridge is None:
            print("[orbslam] IMU alignment skipped (PX4 disabled).", flush=True)
            return np.eye(3, dtype=np.float64), None, "PX4 disabled"
        window_s = max(0.5, float(args.px4_calib_window_s))
        timeout_s = max(1.0, float(args.px4_calib_timeout_s))
        min_samples = max(20, int(window_s * 25))
        accel_med_deg = 3.0
        mag_med_deg = 6.0
        cam_buf: deque[tuple[float, float, float, float]] = deque(maxlen=max(200, int(window_s * 400)))
        deadline = float(time.time() + timeout_s)
        last_log = 0.0
        print("[orbslam] Calibrating IMU alignment (hold still)...", flush=True)
        while time.time() < deadline:
            now = float(time.time())
            try:
                _gyro, accel = framebus.drain_imu()
            except Exception:
                accel = []
            for _t_s, ax, ay, az in accel:
                cam_buf.append((now, float(ax), float(ay), float(az)))
            while cam_buf and (now - float(cam_buf[0][0])) > window_s:
                cam_buf.popleft()
            cam_samples = [(ax, ay, az) for _t, ax, ay, az in cam_buf]
            px4_accel, px4_mag = mav_bridge.get_imu_window(window_s)
            if len(cam_samples) >= min_samples and len(px4_accel) >= min_samples and len(px4_mag) >= min_samples:
                g_cam, g_cam_med = _mean_direction(cam_samples, invert=True)
                g_body, g_body_med = _mean_direction(px4_accel, invert=True)
                m_body, m_body_med = _mean_direction(px4_mag, invert=False)
                if g_cam is not None and g_body is not None and m_body is not None:
                    if g_cam_med <= accel_med_deg and g_body_med <= accel_med_deg and m_body_med <= mag_med_deg:
                        r_body_cam = _rotation_from_to(g_cam, g_body)
                        r_ned_body = _ned_from_acc_mag(g_body, m_body)
                        if r_ned_body is not None:
                            tilt_deg = math.degrees(
                                math.acos(max(-1.0, min(1.0, float(np.dot(g_cam, g_body)))))
                            )
                            rpy_body = _rpy_from_ned(r_ned_body)
                            if rpy_body is not None:
                                roll_b, pitch_b, yaw_b = rpy_body
                                print(
                                    "[orbslam] IMU alignment OK "
                                    f"(cam_tilt={tilt_deg:.2f}deg, "
                                    f"body_rpy={roll_b:+.1f},{pitch_b:+.1f},{yaw_b:+.1f}deg, "
                                    f"acc_med={g_body_med:.2f}deg mag_med={m_body_med:.2f}deg).",
                                    flush=True,
                                )
                            else:
                                print(
                                    "[orbslam] IMU alignment OK "
                                    f"(cam_tilt={tilt_deg:.2f}deg, acc_med={g_body_med:.2f}deg "
                                    f"mag_med={m_body_med:.2f}deg).",
                                    flush=True,
                                )
                            print(
                                f"[orbslam] IMU alignment details: "
                                f"cam_med={g_cam_med:.2f}deg body_med={g_body_med:.2f}deg mag_med={m_body_med:.2f}deg.",
                                flush=True,
                            )
                            return r_body_cam, r_ned_body, "ok"
            if (now - last_log) >= 0.5:
                print(
                    "[orbslam] Calibrating IMU alignment... "
                    f"cam={len(cam_samples)} px4_acc={len(px4_accel)} px4_mag={len(px4_mag)}",
                    flush=True,
                )
                last_log = now
            time.sleep(0.02)
        print("[orbslam] IMU alignment timeout; continuing without NED alignment.", flush=True)
        return np.eye(3, dtype=np.float64), None, "timeout"

    R_body_cam, R_ned_body_ref, _calib_status = _calibrate_alignment()

    slam = make_slam(current_mode)
    atlas_inertial = bool(current_mode.use_imu)

    imu = ImuSynchronizer()
    traj: list[np.ndarray] = []
    map_pts = np.empty((0, 3), dtype=np.float32)
    map_pts_draw = np.empty((0, 3), dtype=np.float32)
    last_img_ts: float | None = None
    last_Twc: np.ndarray | None = None
    R_ned_world: np.ndarray | None = None
    rpy_deg: tuple[float, float, float] | None = None
    odom_pose: np.ndarray | None = np.eye(4, dtype=np.float64)
    odom_prev_pose: np.ndarray | None = None
    odom_delta_xyz: np.ndarray | None = None
    display_align: np.ndarray | None = np.eye(4, dtype=np.float64)
    pending_display_anchor: np.ndarray | None = None
    track_count = 0
    last_frame_seq_any = -1
    last_frame_seq_track = -1
    last_tracking_state = -1
    prev_tracking_state = -1
    last_rgb: np.ndarray | None = None
    last_depth: np.ndarray | None = None
    last_imu_count = 0
    last_ui_wall_s = float(time.time())
    last_track_wall_s = float(time.time())
    last_stall_log_s = 0.0
    slam_time_samples: deque[float] = deque()
    cam_fps: float | None = None
    slam_fps: float | None = None
    switch_timing: dict[str, object] | None = None
    last_cam_poll_perf = 0.0
    last_cam_seq: int | None = None
    last_cam_perf: float | None = None
    last_map_points_perf = 0.0
    non_imu_seq_base: int | None = None
    non_imu_last_seq: int | None = None
    non_imu_ts_fps = 30.0
    map_draw_points_max = min(int(args.max_map_points), 25000)
    odom_display_xyz: np.ndarray | None = None
    odom_frame_label = "MAP"
    odom_ned_q: tuple[float, float, float, float] | None = None
    odom_prev_ned: np.ndarray | None = None
    odom_prev_ned_ts: float | None = None
    odom_reset_counter = 0
    odom_pose_cov = [float("nan")] * 21
    odom_vel_cov = [float("nan")] * 21
    odom_send_period = (1.0 / px4_odom_hz) if px4_odom_enabled and px4_odom_hz > 0 else None
    odom_last_send_perf = 0.0
    odom_waiting_align_log = False
    last_map_points_source: str | None = None
    px4_odom_last_pos: np.ndarray | None = None
    px4_odom_last_vel: np.ndarray | None = None
    px4_odom_last_rpy: tuple[float, float, float] | None = None
    px4_odom_last_ts: float | None = None
    odom_offset = np.eye(4, dtype=np.float64)
    odom_global_pose = np.eye(4, dtype=np.float64)
    last_map_id: int | None = None
    last_map_count: int | None = None
    last_map_id_check_perf = 0.0
    map_id_check_period = 0.0
    odom_jump_trans_m = 0.75
    odom_jump_rot_deg = 35.0
    odom_collapse_hi_m = 0.5
    odom_collapse_lo_m = 0.05
    last_odom_debug_perf = 0.0
    last_odom_norm: float | None = None
    imu_initialized_flag: bool | None = None

    def _sample_map_points(points: np.ndarray) -> np.ndarray:
        if points.size == 0:
            return points
        if int(points.shape[0]) <= int(map_draw_points_max):
            return points
        idx = np.random.choice(int(points.shape[0]), int(map_draw_points_max), replace=False)
        return points[idx]

    def reset_tracking_state(
        reason: str,
        *,
        reset_slam: bool = True,
        preserve_map: bool = False,
        realign_on_first_pose: bool = False,
    ) -> None:
        nonlocal last_img_ts, last_Twc, map_pts, map_pts_draw, track_count
        nonlocal last_frame_seq_any, last_frame_seq_track, last_tracking_state, prev_tracking_state
        nonlocal last_rgb, last_depth, last_ui_wall_s, last_track_wall_s, last_imu_count, last_stall_log_s
        nonlocal display_align, pending_display_anchor
        nonlocal R_ned_world, rpy_deg, odom_pose, odom_prev_pose, odom_delta_xyz
        nonlocal odom_display_xyz, odom_frame_label, odom_ned_q, odom_prev_ned, odom_prev_ned_ts
        nonlocal odom_reset_counter, odom_last_send_perf, odom_waiting_align_log
        nonlocal odom_offset, odom_global_pose, last_map_id, last_map_id_check_perf
        nonlocal last_map_count
        nonlocal slam_time_samples, cam_fps, slam_fps, last_cam_poll_perf, last_cam_seq, last_cam_perf
        nonlocal switch_timing, last_map_points_perf
        nonlocal non_imu_seq_base, non_imu_last_seq
        nonlocal imu_initialized_flag
        nonlocal px4_odom_last_pos, px4_odom_last_vel, px4_odom_last_rpy, px4_odom_last_ts
        print(f"[orbslam] Reset ({reason})", flush=True)
        _odom_log_event("reset", reason)
        if reset_slam:
            try:
                slam.Reset()
            except Exception:
                pass
        imu.reset()
        print(f"[orbslam] IMU sync reset ({reason}).", flush=True)
        if not preserve_map:
            traj.clear()
            map_pts = np.empty((0, 3), dtype=np.float32)
            map_pts_draw = np.empty((0, 3), dtype=np.float32)
            display_align = np.eye(4, dtype=np.float64)
            pending_display_anchor = None
            last_Twc = None
            R_ned_world = None
            rpy_deg = None
            odom_pose = np.eye(4, dtype=np.float64)
            odom_prev_pose = None
            odom_delta_xyz = None
            odom_display_xyz = None
            odom_frame_label = "MAP"
            odom_ned_q = None
            odom_prev_ned = None
            odom_prev_ned_ts = None
            odom_last_send_perf = 0.0
            odom_waiting_align_log = False
            odom_reset_counter = int((odom_reset_counter + 1) % 256)
            print(f"[orbslam] Odom reset counter -> {int(odom_reset_counter)}.", flush=True)
            odom_offset = np.eye(4, dtype=np.float64)
            odom_global_pose = np.eye(4, dtype=np.float64)
            imu_initialized_flag = None
            px4_odom_last_pos = None
            px4_odom_last_vel = None
            px4_odom_last_rpy = None
            px4_odom_last_ts = None
            last_map_id = None
            last_map_count = None
            last_map_id_check_perf = 0.0
        else:
            if realign_on_first_pose:
                pending_display_anchor = last_Twc.copy() if last_Twc is not None else None
                display_align = None
            else:
                pending_display_anchor = None
            map_pts_draw = _sample_map_points(map_pts)
        last_img_ts = None
        track_count = 0
        last_frame_seq_any = -1
        last_frame_seq_track = -1
        last_tracking_state = -1
        prev_tracking_state = -1
        last_rgb = None
        last_depth = None
        last_imu_count = 0
        last_ui_wall_s = float(time.time())
        last_track_wall_s = float(time.time())
        last_stall_log_s = 0.0
        slam_time_samples.clear()
        cam_fps = None
        slam_fps = None
        switch_timing = None
        last_cam_poll_perf = 0.0
        last_cam_seq = None
        last_cam_perf = None
        last_map_points_perf = 0.0
        non_imu_seq_base = None
        non_imu_last_seq = None

    def _get_map_stats() -> tuple[int | None, int | None]:
        try:
            return int(slam.GetCurrentMapId()), int(slam.GetMapCount())
        except Exception:
            return None, None

    def _rot_angle_deg(R: np.ndarray) -> float:
        try:
            tr = float(R[0, 0] + R[1, 1] + R[2, 2])
            val = max(-1.0, min(1.0, 0.5 * (tr - 1.0)))
            return math.degrees(math.acos(val))
        except Exception:
            return 0.0

    def _fmt_vec(v: np.ndarray | None) -> str:
        if v is None:
            return "None"
        try:
            return f"({float(v[0]):+.2f},{float(v[1]):+.2f},{float(v[2]):+.2f})"
        except Exception:
            return "?"

    def _fmt_pose(T: np.ndarray | None) -> str:
        if T is None:
            return "None"
        try:
            t = np.asarray(T[:3, 3], dtype=np.float64).reshape(3)
            return _fmt_vec(t)
        except Exception:
            return "?"

    def _odom_log_line(line: str) -> None:
        nonlocal odom_log_fp
        if odom_log_fp is None:
            return
        try:
            odom_log_fp.write(f"{line}\n")
        except Exception:
            return

    def _odom_log_event(event: str, detail: str | None = None, *, now_s: float | None = None) -> None:
        if odom_log_fp is None:
            return
        try:
            ts = float(time.time()) if now_s is None else float(now_s)
            map_id_dbg, map_count_dbg = _get_map_stats()
            mid = "None" if map_id_dbg is None else str(int(map_id_dbg))
            mcount = "None" if map_count_dbg is None else str(int(map_count_dbg))
            state_name = tracking_state_name(int(last_tracking_state))
            prev_name = tracking_state_name(int(prev_tracking_state))
            traj_n = int(len(traj))
            map_pts_n = int(map_pts.shape[0]) if map_pts is not None else 0
            map_en = int(bool(map_points_enabled))
            map_hz = float(map_points_hz) if map_points_enabled else 0.0
            map_src = "none" if not last_map_points_source else str(last_map_points_source)
            cached_map_count = "None" if last_map_count is None else str(int(last_map_count))
            odom_delta_txt = _fmt_vec(odom_delta_xyz)
            reset_ctr = int(odom_reset_counter)
            imu_n = int(last_imu_count)
            line = (
                f"{ts:.3f} event={event} mode={current_mode.label} "
                f"state={state_name} prev={prev_name} imu_init={imu_initialized_flag} "
                f"map_id={mid} map_count={mcount} map_count_cached={cached_map_count} "
                f"traj_n={traj_n} map_pts={map_pts_n} map_en={map_en} map_hz={map_hz:.2f} map_src={map_src} "
                f"imu_n={imu_n} reset_ctr={reset_ctr} "
                f"T_wc={_fmt_pose(last_Twc)} odom={_fmt_pose(odom_global_pose)} "
                f"odom_delta={odom_delta_txt} display={_fmt_vec(odom_display_xyz)}"
            )
            if detail:
                line = f"{line} detail={detail}"
            _odom_log_line(line)
        except Exception:
            return

    def _debug_odom_snapshot(reason: str) -> None:
        if not debug_odom:
            return
        try:
            map_id_dbg, map_count_dbg = _get_map_stats()
            mid = "?" if map_id_dbg is None else str(int(map_id_dbg))
            mcount = "?" if map_count_dbg is None else str(int(map_count_dbg))
            state_name = tracking_state_name(int(last_tracking_state))
            prev_name = tracking_state_name(int(prev_tracking_state))
            print(
                "[odom][dbg] "
                f"{reason} state={state_name} prev={prev_name} imu_init={imu_initialized_flag} "
                f"map_id={mid} map_count={mcount} "
                f"T_wc={_fmt_pose(last_Twc)} "
                f"odom_pose={_fmt_pose(odom_pose)} "
                f"odom_offset={_fmt_pose(odom_offset)} "
                f"odom_global={_fmt_pose(odom_global_pose)} "
                f"display={_fmt_vec(odom_display_xyz)}",
                flush=True,
            )
        except Exception:
            return

    def _apply_odom_continuity_reset(reason: str, detail: str | None = None) -> None:
        nonlocal odom_offset, odom_pose, odom_prev_pose, odom_global_pose, odom_delta_xyz
        nonlocal odom_prev_ned, odom_prev_ned_ts, odom_display_xyz
        if odom_pose is None:
            odom_pose = np.eye(4, dtype=np.float64)
        odom_offset = odom_offset @ odom_pose
        odom_pose = np.eye(4, dtype=np.float64)
        odom_global_pose = odom_offset.copy()
        if last_Twc is not None:
            odom_prev_pose = last_Twc.copy()
        else:
            odom_prev_pose = None
        odom_prev_ned = None
        odom_prev_ned_ts = None
        odom_delta_xyz = odom_pose[:3, 3].copy()
        odom_display_xyz = odom_global_pose[:3, 3].copy()
        map_id_dbg, map_count_dbg = _get_map_stats()
        msg = f"[orbslam] Odom continuity reset ({reason})"
        if detail:
            msg = f"{msg}: {detail}"
        if map_id_dbg is not None or map_count_dbg is not None:
            mid = "?" if map_id_dbg is None else str(int(map_id_dbg))
            mcount = "?" if map_count_dbg is None else str(int(map_count_dbg))
            msg = f"{msg} map_id={mid} map_count={mcount}"
        print(msg, flush=True)
        _odom_log_event("continuity_reset", detail if detail else reason)
        _debug_odom_snapshot(f"reset:{reason}")

    def _apply_odom_offset_override(reason: str, detail: str | None, offset_pose: np.ndarray) -> None:
        nonlocal odom_offset, odom_pose, odom_prev_pose, odom_global_pose, odom_delta_xyz
        nonlocal odom_prev_ned, odom_prev_ned_ts, odom_display_xyz
        odom_offset = np.asarray(offset_pose, dtype=np.float64).reshape(4, 4)
        odom_pose = np.eye(4, dtype=np.float64)
        odom_global_pose = odom_offset.copy()
        odom_delta_xyz = odom_pose[:3, 3].copy()
        if last_Twc is not None:
            odom_prev_pose = last_Twc.copy()
        else:
            odom_prev_pose = None
        odom_prev_ned = None
        odom_prev_ned_ts = None
        odom_display_xyz = odom_global_pose[:3, 3].copy()
        msg = f"[orbslam] Odom continuity override ({reason})"
        if detail:
            msg = f"{msg}: {detail}"
        print(msg, flush=True)
        _odom_log_event("continuity_override", detail if detail else reason)
        _debug_odom_snapshot(f"override:{reason}")

    def _handle_map_change(new_map_id: int | None) -> bool:
        nonlocal last_map_id
        if new_map_id is None:
            return False
        if last_map_id is None:
            last_map_id = int(new_map_id)
            return False
        if int(new_map_id) == int(last_map_id):
            return False
        _apply_odom_continuity_reset(
            "map change",
            f"map_id {int(last_map_id)}->{int(new_map_id)}",
        )
        last_map_id = int(new_map_id)
        return True

    def _supports_switch_sensor() -> bool:
        return callable(getattr(slam, "SwitchSensor", None))

    def _supports_tracking_state() -> bool:
        return callable(getattr(slam, "GetTrackingState", None))

    def _supports_loc_mode() -> bool:
        return callable(getattr(slam, "ActivateLocalizationMode", None))

    def _supports_tracked_points() -> bool:
        inner = getattr(slam, "slam", None)
        if inner is not None:
            return callable(getattr(inner, "_GetTrackedMapPoints", None))
        return callable(getattr(slam, "GetTrackedMapPoints", None))

    def _supports_map_points() -> bool:
        inner = getattr(slam, "slam", None)
        if inner is not None:
            return callable(getattr(inner, "_GetMapPoints", None))
        return callable(getattr(slam, "GetMapPoints", None))

    def _supports_map_changed() -> bool:
        return callable(getattr(slam, "MapChanged", None))

    def _supports_imu_initialized() -> bool:
        return callable(getattr(slam, "IsImuInitialized", None))

    def _atlas_is_inertial() -> bool:
        try:
            return bool(slam.IsAtlasInertial())
        except Exception:
            return bool(atlas_inertial)

    def _get_imu_status_text() -> Optional[str]:
        if not current_mode.use_imu:
            return None
        if not _supports_imu_initialized():
            return None
        try:
            if not bool(slam.IsImuInitialized()):
                return "Move camera to initialize IMU"
        except Exception:
            return None
        return None

    def _select_map_points_fetch():
        if _supports_tracked_points():
            return "tracked", slam.GetTrackedMapPoints
        if _supports_map_points():
            return "map", slam.GetMapPoints
        return None, None

    def _map_active_label() -> str:
        if not map_points_enabled:
            return "map_off"
        return "map_5hz" if float(map_points_hz) >= 3.0 else "map_1hz"

    def _set_map_mode(enable: bool, hz: float | None = None, *, log: bool = True) -> None:
        nonlocal map_points_enabled, map_points_hz, map_points_period
        nonlocal map_pts, map_pts_draw, last_map_points_perf, map_button_active
        if not enable:
            map_points_enabled = False
            map_points_period = None
            map_pts = np.empty((0, 3), dtype=np.float32)
            map_pts_draw = np.empty((0, 3), dtype=np.float32)
            last_map_points_perf = 0.0
            map_button_active = _map_active_label()
            if log:
                print("[orbslam] Map points disabled.", flush=True)
            return
        if hz is None or float(hz) <= 0:
            hz = 1.0
        if not (_supports_map_points() or _supports_tracked_points()):
            if log:
                print("[orbslam] Map points unavailable: no map-point API.", flush=True)
            map_points_enabled = False
            map_points_period = None
            map_pts = np.empty((0, 3), dtype=np.float32)
            map_pts_draw = np.empty((0, 3), dtype=np.float32)
            map_button_active = _map_active_label()
            return
        source, fetch = _select_map_points_fetch()
        if fetch is None:
            if log:
                print("[orbslam] Map points unavailable: no map-point API.", flush=True)
            map_points_enabled = False
            map_points_period = None
            map_pts = np.empty((0, 3), dtype=np.float32)
            map_pts_draw = np.empty((0, 3), dtype=np.float32)
            map_button_active = _map_active_label()
            return
        map_points_enabled = True
        map_points_hz = float(hz)
        map_points_period = (1.0 / map_points_hz) if map_points_hz > 0 else None
        map_pts_draw = _sample_map_points(map_pts)
        last_map_points_perf = 0.0
        map_button_active = _map_active_label()
        if log:
            print(f"[orbslam] Map points source: {source}.", flush=True)
            print(f"[orbslam] Map points enabled @ {map_points_hz:.1f} Hz.", flush=True)

    def _camera_group(mode) -> str:
        if mode.uses_rgbd():
            return "rgbd"
        if mode.uses_stereo() or mode.uses_mono():
            return "ir"
        return "other"

    def _needs_atlas_refresh(mode) -> bool:
        if _camera_group(mode) != _camera_group(current_mode):
            return True
        return bool((not _atlas_is_inertial()) and mode.use_imu)

    def _refresh_slam(new_mode) -> None:
        nonlocal slam, atlas_inertial
        try:
            slam.Shutdown()
        except Exception:
            pass
        try:
            del slam
        except Exception:
            pass
        slam = make_slam(new_mode)
        atlas_inertial = bool(new_mode.use_imu)

    def switch_mode(new_mode, reason: str) -> None:
        nonlocal current_mode, current_mode_id, slam, handoff_restart, switch_timing, atlas_inertial
        if new_mode.key == current_mode.key:
            return
        use_restart = bool(args.mode_switch_restart) or not _supports_switch_sensor()
        if use_restart:
            restart_note = "restart" if bool(args.mode_switch_restart) else "restart (no SwitchSensor)"
            print(
                f"[orbslam] Switching mode {current_mode.label} -> {new_mode.label} ({reason}, {restart_note})",
                flush=True,
            )
            try:
                framebus.request_mode(int(mode_id_for(new_mode)))
            except Exception:
                pass
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass

            argv = [sys.executable, str(Path(__file__).resolve())]
            argv += ["--vocab", str(args.vocab)]
            argv += ["--vocab-tar", str(args.vocab_tar)]
            argv += ["--mode", str(new_mode.key)]
            argv.append("--mode-switch-restart")
            if bool(args.viewer):
                argv.append("--viewer")
            if bool(args.slam_shutdown):
                argv.append("--slam-shutdown")
            else:
                argv.append("--no-slam-shutdown")
            argv += ["--map-size", str(args.map_size)]
            argv += ["--max-map-points", str(args.max_map_points)]
            argv += ["--depth-max-m", str(args.depth_max_m)]
            if bool(args.debug):
                argv.append("--debug")
            if float(args.auto_exit_s) > 0:
                argv += ["--auto-exit-s", str(args.auto_exit_s)]
            argv += ["--framebus-name", str(args.framebus_name)]
            if not bool(args.framebus_auto_start):
                argv.append("--no-framebus-auto-start")
            if not bool(args.framebus_auto_stop):
                argv.append("--no-framebus-auto-stop")
            argv += ["--framebus-start-wait-s", str(args.framebus_start_wait_s)]
            if bool(args.framebus_force):
                argv.append("--framebus-force")
            if bool(args.framebus_fixed_streams):
                argv.append("--framebus-fixed-streams")
            else:
                argv.append("--no-framebus-fixed-streams")
            if framebus_owned:
                argv.append("--framebus-owned")
            if test_switch_interval_s > 0:
                remaining = None
                if test_switch_count > 0:
                    remaining = int(test_switch_count - test_switch_done)
                    if remaining <= 0:
                        remaining = None
                if remaining is None and test_switch_count > 0:
                    pass
                else:
                    argv += ["--test-switch-interval-s", str(test_switch_interval_s)]
                    if remaining is not None:
                        argv += ["--test-switch-count", str(remaining)]
                    if str(args.test_switch_order).strip():
                        argv += ["--test-switch-order", str(args.test_switch_order)]
                if test_log_stall_s > 0:
                    argv += ["--test-log-stall-s", str(test_log_stall_s)]
            if test_switch_log:
                argv.append("--test-switch-log")
            creationflags = 0
            if hasattr(subprocess, "CREATE_NEW_PROCESS_GROUP"):
                creationflags = subprocess.CREATE_NEW_PROCESS_GROUP
            try:
                subprocess.Popen(argv, cwd=str(PROJECT_ROOT), creationflags=creationflags)
                handoff_restart = True
            except Exception as exc:
                print(f"[orbslam] Failed to restart: {exc}", flush=True)
            raise SystemExit(0)

        print(f"[orbslam] Switching mode {current_mode.label} -> {new_mode.label} ({reason})", flush=True)
        if test_switch_log:
            map_id_before, map_count_before = _get_map_stats()
            switch_timing = {
                "start": float(time.perf_counter()),
                "from_label": str(current_mode.label),
                "to_label": str(new_mode.label),
                "traj_n": float(len(traj)),
                "map_n": float(map_pts.shape[0]),
                "map_id_before": map_id_before,
                "map_count_before": map_count_before,
            }
        if _needs_atlas_refresh(new_mode):
            print("[orbslam] Refreshing atlas for IMU switch.", flush=True)
            try:
                framebus.request_mode(int(mode_id_for(new_mode)))
            except Exception:
                pass
            _refresh_slam(new_mode)
            current_mode = new_mode
            current_mode_id = mode_id_for(current_mode)
            atlas_inertial = bool(current_mode.use_imu)
            if _supports_loc_mode():
                try:
                    if current_mode.use_imu:
                        slam.DeactivateLocalizationMode()
                        print("[orbslam] Localization-only OFF (IMU mode).", flush=True)
                    else:
                        slam.ActivateLocalizationMode()
                        print("[orbslam] Localization-only ON (non-IMU mode).", flush=True)
                except Exception:
                    pass
            if switch_timing is not None:
                switch_timing["slam_ready"] = float(time.perf_counter())
                map_id_after, map_count_after = _get_map_stats()
                if map_id_after is not None:
                    switch_timing["map_id_after"] = map_id_after
                if map_count_after is not None:
                    switch_timing["map_count_after"] = map_count_after
            reset_tracking_state("mode switch refresh", reset_slam=False, preserve_map=False, realign_on_first_pose=False)
            return
        try:
            framebus.request_mode(int(mode_id_for(new_mode)))
        except Exception:
            pass
        ok = False
        try:
            localization_only = not bool(new_mode.use_imu)
            ok = bool(slam.SwitchSensor(str(new_mode.settings), new_mode.sensor, localization_only))
        except Exception as exc:
            print(f"[orbslam] SwitchSensor failed: {exc}", flush=True)
        if not ok:
            print("[orbslam] SwitchSensor failed, staying on current mode.", flush=True)
            return
        current_mode = new_mode
        current_mode_id = mode_id_for(current_mode)
        if bool(new_mode.use_imu):
            atlas_inertial = True
        if _supports_loc_mode():
            try:
                if current_mode.use_imu:
                    slam.DeactivateLocalizationMode()
                    print("[orbslam] Localization-only OFF (IMU mode).", flush=True)
                else:
                    slam.ActivateLocalizationMode()
                    print("[orbslam] Localization-only ON (non-IMU mode).", flush=True)
            except Exception:
                pass
        if switch_timing is not None:
            switch_timing["slam_ready"] = float(time.perf_counter())
            map_id_after, map_count_after = _get_map_stats()
            if map_id_after is not None:
                switch_timing["map_id_after"] = map_id_after
            if map_count_after is not None:
                switch_timing["map_count_after"] = map_count_after
        reset_tracking_state("mode switch", reset_slam=False, preserve_map=True, realign_on_first_pose=False)

    win_map = "ORB-SLAM3 Top View"
    win_rgb = "RGB"
    win_depth = "Depth"
    cv2.namedWindow(win_map, cv2.WINDOW_NORMAL)
    cv2.namedWindow(win_rgb, cv2.WINDOW_NORMAL)
    cv2.namedWindow(win_depth, cv2.WINDOW_NORMAL)

    def _map_click_to_image(x: int, y: int) -> tuple[int, int]:
        try:
            _, _, w, h = cv2.getWindowImageRect(win_map)
            if int(w) > 0 and int(h) > 0 and (int(w) != int(map_size) or int(h) != int(map_size)):
                sx = float(map_size) / float(w)
                sy = float(map_size) / float(h)
                return int(x * sx), int(y * sy)
        except Exception:
            pass
        return int(x), int(y)

    def _build_mode_buttons() -> list[object]:
        labels = []
        if bool(args.mode_switch_restart) or not _supports_switch_sensor():
            for mode in MODE_SPECS:
                labels.append((mode.label, f"{mode.label} (Refresh Atlas)"))
            return layout_mode_buttons(size=map_size, labels=labels, button_w=260)

        for mode in MODE_SPECS:
            action = "Refresh Atlas" if _needs_atlas_refresh(mode) else "Keep Atlas"
            labels.append((mode.label, f"{mode.label} ({action})"))
        return layout_mode_buttons(size=map_size, labels=labels, button_w=260)

    mode_buttons = _build_mode_buttons()
    map_buttons = layout_bottom_right_buttons(
        size=map_size,
        labels=[
            ("map_off", "Disable Map"),
            ("map_1hz", "1Hz map"),
            ("map_5hz", "5Hz map"),
        ],
        button_w=170,
    )
    map_button_active = _map_active_label()
    pending_mode: Optional[object] = None
    pending_map_button: Optional[str] = None

    if map_points_enabled:
        _set_map_mode(True, map_points_hz, log=False)
    else:
        _set_map_mode(False, log=False)

    def on_mouse(event, x, y, _flags, _param) -> None:
        nonlocal pending_mode, pending_map_button
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        ix, iy = _map_click_to_image(int(x), int(y))
        for label, _display, (x1, y1, x2, y2) in map_buttons:
            if x1 <= ix <= x2 and y1 <= iy <= y2:
                pending_map_button = str(label)
                return
        for label, _display, (x1, y1, x2, y2) in mode_buttons:
            if x1 <= ix <= x2 and y1 <= iy <= y2:
                pending_mode = MODE_BY_LABEL.get(label)
                return

    cv2.setMouseCallback(win_map, on_mouse)

    print(
        "[orbslam] Running. Press 'q' to quit, 'r' to reset, 'm' to cycle map overlay (off/1Hz/5Hz). "
        "Click mode/map buttons to switch."
    )
    auto_exit_s = float(args.auto_exit_s)
    t_start = time.time()
    if auto_exit_s > 0:
        print(f"[orbslam] Auto-exit after {auto_exit_s:.1f}s")
    t_last_stat = 0.0

    rgb_na = np.zeros((framebus_cfg.height, framebus_cfg.width, 3), dtype=np.uint8)
    cv2.putText(rgb_na, "RGB N/A", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2)
    depth_na = np.zeros((framebus_cfg.height, framebus_cfg.width, 3), dtype=np.uint8)
    cv2.putText(depth_na, "Depth N/A", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2)

    def _render_and_show(
        rgb_img: np.ndarray | None,
        depth_img: np.ndarray | None,
        tracking_state: int,
        prev_state: int,
        camera_fps: float | None,
        slam_fps: float | None,
        status_text: str | None,
    ) -> None:
        traj_arr = np.asarray(traj, dtype=np.float32).reshape(-1, 3) if traj else np.empty((0, 3), dtype=np.float32)
        nonlocal mode_buttons
        mode_buttons = _build_mode_buttons()
        px4_age_s = None
        px4_status = None
        if px4_odom_enabled:
            if px4_odom_last_ts is not None:
                try:
                    px4_age_s = float(time.time() - float(px4_odom_last_ts))
                except Exception:
                    px4_age_s = None
            else:
                px4_status = "PX4 ODOM: waiting for alignment"
        map_img = render_top_view(
            traj_xyz=traj_arr,
            map_xyz=map_pts_draw if map_points_enabled else np.empty((0, 3), dtype=np.float32),
            Twc=last_Twc,
            tracking_state=int(tracking_state),
            prev_tracking_state=int(prev_state),
            camera_fps=camera_fps,
            slam_fps=slam_fps,
            odom_delta_xyz=tuple(float(v) for v in odom_display_xyz) if odom_display_xyz is not None else None,
            odom_rpy_deg=rpy_deg,
            odom_frame=odom_frame_label,
            px4_odom_xyz=tuple(float(v) for v in px4_odom_last_pos) if px4_odom_last_pos is not None else None,
            px4_odom_vel_xyz=tuple(float(v) for v in px4_odom_last_vel) if px4_odom_last_vel is not None else None,
            px4_odom_rpy_deg=px4_odom_last_rpy,
            px4_odom_age_s=px4_age_s,
            px4_odom_status=px4_status,
            map_points_enabled=bool(map_points_enabled),
            map_points_count=int(map_pts.shape[0]),
            map_points_hz=float(map_points_hz) if map_points_enabled else None,
            size=map_size,
            max_draw_points=map_draw_points_max,
            buttons=mode_buttons,
            active_button=current_mode.label,
            map_buttons=map_buttons,
            map_active_button=map_button_active,
            status_text=status_text,
        )
        cv2.imshow(win_map, map_img)

        if rgb_img is not None:
            bgr = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
            cv2.imshow(win_rgb, bgr)
        else:
            cv2.imshow(win_rgb, rgb_na)

        if current_mode.uses_rgbd() and depth_img is not None:
            depth_color = _colorize_depth(depth_img, max_depth_m=depth_max_m)
            cv2.imshow(win_depth, depth_color)
        else:
            cv2.imshow(win_depth, depth_na)

    try:
        while True:
            now = time.time()
            now_perf = time.perf_counter()
            imu_status_text = _get_imu_status_text()
            align_status_text = None
            if R_ned_body_ref is not None and R_ned_world is None:
                align_status_text = "Aligning to NED"
            status_text = imu_status_text if imu_status_text is not None else align_status_text
            if auto_exit_s > 0:
                if float(now - t_start) >= float(auto_exit_s):
                    print("[orbslam] Auto-exit")
                    break

            if test_switch_interval_s > 0 and next_test_switch_t is not None and pending_mode is None:
                if float(now) >= float(next_test_switch_t):
                    pending_mode = _next_test_mode()
                    test_switch_done = int(test_switch_done + 1)
                    if test_switch_count > 0 and int(test_switch_done) >= int(test_switch_count):
                        test_switch_interval_s = 0.0
                        next_test_switch_t = None
                    else:
                        next_test_switch_t = float(now + test_switch_interval_s)

            if pending_map_button is not None:
                if pending_map_button == "map_off":
                    _set_map_mode(False)
                elif pending_map_button == "map_5hz":
                    _set_map_mode(True, 5.0)
                else:
                    _set_map_mode(True, 1.0)
                pending_map_button = None

            if pending_mode is not None:
                switch_mode(pending_mode, "ui")
                pending_mode = None
                continue

            if current_mode.use_imu:
                gyro, accel = framebus.drain_imu()
                for t_s, gx, gy, gz in gyro:
                    imu.add_gyro(t_s=float(t_s), gx=float(gx), gy=float(gy), gz=float(gz))
                for t_s, ax, ay, az in accel:
                    imu.add_accel(t_s=float(t_s), ax=float(ax), ay=float(ay), az=float(az))

            ui_frame = framebus.read_latest(
                last_seq=last_frame_seq_any,
                required_flags=FLAG_RGB,
                mode_id=int(current_mode_id),
                copy=True,
            )
            if ui_frame is None:
                ui_frame = framebus.read_latest(
                    last_seq=last_frame_seq_any,
                    required_flags=FLAG_RGB,
                    mode_id=None,
                    copy=True,
                )
            if ui_frame is not None:
                last_frame_seq_any = int(ui_frame.seq)
                last_ui_wall_s = float(now)
                if ui_frame.rgb is not None:
                    last_rgb = ui_frame.rgb
                if current_mode.uses_rgbd() and ui_frame.depth is not None:
                    last_depth = ui_frame.depth
                elif not current_mode.uses_rgbd():
                    last_depth = None

            track_frame = framebus.read_latest(
                last_seq=last_frame_seq_track,
                required_flags=_required_flags_for_mode(current_mode),
                mode_id=int(current_mode_id),
                copy=True,
            )
            if track_frame is None:
                # Allow frames tagged with a different mode during mode-switch handoff.
                track_frame = framebus.read_latest(
                    last_seq=last_frame_seq_track,
                    required_flags=_required_flags_for_mode(current_mode),
                    mode_id=None,
                    copy=True,
                )

            stall_reason = None
            imu_batch = np.empty((0, 7), dtype=np.float64)

            if track_frame is None:
                stall_reason = "waiting for tracking frames"
                _render_and_show(
                    last_rgb,
                    last_depth,
                    last_tracking_state,
                    prev_tracking_state,
                    cam_fps,
                    slam_fps,
                    status_text,
                )
            else:
                last_frame_seq_track = int(track_frame.seq)
                ts = float(track_frame.ts_s)
                if not current_mode.use_imu:
                    seq = int(track_frame.seq)
                    if non_imu_seq_base is None or (non_imu_last_seq is not None and seq <= non_imu_last_seq):
                        non_imu_seq_base = int(seq)
                        last_img_ts = None
                    non_imu_last_seq = int(seq)
                    ts = float(seq - int(non_imu_seq_base)) / float(non_imu_ts_fps)

                if track_frame.rgb is not None:
                    last_rgb = track_frame.rgb
                if current_mode.uses_rgbd() and track_frame.depth is not None:
                    last_depth = track_frame.depth
                elif not current_mode.uses_rgbd():
                    last_depth = None

                if last_img_ts is None:
                    last_img_ts = float(ts)
                    stall_reason = "waiting for second frame"
                elif float(ts) <= float(last_img_ts):
                    if current_mode.use_imu:
                        imu.reset()
                    last_img_ts = None
                    stall_reason = "timestamp reset"

                if stall_reason is None and current_mode.use_imu:
                    _gmin, gmax = imu.gyro_time_range()
                    if gmax is None or float(gmax) < float(ts):
                        stall_reason = "waiting for IMU"
                    else:
                        imu_batch = imu.pop_batch(t0_s=float(last_img_ts), t1_s=float(ts), min_samples=2)
                        if imu_batch.size == 0:
                            stall_reason = "waiting for IMU"

                if stall_reason is not None:
                    _render_and_show(
                        last_rgb,
                        last_depth,
                        last_tracking_state,
                        prev_tracking_state,
                        cam_fps,
                        slam_fps,
                        status_text,
                    )
                else:
                    last_img_ts = float(ts)

                    rgb = track_frame.rgb
                    depth_raw = track_frame.depth
                    ir_left = track_frame.ir_left
                    ir_right = track_frame.ir_right

                    if debug:
                        track_count += 1
                        if track_count <= 3:
                            parts = [f"[orbslam] Track{current_mode.label}#{track_count} ts={ts:.6f}"]
                            if rgb is not None:
                                parts.append(f"rgb={tuple(rgb.shape)} {rgb.dtype}")
                            if depth_raw is not None:
                                parts.append(f"depth={tuple(depth_raw.shape)} {depth_raw.dtype}")
                            if ir_left is not None and ir_right is not None:
                                parts.append(f"left={tuple(ir_left.shape)} {ir_left.dtype}")
                                parts.append(f"right={tuple(ir_right.shape)} {ir_right.dtype}")
                            elif ir_left is not None:
                                parts.append(f"mono={tuple(ir_left.shape)} {ir_left.dtype}")
                            if current_mode.use_imu:
                                parts.append(f"imu={tuple(imu_batch.shape)} {imu_batch.dtype}")
                            print(" ".join(parts), flush=True)

                    if current_mode.uses_rgbd():
                        Tcw = slam.TrackRGBD(rgb, depth_raw, float(ts), imu_batch if current_mode.use_imu else None)
                    elif current_mode.uses_stereo():
                        Tcw = slam.TrackStereo(ir_left, ir_right, float(ts), imu_batch if current_mode.use_imu else None)
                    else:
                        Tcw = slam.TrackMonocular(ir_left, float(ts), imu_batch if current_mode.use_imu else None)
                    if switch_timing is not None and "first_track" not in switch_timing:
                        switch_timing["first_track"] = float(time.perf_counter())
                    if debug and track_count <= 3:
                        print(f"[orbslam] Track{current_mode.label}#{track_count} returned", flush=True)
                    Tcw = np.asarray(Tcw, dtype=np.float64).reshape(4, 4)
                    Twc = None
                    if np.all(np.isfinite(Tcw)):
                        try:
                            Twc = np.linalg.inv(Tcw)
                            if display_align is None:
                                if pending_display_anchor is not None and np.all(np.isfinite(pending_display_anchor)):
                                    display_align = pending_display_anchor @ np.linalg.inv(Twc)
                                else:
                                    display_align = np.eye(4, dtype=np.float64)
                                pending_display_anchor = None
                            if display_align is not None:
                                Twc = display_align @ Twc
                                last_Twc = Twc.copy()
                                traj.append(Twc[:3, 3].astype(np.float32, copy=False))
                                if switch_timing is not None and "first_pose" not in switch_timing:
                                    switch_timing["first_pose"] = float(time.perf_counter())
                        except Exception:
                            Twc = last_Twc
                    else:
                        Twc = last_Twc

                    if last_Twc is not None and R_ned_body_ref is not None:
                        try:
                            R_wc = np.asarray(last_Twc[:3, :3], dtype=np.float64)
                            if R_ned_world is None:
                                R_ned_world = R_ned_body_ref @ R_body_cam @ R_wc.T
                                odom_waiting_align_log = False
                            R_ned_body = R_ned_world @ R_wc @ R_body_cam.T
                            rpy_deg = _rpy_from_ned(R_ned_body)
                            odom_ned_q = _quat_from_R(R_ned_body)
                            odom_frame_label = "NED"
                        except Exception:
                            rpy_deg = None
                            odom_ned_q = None

                    if _supports_tracking_state():
                        try:
                            state = int(slam.GetTrackingState())
                        except Exception:
                            state = -1
                    else:
                        try:
                            if bool(slam.isLost()):
                                state = 4
                            elif last_Twc is not None:
                                state = 2
                            else:
                                state = 1
                        except Exception:
                            state = -1
                    state_changed = int(state) != int(last_tracking_state)
                    if state_changed:
                        prev_tracking_state = int(last_tracking_state)
                        try:
                            prev_name = str(tracking_state_name(int(prev_tracking_state)))
                            now_name = str(tracking_state_name(int(state)))
                        except Exception:
                            prev_name = str(int(prev_tracking_state))
                            now_name = str(int(state))
                        print(f"[orbslam] Tracking state {prev_name} -> {now_name}", flush=True)
                        if int(state) in (1, 3, 4):
                            map_id_dbg, map_count_dbg = _get_map_stats()
                            mid = "?" if map_id_dbg is None else str(int(map_id_dbg))
                            mcount = "?" if map_count_dbg is None else str(int(map_count_dbg))
                            print(
                                f"[orbslam] Tracking reset suspected (state transition). map_id={mid} map_count={mcount}",
                                flush=True,
                            )
                    last_tracking_state = int(state)

                    imu_initialized_now = True
                    if current_mode.use_imu and _supports_imu_initialized():
                        try:
                            imu_initialized_now = bool(slam.IsImuInitialized())
                        except Exception:
                            imu_initialized_now = True
                    imu_initialized_flag = imu_initialized_now
                    if current_mode.use_imu and _supports_imu_initialized():
                        if not imu_initialized_now:
                            if debug_odom:
                                _debug_odom_snapshot("imu_not_initialized")
                        elif debug_odom:
                            _debug_odom_snapshot("imu_initialized")

                    now_track = float(time.time())
                    now_track_perf = float(time.perf_counter())
                    map_changed = False
                    if (now_track_perf - last_map_id_check_perf) >= map_id_check_period:
                        last_map_id_check_perf = float(now_track_perf)
                        try:
                            map_id_now, map_count_now = _get_map_stats()
                        except Exception:
                            map_id_now = None
                            map_count_now = None
                        if map_count_now is not None:
                            if last_map_count is not None and int(map_count_now) != int(last_map_count):
                                print(
                                    f"[orbslam] Map count changed {int(last_map_count)} -> {int(map_count_now)}",
                                    flush=True,
                                )
                            last_map_count = int(map_count_now)
                        map_changed = _handle_map_change(map_id_now)
                        if map_changed:
                            _debug_odom_snapshot("map_change")
                    if not map_changed and _supports_map_changed():
                        try:
                            if bool(slam.MapChanged()):
                                _odom_log_event("map_changed", "MapChanged")
                        except Exception:
                            pass
                    if not map_changed and odom_prev_pose is not None and last_Twc is not None:
                        try:
                            delta_check = np.linalg.inv(odom_prev_pose) @ last_Twc
                            trans_norm = float(np.linalg.norm(delta_check[:3, 3]))
                            rot_deg = float(_rot_angle_deg(delta_check[:3, :3]))
                            jump_state = bool(state_changed) and int(prev_tracking_state) in (-1, 0, 1, 3, 4)
                            if jump_state and (trans_norm > odom_jump_trans_m or rot_deg > odom_jump_rot_deg):
                                _apply_odom_continuity_reset(
                                    "pose jump",
                                    f"d={trans_norm:.2f}m rot={rot_deg:.1f}deg",
                                )
                                map_changed = True
                                _debug_odom_snapshot("pose_jump")
                        except Exception:
                            pass

                    if last_Twc is not None and int(state) in (2, 5):
                        try:
                            if map_changed:
                                odom_prev_pose = last_Twc.copy()
                                odom_pose = np.eye(4, dtype=np.float64)
                                odom_global_pose = odom_offset.copy()
                                odom_delta_xyz = odom_pose[:3, 3].copy()
                                if odom_global_pose is not None:
                                    odom_pos = odom_global_pose[:3, 3].astype(np.float64, copy=False)
                                    if R_ned_world is not None:
                                        odom_display_xyz = (R_ned_world @ odom_pos.reshape(3)).astype(
                                            np.float64, copy=False
                                        )
                                        odom_frame_label = "NED"
                                    else:
                                        odom_display_xyz = odom_pos.copy()
                                        odom_frame_label = "MAP"
                                raise StopIteration
                            if odom_prev_pose is None:
                                odom_prev_pose = last_Twc.copy()
                                odom_pose = np.eye(4, dtype=np.float64)
                                odom_delta_xyz = odom_pose[:3, 3].copy()
                            else:
                                delta = np.linalg.inv(odom_prev_pose) @ last_Twc
                                odom_pose = odom_pose @ delta
                                odom_prev_pose = last_Twc.copy()
                                odom_delta_xyz = odom_pose[:3, 3].copy()
                            if odom_pose is not None:
                                odom_global_prev = odom_global_pose.copy()
                                odom_global_pose = odom_offset @ odom_pose
                                odom_pos = odom_global_pose[:3, 3].astype(np.float64, copy=False)
                                if R_ned_world is not None:
                                    odom_display_xyz = (R_ned_world @ odom_pos.reshape(3)).astype(
                                        np.float64, copy=False
                                    )
                                    odom_frame_label = "NED"
                                else:
                                    odom_display_xyz = odom_pos.copy()
                                    odom_frame_label = "MAP"
                                try:
                                    norm_now = float(np.linalg.norm(odom_display_xyz))
                                    prev_norm = last_odom_norm
                                    if debug_odom:
                                        if prev_norm is not None and prev_norm > 0.2 and norm_now < 0.05:
                                            print(
                                                f"[odom][dbg] display norm drop {prev_norm:.2f} -> {norm_now:.2f}",
                                                flush=True,
                                            )
                                            _debug_odom_snapshot("norm_drop")
                                        if debug_odom_hz > 0:
                                            period = 1.0 / debug_odom_hz
                                            if (now_track_perf - last_odom_debug_perf) >= period:
                                                last_odom_debug_perf = float(now_track_perf)
                                                _debug_odom_snapshot("periodic")
                                    if (
                                        prev_norm is not None
                                        and prev_norm > float(odom_collapse_hi_m)
                                        and norm_now < float(odom_collapse_lo_m)
                                    ):
                                        _apply_odom_offset_override(
                                            "collapse",
                                            f"{prev_norm:.2f} -> {norm_now:.2f}",
                                            odom_global_prev,
                                        )
                                        odom_display_xyz = odom_global_pose[:3, 3].copy()
                                        norm_now = float(np.linalg.norm(odom_display_xyz))
                                    last_odom_norm = norm_now
                                except Exception:
                                    pass
                        except StopIteration:
                            pass
                        except Exception:
                                odom_prev_pose = last_Twc.copy()
                    if odom_log_fp is not None and odom_log_period is not None:
                        if (now_track_perf - last_odom_log_perf) >= odom_log_period:
                            last_odom_log_perf = float(now_track_perf)
                            _odom_log_event("tick", f"map_changed={int(map_changed)}", now_s=now_track)
                    slam_time_samples.append(now_track_perf)
                    while slam_time_samples and (now_track_perf - slam_time_samples[0]) > 1.0:
                        slam_time_samples.popleft()
                    if len(slam_time_samples) >= 2:
                        dt = float(now_track_perf - slam_time_samples[0])
                        if dt > 0:
                            slam_fps = float(len(slam_time_samples) - 1) / dt
                    last_track_wall_s = float(now_track)
                    last_imu_count = int(imu_batch.shape[0]) if current_mode.use_imu else 0

                    if (
                        px4_odom_enabled
                        and mav_bridge is not None
                        and odom_send_period is not None
                        and last_Twc is not None
                        and int(state) in (2, 5)
                    ):
                        if R_ned_world is None or odom_ned_q is None:
                            if not odom_waiting_align_log:
                                print("[orbslam] PX4 odom waiting for NED alignment.", flush=True)
                                odom_waiting_align_log = True
                        elif odom_pose is not None:
                            if (now_track_perf - odom_last_send_perf) >= odom_send_period:
                                pos_world = odom_global_pose[:3, 3].astype(np.float64, copy=False)
                                pos_ned = (R_ned_world @ pos_world.reshape(3)).astype(np.float64, copy=False)
                                ts_s = float(now_track)
                                usec = int(ts_s * 1_000_000)
                                vx = vy = vz = float("nan")
                                if odom_prev_ned is not None and odom_prev_ned_ts is not None:
                                    dt = float(ts_s - odom_prev_ned_ts)
                                    if dt > 1e-3:
                                        vel = (pos_ned - odom_prev_ned) / dt
                                        vx = float(vel[0])
                                        vy = float(vel[1])
                                        vz = float(vel[2])
                                mav_bridge.send_odometry(
                                    {
                                        "usec": usec,
                                        "x": float(pos_ned[0]),
                                        "y": float(pos_ned[1]),
                                        "z": float(pos_ned[2]),
                                        "q": list(odom_ned_q),
                                        "vx": vx,
                                        "vy": vy,
                                        "vz": vz,
                                        "pose_cov": odom_pose_cov,
                                        "vel_cov": odom_vel_cov,
                                        "reset_counter": int(odom_reset_counter),
                                    }
                                )
                                try:
                                    px4_odom_last_pos = pos_ned.astype(np.float64, copy=True)
                                except Exception:
                                    px4_odom_last_pos = None
                                try:
                                    if math.isfinite(vx) and math.isfinite(vy) and math.isfinite(vz):
                                        px4_odom_last_vel = np.array([vx, vy, vz], dtype=np.float64)
                                    else:
                                        px4_odom_last_vel = None
                                except Exception:
                                    px4_odom_last_vel = None
                                px4_odom_last_rpy = rpy_deg if rpy_deg is not None else None
                                px4_odom_last_ts = ts_s
                                odom_prev_ned = pos_ned
                                odom_prev_ned_ts = ts_s
                                odom_last_send_perf = float(now_track_perf)

                    pts = np.empty((0, 3), dtype=np.float32)
                    source = None
                    if map_points_enabled and map_points_period is not None:
                        if (now_track_perf - last_map_points_perf) >= map_points_period:
                            last_map_points_perf = float(now_track_perf)
                            try:
                                source, fetch = _select_map_points_fetch()
                                if fetch is not None:
                                    pts = fetch()
                                    pts = np.asarray(pts, dtype=np.float32).reshape(-1, 3)
                                    last_map_points_source = source
                            except Exception:
                                pts = np.empty((0, 3), dtype=np.float32)
                            if pts.size == 0 and source == "map" and _supports_tracked_points():
                                try:
                                    pts = slam.GetTrackedMapPoints()
                                    pts = np.asarray(pts, dtype=np.float32).reshape(-1, 3)
                                    source = "tracked"
                                    last_map_points_source = source
                                except Exception:
                                    pts = np.empty((0, 3), dtype=np.float32)

                    if source is not None:
                        if pts.size > 0:
                            if display_align is not None:
                                R = display_align[:3, :3].astype(np.float32, copy=False)
                                t = display_align[:3, 3].astype(np.float32, copy=False)
                                pts = (pts @ R.T) + t
                            map_pts = pts
                            map_pts_draw = _sample_map_points(map_pts)
                        else:
                            map_pts = np.empty((0, 3), dtype=np.float32)
                            map_pts_draw = map_pts

                    _render_and_show(
                        rgb if rgb is not None else last_rgb,
                        depth_raw,
                        last_tracking_state,
                        prev_tracking_state,
                        cam_fps,
                        slam_fps,
                        status_text,
                    )

            k = int(cv2.waitKey(1) & 0xFF)
            if k in (27, ord("q")):
                break
            if k == ord("r"):
                reset_tracking_state("user reset")
            if k == ord("m"):
                current_label = _map_active_label()
                if current_label == "map_off":
                    _set_map_mode(True, 1.0)
                elif current_label == "map_1hz":
                    _set_map_mode(True, 5.0)
                else:
                    _set_map_mode(False)

            now = time.time()
            if stall_reason and test_log_stall_s > 0:
                if (now - last_track_wall_s) >= test_log_stall_s and (now - last_stall_log_s) >= test_log_stall_s:
                    try:
                        status = framebus.status()
                        print(
                            "[orbslam][stall] "
                            f"{stall_reason} mode={current_mode.label} "
                            f"active={status['active_mode']} req={status['request_mode']} "
                            f"seq={status['frame_seq']} gyro={status['gyro_count']} accel={status['accel_count']}",
                            flush=True,
                        )
                    except Exception:
                        print(f"[orbslam][stall] {stall_reason}", flush=True)
                    last_stall_log_s = float(now)

            if (now_perf - last_cam_poll_perf) >= 0.2:
                try:
                    status = framebus.status()
                    seq = int(status["frame_seq"])
                    if switch_timing is not None and "framebus_active" not in switch_timing:
                        if int(status["active_mode"]) == int(current_mode_id):
                            switch_timing["framebus_active"] = float(now_perf)
                    if last_cam_seq is None or seq < int(last_cam_seq):
                        last_cam_seq = int(seq)
                        last_cam_perf = float(now_perf)
                        cam_fps = None
                    else:
                        if last_cam_perf is not None:
                            dt = float(now_perf - last_cam_perf)
                            delta = int(seq) - int(last_cam_seq)
                            if dt > 0 and delta > 0:
                                inst_fps = float(delta) / dt
                                if cam_fps is None:
                                    cam_fps = inst_fps
                                else:
                                    cam_fps = (cam_fps * 0.8) + (inst_fps * 0.2)
                        last_cam_seq = int(seq)
                        last_cam_perf = float(now_perf)
                    last_cam_poll_perf = float(now_perf)
                except Exception:
                    last_cam_poll_perf = float(now_perf)

            if switch_timing is not None and "first_pose" in switch_timing:
                start = float(switch_timing.get("start", now_perf))
                parts = [
                    "[orbslam][switch]",
                    f"{switch_timing.get('from_label', '?')}->{switch_timing.get('to_label', '?')}",
                ]
                if "framebus_active" in switch_timing:
                    parts.append(f"fb={switch_timing['framebus_active'] - start:.2f}s")
                if "slam_ready" in switch_timing:
                    parts.append(f"slam_init={switch_timing['slam_ready'] - start:.2f}s")
                if "first_track" in switch_timing:
                    parts.append(f"first_track={switch_timing['first_track'] - start:.2f}s")
                parts.append(f"first_pose={switch_timing['first_pose'] - start:.2f}s")
                map_id_before = switch_timing.get("map_id_before")
                map_id_after = switch_timing.get("map_id_after", map_id_before)
                if map_id_before is not None or map_id_after is not None:
                    before_txt = "?" if map_id_before is None else str(int(map_id_before))
                    after_txt = "?" if map_id_after is None else str(int(map_id_after))
                    parts.append(f"map_id={before_txt}->{after_txt}")
                map_count_before = switch_timing.get("map_count_before")
                map_count_after = switch_timing.get("map_count_after", map_count_before)
                if map_count_before is not None or map_count_after is not None:
                    before_txt = "?" if map_count_before is None else str(int(map_count_before))
                    after_txt = "?" if map_count_after is None else str(int(map_count_after))
                    parts.append(f"maps={before_txt}->{after_txt}")
                parts.append(
                    f"map_keep={int(switch_timing.get('map_n', 0))} traj_keep={int(switch_timing.get('traj_n', 0))}"
                )
                print(" ".join(parts), flush=True)
                switch_timing = None

            if (now - t_last_stat) >= 1.0:
                t_last_stat = now
                try:
                    imu_n = int(last_imu_count) if current_mode.use_imu else 0
                    msg = (
                        f"[orbslam] state={last_tracking_state} "
                        f"traj_n={len(traj)} map_n={int(map_pts.shape[0])} imu_n={imu_n}"
                    )
                    if bool(args.test_log_fps):
                        cam_txt = "?" if cam_fps is None else f"{cam_fps:.1f}"
                        slam_txt = "?" if slam_fps is None else f"{slam_fps:.1f}"
                        msg = f"{msg} cam_fps={cam_txt} slam_fps={slam_txt}"
                    print(msg)
                except Exception:
                    pass

            if stall_reason and track_frame is None:
                time.sleep(0.001)

    finally:
        if not handoff_restart and bool(args.slam_shutdown):
            try:
                slam.Shutdown()
            except Exception:
                pass
        if mav_bridge is not None:
            try:
                mav_bridge.stop()
            except Exception:
                pass
        if odom_log_fp is not None:
            try:
                odom_log_fp.close()
            except Exception:
                pass
        if bool(args.framebus_auto_stop) and (framebus_proc is not None or framebus_owned) and not handoff_restart:
            try:
                framebus.request_stop()
            except Exception:
                pass
            if framebus_proc is not None:
                try:
                    framebus_proc.wait(timeout=5)
                except Exception:
                    try:
                        framebus_proc.terminate()
                        framebus_proc.wait(timeout=5)
                    except Exception:
                        try:
                            framebus_proc.kill()
                        except Exception:
                            pass
        try:
            framebus.close()
        except Exception:
            pass
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
