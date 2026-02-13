#!/usr/bin/env python3
# mavlink_bridge.py — MAVLink I/O + RC normalization + gates for PID & MANUAL_CONTROL
# RC channels are normalized to [-1.0 .. +1.0] and exposed via self.rc_norm (1-based indexing).
# Gates use normalized thresholds from YAML:
#   pid.remote_gate.{require_remote_enable, ch|channel, threshold}
#   mav_control.rc_gate.{require_remote_enable, channel, threshold}

import threading
import time
import math
from typing import Any, Dict, Optional, Tuple
import socket
import errno
from collections import defaultdict, deque

from pymavlink import mavutil  # type: ignore
from orbslam_app.udp import parse_hostport


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else (hi if v > hi else v)


def _norm_rc_value(val: Any) -> float:
    """Normalize a variety of RC encodings to [-1..+1].
    Accepts:
      - PWM 1000..2000          -> (v-1500)/500
      - scaled -1000..+1000     -> v/1000
      - scaled 0..1000          -> (v-500)/500
      - already normalized -1..+1
    Returns 0.0 if input invalid.
    """
    try:
        v = float(val)
    except Exception:
        return 0.0

    # PWM takes precedence when in 800..2200 range
    if 800.0 <= v <= 2200.0:
        return _clamp((v - 1500.0) / 500.0, -1.0, 1.0)
    # Scaled ranges
    if -1.5 <= v <= 1.5:
        return _clamp(v, -1.0, 1.0)
    if -1500.0 <= v <= 1500.0:
        return _clamp(v / 1000.0, -1.0, 1.0)
    if -100.0 <= v <= 1100.0:
        return _clamp((v - 500.0) / 500.0, -1.0, 1.0)
    return _clamp(v, -1.0, 1.0)


class MavlinkBridge:
    """Thin wrapper around pymavlink for RX (RC channels) and TX (MANUAL_CONTROL, vision)."""
    def __init__(self, cfg, csvlog=None, print_fn=print):
        self.cfg = cfg if isinstance(cfg, dict) else {}
        self.csvlog = csvlog
        self.print = print_fn

        px4_cfg = self.cfg.get("px4", {}) or {}
        self.port = px4_cfg.get("serial", "/dev/ttyUSB0")
        self.baud = int(px4_cfg.get("baud", 115200))
        # IMPORTANT: pymavlink defaults `source_component=0` (reserved) unless explicitly set.
        # Many autopilots/inspectors will ignore or hide messages with compid=0, so we set a sane default.
        self.source_system = int(px4_cfg.get("source_system", 1))
        self.source_component = int(px4_cfg.get("source_component", getattr(mavutil.mavlink, "MAV_COMP_ID_VISUAL_INERTIAL_ODOMETRY", 197)))

        hb_cfg = px4_cfg.get("heartbeat", {}) or {}
        self._hb_enabled = bool(hb_cfg.get("enabled", True))
        self._hb_rate_hz = float(hb_cfg.get("rate_hz", 1.0))
        self._hb_type = int(hb_cfg.get("type", getattr(mavutil.mavlink, "MAV_TYPE_ONBOARD_CONTROLLER", 18)))
        self._hb_autopilot = int(hb_cfg.get("autopilot", getattr(mavutil.mavlink, "MAV_AUTOPILOT_INVALID", 8)))
        self._hb_base_mode = int(hb_cfg.get("base_mode", 0))
        self._hb_custom_mode = int(hb_cfg.get("custom_mode", 0))
        self._hb_system_status = int(hb_cfg.get("system_status", getattr(mavutil.mavlink, "MAV_STATE_ACTIVE", 4)))

        status_cfg = px4_cfg.get("status", {}) or {}
        self._status_enabled = bool(status_cfg.get("enabled", False))
        self._status_rate_hz = float(status_cfg.get("rate_hz", 1.0))

        self._master: Optional["mavutil.mavlink_connection"] = None
        self._rx_thread: Optional[threading.Thread] = None
        self._mc_thread: Optional[threading.Thread] = None
        self._hb_thread: Optional[threading.Thread] = None
        self._status_thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._tx_lock = threading.Lock()

        # RC normalized values (1-based indexing)
        self.rc_norm: Dict[int, float] = {}

        # Mode string (if we parse it)
        self._mode_name: str = ""

        # Gates (will be recomputed upon RC updates)
        self._pid_gate_on: Optional[bool] = None
        self._mc_gate_on: Optional[bool] = None

        # Optional MAVLink mirror (raw UDP forward)
        self._mirror_sock: Optional[socket.socket] = None
        self._mirror_addr: Optional[Tuple[str, int]] = None
        self._mirror_drops: int = 0

        # MANUAL_CONTROL TX diagnostics (rate-limited console logging)
        self._mc_tx_errs: int = 0
        self._mc_tx_last_err_t: float = 0.0
        self._hb_tx_errs: int = 0
        self._hb_tx_last_err_t: float = 0.0
        self._vpe_tx_errs: int = 0
        self._vpe_tx_last_err_t: float = 0.0
        self._vse_tx_errs: int = 0
        self._vse_tx_last_err_t: float = 0.0
        self._odom_tx_errs: int = 0
        self._odom_tx_last_err_t: float = 0.0

        # Basic RX/TX counters for debugging.
        self._rx_total: int = 0
        self._rx_by_type: Dict[str, int] = defaultdict(int)
        self._rx_last_t: float = 0.0
        self._rx_last_type: str = ""
        self._rx_last_hb_t: float = 0.0
        self._rx_last_hb_sysid: int = 0
        self._rx_last_hb_compid: int = 0

        # RAW_IMU / HIGHRES_IMU buffers (for calibration).
        self._imu_lock = threading.Lock()
        self._accel_samples = deque(maxlen=800)
        self._mag_samples = deque(maxlen=800)
        self._accel_latest: Optional[Tuple[float, float, float]] = None
        self._mag_latest: Optional[Tuple[float, float, float]] = None

        self._tx_total: int = 0
        self._tx_by_type: Dict[str, int] = defaultdict(int)
        self._tx_last_t: Dict[str, float] = {}
        self._open_error: Optional[str] = None

    def _record_tx(self, name: str) -> None:
        try:
            key = str(name)
            self._tx_total += 1
            self._tx_by_type[key] = int(self._tx_by_type.get(key, 0)) + 1
            self._tx_last_t[key] = float(time.time())
        except Exception:
            return

    def _status_snapshot(self) -> dict:
        m = self._master
        now = float(time.time())
        if m is None:
            return {"ok": False, "port": str(self.port), "baud": int(self.baud)}
        try:
            tgt_sys = int(getattr(m, "target_system", 0) or 0)
        except Exception:
            tgt_sys = 0
        try:
            tgt_comp = int(getattr(m, "target_component", 0) or 0)
        except Exception:
            tgt_comp = 0

        age_rx = (now - float(self._rx_last_t)) if self._rx_last_t else float("inf")
        age_hb = (now - float(self._rx_last_hb_t)) if self._rx_last_hb_t else float("inf")
        try:
            src_sys = int(getattr(m.mav, "srcSystem", int(self.source_system)))
        except Exception:
            src_sys = int(self.source_system)
        try:
            src_comp = int(getattr(m.mav, "srcComponent", int(self.source_component)))
        except Exception:
            src_comp = int(self.source_component)
        return {
            "ok": True,
            "port": str(self.port),
            "baud": int(self.baud),
            "src": (int(src_sys), int(src_comp)),
            "tgt": (int(tgt_sys), int(tgt_comp)),
            "rx_total": int(self._rx_total),
            "rx_last_type": str(self._rx_last_type),
            "rx_age_s": float(age_rx),
            "hb_age_s": float(age_hb),
            "hb_last_src": (int(self._rx_last_hb_sysid), int(self._rx_last_hb_compid)),
            "tx_total": int(self._tx_total),
            "tx_hb": int(self._tx_by_type.get("HEARTBEAT", 0)),
            "tx_vse": int(self._tx_by_type.get("VISION_SPEED_ESTIMATE", 0)),
            "tx_vpe": int(self._tx_by_type.get("VISION_POSITION_ESTIMATE", 0)),
            "tx_odom": int(self._tx_by_type.get("ODOMETRY", 0)),
            "tx_err_hb": int(self._hb_tx_errs),
            "tx_err_vse": int(self._vse_tx_errs),
            "tx_err_vpe": int(self._vpe_tx_errs),
            "tx_err_odom": int(self._odom_tx_errs),
        }

    def print_status_once(self) -> None:
        try:
            s = self._status_snapshot()
            if not bool(s.get("ok", False)):
                self.print(f"[MAV-STAT] DISCONNECTED port={s.get('port')} baud={s.get('baud')}", flush=True)
                return
            src_sys, src_comp = s.get("src", (0, 0))
            tgt_sys, tgt_comp = s.get("tgt", (0, 0))
            hb_sys, hb_comp = s.get("hb_last_src", (0, 0))
            rx_age = float(s.get("rx_age_s", 0.0))
            hb_age = float(s.get("hb_age_s", 0.0))
            self.print(
                "[MAV-STAT] "
                f"link={s.get('port')}@{s.get('baud')} "
                f"src={int(src_sys)}/{int(src_comp)} tgt={int(tgt_sys)}/{int(tgt_comp)} "
                f"rx={int(s.get('rx_total', 0))} last_rx={rx_age:.2f}s({s.get('rx_last_type','')}) "
                f"last_hb={hb_age:.2f}s(from {int(hb_sys)}/{int(hb_comp)}) "
                f"tx(hb/vse/vpe/odom)={int(s.get('tx_hb', 0))}/{int(s.get('tx_vse', 0))}/"
                f"{int(s.get('tx_vpe', 0))}/{int(s.get('tx_odom', 0))} "
                f"errs(hb/vse/vpe/odom)={int(s.get('tx_err_hb', 0))}/{int(s.get('tx_err_vse', 0))}/"
                f"{int(s.get('tx_err_vpe', 0))}/{int(s.get('tx_err_odom', 0))}",
                flush=True,
            )
        except Exception:
            return

    def is_connected(self) -> bool:
        return self._master is not None

    def open_error(self) -> Optional[str]:
        return self._open_error

    def _status_loop(self) -> None:
        period = 1.0
        try:
            hz = float(self._status_rate_hz)
            if hz > 0.0:
                period = 1.0 / hz
        except Exception:
            period = 1.0
        if period < 0.2:
            period = 0.2
        t_next = time.time()
        while not self._stop.is_set():
            now = time.time()
            if now >= t_next:
                self.print_status_once()
                t_next = now + float(period)
            time.sleep(0.01)

    # ---------------- public API used by server.py ----------------
    def try_open_serial_only(self) -> Tuple[bool, str]:
        try:
            m = mavutil.mavlink_connection(
                self.port,
                baud=self.baud,
                source_system=int(self.source_system),
                source_component=int(self.source_component),
            )
            m.close()
            return True, f"{self.port}@{self.baud}"
        except Exception as e:
            return False, str(e)

    def wait_heartbeat_once(self, timeout: float = 5.0) -> Tuple[bool, str]:
        try:
            m = mavutil.mavlink_connection(
                self.port,
                baud=self.baud,
                source_system=int(self.source_system),
                source_component=int(self.source_component),
            )
            msg = m.wait_heartbeat(timeout=timeout)
            try:
                sysid = int(getattr(m, "target_system", 0) or 0)
                compid = int(getattr(m, "target_component", 0) or 0)
            except Exception:
                sysid = 0
                compid = 0
            m.close()
            if msg is None:
                return False, f"no heartbeat within {timeout:.1f}s"
            return True, f"sys={sysid} comp={compid}"
        except Exception as e:
            return False, str(e)

    def apply_rates(self, rates: Dict[str, Any]) -> None:
        """Optionally request message intervals using MAV_CMD_SET_MESSAGE_INTERVAL."""
        if self._master is None:
            return
        m = self._master
        try:
            for name, hz in (rates or {}).items():
                try:
                    hz = float(hz)
                except Exception:
                    continue
                if hz is None:
                    continue
                try:
                    msg_id = mavutil.mavlink.enums["MAVLINK_MSG_ID"][name].value
                except Exception:
                    msg_id = getattr(mavutil.mavlink, f"MAVLINK_MSG_ID_{name}", None)
                if msg_id is None:
                    continue
                # MAV_CMD_SET_MESSAGE_INTERVAL: param2 = interval_us, -1 disables, 0 requests default rate.
                interval_us = -1 if hz <= 0.0 else int(1_000_000.0 / hz)
                try:
                    with self._tx_lock:
                        m.mav.command_long_send(
                            m.target_system if hasattr(m, "target_system") else 1,
                            m.target_component if hasattr(m, "target_component") else 0,
                            mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
                            0,
                            float(msg_id),
                            float(interval_us),
                            0, 0, 0, 0, 0,
                        )
                except Exception:
                    pass
        except Exception:
            pass

    def send_heartbeat_once(self) -> None:
        m = self._master
        if m is None:
            return
        try:
            with self._tx_lock:
                m.mav.heartbeat_send(
                    int(self._hb_type),
                    int(self._hb_autopilot),
                    int(self._hb_base_mode),
                    int(self._hb_custom_mode),
                    int(self._hb_system_status),
                )
            self._record_tx("HEARTBEAT")
        except Exception as e:
            self._hb_tx_errs += 1
            now = time.time()
            if (now - float(self._hb_tx_last_err_t)) >= 1.0:
                self._hb_tx_last_err_t = float(now)
                try:
                    self.print(f"[MAV] HEARTBEAT send error: {e} (errs={self._hb_tx_errs})", flush=True)
                except Exception:
                    pass

    def _heartbeat_loop(self) -> None:
        # Keep-alive heartbeat so PX4/QGC see us as a component (and to avoid compid=0 pitfalls).
        period = 1.0
        try:
            hz = float(self._hb_rate_hz)
            if hz > 0.0:
                period = 1.0 / hz
        except Exception:
            period = 1.0
        if period < 0.05:
            period = 0.05
        t_next = time.time()
        while not self._stop.is_set():
            now = time.time()
            if now >= t_next:
                self.send_heartbeat_once()
                t_next = now + float(period)
            time.sleep(0.01)

    def start(self):
        self._open_error = None
        try:
            self._master = mavutil.mavlink_connection(
                self.port,
                baud=self.baud,
                source_system=int(self.source_system),
                source_component=int(self.source_component),
            )
        except Exception as e:
            self._open_error = str(e)
            self.print(f"[MAV] open error: {e}", flush=True)
            return
        try:
            self.print(
                f"[MAV] opened {self.port}@{int(self.baud)} src=sys{int(self.source_system)} comp{int(self.source_component)}",
                flush=True,
            )
        except Exception:
            pass

        self._stop.clear()

        # Initiate heartbeat immediately (and keep sending it).
        if bool(self._hb_enabled):
            self.send_heartbeat_once()
            self._hb_thread = threading.Thread(target=self._heartbeat_loop, daemon=True, name="mav-hb")
            self._hb_thread.start()

        # Optional periodic status prints for debugging.
        if bool(self._status_enabled):
            self._status_thread = threading.Thread(target=self._status_loop, daemon=True, name="mav-stat")
            self._status_thread.start()

        # Optional: mirror PX4 MAVLink stream to a UDP target (e.g., ground station / GUI).
        try:
            mirror_cfg = (self.cfg.get("px4", {}) or {}).get("mirror", {}) or {}
            mirror_en = bool(mirror_cfg.get("enabled", False))
            mirror_udp = mirror_cfg.get("udp", None)
            if mirror_en and mirror_udp:
                host, port = parse_hostport(str(mirror_udp))
                s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                try:
                    s.setblocking(False)
                except Exception:
                    pass
                self._mirror_sock = s
                self._mirror_addr = (host, int(port))
                self._mirror_drops = 0
                self.print(f"[MAV] mirror enabled -> udp://{host}:{int(port)}", flush=True)
            else:
                self._mirror_sock = None
                self._mirror_addr = None
        except Exception as e:
            self._mirror_sock = None
            self._mirror_addr = None
            self.print(f"[MAV] mirror init error: {e}", flush=True)

        self._rx_thread = threading.Thread(target=self._rx_loop, daemon=True, name="mav-rx")
        self._rx_thread.start()

        # Optional MANUAL_CONTROL passthrough from RC sticks (kept OFF by default to avoid
        # fighting the PID-driven MANUAL_CONTROL loop).
        try:
            mc_cfg = self.cfg.get("mav_control", {}) or {}
            passthrough = (mc_cfg.get("passthrough") or {}) if isinstance(mc_cfg, dict) else {}
            passthrough_en = bool(passthrough.get("enable", False))
        except Exception:
            passthrough_en = False
        if passthrough_en:
            self._mc_thread = threading.Thread(target=self._manual_control_loop, daemon=True, name="mav-mc")
            self._mc_thread.start()

    def stop(self):
        self._stop.set()
        try:
            if self._mc_thread and self._mc_thread.is_alive():
                self._mc_thread.join(timeout=0.5)
        except Exception:
            pass
        try:
            if self._hb_thread and self._hb_thread.is_alive():
                self._hb_thread.join(timeout=0.5)
        except Exception:
            pass
        try:
            if self._status_thread and self._status_thread.is_alive():
                self._status_thread.join(timeout=0.5)
        except Exception:
            pass
        try:
            if self._rx_thread and self._rx_thread.is_alive():
                self._rx_thread.join(timeout=0.5)
        except Exception:
            pass
        try:
            if self._master:
                self._master.close()
        except Exception:
            pass
        try:
            if self._mirror_sock:
                self._mirror_sock.close()
        except Exception:
            pass
        self._mirror_sock = None
        self._mirror_addr = None
        self._master = None

    # Convenience getters for PID / gating
    def get_rc_norm(self, ch_1based: int) -> float:
        try:
            return float(self.rc_norm.get(int(ch_1based), 0.0))
        except Exception:
            return 0.0

    def mode_name(self) -> str:
        return self._mode_name

    def get_imu_window(self, window_s: float = 1.0) -> Tuple[list[Tuple[float, float, float]], list[Tuple[float, float, float]]]:
        now = float(time.time())
        acc: list[Tuple[float, float, float]] = []
        mag: list[Tuple[float, float, float]] = []
        with self._imu_lock:
            for t, ax, ay, az in list(self._accel_samples):
                if (now - float(t)) <= float(window_s):
                    acc.append((float(ax), float(ay), float(az)))
            for t, mx, my, mz in list(self._mag_samples):
                if (now - float(t)) <= float(window_s):
                    mag.append((float(mx), float(my), float(mz)))
        return acc, mag

    # Direct MANUAL_CONTROL sender for controllers (x=pitch, y=roll, z=throttle, r=yaw)
    def send_manual_control(self, x_norm: float, y_norm: float, z_norm: float, r_norm: float) -> bool:
        m = self._master
        if m is None:
            return False
        try:
            x = int(_clamp(x_norm, -1.0, 1.0) * 1000.0)
            y = int(_clamp(y_norm, -1.0, 1.0) * 1000.0)
            z = int(_clamp(z_norm, 0.0, 1.0) * 1000.0)   # PX4 expects 0..1000 for throttle
            r = int(_clamp(r_norm, -1.0, 1.0) * 1000.0)
            with self._tx_lock:
                m.mav.manual_control_send(
                    m.target_system if hasattr(m, "target_system") else 1,
                    x, y, z, r, 0
                )
            self._record_tx("MANUAL_CONTROL")
            return True
        except Exception as e:
            self._mc_tx_errs += 1
            now = time.time()
            if (now - self._mc_tx_last_err_t) >= 1.0:
                self._mc_tx_last_err_t = now
                try:
                    self.print(f"[MAV] MANUAL_CONTROL send error: {e} (errs={self._mc_tx_errs})", flush=True)
                except Exception:
                    pass
            return False

    # External-vision pose to PX4 (VISION_POSITION_ESTIMATE)
    def send_external_vision_pose(self, pose: dict) -> None:
        """Send world pose to PX4 as VISION_POSITION_ESTIMATE."""
        m = self._master
        if m is None:
            return
        try:
            pos = pose.get("position_m", {}) if isinstance(pose, dict) else {}
            ang = pose.get("orientation_deg", {})
            x = float(pos.get("x", 0.0))
            y = float(pos.get("y", 0.0))
            z = float(pos.get("z", 0.0))
            roll  = math.radians(float(ang.get("roll", 0.0)))
            pitch = math.radians(float(ang.get("pitch", 0.0)))
            yaw   = math.radians(float(ang.get("yaw", 0.0)))
            usec = int(time.time() * 1_000_000)
            with self._tx_lock:
                m.mav.vision_position_estimate_send(usec, x, y, z, roll, pitch, yaw)
            self._record_tx("VISION_POSITION_ESTIMATE")
        except Exception:
            self._vpe_tx_errs += 1
            now = time.time()
            if (now - self._vpe_tx_last_err_t) >= 1.0:
                self._vpe_tx_last_err_t = now
                try:
                    self.print(f"[MAV] VISION_POSITION_ESTIMATE send error (errs={self._vpe_tx_errs})", flush=True)
                except Exception:
                    pass

    # External-vision velocity to PX4 (VISION_SPEED_ESTIMATE)
    def send_external_vision_speed(self, vel: dict) -> None:
        """Send velocity to PX4 as VISION_SPEED_ESTIMATE (message 103).

        Expected `vel` fields:
          - usec: int microseconds (optional; defaults to host time)
          - x/y/z: float m/s
          - covariance: optional float[9] (MAVLink2 extension)
          - reset_counter: optional uint8 (MAVLink2 extension)
        """
        m = self._master
        if m is None:
            return
        try:
            usec = int(vel.get("usec", int(time.time() * 1_000_000)))
            x = float(vel.get("x", 0.0))
            y = float(vel.get("y", 0.0))
            z = float(vel.get("z", 0.0))
            cov = vel.get("covariance", None)
            if cov is None:
                cov = [float("nan")] + [0.0] * 8  # spec: unknown covariance -> first element NaN
            rc = int(vel.get("reset_counter", 0)) & 0xFF
            try:
                with self._tx_lock:
                    m.mav.vision_speed_estimate_send(usec, x, y, z, cov, rc)
            except TypeError:
                # Dialects without extension fields (MAVLink1) only accept the base fields.
                with self._tx_lock:
                    m.mav.vision_speed_estimate_send(usec, x, y, z)
            self._record_tx("VISION_SPEED_ESTIMATE")
        except Exception:
            self._vse_tx_errs += 1
            now = time.time()
            if (now - self._vse_tx_last_err_t) >= 1.0:
                self._vse_tx_last_err_t = now
                try:
                    self.print(f"[MAV] VISION_SPEED_ESTIMATE send error (errs={self._vse_tx_errs})", flush=True)
                except Exception:
                    pass
            return

    def send_odometry(self, odom: dict) -> None:
        """Send pose/velocity as MAVLink ODOMETRY (message 331)."""
        m = self._master
        if m is None:
            return
        try:
            usec = int(odom.get("usec", int(time.time() * 1_000_000)))
            frame_id = int(odom.get("frame_id", getattr(mavutil.mavlink, "MAV_FRAME_LOCAL_NED", 1)))
            child_frame_id = int(
                odom.get(
                    "child_frame_id",
                    getattr(
                        mavutil.mavlink,
                        "MAV_FRAME_BODY_FRD",
                        getattr(mavutil.mavlink, "MAV_FRAME_BODY_NED", 8),
                    ),
                )
            )
            x = float(odom.get("x", 0.0))
            y = float(odom.get("y", 0.0))
            z = float(odom.get("z", 0.0))
            q = odom.get("q", None)
            if not (isinstance(q, (list, tuple)) and len(q) >= 4):
                q = [1.0, 0.0, 0.0, 0.0]
            else:
                q = [float(q[0]), float(q[1]), float(q[2]), float(q[3])]
            vx = float(odom.get("vx", float("nan")))
            vy = float(odom.get("vy", float("nan")))
            vz = float(odom.get("vz", float("nan")))
            rollspeed = float(odom.get("rollspeed", float("nan")))
            pitchspeed = float(odom.get("pitchspeed", float("nan")))
            yawspeed = float(odom.get("yawspeed", float("nan")))
            pose_cov = odom.get("pose_cov", None)
            if not (isinstance(pose_cov, (list, tuple)) and len(pose_cov) == 21):
                pose_cov = [float("nan")] * 21
            vel_cov = odom.get("vel_cov", None)
            if not (isinstance(vel_cov, (list, tuple)) and len(vel_cov) == 21):
                vel_cov = [float("nan")] * 21
            reset_counter = int(odom.get("reset_counter", 0)) & 0xFF
            estimator_type = int(
                odom.get(
                    "estimator_type",
                    getattr(mavutil.mavlink, "MAV_ESTIMATOR_TYPE_VISION", 2),
                )
            )
            with self._tx_lock:
                try:
                    m.mav.odometry_send(
                        usec,
                        frame_id,
                        child_frame_id,
                        x,
                        y,
                        z,
                        q,
                        vx,
                        vy,
                        vz,
                        rollspeed,
                        pitchspeed,
                        yawspeed,
                        pose_cov,
                        vel_cov,
                        reset_counter,
                        estimator_type,
                    )
                except TypeError:
                    m.mav.odometry_send(
                        usec,
                        frame_id,
                        child_frame_id,
                        x,
                        y,
                        z,
                        q,
                        vx,
                        vy,
                        vz,
                        rollspeed,
                        pitchspeed,
                        yawspeed,
                        pose_cov,
                        vel_cov,
                        reset_counter,
                    )
            self._record_tx("ODOMETRY")
        except Exception as e:
            self._odom_tx_errs += 1
            now = time.time()
            if (now - self._odom_tx_last_err_t) >= 1.0:
                self._odom_tx_last_err_t = now
                try:
                    self.print(f"[MAV] ODOMETRY send error: {e} (errs={self._odom_tx_errs})", flush=True)
                except Exception:
                    pass

    # ---------------- internal: RX / RC handling ----------------
    def _parse_mode_name(self, hb) -> None:
        """Best-effort mode string parsing from HEARTBEAT (very rough)."""
        try:
            cm = int(getattr(hb, "custom_mode", 0))
            base = int(getattr(hb, "base_mode", 0))
            mode = ""
            if cm in (1, 2, 3):   # heuristic PX4 mapping
                mode = ("MANUAL", "ALTCTL", "POSCTL")[cm-1]
            elif (base & 0b10000000):
                mode = ""
            try:
                mode = mavutil.mode_string_v10(hb)
            except Exception:
                pass
            self._mode_name = mode
        except Exception:
            self._mode_name = ""

    def _rx_loop(self):
        m = self._master
        if m is None:
            return
        rates = None
        try:
            rates = (self.cfg.get("px4", {}) or {}).get("rates_hz") or {}
        except Exception:
            rates = None
        try:
            hb = m.wait_heartbeat(timeout=5.0)
            try:
                self._parse_mode_name(hb)
            except Exception:
                pass
            try:
                if rates is not None:
                    self.apply_rates(rates)
            except Exception:
                pass
        except Exception:
            try:
                self.print("[MAV] warning: no HEARTBEAT within 5s; continuing to listen", flush=True)
            except Exception:
                pass

        while not self._stop.is_set():
            try:
                msg = m.recv_match(blocking=True, timeout=0.2)
                if msg is None:
                    continue

                mtype = msg.get_type()
                try:
                    self._rx_total += 1
                    self._rx_by_type[str(mtype)] = int(self._rx_by_type.get(str(mtype), 0)) + 1
                    self._rx_last_t = float(time.time())
                    self._rx_last_type = str(mtype)
                    if str(mtype) == "HEARTBEAT":
                        self._rx_last_hb_t = float(self._rx_last_t)
                        try:
                            self._rx_last_hb_sysid = int(msg.get_srcSystem())
                            self._rx_last_hb_compid = int(msg.get_srcComponent())
                        except Exception:
                            self._rx_last_hb_sysid = 0
                            self._rx_last_hb_compid = 0
                except Exception:
                    pass

                # Mirror raw bytes out over UDP (best-effort; never block RX loop).
                try:
                    if self._mirror_sock is not None and self._mirror_addr is not None and mtype != "BAD_DATA":
                        raw = None
                        try:
                            raw = msg.get_msgbuf()
                        except Exception:
                            raw = None
                        if raw:
                            try:
                                self._mirror_sock.sendto(bytes(raw), self._mirror_addr)
                            except (BlockingIOError, InterruptedError):
                                self._mirror_drops += 1
                            except OSError as e:
                                if getattr(e, "errno", None) in (errno.EWOULDBLOCK, errno.EAGAIN, errno.ENOBUFS):
                                    self._mirror_drops += 1
                except Exception:
                    pass

                if mtype == "HEARTBEAT":
                    self._parse_mode_name(msg)

                # RC updates
                if mtype in ("RC_CHANNELS", "RC_CHANNELS_RAW"):
                    self._on_rc(msg)

                # IMU samples (for calibration).
                if mtype in ("RAW_IMU", "HIGHRES_IMU"):
                    self._on_imu(msg)

                # Optional CSV log
                try:
                    if self.csvlog:
                        self.csvlog.log_mavlink(msg)  # type: ignore
                except Exception:
                    pass

            except Exception:
                continue

    def _on_rc(self, msg: Any):
        """Update rc_norm from RC_* message, evaluate gates, print on changes."""
        raw: Dict[int, float] = {}

        if msg.get_type() == "RC_CHANNELS":
            for i in range(1, 9):
                raw[i] = float(getattr(msg, f"chan{i}_raw", 1500) or 1500)
            for i in range(9, 19):
                v = getattr(msg, f"chan{i}_raw", None)
                if v is not None:
                    raw[i] = float(v)

        elif msg.get_type() == "RC_CHANNELS_RAW":
            for i, name in enumerate(("chan1_raw","chan2_raw","chan3_raw","chan4_raw","chan5_raw","chan6_raw","chan7_raw","chan8_raw"), start=1):
                v = getattr(msg, name, None)
                if v is not None:
                    raw[i] = float(v)

        # Normalize
        for ch, v in raw.items():
            self.rc_norm[ch] = _norm_rc_value(v)

        # Evaluate gates and print on change
        self._eval_pid_gate_print()
        self._eval_mc_gate_print()

    def _on_imu(self, msg: Any) -> None:
        try:
            mtype = msg.get_type()
        except Exception:
            return
        ax = ay = az = None
        mx = my = mz = None
        try:
            if mtype == "RAW_IMU":
                ax = float(getattr(msg, "xacc", None))
                ay = float(getattr(msg, "yacc", None))
                az = float(getattr(msg, "zacc", None))
                mx = float(getattr(msg, "xmag", None))
                my = float(getattr(msg, "ymag", None))
                mz = float(getattr(msg, "zmag", None))
            elif mtype == "HIGHRES_IMU":
                ax = float(getattr(msg, "xacc", None))
                ay = float(getattr(msg, "yacc", None))
                az = float(getattr(msg, "zacc", None))
                mx = float(getattr(msg, "xmag", None))
                my = float(getattr(msg, "ymag", None))
                mz = float(getattr(msg, "zmag", None))
        except Exception:
            return
        if ax is None or ay is None or az is None:
            return
        now = float(time.time())
        with self._imu_lock:
            self._accel_samples.append((now, float(ax), float(ay), float(az)))
            self._accel_latest = (float(ax), float(ay), float(az))
            if mx is not None and my is not None and mz is not None:
                if not (math.isfinite(mx) and math.isfinite(my) and math.isfinite(mz)):
                    return
                self._mag_samples.append((now, float(mx), float(my), float(mz)))
                self._mag_latest = (float(mx), float(my), float(mz))

    def _eval_pid_gate_print(self):
        pid_cfg = self.cfg.get("pid", {}) or {}
        gate = pid_cfg.get("remote_gate", {}) or {}
        req = bool(gate.get("require_remote_enable", True))
        ch = int(gate.get("ch", gate.get("channel", 7)))
        th = float(gate.get("threshold", 0.6))  # normalized -1..+1

        val = self.rc_norm.get(ch, 0.0)
        high = (val >= th)
        ok = (high if req else True)

        if self._pid_gate_on is None and not req:
            self.print("[PID] CH7 ignored (pid.remote_gate.require_remote_enable=false)", flush=True)
        if self._pid_gate_on is None:
            self._pid_gate_on = ok
        elif self._pid_gate_on != ok:
            self._pid_gate_on = ok
            self.print("[PID] remote ENABLED (CH7 high)" if ok else "[PID] remote DISABLED (CH7 low)", flush=True)

    def _eval_mc_gate_print(self):
        mc_cfg = self.cfg.get("mav_control", {}) or {}
        gate = mc_cfg.get("rc_gate", {}) or {}
        req = bool(gate.get("require_remote_enable", True))
        ch = int(gate.get("channel", 7))
        th = float(gate.get("threshold", 0.6))  # normalized -1..+1

        val = self.rc_norm.get(ch, 0.0)
        high = (val >= th)
        ok = (high if req else True)

        if self._mc_gate_on is None and not req:
            self.print("[MC] CH7 ignored (mav_control.rc_gate.require_remote_enable=false)", flush=True)
        if self._mc_gate_on is None:
            self._mc_gate_on = ok
        elif self._mc_gate_on != ok:
            self._mc_gate_on = ok
            self.print("[MC] remote ENABLED (CH7 high)" if ok else "[MC] remote DISABLED (CH7 low)", flush=True)

    def _manual_control_loop(self):
        """Publish MANUAL_CONTROL to PX4 when the gate allows, using normalized sticks."""
        m = self._master
        if m is None:
            return

        mc_cfg = self.cfg.get("mav_control", {}) or {}
        passthrough = (mc_cfg.get("passthrough") or {}) if isinstance(mc_cfg, dict) else {}
        gate = mc_cfg.get("rc_gate", {}) or {}
        req = bool(gate.get("require_remote_enable", True))
        ch = int(gate.get("channel", 7))
        th = float(gate.get("threshold", 0.6))
        period = 1.0 / float(mc_cfg.get("manual_loop_hz", 30.0))

        axis_map = (passthrough.get("axis_map") if isinstance(passthrough, dict) else None) or {
            "x": 2,  # pitch
            "y": 1,  # roll
            "z": 3,  # throttle
            "r": 4,  # yaw
        }
        if not isinstance(axis_map, dict):
            axis_map = {"x": 2, "y": 1, "z": 3, "r": 4}

        while not self._stop.is_set():
            try:
                gate_high = (self.rc_norm.get(ch, 0.0) >= th)
                gate_ok = (gate_high if req else True)

                if gate_ok:
                    def _s(ch_idx: int) -> int:
                        v = float(self.rc_norm.get(ch_idx, 0.0))
                        if ch_idx == axis_map.get("z", 3):
                            return int(_clamp((v + 1.0) / 2.0, 0.0, 1.0) * 1000.0)  # 0..1000
                        return int(_clamp(v, -1.0, 1.0) * 1000.0)  # -1000..+1000

                    x = _s(axis_map.get("x", 2))
                    y = _s(axis_map.get("y", 1))
                    z = _s(axis_map.get("z", 3))
                    r = _s(axis_map.get("r", 4))

                    try:
                        m.mav.manual_control_send(
                            m.target_system if hasattr(m, "target_system") else 1,
                            x, y, z, r, 0  # buttons=0
                        )
                    except Exception:
                        pass

                time.sleep(period)
            except Exception:
                time.sleep(period)
