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

try:
    from pymavlink import mavutil  # type: ignore
except Exception:  # pragma: no cover (optional on camera-only setups)
    mavutil = None  # type: ignore
from udp import parse_hostport


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
        self.enabled = bool(px4_cfg.get("enabled", True))
        self.port = px4_cfg.get("serial", "/dev/ttyUSB0")
        self.baud = int(px4_cfg.get("baud", 115200))

        self._master: Optional["mavutil.mavlink_connection"] = None
        self._rx_thread: Optional[threading.Thread] = None
        self._mc_thread: Optional[threading.Thread] = None
        self._stop = threading.Event()

        # RC normalized values (1-based indexing). Missing/stale channels default to -1.0 (safe OFF).
        self.rc_norm: Dict[int, float] = {i: -1.0 for i in range(1, 19)}
        self._rc_last_ch_t: Dict[int, float] = {i: 0.0 for i in range(1, 19)}
        try:
            mc_cfg = self.cfg.get("mav_control", {}) or {}
            gate_cfg = (mc_cfg.get("rc_gate") or {}) if isinstance(mc_cfg, dict) else {}
            self._rc_stale_timeout_s = float(gate_cfg.get("stale_timeout_s", 0.5))
        except Exception:
            self._rc_stale_timeout_s = 0.5
        self._rc_stale_timeout_s = float(max(0.0, min(5.0, float(self._rc_stale_timeout_s))))

        # Serialize TX to avoid interleaving MAVLink frames across threads.
        self._tx_lock = threading.Lock()

        # Keep vision message timestamps monotonically increasing even if the system clock adjusts
        # (some GUIs compute "Hz" from successive usec deltas).
        self._vision_tx_last_usec: int = 0

        # Mode string (if we parse it)
        self._mode_name: str = ""

        # Latest attitude from FCU (ATTITUDE), radians.
        # Stored as (roll,pitch,yaw,t_ms_host) and updated in RX thread.
        self._att_lock = threading.Lock()
        self._att_rpy: Optional[Tuple[float, float, float, int]] = None

        # Latest downward rangefinder distance (meters).
        # Updated in RX thread from one of: DISTANCE_SENSOR / RANGEFINDER / ALTITUDE.bottom_clearance.
        # Stored as (distance_m, t_ms_host, source).
        self._rf_lock = threading.Lock()
        self._rf_down: Optional[Tuple[float, int, str]] = None

        # Gates (will be recomputed upon RC updates)
        self._pid_gate_on: Optional[bool] = None
        self._mc_gate_on: Optional[str] = None  # OFF / PID_ALIGN / PID_MOVE (string for tri-state)

        # Optional MAVLink mirror (raw UDP forward)
        self._mirror_sock: Optional[socket.socket] = None
        self._mirror_addr: Optional[Tuple[str, int]] = None
        self._mirror_drops: int = 0

        # MANUAL_CONTROL TX diagnostics (rate-limited console logging)
        self._mc_tx_errs: int = 0
        self._mc_tx_last_err_t: float = 0.0
        self._params_applied: bool = False

    def _mirror_tx_bytes(self, raw: bytes | bytearray | memoryview | None) -> None:
        """Best-effort TX mirror to the same UDP target as RX mirroring.

        This lets the GUI observe outbound messages (e.g. VISION_POSITION_ESTIMATE) that are
        otherwise only visible on the serial link to PX4.
        """
        try:
            if self._mirror_sock is None or self._mirror_addr is None:
                return
            if not raw:
                return
            try:
                self._mirror_sock.sendto(bytes(raw), self._mirror_addr)
            except (BlockingIOError, InterruptedError):
                self._mirror_drops += 1
            except OSError as e:
                if getattr(e, "errno", None) in (errno.EWOULDBLOCK, errno.EAGAIN, errno.ENOBUFS):
                    self._mirror_drops += 1
        except Exception:
            pass

    # ---------------- public API used by server.py ----------------
    def set_mirror_destination(self, host: str, port: Optional[int] = None) -> bool:
        """Update MAVLink mirror UDP destination at runtime."""
        try:
            host = str(host).strip()
            if not host:
                return False
            if port is None:
                try:
                    if self._mirror_addr is not None:
                        port = int(self._mirror_addr[1])
                except Exception:
                    port = None
            if port is None:
                try:
                    mirror_cfg = (self.cfg.get("px4", {}) or {}).get("mirror", {}) or {}
                    cur_udp = mirror_cfg.get("udp", None)
                    if cur_udp:
                        _, port2 = parse_hostport(str(cur_udp))
                        port = int(port2)
                except Exception:
                    port = None
            if port is None:
                port = 14550
            port_i = int(port)

            # Update runtime target (used by RX thread).
            self._mirror_addr = (host, port_i)

            # Best-effort: keep local cfg dict in sync for diagnostics.
            try:
                px4_cfg = self.cfg.get("px4", None)
                if not isinstance(px4_cfg, dict):
                    px4_cfg = {}
                    self.cfg["px4"] = px4_cfg
                mirror_cfg = px4_cfg.get("mirror", None)
                if not isinstance(mirror_cfg, dict):
                    mirror_cfg = {}
                    px4_cfg["mirror"] = mirror_cfg
                mirror_cfg["udp"] = f"{host}:{port_i}"
            except Exception:
                pass

            return True
        except Exception:
            return False

    def latest_rangefinder_down_m(self, max_age_s: float = 0.5) -> Optional[float]:
        """Return latest downward rangefinder distance in meters (or None if missing/stale)."""
        try:
            with self._rf_lock:
                cur = self._rf_down
            if cur is None:
                return None
            dist_m, t_ms, _src = cur
            dist_m = float(dist_m)
            if (not math.isfinite(float(dist_m))) or float(dist_m) <= 0.0:
                return None
            if max_age_s is not None:
                try:
                    max_age_s = float(max_age_s)
                except Exception:
                    max_age_s = 0.0
                if math.isfinite(float(max_age_s)) and float(max_age_s) > 0.0:
                    age_s = (float(time.time() * 1000.0) - float(t_ms)) / 1000.0
                    if float(age_s) > float(max_age_s):
                        return None
            return float(dist_m)
        except Exception:
            return None

    def _set_rangefinder_down(self, dist_m: float, source: str, *, prefer: bool = False) -> None:
        try:
            dist_m = float(dist_m)
        except Exception:
            return
        if (not math.isfinite(float(dist_m))) or float(dist_m) <= 0.0:
            return
        try:
            t_ms = int(time.time() * 1000)
        except Exception:
            t_ms = 0
        with self._rf_lock:
            if bool(prefer) or self._rf_down is None:
                self._rf_down = (float(dist_m), int(t_ms), str(source or ""))
                return
            # Update stale values even when not preferred.
            try:
                _dist0, t0, _src0 = self._rf_down
                if int(t0) <= 0 or (int(t_ms) - int(t0)) >= 1000:
                    self._rf_down = (float(dist_m), int(t_ms), str(source or ""))
            except Exception:
                self._rf_down = (float(dist_m), int(t_ms), str(source or ""))

    def _on_distance_sensor(self, msg: Any) -> None:
        # DISTANCE_SENSOR.current_distance is in cm (uint16).
        try:
            cur_cm = int(getattr(msg, "current_distance", 0) or 0)
        except Exception:
            return
        if cur_cm <= 0 or cur_cm >= 65535:
            return
        dist_m = float(cur_cm) * 0.01
        try:
            orient = int(getattr(msg, "orientation", -1) or -1)
        except Exception:
            orient = -1
        # Downward-facing sensors are typically PITCH_270 (25). Accept 24 as well (some stacks differ).
        is_down = int(orient) in (24, 25)
        self._set_rangefinder_down(float(dist_m), "DISTANCE_SENSOR", prefer=bool(is_down))

    def _on_rangefinder(self, msg: Any) -> None:
        # RANGEFINDER.distance is in meters (float).
        try:
            dist_m = float(getattr(msg, "distance", 0.0) or 0.0)
        except Exception:
            return
        if (not math.isfinite(float(dist_m))) or float(dist_m) <= 0.0:
            return
        self._set_rangefinder_down(float(dist_m), "RANGEFINDER", prefer=True)

    def _on_altitude(self, msg: Any) -> None:
        # ALTITUDE.bottom_clearance is in meters (float), often sourced from a downward rangefinder.
        try:
            bc = float(getattr(msg, "bottom_clearance", 0.0) or 0.0)
        except Exception:
            return
        if (not math.isfinite(float(bc))) or float(bc) <= 0.0:
            return
        # Treat as a non-preferred fallback (low rate / may be missing).
        self._set_rangefinder_down(float(bc), "ALTITUDE.bottom_clearance", prefer=False)

    def try_open_serial_only(self) -> Tuple[bool, str]:
        if not bool(getattr(self, "enabled", True)):
            return True, "DISABLED (px4.enabled=false)"
        if mavutil is None:
            return False, "pymavlink not installed"
        try:
            m = mavutil.mavlink_connection(self.port, baud=self.baud)
            m.close()
            return True, f"{self.port}@{self.baud}"
        except Exception as e:
            return False, str(e)

    def wait_heartbeat_once(self, timeout: float = 5.0) -> Tuple[bool, str]:
        if not bool(getattr(self, "enabled", True)):
            return True, "DISABLED (px4.enabled=false)"
        if mavutil is None:
            return False, "pymavlink not installed"
        try:
            m = mavutil.mavlink_connection(self.port, baud=self.baud)
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

    def apply_params(self, params: Dict[str, Any]) -> None:
        """Best-effort parameter set after a confirmed connection (heartbeat)."""
        if self._master is None:
            return
        if mavutil is None:
            return
        if not isinstance(params, dict) or not params:
            return

        m = self._master
        try:
            sysid = int(getattr(m, "target_system", 1) or 1)
        except Exception:
            sysid = 1
        try:
            compid = int(getattr(m, "target_component", 0) or 0)
        except Exception:
            compid = 0

        applied = 0
        for key, value in params.items():
            if value is None:
                continue
            try:
                name = str(key).strip()
            except Exception:
                continue
            if not name:
                continue
            if len(name.encode("utf-8", errors="ignore")) > 16:
                self.print(f"[MAV] param name too long (max 16): {name}", flush=True)
                continue

            # PARAM_SET uses a float payload but type matters (especially on ArduPilot).
            ptype = mavutil.mavlink.MAV_PARAM_TYPE_REAL32
            fval: float
            try:
                if isinstance(value, bool):
                    ptype = mavutil.mavlink.MAV_PARAM_TYPE_UINT8
                    fval = 1.0 if value else 0.0
                elif isinstance(value, int) and not isinstance(value, bool):
                    ptype = mavutil.mavlink.MAV_PARAM_TYPE_INT32
                    fval = float(value)
                else:
                    ptype = mavutil.mavlink.MAV_PARAM_TYPE_REAL32
                    fval = float(value)
            except Exception:
                self.print(f"[MAV] param {name} invalid value: {value!r}", flush=True)
                continue

            try:
                m.mav.param_set_send(
                    sysid,
                    compid,
                    name.encode("utf-8"),
                    float(fval),
                    int(ptype),
                )
                applied += 1
                self.print(f"[MAV] param set {name}={fval}", flush=True)
            except Exception as e:
                self.print(f"[MAV] param set failed {name}={value!r}: {e}", flush=True)

        if applied:
            self._params_applied = True

    def start(self):
        if not bool(getattr(self, "enabled", True)):
            try:
                self.print("[MAV] disabled (px4.enabled=false)", flush=True)
            except Exception:
                pass
            return
        if mavutil is None:
            try:
                self.print("[MAV] disabled (pymavlink not installed)", flush=True)
            except Exception:
                pass
            return
        try:
            self._master = mavutil.mavlink_connection(self.port, baud=self.baud)
        except Exception as e:
            self.print(f"[MAV] open error: {e}", flush=True)
            return

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

        self._stop.clear()
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
        ch = int(ch_1based)
        now = float(time.monotonic())
        try:
            v = float(self.rc_norm.get(ch, -1.0))
        except Exception:
            v = -1.0
        try:
            to_s = float(getattr(self, "_rc_stale_timeout_s", 0.0) or 0.0)
            if to_s > 0.0:
                t0 = float(self._rc_last_ch_t.get(ch, 0.0))
                if t0 <= 0.0 or (now - t0) > to_s:
                    return -1.0
        except Exception:
            pass
        return float(v)

    def mode_name(self) -> str:
        return self._mode_name

    def get_attitude_rpy(self) -> Optional[Tuple[float, float, float, int]]:
        with self._att_lock:
            return self._att_rpy

    def get_attitude_yaw_rad(self) -> Optional[float]:
        a = self.get_attitude_rpy()
        if a is None:
            return None
        return float(a[2])

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
            sysid = m.target_system if hasattr(m, "target_system") else 1
            msg = m.mav.manual_control_encode(sysid, x, y, z, r, 0)
            with self._tx_lock:
                m.mav.send(msg)
                try:
                    self._mirror_tx_bytes(msg.get_msgbuf())
                except Exception:
                    pass
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
    def send_external_vision_pose(
        self,
        pose: dict,
        *,
        usec: Optional[int] = None,
        covariance: Optional[list] = None,
        reset_counter: Optional[int] = None,
    ) -> bool:
        """Send a pose to PX4 as VISION_POSITION_ESTIMATE (msg 102).

        `pose` is expected to look like the VO JSON we already emit:
          - position_m: {x,y,z}
          - yaw_deg: <float> (optional)
        It also accepts:
          - orientation_deg: {roll,pitch,yaw} (optional)
        MAVLink2 extensions (if supported by the pymavlink dialect):
          - covariance: float[21] (set covariance[0]=NaN to mark unknown covariance)
          - reset_counter: uint8
        """
        m = self._master
        if m is None:
            return False
        if not isinstance(pose, dict):
            return False
        try:
            pos = pose.get("position_m", {}) if isinstance(pose.get("position_m", None), dict) else {}
            x = float(pos.get("x", 0.0))
            y = float(pos.get("y", 0.0))
            z = float(pos.get("z", 0.0))

            # Orientation:
            # Prefer explicit roll/pitch/yaw dict, but accept yaw_deg-only for current track3d VO.
            ang = pose.get("orientation_deg", None)
            if not isinstance(ang, dict):
                ang = {}
            roll_deg = float(ang.get("roll", 0.0))
            pitch_deg = float(ang.get("pitch", 0.0))
            if "yaw" in ang:
                yaw_deg = float(ang.get("yaw", 0.0))
            else:
                yv = pose.get("yaw_deg", None)
                yaw_deg = float(yv) if isinstance(yv, (int, float)) else 0.0

            roll = math.radians(float(roll_deg))
            pitch = math.radians(float(pitch_deg))
            yaw = math.radians(float(yaw_deg))

            cov = covariance
            if cov is None:
                cov = pose.get("covariance", None) if isinstance(pose, dict) else None
            rc = reset_counter
            if rc is None:
                rc0 = pose.get("reset_counter", None) if isinstance(pose, dict) else None
                rc = int(rc0) if isinstance(rc0, (int, float)) else None

            with self._tx_lock:
                try:
                    t_usec = int(usec) if usec is not None else int(time.time() * 1_000_000)
                except Exception:
                    t_usec = int(time.time() * 1_000_000)
                if int(t_usec) <= int(self._vision_tx_last_usec):
                    t_usec = int(self._vision_tx_last_usec) + 1
                self._vision_tx_last_usec = int(t_usec)

                # Try MAVLink2 extension fields first (if present).
                if cov is not None or rc is not None:
                    cov21 = None
                    if isinstance(cov, (list, tuple)):
                        try:
                            cov21 = [float(v) for v in cov[:21]]
                            if len(cov21) < 21:
                                cov21 += [0.0] * (21 - len(cov21))
                        except Exception:
                            cov21 = None
                    if cov21 is None:
                        cov21 = [float("nan")] + [0.0] * 20
                    rc_i = int(rc) if rc is not None else 0
                    try:
                        msg = m.mav.vision_position_estimate_encode(t_usec, x, y, z, roll, pitch, yaw, cov21, rc_i)
                        m.mav.send(msg)
                        try:
                            self._mirror_tx_bytes(msg.get_msgbuf())
                        except Exception:
                            pass
                        return True
                    except TypeError:
                        # Older dialects don't include extension fields.
                        pass

                msg = m.mav.vision_position_estimate_encode(t_usec, x, y, z, roll, pitch, yaw)
                m.mav.send(msg)
                try:
                    self._mirror_tx_bytes(msg.get_msgbuf())
                except Exception:
                    pass
            return True
        except Exception:
            return False

    def send_external_vision_speed(
        self,
        payload: dict,
        *,
        usec: Optional[int] = None,
    ) -> bool:
        """Send velocity to PX4 as VISION_SPEED_ESTIMATE (msg 103).

        Expected `payload`:
          - velocity_mps: {x,y,z}
        """
        m = self._master
        if m is None:
            return False
        if not isinstance(payload, dict):
            return False
        try:
            vel = payload.get("velocity_mps", {}) if isinstance(payload.get("velocity_mps", None), dict) else {}
            vx = float(vel.get("x", 0.0))
            vy = float(vel.get("y", 0.0))
            vz = float(vel.get("z", 0.0))

            enc = getattr(getattr(m, "mav", None), "vision_speed_estimate_encode", None)
            if not callable(enc):
                return False

            with self._tx_lock:
                try:
                    t_usec = int(usec) if usec is not None else int(time.time() * 1_000_000)
                except Exception:
                    t_usec = int(time.time() * 1_000_000)
                if int(t_usec) <= int(self._vision_tx_last_usec):
                    t_usec = int(self._vision_tx_last_usec) + 1
                self._vision_tx_last_usec = int(t_usec)

                msg = enc(int(t_usec), float(vx), float(vy), float(vz))
                m.mav.send(msg)
                try:
                    self._mirror_tx_bytes(msg.get_msgbuf())
                except Exception:
                    pass
            return True
        except Exception:
            return False

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
        params = None
        try:
            rates = (self.cfg.get("px4", {}) or {}).get("rates_hz") or {}
        except Exception:
            rates = None
        try:
            params = (self.cfg.get("px4", {}) or {}).get("params") or {}
        except Exception:
            params = None
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
            try:
                if not self._params_applied and params is not None:
                    self.apply_params(params)
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
                    try:
                        if not self._params_applied and params is not None:
                            self.apply_params(params)
                    except Exception:
                        pass

                if mtype == "ATTITUDE":
                    try:
                        self._on_attitude(msg)
                    except Exception:
                        pass

                if mtype in ("DISTANCE_SENSOR", "RANGEFINDER", "ALTITUDE"):
                    try:
                        if mtype == "DISTANCE_SENSOR":
                            self._on_distance_sensor(msg)
                        elif mtype == "RANGEFINDER":
                            self._on_rangefinder(msg)
                        elif mtype == "ALTITUDE":
                            self._on_altitude(msg)
                    except Exception:
                        pass

                # RC updates
                if mtype in ("RC_CHANNELS", "RC_CHANNELS_RAW"):
                    self._on_rc(msg)

                # Optional CSV log
                try:
                    if self.csvlog:
                        self.csvlog.log_mavlink(msg)  # type: ignore
                except Exception:
                    pass

            except Exception:
                continue

    def _on_attitude(self, msg: Any) -> None:
        try:
            roll = float(getattr(msg, "roll", 0.0))
            pitch = float(getattr(msg, "pitch", 0.0))
            yaw = float(getattr(msg, "yaw", 0.0))
        except Exception:
            return
        with self._att_lock:
            self._att_rpy = (roll, pitch, yaw, int(time.time() * 1000))

    def _on_rc(self, msg: Any):
        """Update rc_norm from RC_* message, evaluate gates, print on changes."""
        raw: Dict[int, float] = {}

        def _maybe_add(ch: int, v: Any) -> None:
            try:
                vf = float(v)
            except Exception:
                return
            # RC_* messages carry PWM microseconds. Treat out-of-range (0/65535/etc) as invalid and ignore it.
            if not (800.0 <= vf <= 2200.0):
                return
            raw[int(ch)] = float(vf)

        if msg.get_type() == "RC_CHANNELS":
            for i in range(1, 9):
                _maybe_add(i, getattr(msg, f"chan{i}_raw", None))
            for i in range(9, 19):
                v = getattr(msg, f"chan{i}_raw", None)
                if v is not None:
                    _maybe_add(i, v)

        elif msg.get_type() == "RC_CHANNELS_RAW":
            for i, name in enumerate(("chan1_raw","chan2_raw","chan3_raw","chan4_raw","chan5_raw","chan6_raw","chan7_raw","chan8_raw"), start=1):
                v = getattr(msg, name, None)
                if v is not None:
                    _maybe_add(i, v)

        # Normalize
        now = float(time.monotonic())
        for ch, v in raw.items():
            self.rc_norm[ch] = _norm_rc_value(v)
            try:
                self._rc_last_ch_t[int(ch)] = float(now)
            except Exception:
                pass

        # Evaluate gates and print on change
        self._eval_pid_gate_print()
        self._eval_mc_gate_print()

    def _eval_pid_gate_print(self):
        pid_cfg = self.cfg.get("pid", {}) or {}
        gate = pid_cfg.get("remote_gate", {}) or {}
        req = bool(gate.get("require_remote_enable", True))
        ch = int(gate.get("ch", gate.get("channel", 7)))
        th = float(gate.get("threshold", 0.6))  # normalized -1..+1

        val = float(self.get_rc_norm(ch))
        high = (val >= th)
        ok = (high if req else True)

        if self._pid_gate_on is None and not req:
            self.print(f"[PID] CH{ch} ignored (pid.remote_gate.require_remote_enable=false)", flush=True)
        if self._pid_gate_on is None:
            self._pid_gate_on = ok
        elif self._pid_gate_on != ok:
            self._pid_gate_on = ok
            self.print(
                f"[PID] remote ENABLED (CH{ch} high)" if ok else f"[PID] remote DISABLED (CH{ch} low)",
                flush=True,
            )

    def _eval_mc_gate_print(self):
        mc_cfg = self.cfg.get("mav_control", {}) or {}
        gate = mc_cfg.get("rc_gate", {}) or {}
        req = bool(gate.get("require_remote_enable", True))
        ch = int(gate.get("channel", 7))

        if self._mc_gate_on is None and not req:
            self.print(f"[MC] CH{ch} ignored (mav_control.rc_gate.require_remote_enable=false)", flush=True)

        # Map rc_norm [-1..+1] to [0..1] with mid at ~0.5.
        rc01 = _clamp((float(self.get_rc_norm(ch)) + 1.0) * 0.5, 0.0, 1.0)
        off_max = float(gate.get("off_max", 0.33))
        align_min = float(gate.get("align_min", 0.30))
        align_max = float(gate.get("align_max", 0.66))
        off_max = _clamp(off_max, 0.0, 1.0)
        align_min = _clamp(align_min, 0.0, 1.0)
        align_max = _clamp(align_max, 0.0, 1.0)
        if align_max < align_min:
            align_max = align_min
        if off_max < align_min:
            off_max = align_min

        prev = self._mc_gate_on or "OFF"
        if not req:
            state = "PID_MOVE"
        elif prev == "OFF":
            state = "PID_MOVE" if rc01 >= align_max else ("PID_ALIGN" if rc01 >= off_max else "OFF")
        elif prev == "PID_ALIGN":
            state = "PID_MOVE" if rc01 >= align_max else ("OFF" if rc01 < align_min else "PID_ALIGN")
        else:  # PID_MOVE
            state = "PID_MOVE" if rc01 >= align_max else ("OFF" if rc01 < align_min else "PID_ALIGN")

        if self._mc_gate_on is None:
            self._mc_gate_on = state
        elif self._mc_gate_on != state:
            self._mc_gate_on = state
            self.print(f"[MC] gate {state} (ch{ch}={rc01:.2f})", flush=True)

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
        align_max = float(gate.get("align_max", gate.get("threshold", 0.6)))
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
                rc01 = _clamp((float(self.get_rc_norm(ch)) + 1.0) * 0.5, 0.0, 1.0)
                gate_high = (rc01 >= float(align_max))
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
