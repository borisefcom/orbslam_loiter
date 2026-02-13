#!/usr/bin/env python3
# pid.py — Event-driven MANUAL_CONTROL publisher (PX4) using pixel-space errors.

from __future__ import annotations
from dataclasses import dataclass, field, is_dataclass
from typing import Callable, Optional, Literal, Dict, Any, Tuple
import time

StateT = Literal["IDLE", "TRACKING", "LOST"]
GateStateT = Literal["OFF", "PID_ALIGN", "PID_MOVE"]

# ---------------- Small utils ----------------
def clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else (hi if v > hi else v)

# Last gate snapshot (for external status queries)
_last_gate_status: dict = {}
def get_last_gate_status() -> dict:
    return dict(_last_gate_status) if _last_gate_status else {}

class SlewRateLimiter:
    def __init__(self, rate_per_s: float):
        self.rate = float(max(0.0, rate_per_s))
        self.prev: float = 0.0
        self.t_prev: Optional[float] = None

    def reset(self, value: float = 0.0):
        self.prev = float(value)
        # Start timing immediately so first step ramps instead of teleporting
        self.t_prev = time.perf_counter()

    def step(self, target: float, now: Optional[float] = None) -> float:
        t = time.perf_counter() if now is None else now
        if self.t_prev is None:
            self.t_prev = t
            self.prev = float(target)
            return self.prev
        dt = max(1e-3, t - self.t_prev)
        self.t_prev = t
        if self.rate <= 0.0:
            self.prev = float(target)
            return self.prev
        max_delta = self.rate * dt
        lo = self.prev - max_delta
        hi = self.prev + max_delta
        self.prev = clamp(float(target), lo, hi)
        return self.prev

# ---------------- Config dataclasses ----------------
@dataclass
class AxisCfg:
    enabled: bool = True
    invert: bool = False
    kp: float = 0.0
    ki: float = 0.0
    kd: float = 0.0
    deadband_norm: float = 0.0   # in [0..1]
    out_limit: float = 1.0       # clamp in [-1..+1]
    rate_limit: float = 0.0      # 0 => no slew
    soft_start_s: float = 0.0    # 0 => no soft-start (instant)
    soft_start_gamma: float = 1.0  # cap curve exponent; >1 slower for small, faster for large
    enable_gamma: bool = True    # if false, ignore soft_start_gamma (use gamma=1)

@dataclass
class ThrustCfg:
    enabled: bool = False
    min_value: float = 0.10      # floor (abs)
    max_value: float = 0.50      # ceiling (abs)
    ramp_time_s: float = 1.0     # seconds to go from min -> max (<=0 => instant)
    down_ramp_time_s: float = 0.0  # seconds to keep max after max-bbox stop (<=0 disables)
    invert: bool = False         # reverse direction if True

@dataclass
class RCGateCfg:
    channel: int = 7
    require_remote_enable: bool = True
    off_max: float = 0.33        # normalized [0..1], OFF when rc01 < off_max
    align_min: float = 0.30      # normalized [0..1], ALIGN when rc01 >= align_min (hysteresis)
    align_max: float = 0.66      # normalized [0..1], ALIGN when rc01 < align_max, else MOVE

@dataclass
class ModeGateCfg:
    allow_in_all_modes: bool = False
    allow_in_alt_hold: bool = True
    allow_in_pos_hold: bool = False

@dataclass
class AlignExitCfg:
    enabled: bool = True
    delta_horizontal_deg: float = 1.0
    delta_vertical_deg: float = 1.0

@dataclass
class PidProfileCfg:
    mode_gate: ModeGateCfg = field(default_factory=ModeGateCfg)
    pitch: AxisCfg = field(default_factory=lambda: AxisCfg(
        enabled=True, invert=True, kp=0.004, ki=0.0, kd=0.0,
        deadband_norm=0.01, out_limit=0.6, rate_limit=0.0  # P-only, no slew
    ))
    roll: AxisCfg = field(default_factory=lambda: AxisCfg(
        enabled=False, invert=False, kp=0.0, ki=0.0, kd=0.0,
        deadband_norm=0.01, out_limit=0.6, rate_limit=0.0  # P-only, no slew
    ))
    yaw: AxisCfg = field(default_factory=lambda: AxisCfg(
        enabled=True, invert=False, kp=0.0018, ki=0.0, kd=0.0,
        deadband_norm=0.01, out_limit=0.6, rate_limit=0.0  # P-only, no slew
    ))
    thrust: ThrustCfg = field(default_factory=lambda: ThrustCfg(
        enabled=False, min_value=0.10, max_value=0.50, ramp_time_s=1.0, invert=False
    ))
    auto_exit: Optional[AlignExitCfg] = None

@dataclass
class MavControlCfg:
    enable: bool = True
    dry_run: bool = False  # compute + emit debug, but never send MANUAL_CONTROL
    rc_gate: RCGateCfg = field(default_factory=RCGateCfg)
    pid_move: PidProfileCfg = field(default_factory=PidProfileCfg)
    pid_align: PidProfileCfg = field(default_factory=lambda: PidProfileCfg(
        auto_exit=AlignExitCfg()
    ))

# --------------- Runtime measurement ----------------
@dataclass
class Measurement:
    state: StateT
    yaw_err_rad: Optional[float]
    pitch_err_rad: Optional[float]
    track_id: Optional[int] = None
    t_ms: Optional[int] = None

# ---------------- PidManager ----------------
class PidManager:
    """
    Call on_measurement() each detector/track tick.
    Sends MANUAL_CONTROL only if:
      - cfg.enable
      - RC gate state is PID_ALIGN or PID_MOVE (3-state gate) if required
      - mode gate passes (ALT_HOLD / POS_HOLD, or allow_in_all_modes) for the active profile
      - m.state == "TRACKING"

    NOTE: The order used in send_manual_control(...) below is preserved exactly as in your working setup.
    """
    def __init__(
        self,
        cfg: MavControlCfg,
        *,
        get_rc_norm: Callable[[int], float],
        get_mode_name: Callable[[], str],
        send_manual_control: Callable[[float, float, float, float], bool],
        emit_debug_json: Optional[Callable[[Dict[str, Any]], None]] = None
    ):
        self.cfg = cfg
        self.get_rc_norm = get_rc_norm
        self.get_mode_name = get_mode_name
        self.send_manual_control = send_manual_control
        self.emit_debug_json = emit_debug_json

        # Limiters exist but rate=0 => pass-through
        base_prof = self.cfg.pid_move
        self.slew_pitch = SlewRateLimiter(base_prof.pitch.rate_limit)
        self.slew_roll  = SlewRateLimiter(base_prof.roll.rate_limit)
        self.slew_yaw   = SlewRateLimiter(base_prof.yaw.rate_limit)

        # Thrust limiter rate derived from ramp_time_s
        self.slew_thrust = SlewRateLimiter(0.0)
        self._update_thrust_rate(base_prof.thrust)

        self._gate_prev = False  # edge-detect on (RC+mode) gate
        self._gate_state: GateStateT = "OFF"
        self._align_done: bool = False
        self._ramp_t0: Optional[float] = None
        self._last_dbg_print = 0.0
        self._force_thrust_until: float = 0.0
        self._force_thrust_reason: str = ""

    def apply_updates(self, new_cfg: MavControlCfg):
        self.cfg = new_cfg
        # Keep current gate state; update rates from the active profile if possible.
        prof = self._active_profile(self._gate_state) or self.cfg.pid_move
        self.slew_pitch.rate = prof.pitch.rate_limit
        self.slew_roll.rate  = prof.roll.rate_limit
        self.slew_yaw.rate   = prof.yaw.rate_limit
        self._update_thrust_rate(prof.thrust)

    def gate_state(self) -> GateStateT:
        return self._gate_state

    def mark_align_done(self):
        """Latch PID_ALIGN as complete until the user changes the switch state."""
        if self._gate_state == "PID_ALIGN":
            self._align_done = True

    def start_thrust_hold_max(self, *, reason: str = "max_bbox_area"):
        """
        Timed thrust-only hold (no yaw/pitch adjustments) for indoor experiments.
        Triggered when the tracker intentionally stops because the bbox is too large.
        """
        # Hold behavior is only defined for PID_MOVE.
        if self._gate_state != "PID_MOVE":
            return
        try:
            dur = float(getattr(self.cfg.pid_move.thrust, "down_ramp_time_s", 0.0) or 0.0)
        except Exception:
            dur = 0.0
        if dur <= 0.0:
            return
        now = time.perf_counter()
        self._force_thrust_until = max(float(self._force_thrust_until), float(now) + float(dur))
        self._force_thrust_reason = str(reason or "thrust_hold")

    def _soft_start_cap(self, axis_cfg: AxisCfg, now_perf: float) -> float:
        """
        Soft-start via a time-ramped *output cap* (preferred over scaling so small commands pass through sooner).
          cap(t) = out_limit * r^gamma, r=t/T in [0..1]
          cmd_out = clamp(cmd, -cap, +cap)
        """
        try:
            s = float(getattr(axis_cfg, "soft_start_s", 0.0) or 0.0)
        except Exception:
            s = 0.0
        try:
            gamma = float(getattr(axis_cfg, "soft_start_gamma", 1.0) or 1.0)
        except Exception:
            gamma = 1.0
        try:
            enable_gamma = bool(getattr(axis_cfg, "enable_gamma", True))
        except Exception:
            enable_gamma = True
        if not enable_gamma:
            gamma = 1.0
        if gamma <= 0.0:
            gamma = 1.0
        t0 = self._ramp_t0
        if s <= 0.0 or t0 is None:
            return float(max(0.0, abs(float(axis_cfg.out_limit))))
        r = clamp((float(now_perf) - float(t0)) / float(s), 0.0, 1.0)
        return float(max(0.0, abs(float(axis_cfg.out_limit)))) * (float(r) ** float(gamma))

    def _update_thrust_rate(self, thrust_cfg: ThrustCfg):
        delta = abs(float(thrust_cfg.max_value) - float(thrust_cfg.min_value))
        if float(thrust_cfg.ramp_time_s) <= 0.0 or delta <= 1e-6:
            self.slew_thrust.rate = 0.0  # 0 => no slew (instant)
        else:
            self.slew_thrust.rate = delta / float(thrust_cfg.ramp_time_s)

    def on_measurement(self, m: Measurement):
        now_perf = time.perf_counter()
        gate_state, rc_norm = self._rc_gate_state()
        mode_bucket = "ANY"
        mode_str = ""
        gate_rc = False
        gate_mode_ok = False
        gate_now = False

        pitch_cmd = 0.0
        roll_cmd = 0.0
        yaw_cmd = 0.0
        thrust_cmd = 0.0
        yaw_err = float(m.yaw_err_rad) if m.yaw_err_rad is not None else None
        pitch_err = float(m.pitch_err_rad) if m.pitch_err_rad is not None else None
        valid_measure = (yaw_err is not None) and (pitch_err is not None)
        sent = False
        reason = ""

        def _emit_pid_debug(*, sent_override: Optional[bool] = None, reason_override: Optional[str] = None) -> None:
            if not self.emit_debug_json:
                return
            sent0 = bool(sent) if sent_override is None else bool(sent_override)
            reason0 = str(reason) if reason_override is None else str(reason_override)
            try:
                rad2deg0 = 180.0 / 3.141592653589793
                yaw_err_deg0 = (yaw_err * rad2deg0) if yaw_err is not None else None
                pitch_err_deg0 = (pitch_err * rad2deg0) if pitch_err is not None else None
            except Exception:
                yaw_err_deg0 = None
                pitch_err_deg0 = None
            dbg = {
                "type": "pid_debug",
                "t_ms": int(time.time() * 1000),
                "state": m.state,
                "gate_state": gate_state,
                "mode": mode_str,
                "mode_bucket": mode_bucket,
                "rc_ch_norm": rc_norm,
                "gate": {
                    "rc": gate_rc,
                    "mode": gate_mode_ok,
                },
                "measurement": {
                    "yaw_err_rad": yaw_err,
                    "pitch_err_rad": pitch_err,
                    "yaw_err_deg": yaw_err_deg0,
                    "pitch_err_deg": pitch_err_deg0,
                    "track_id": m.track_id,
                },
                "result": {
                    "pitch": round(pitch_cmd, 6),
                    "roll": round(roll_cmd, 6),
                    "yaw": round(yaw_cmd, 6),
                    "thrust": round(thrust_cmd, 6),
                    "sent": sent0,
                    "reason": reason0,
                },
            }
            try:
                self.emit_debug_json(dbg)
            except Exception:
                pass

        if not self.cfg.enable:
            reason = "disabled"
            self._gate_prev = False
            self._ramp_t0 = None
            self._reset_limiters(hard=True)
        else:
            # OFF state: suppress MANUAL_CONTROL, but still emit pid_debug JSON for visibility.
            if gate_state == "OFF":
                reason = "gate_off"
                if self._gate_state != "OFF" or self._gate_prev:
                    self._reset_limiters(hard=True)
                    self._ramp_t0 = None
                    self._gate_prev = False
                self._gate_state = "OFF"
                self._align_done = False
                try:
                    _last_gate_status.update(
                        {
                            "gate_state": "OFF",
                            "rc_norm": float(rc_norm),
                            "gate_rc": False,
                            "gate_mode": False,
                            "gate_now": False,
                            "mode": "",
                            "mode_bucket": "ANY",
                            "state": str(m.state),
                            "reason": str(reason),
                            "sent": False,
                        }
                    )
                except Exception:
                    pass
                _emit_pid_debug(sent_override=False)
                return

            # Track gate-state transitions (ALIGN/MOVE) and reset latch when leaving/entering ALIGN.
            prev_gate_state = self._gate_state
            if gate_state != prev_gate_state:
                if gate_state == "PID_ALIGN":
                    self._align_done = False
                elif prev_gate_state == "PID_ALIGN":
                    self._align_done = False
            self._gate_state = gate_state

            # ALIGN complete latch: once done, remain silent until the user changes the switch.
            if gate_state == "PID_ALIGN" and self._align_done:
                reason = "align_done"
                gate_rc = True
                try:
                    _last_gate_status.update(
                        {
                            "gate_state": "PID_ALIGN",
                            "rc_norm": float(rc_norm),
                            "gate_rc": True,
                            "gate_mode": False,
                            "gate_now": False,
                            "mode": "",
                            "mode_bucket": "ANY",
                            "state": str(m.state),
                            "reason": str(reason),
                            "sent": False,
                        }
                    )
                except Exception:
                    pass
                _emit_pid_debug(sent_override=False)
                return

            prof = self._active_profile(gate_state)
            if prof is None:
                reason = "no_profile"
                gate_rc = True
                try:
                    _last_gate_status.update(
                        {
                            "gate_state": str(gate_state),
                            "rc_norm": float(rc_norm),
                            "gate_rc": bool(gate_rc),
                            "gate_mode": False,
                            "gate_now": False,
                            "mode": "",
                            "mode_bucket": "ANY",
                            "state": str(m.state),
                            "reason": str(reason),
                            "sent": False,
                        }
                    )
                except Exception:
                    pass
                _emit_pid_debug(sent_override=False)
                return

            # Update per-profile rates (cheap; just assigns floats).
            self.slew_pitch.rate = float(prof.pitch.rate_limit)
            self.slew_roll.rate = float(prof.roll.rate_limit)
            self.slew_yaw.rate = float(prof.yaw.rate_limit)
            self._update_thrust_rate(prof.thrust)

            gate_rc = True
            gate_mode_ok, mode_bucket, mode_str = self._mode_gate_ok(prof.mode_gate)
            gate_now = bool(gate_mode_ok)

            # ----- Edge handling for limiters / ramps -----
            # All ramps start on gate open (RC+mode) or when switching between ALIGN/MOVE,
            # even if we're not yet TRACKING.
            profile_changed = bool(gate_state != prev_gate_state)
            if gate_now and (not self._gate_prev or profile_changed):
                self._reset_limiters(hard=False)
                self._ramp_t0 = now_perf
                if prof.thrust.enabled:
                    sign = -1.0 if prof.thrust.invert else 1.0
                    self.slew_thrust.reset(sign * float(prof.thrust.min_value))
            elif not gate_now and self._gate_prev:
                self._reset_limiters(hard=True)
                self._ramp_t0 = None
            self._gate_prev = gate_now

            if gate_state == "PID_MOVE" and gate_now and bool(prof.thrust.enabled) and (now_perf < float(self._force_thrust_until)) and str(m.state).upper() != "TRACKING":
                # Thrust-only override: keep pushing at max_value for a fixed time.
                sign = -1.0 if prof.thrust.invert else 1.0
                thrust_cmd = clamp(
                    sign * float(prof.thrust.max_value),
                    -float(prof.thrust.max_value),
                    float(prof.thrust.max_value),
                )
                pitch_cmd = 0.0
                yaw_cmd = 0.0
                roll_cmd = 0.0
                if bool(self.cfg.dry_run):
                    sent = False
                    reason = f"{self._force_thrust_reason}_hold_dry_run"
                else:
                    try:
                        sent = bool(self.send_manual_control(thrust_cmd * -1, 0.0, 0.5, 0.0))
                    except Exception:
                        sent = False
                    reason = f"{self._force_thrust_reason}_hold" if sent else "send_failed"
            elif not gate_now:
                reason = "gate_blocked"
            elif m.state != "TRACKING":
                reason = "state"
            elif not valid_measure:
                reason = "invalid_measurement"
            else:
                pitch_cap = self._soft_start_cap(prof.pitch, now_perf)
                roll_cap = self._soft_start_cap(prof.roll, now_perf)
                yaw_cap = self._soft_start_cap(prof.yaw, now_perf)

                # P-only for pitch/yaw (deadband + clamp), no slew on these axes
                if prof.pitch.enabled:
                    e_y = (-pitch_err) if prof.pitch.invert else pitch_err
                    if abs(e_y) < prof.pitch.deadband_norm:
                        e_y = 0.0
                    pitch_cmd = clamp(prof.pitch.kp * e_y, -prof.pitch.out_limit, prof.pitch.out_limit)
                    pitch_cmd = clamp(float(pitch_cmd), -float(pitch_cap), float(pitch_cap))
                else:
                    self.slew_pitch.reset(0.0)

                if prof.yaw.enabled:
                    e_x = (-yaw_err) if prof.yaw.invert else yaw_err
                    if abs(e_x) < prof.yaw.deadband_norm:
                        e_x = 0.0
                    yaw_cmd = clamp(prof.yaw.kp * e_x, -prof.yaw.out_limit, prof.yaw.out_limit)
                    yaw_cmd = clamp(float(yaw_cmd), -float(yaw_cap), float(yaw_cap))
                else:
                    self.slew_yaw.reset(0.0)

                if prof.roll.enabled:
                    e_r = (-yaw_err) if prof.roll.invert else yaw_err
                    if abs(e_r) < prof.roll.deadband_norm:
                        e_r = 0.0
                    roll_cmd = clamp(prof.roll.kp * e_r, -prof.roll.out_limit, prof.roll.out_limit)
                    roll_cmd = clamp(float(roll_cmd), -float(roll_cap), float(roll_cap))
                else:
                    self.slew_roll.reset(0.0)

                # Thrust (ONLY axis with slew + limiter)
                if prof.thrust.enabled:
                    sign = -1.0 if prof.thrust.invert else 1.0
                    target = sign * float(prof.thrust.max_value)
                    bias = self.slew_thrust.step(target, now=now_perf)
                    # enforce minimum magnitude while active
                    if 0.0 < abs(bias) < float(prof.thrust.min_value):
                        bias = sign * float(prof.thrust.min_value)
                    # guard against numeric overshoot
                    thrust_cmd = clamp(bias, -float(prof.thrust.max_value), float(prof.thrust.max_value))
                else:
                    self.slew_thrust.reset(0.0)
                    thrust_cmd = 0.0

                # *** DO NOT CHANGE THIS ORDER (your verified mapping) ***
                if bool(self.cfg.dry_run):
                    sent = False
                    reason = "dry_run"
                else:
                    try:
                        sent = bool(self.send_manual_control(thrust_cmd * -1, roll_cmd, pitch_cmd + 0.5, yaw_cmd))
                    except Exception:
                        sent = False
                    reason = "sent" if sent else "send_failed"

        rad2deg = 180.0 / 3.141592653589793
        yaw_err_deg = (yaw_err * rad2deg) if yaw_err is not None else None
        pitch_err_deg = (pitch_err * rad2deg) if pitch_err is not None else None

        _emit_pid_debug()
        # Console debug (lightly rate-limited)
        now_print = time.time()
        if (now_print - self._last_dbg_print) >= 1:
            try:
                yaw_err_deg_s = "None" if yaw_err_deg is None else f"{yaw_err_deg:+.2f}"
                pitch_err_deg_s = "None" if pitch_err_deg is None else f"{pitch_err_deg:+.2f}"
                print(
                    f"[PID] state={m.state} gate={gate_state} mode={mode_str} gate(rc={gate_rc} mode={gate_mode_ok}) "
                    f"sent={sent} reason={reason} yaw_err_deg={yaw_err_deg_s} pitch_err_deg={pitch_err_deg_s} "
                    f"cmd yaw={yaw_cmd:.3f} roll={roll_cmd:.3f} pitch={pitch_cmd:.3f} thrust={thrust_cmd:.3f}"
                )
            except Exception:
                pass
            self._last_dbg_print = now_print
        # Update last gate snapshot for external status (best-effort, no locking for speed).
        try:
            _last_gate_status.update(
                {
                    "gate_state": str(gate_state),
                    "rc_norm": float(rc_norm),
                    "gate_rc": bool(gate_rc),
                    "gate_mode": bool(gate_mode_ok),
                    "gate_now": bool(gate_now),
                    "mode": str(mode_str),
                    "mode_bucket": str(mode_bucket),
                    "state": str(m.state),
                    "reason": reason,
                    "sent": bool(sent),
                }
            )
        except Exception:
            pass



    # ---------------- helpers ----------------
    def _reset_limiters(self, hard: bool=False):
        self.slew_pitch.reset(0.0)
        self.slew_roll.reset(0.0)
        self.slew_yaw.reset(0.0)
        if hard:
            self.slew_thrust.reset(0.0)

    def _rc_norm_safe(self, ch_1based: int) -> float:
        """Return RC channel value normalized to [0..1] with center at ~0.5."""
        try:
            v = float(self.get_rc_norm(ch_1based))
        except Exception:
            v = 0.0
        # get_rc_norm() is normalized to [-1..+1]; gate uses a 0..1 scale.
        return clamp((v + 1.0) * 0.5, 0.0, 1.0)

    def _rc_gate_state(self) -> Tuple[GateStateT, float]:
        """Three-state RC gate with hysteresis bands."""
        ch = int(max(1, getattr(self.cfg.rc_gate, "channel", 7)))
        if not bool(getattr(self.cfg.rc_gate, "require_remote_enable", True)):
            return "PID_MOVE", 1.0

        rc01 = self._rc_norm_safe(ch)
        try:
            off_max = float(getattr(self.cfg.rc_gate, "off_max", 0.33))
        except Exception:
            off_max = 0.33
        try:
            align_min = float(getattr(self.cfg.rc_gate, "align_min", 0.30))
        except Exception:
            align_min = 0.30
        try:
            align_max = float(getattr(self.cfg.rc_gate, "align_max", 0.66))
        except Exception:
            align_max = 0.66

        off_max = clamp(off_max, 0.0, 1.0)
        align_min = clamp(align_min, 0.0, 1.0)
        align_max = clamp(align_max, 0.0, 1.0)
        if align_max < align_min:
            align_max = align_min
        if off_max < align_min:
            off_max = align_min

        prev = getattr(self, "_gate_state", "OFF")
        if prev == "OFF":
            if rc01 >= align_max:
                return "PID_MOVE", rc01
            if rc01 >= off_max:
                return "PID_ALIGN", rc01
            return "OFF", rc01

        if prev == "PID_ALIGN":
            if rc01 >= align_max:
                return "PID_MOVE", rc01
            if rc01 < align_min:
                return "OFF", rc01
            return "PID_ALIGN", rc01

        # prev == PID_MOVE
        if rc01 >= align_max:
            return "PID_MOVE", rc01
        if rc01 < align_min:
            return "OFF", rc01
        return "PID_ALIGN", rc01

    def _active_profile(self, gate_state: GateStateT) -> Optional[PidProfileCfg]:
        if gate_state == "PID_MOVE":
            return self.cfg.pid_move
        if gate_state == "PID_ALIGN":
            return self.cfg.pid_align
        return None

    def _mode_gate_ok(self, mode_gate: ModeGateCfg) -> Tuple[bool, str, str]:
        """
        Returns (ok, bucket, mode_str)
        bucket ∈ {"ANY","ALT","POS","OTHER"}
        mode_str is built as "{MODE}:{Allowed|Not Allowed}" based on config gates.
        """
        try:
            mode_raw = self.get_mode_name() or ""
            m = mode_raw.upper()
        except Exception:
            mode_raw = ""
            m = ""
        if bool(getattr(mode_gate, "allow_in_all_modes", False)):
            return True, "ANY", f"{m or 'ANY'}:Allowed"
        if m in ("ALTCTL", "ALT_HOLD"):
            ok = bool(getattr(mode_gate, "allow_in_alt_hold", True))
            return ok, "ALT", f"{m}:{'Allowed' if ok else 'Not Allowed'}"
        if m in ("POSCTL", "POS_HOLD", "LOITER"):
            ok = bool(getattr(mode_gate, "allow_in_pos_hold", False))
            return ok, "POS", f"{m}:{'Allowed' if ok else 'Not Allowed'}"
        return False, "OTHER", f"{m or 'OTHER'}:Not Allowed"


def _apply_overrides(dc_obj, updates: Dict[str, Any]):
    if not isinstance(updates, dict):
        return
    for key, value in updates.items():
        if not hasattr(dc_obj, key):
            continue
        current = getattr(dc_obj, key)
        if is_dataclass(current):
            if not isinstance(value, dict):
                continue
            _apply_overrides(current, value)
        else:
            try:
                if isinstance(current, bool):
                    setattr(dc_obj, key, bool(value))
                elif isinstance(current, int) and not isinstance(current, bool):
                    setattr(dc_obj, key, int(value))
                elif isinstance(current, float):
                    setattr(dc_obj, key, float(value))
                else:
                    setattr(dc_obj, key, value)
            except Exception:
                setattr(dc_obj, key, current)


def config_from_dict(data: Optional[Dict[str, Any]]) -> MavControlCfg:
    cfg = MavControlCfg()
    if isinstance(data, dict):
        d = dict(data)

        # Normalize profile keys (YAML uses PID_MOVE / PID_ALIGN).
        try:
            if "PID_MOVE" in d and "pid_move" not in d:
                d["pid_move"] = d.pop("PID_MOVE")
        except Exception:
            pass
        try:
            if "PID_ALIGN" in d and "pid_align" not in d:
                d["pid_align"] = d.pop("PID_ALIGN")
        except Exception:
            pass

        # Back-compat: older configs had axis settings at mav_control root (no profiles).
        if "pid_move" not in d:
            root_keys = ("mode_gate", "pitch", "yaw", "roll", "thrust", "forward")
            if any(k in d for k in root_keys):
                pm: Dict[str, Any] = {}
                for k in root_keys:
                    if k in d:
                        pm[k] = d.pop(k)
                d["pid_move"] = pm

        # Back-compat: older configs used mav_control.forward; prefer thrust (inside profiles).
        for prof_key in ("pid_move", "pid_align"):
            try:
                prof = d.get(prof_key)
                if isinstance(prof, dict) and ("thrust" not in prof) and ("forward" in prof):
                    prof = dict(prof)
                    prof["thrust"] = prof.get("forward")
                    d[prof_key] = prof
            except Exception:
                pass

        # NOTE: PID profiles should be fully defined in YAML (PID_MOVE and PID_ALIGN).
        # Do not auto-copy between profiles here; it makes tuning/validation ambiguous.

        # Back-compat: rc_gate.threshold -> binary gating (no ALIGN state).
        try:
            rg = d.get("rc_gate")
            if isinstance(rg, dict) and ("threshold" in rg):
                if not any(k in rg for k in ("off_max", "align_min", "align_max")):
                    th = float(rg.get("threshold", 0.66))
                    rg = dict(rg)
                    rg["off_max"] = th
                    rg["align_min"] = th
                    rg["align_max"] = th
                    d["rc_gate"] = rg
        except Exception:
            pass

        _apply_overrides(cfg, d)
    return cfg
