from __future__ import annotations

import math
import os
import threading
from pathlib import Path
from typing import Any, Optional

from .yaml_config import load_yaml_dict


class Px4VisionState:
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


class Px4VisionPublisher:
    """Optional PX4 external-vision publisher (VISION_POSITION_ESTIMATE / VISION_SPEED_ESTIMATE)."""

    def __init__(self, *, cfg_path: Path, state: Px4VisionState, print_fn=print) -> None:
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
        cfg = load_yaml_dict(self.cfg_path)
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

