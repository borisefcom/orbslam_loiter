import os
import csv
import time
import yaml
import datetime
from typing import Any, Dict, Tuple, Optional, List

# -------------------------------
# Small utils
# -------------------------------

def _ensure_dir(path: str):
    d = os.path.dirname(path)
    os.makedirs(d if d else ".", exist_ok=True)

def _timestamp_utc() -> str:
    return datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

def _now_ms() -> int:
    return int(time.time() * 1000)

def _split_base_ext(path: str) -> Tuple[str, str]:
    base, ext = os.path.splitext(path)
    if not ext:
        ext = ".csv"
    return base, ext

# -------------------------------
# Config loader with dot-getter
# -------------------------------

class Config:
    def __init__(self, path: str):
        self._path = path
        with open(path, "r", encoding="utf-8") as f:
            self._cfg: Dict[str, Any] = yaml.safe_load(f) or {}
        self._default_rates: Dict[str, float] = {
            "HEARTBEAT": 1,
            "RC_CHANNELS": 20,
            "ATTITUDE": 10,
            "GLOBAL_POSITION_INT": 10,
            "GPS_RAW_INT": 5,
            "VFR_HUD": 5,
            "SYS_STATUS": 1,
            "BATTERY_STATUS": 1,
            "SERVO_OUTPUT_RAW": 5,
            "HIGHRES_IMU": 50,
            "SCALED_IMU2": 25,
            "SCALED_IMU3": 25,
            "RAW_IMU": 0,
            "SCALED_PRESSURE": 5,
            "SCALED_PRESSURE2": 5,
            "SCALED_PRESSURE3": 5,
            "WIND": 1,
            "LOCAL_POSITION_NED": 0,
            "LOCAL_POSITION_NED_COV": 0,
            "ATTITUDE_QUATERNION": 0,
            "AHRS2": 0,
            "AHRS3": 0,
            "EKF_STATUS_REPORT": 0,
            "NAV_CONTROLLER_OUTPUT": 0,
            "EXTENDED_SYS_STATE": 1,
            "MISSION_CURRENT": 0,
            "HOME_POSITION": 0,
            "NAMED_VALUE_FLOAT": 0,
            "STATUSTEXT": 0,
        }

    def as_dict(self) -> Dict[str, Any]:
        return dict(self._cfg)

    def get(self, dotted_key: str, default: Any = None) -> Any:
        node: Any = self._cfg
        if not dotted_key:
            return default
        for part in dotted_key.split("."):
            if isinstance(node, dict) and part in node:
                node = node[part]
            else:
                return default
        return node

    def set(self, dotted_key: str, value: Any) -> None:
        parts = dotted_key.split(".")
        node = self._cfg
        for p in parts[:-1]:
            if p not in node or not isinstance(node[p], dict):
                node[p] = {}
            node = node[p]
        node[parts[-1]] = value

    def get_default_mavlink_rates(self) -> Dict[str, float]:
        return dict(self._default_rates)

# -------------------------------
# CSV logger
# -------------------------------

class SingleCsvLogger:
    """
    Minimal CSV logger with auto-header expansion and optional rotation by bytes.

    Public attrs/APIs expected elsewhere:
      - enabled (bool)
      - path (str)
      - log(kind, name, payload, channel="", src="")
      - close()

    Back-compat shims for legacy callers:
      - write_json(name, payload, channel="", src="server")
      - write_mavlink(name, payload, channel="", src="mav")
    """

    def __init__(
        self,
        path: str,
        *,
        enabled: bool = True,
        max_bytes: int = 10 * 1024 * 1024,
        include_json: Optional[Dict[str, bool]] = None,
        include_mavlink: Optional[Dict[str, bool]] = None,
        **kwargs,
    ):
        self.path = path
        self.enabled = bool(enabled)
        self.max_bytes = max(0, int(max_bytes))
        self.include_json = include_json or {}
        self.include_mavlink = include_mavlink or {}

        self._base = ""
        self._ext = ""
        self._f = None
        self._writer = None
        self._header: List[str] = []
        self._bytes = 0

        if self.enabled:
            _ensure_dir(self.path)
            self._base, self._ext = _split_base_ext(self.path)
            self._open()

    # Legacy shims
    def write_json(self, name: str, payload: dict, channel: str = "", src: str = "server"):
        return self.log(kind="json", name=name, payload=payload, channel=channel, src=src)

    def write_mavlink(self, name: str, payload: dict, channel: str = "", src: str = "mav"):
        return self.log(kind="mavlink", name=name, payload=payload, channel=channel, src=src)

    # Unified API
    def log(self, *, kind: str, name: str, payload: Dict[str, Any], channel: str = "", src: str = ""):
        if not self.enabled:
            return
        if not isinstance(payload, dict):
            payload = {"value": payload}
        # selection filters
        if kind == "json" and self.include_json:
            if not self.include_json.get(name, False):
                return
        if kind == "mavlink" and self.include_mavlink:
            if not self.include_mavlink.get(name, False):
                return
        row = {
            "ts_utc": _timestamp_utc(),
            "ts_ms": _now_ms(),
            "kind": kind,
            "name": name,
            "channel": channel,
            "src": src,
        }
        for k, v in payload.items():
            row[k] = v
        self._maybe_expand_header(row)
        self._writer.writerow([row.get(col, "") for col in self._header])
        try:
            self._f.flush()
        except Exception:
            pass
        self._bytes += sum(len(str(x)) for x in row.values()) + len(self._header)
        if self.max_bytes > 0 and self._bytes >= self.max_bytes:
            self._rotate()

    # internals
    def _open(self):
        exists = os.path.exists(self.path)
        self._f = open(self.path, "a", newline="", encoding="utf-8")
        self._writer = csv.writer(self._f)
        if not exists:
            self._header = ["ts_utc", "ts_ms", "kind", "name", "channel", "src"]
            self._writer.writerow(self._header)
            try: self._f.flush()
            except Exception: pass
            self._bytes = 0
        else:
            try:
                with open(self.path, "r", encoding="utf-8") as r:
                    first = r.readline()
                    hdr = [h.strip() for h in first.strip().split(",")] if first else []
                    self._header = hdr if hdr else ["ts_utc", "ts_ms", "kind", "name", "channel", "src"]
            except Exception:
                self._header = ["ts_utc", "ts_ms", "kind", "name", "channel", "src"]

    def _maybe_expand_header(self, row: Dict[str, Any]):
        new_cols = [k for k in row.keys() if k not in self._header]
        if not new_cols:
            return
        self._header.extend(sorted(new_cols))
        try:
            self._f.close()
        except Exception:
            pass
        try:
            with open(self.path, "r", encoding="utf-8") as rf:
                lines = rf.readlines()
        except Exception:
            lines = []
        self._f = open(self.path, "w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._f)
        self._writer.writerow(self._header)
        if lines:
            old_header = [h.strip() for h in lines[0].strip().split(",")] if lines[0:1] else []
            if old_header:
                for line in lines[1:]:
                    cells = [c.strip() for c in line.rstrip("\n").split(",")]
                    row_map = {old_header[i]: (cells[i] if i < len(cells) else "") for i in range(len(old_header))}
                    self._writer.writerow([row_map.get(col, "") for col in self._header])
        try:
            self._f.flush()
        except Exception:
            pass

    def _rotate(self):
        ts = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%SZ")
        rotated = f"{self._base}_{ts}{self._ext}"
        try:
            try: self._f.close()
            except Exception: pass
            os.replace(self.path, rotated)
        except Exception:
            pass
        finally:
            self._open()
            self._bytes = 0

    def close(self):
        if not self.enabled:
            return
        try:
            if self._f:
                self._f.close()
        except Exception:
            pass

# -------------------------------
# Helper
# -------------------------------

def diff_mavlink_rates(defaults: Dict[str, float], chosen: Dict[str, float]) -> Dict[str, Tuple[Optional[float], Optional[float]]]:
    out: Dict[str, Tuple[Optional[float], Optional[float]]] = {}
    keys = set(defaults.keys()) | set((chosen or {}).keys())
    for k in keys:
        dv = defaults.get(k)
        cv = (chosen or {}).get(k, dv)
        if dv != cv:
            out[k] = (dv, cv)
    return out
