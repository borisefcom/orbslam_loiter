# telemetry.py
import time
import threading
import math
from typing import Optional, Dict, Any, Tuple
from copy import deepcopy

from config import SingleCsvLogger

# --- robust cellular import (handles different function names / signatures) ---
_cellular_fn = None
try:
    from cellular import cellular_snapshot as _cellular_fn  # preferred
except Exception:
    try:
        import cellular as _cell_mod  # type: ignore
        _cellular_fn = (
            getattr(_cell_mod, "cellular_snapshot", None)
            or getattr(_cell_mod, "snapshot", None)
            or getattr(_cell_mod, "cellular_status", None)
            or getattr(_cell_mod, "get_status", None)
        )
    except Exception:
        _cellular_fn = None

# --- robust sys-stats import (optional on Linux only) ---
_sys_fn = None
try:
    from sys_stats import system_snapshot as _sys_fn  # type: ignore
except Exception:
    _sys_fn = None

def _now_ms() -> int:
    return int(time.time() * 1000)

# ---------------- Cellular helpers ----------------

def _norm_cellular(raw: Any) -> dict:
    ts = _now_ms()
    base: dict = {
        "type": "cellular",
        "ts": ts,
        "available": None,
        "status": {
            "state": None,
            "power": None,
            "access_tech": None,
            "signal_quality": None
        },
        "signal": {
            "dbm": None,
        },
        "registration": {
            "state": None,
            "operator_id": None,
            "operator_name": None
        },
        "connection": {
            "connected": None,
            "iface": None,
            "ipv4": None,
            "ipv6": None,
            "mtu": None
        }
    }

    if not isinstance(raw, dict):
        base["available"] = False
        return base

    base["available"] = raw.get("available", raw.get("ok", raw.get("reachable", None)))

    st = raw.get("status", {})
    if isinstance(st, dict):
        base["status"]["state"] = st.get("state", raw.get("state"))
        base["status"]["power"] = st.get("power", raw.get("power"))
        base["status"]["access_tech"] = st.get("access_tech", st.get("rat", raw.get("access")))
        base["status"]["signal_quality"] = st.get("signal_quality", st.get("sq", raw.get("signal_quality")))
    else:
        base["status"]["state"] = raw.get("state")
        base["status"]["power"] = raw.get("power")
        base["status"]["access_tech"] = raw.get("access_tech")
        base["status"]["signal_quality"] = raw.get("signal_quality")

    sig = raw.get("signal", {})
    if isinstance(sig, dict):
        base["signal"]["dbm"] = sig.get("dbm", raw.get("rssi_dbm"))
        for k in ("rsrp", "rsrq", "sinr"):
            if k in sig:
                base["signal"][k] = sig.get(k)
            elif k in raw:
                base["signal"][k] = raw.get(k)
    else:
        base["signal"]["dbm"] = raw.get("dbm", raw.get("rssi_dbm"))

    reg = raw.get("registration", {})
    if isinstance(reg, dict):
        base["registration"]["state"] = reg.get("state", raw.get("reg_state"))
        base["registration"]["operator_id"] = reg.get("operator_id", raw.get("operator_id"))
        base["registration"]["operator_name"] = reg.get("operator_name", raw.get("operator"))
    else:
        base["registration"]["state"] = raw.get("reg_state")
        base["registration"]["operator_id"] = raw.get("operator_id")
        base["registration"]["operator_name"] = raw.get("operator")

    conn = raw.get("connection", {})
    if isinstance(conn, dict):
        base["connection"]["connected"] = conn.get("connected", raw.get("connected"))
        base["connection"]["iface"] = conn.get("iface", raw.get("iface"))
        base["connection"]["ipv4"] = conn.get("ipv4", raw.get("ipv4"))
        base["connection"]["ipv6"] = conn.get("ipv6", raw.get("ipv6"))
        base["connection"]["mtu"] = conn.get("mtu", raw.get("mtu"))
    else:
        base["connection"]["connected"] = raw.get("connected")
        base["connection"]["iface"] = raw.get("iface")
        base["connection"]["ipv4"] = raw.get("ipv4")
        base["connection"]["ipv6"] = raw.get("ipv6")
        base["connection"]["mtu"] = raw.get("mtu")

    return base

def _quantize_cellular(cell: dict) -> dict:
    c = deepcopy(cell)
    try:
        sig = c.get("signal", {})
        for k in ("dbm", "rsrp", "rsrq", "sinr"):
            if k in sig and isinstance(sig[k], (int, float)):
                sig[k] = round(float(sig[k]), 1)
        st = c.get("status", {})
        if "signal_quality" in st and isinstance(st["signal_quality"], (int, float)):
            st["signal_quality"] = int(round(float(st["signal_quality"])))
    except Exception:
        pass
    return c

# ---------------- Sys helpers ----------------

def _norm_sys(raw: Any) -> dict:
    ts = _now_ms()
    base: dict = {
        "type": "sys",
        "ts": ts,
        "available": None,
        "ok": None,
        "source": None,
        "cpu_pct": None,   # [core1..coreN] percent
        "mem_pct": None,
        "swap_pct": None,
        "err": None,
    }

    if not isinstance(raw, dict):
        base["available"] = False
        base["ok"] = False
        base["err"] = "invalid"
        return base

    base["available"] = bool(raw.get("available", True))
    base["ok"] = bool(raw.get("ok", True))
    base["source"] = raw.get("source")
    base["cpu_pct"] = raw.get("cpu_pct")
    base["mem_pct"] = raw.get("mem_pct")
    base["swap_pct"] = raw.get("swap_pct")
    base["err"] = raw.get("err")
    return base


def _quantize_sys(sysm: dict) -> dict:
    s = deepcopy(sysm)
    try:
        cpu = s.get("cpu_pct", None)
        if isinstance(cpu, list):
            s["cpu_pct"] = [round(float(x), 1) for x in cpu[:8]]
        if isinstance(s.get("mem_pct"), (int, float)):
            s["mem_pct"] = round(float(s["mem_pct"]), 1)
        if isinstance(s.get("swap_pct"), (int, float)):
            s["swap_pct"] = round(float(s["swap_pct"]), 1)
    except Exception:
        pass
    return s

# ---------------- Compare helpers ----------------

def _without_ts(payload: dict) -> dict:
    p = deepcopy(payload)
    p.pop("ts", None)
    return p

def _deep_equal(a, b) -> bool:
    if type(a) != type(b):
        return False
    if isinstance(a, dict):
        if set(a.keys()) != set(b.keys()):
            return False
        return all(_deep_equal(a[k], b[k]) for k in a.keys())
    if isinstance(a, (list, tuple)):
        if len(a) != len(b):
            return False
        return all(_deep_equal(x, y) for x, y in zip(a, b))
    return a == b


class _Periodic(threading.Thread):
    def __init__(self, name: str, hz: float, fn):
        super().__init__(daemon=True, name=name)
        # Allow rates below 1 Hz (e.g., 0.2 Hz -> 5s), but clamp away from 0 to avoid div-by-zero.
        self._period = 1.0 / max(1e-6, float(hz))
        self._fn = fn
        self._stop = threading.Event()

    def run(self):
        next_t = time.time()
        while not self._stop.is_set():
            next_t += self._period
            try:
                self._fn()
            except Exception as e:
                print(f"[{self.name}] tick error: {e}", flush=True)
            delay = next_t - time.time()
            if delay > 0:
                time.sleep(delay)
            else:
                next_t = time.time()

    def stop(self):
        self._stop.set()


class TelemetryManager:
    """
    Periodic JSON telemetry for pose and cellular modem.
    Incoming control handler supports:
      - Video: video_to_sender, dest_ip/dest_port, bitrate_kbps, rc_mode, gop, idr, stream_method
      - MAVLink: mav_mirror_ip
      - Telemetry: telemetry_ip / telemetry_port
      - Tracking: track_select (int or {track_id}), roi_select, track_cancel
      - Mouse (future): type="mouse_move" | "mouse_stop" | "mouse_click"
    """

    def __init__(
        self,
        cfg,
        json_tx,
        csvlog: SingleCsvLogger,
        mav,
        video,
        tracker=None,
        *,
        pose_hz: float = None,
        cell_hz: float = None,
        modem_index: int = None,
        **kwargs
    ):
        self.cfg = cfg
        self.json_tx = json_tx
        self.csvlog = csvlog
        self.mav = mav
        self.video = video
        self.tracker = tracker

        self.pose_hz = float(pose_hz if pose_hz is not None else (cfg.get("telemetry.json.pose_hz", 10.0)))
        self.cell_hz = float(cell_hz if cell_hz is not None else (cfg.get("telemetry.json.cellular_hz", 2.0)))
        self.sys_hz = float(cfg.get("telemetry.json.sys_hz", 2.0))
        self.modem_index = int(modem_index if modem_index is not None else (cfg.get("cellular.modem_index", 0)))

        self._selected_id: Optional[int] = None
        self._last_pose_no_ts: Optional[dict] = None
        self._last_pose_send_wall_s: float = 0.0
        self._last_cell_no_ts: Optional[dict] = None
        self._last_sys_no_ts: Optional[dict] = None
        self._last_mouse_move: Optional[Dict[str, Any]] = None
        self._last_mouse_stop: Optional[Dict[str, Any]] = None
        self._last_mouse_click: Optional[Dict[str, Any]] = None

        self.pose_th: Optional[_Periodic] = None
        self.cell_th: Optional[_Periodic] = None
        self.sys_th: Optional[_Periodic] = None

        self._vo_pose: Optional[dict] = None
        self._vo_pose_wall_s: float = 0.0
        self._vo_lock = threading.Lock()
        self._vo_cond = threading.Condition(self._vo_lock)
        self._vo_seq: int = 0

        # Emit a periodic pose keepalive even if the pose hasn't changed (UDP is lossy and
        # the client may start after the last "pose unavailable" send).
        try:
            self.pose_keepalive_s = float(cfg.get("telemetry.json.pose_keepalive_s", 1.0))
        except Exception:
            self.pose_keepalive_s = 1.0
        if float(self.pose_keepalive_s) < 0.0:
            self.pose_keepalive_s = 0.0

    # ---------------- External inputs (from other threads) ----------------

    def set_vo_pose(self, vo_pose: Optional[dict]) -> None:
        """Update latest VO pose snapshot (consumed by periodic pose publisher)."""
        with self._vo_cond:
            self._vo_pose = dict(vo_pose) if isinstance(vo_pose, dict) else None
            self._vo_pose_wall_s = float(time.time()) if isinstance(self._vo_pose, dict) else 0.0
            self._vo_seq += 1
            try:
                self._vo_cond.notify_all()
            except Exception:
                pass

    def get_vo_pose_snapshot(self) -> Tuple[Optional[dict], float]:
        """Return a (deepcopy(vo_pose), ts_wall_s) snapshot for cross-thread consumers."""
        with self._vo_lock:
            vo_pose = deepcopy(self._vo_pose) if isinstance(self._vo_pose, dict) else None
            ts_wall = float(self._vo_pose_wall_s)
        return vo_pose, ts_wall

    def wait_vo_pose_snapshot(self, *, last_seq: int, timeout_s: float) -> Tuple[int, Optional[dict], float]:
        """Block until the VO pose updates (seq changes) or timeout.

        Returns (seq, vo_pose, vo_pose_wall_s). If seq == last_seq, it timed out.
        """
        timeout_s = float(timeout_s) if timeout_s is not None else 0.0
        if timeout_s < 0.0:
            timeout_s = 0.0
        with self._vo_cond:
            if int(self._vo_seq) == int(last_seq):
                try:
                    self._vo_cond.wait(timeout=timeout_s)
                except Exception:
                    pass
            seq = int(self._vo_seq)
            vo_pose = deepcopy(self._vo_pose) if isinstance(self._vo_pose, dict) else None
            ts_wall = float(self._vo_pose_wall_s)
        return seq, vo_pose, ts_wall

    # ---------------- Periodic publishers ----------------

    def start(self):
        """Start periodic pose & cellular publishers."""

        def tick_pose():
            # Pose is now sourced from the 3D tracker VO (track3d), when available.
            with self._vo_lock:
                vo_pose = deepcopy(self._vo_pose) if isinstance(self._vo_pose, dict) else None

            if vo_pose is None:
                payload = {"type": "pose", "ts": _now_ms(), "vo_pose": None, "reason": "unavailable"}
            else:
                # Reduce UI noise / payload churn: round VO position to 0.1 m.
                try:
                    pos = vo_pose.get("position_m", None) if isinstance(vo_pose, dict) else None
                    if isinstance(pos, dict):
                        for k in ("x", "y", "z"):
                            v = pos.get(k, None)
                            if isinstance(v, (int, float)):
                                pos[k] = round(float(v), 1)
                except Exception:
                    pass
                payload = {"type": "pose", "ts": _now_ms(), "vo_pose": vo_pose, "reason": None}

            # Back-compat alias (older clients may still look for this key).
            payload.setdefault("zed_pose", payload.get("vo_pose"))

            cmp_payload = _without_ts(payload)
            now_s = float(time.time())
            do_send = False
            if vo_pose is not None:
                # When VO is available, publish at the configured `pose_hz` even if the pose is unchanged.
                # The GUI rate indicators depend on a steady cadence.
                do_send = True
            else:
                keepalive_ok = float(self.pose_keepalive_s) > 0.0 and (
                    float(now_s) - float(self._last_pose_send_wall_s)
                ) >= float(self.pose_keepalive_s)
                do_send = (
                    self._last_pose_no_ts is None
                    or not _deep_equal(cmp_payload, self._last_pose_no_ts)
                    or bool(keepalive_ok)
                )

            if bool(do_send):
                self.json_tx.send(payload)
                self.csvlog.log(kind="json", name="pose", payload=payload, channel="", src="server")
                self._last_pose_no_ts = cmp_payload
                self._last_pose_send_wall_s = float(now_s)

        def tick_cell():
            if _cellular_fn is None:
                raw = {"available": False}
            else:
                try:
                    try:
                        raw = _cellular_fn(self.modem_index)  # type: ignore[call-arg]
                    except TypeError:
                        raw = _cellular_fn()  # type: ignore[misc]
                except Exception as e:
                    raw = {"available": False, "error": repr(e)}

            norm = _norm_cellular(raw)
            normq = _quantize_cellular(norm)

            cmp_payload = _without_ts(normq)
            if self._last_cell_no_ts is None or not _deep_equal(cmp_payload, self._last_cell_no_ts):
                self.json_tx.send(normq)
                self.csvlog.log(kind="json", name="cellular", payload=normq, channel="", src="server")
                self._last_cell_no_ts = cmp_payload

        def tick_sys():
            if _sys_fn is None:
                raw = {"available": False, "ok": False, "err": "sys_stats_unavailable"}
            else:
                try:
                    raw = _sys_fn()  # type: ignore[misc]
                except Exception as e:
                    raw = {"available": False, "ok": False, "err": repr(e)}

            norm = _norm_sys(raw)
            normq = _quantize_sys(norm)

            cmp_payload = _without_ts(normq)
            if self._last_sys_no_ts is None or not _deep_equal(cmp_payload, self._last_sys_no_ts):
                self.json_tx.send(normq)
                self.csvlog.log(kind="json", name="sys", payload=normq, channel="", src="server")
                self._last_sys_no_ts = cmp_payload

        if self.pose_hz > 0.0:
            self.pose_th = _Periodic("pose", self.pose_hz, tick_pose)
            self.pose_th.start()

        # cellular_hz<=0 disables cellular telemetry entirely
        if self.cell_hz > 0.0:
            self.cell_th = _Periodic("cellular", self.cell_hz, tick_cell)
            self.cell_th.start()

        # sys_hz<=0 disables sys telemetry entirely
        if self.sys_hz > 0.0:
            self.sys_th = _Periodic("sys", self.sys_hz, tick_sys)
            self.sys_th.start()

    def stop(self):
        if self.pose_th: self.pose_th.stop()
        if self.cell_th: self.cell_th.stop()
        if self.sys_th: self.sys_th.stop()

    # ---------------- Control JSON handler ----------------

    def handle_control(self, msg: Dict[str, Any], *, sender_addr: Optional[Tuple[str,int]] = None) -> Dict[str, Any]:
        """
        Apply control dictionary. Returns dict of applied changes (used only for local prints).
        No JSON acknowledgements are emitted from here.
        """
        changed: Dict[str, Any] = {}
        if not isinstance(msg, dict):
            return changed

        # ---- Mouse position (future functionality) ----
        try:
            if str(msg.get("type", "")).strip().lower() == "mouse_move":
                # Store a normalized snapshot for future use; do not act on it yet.
                self._last_mouse_move = {
                    "ts": int(msg.get("ts") or msg.get("t_ms") or _now_ms()),
                    "surface": msg.get("surface", "osd_main"),
                    "x_fb": msg.get("x_fb"),
                    "y_fb": msg.get("y_fb"),
                    "fb_w": msg.get("fb_w"),
                    "fb_h": msg.get("fb_h"),
                    "x_norm": msg.get("x_norm"),
                    "y_norm": msg.get("y_norm"),
                }
            if str(msg.get("type", "")).strip().lower() == "mouse_click":
                # Store a normalized snapshot for future use; do not act on it yet.
                self._last_mouse_click = {
                    "ts": int(msg.get("ts") or msg.get("t_ms") or _now_ms()),
                    "surface": msg.get("surface", "osd_main"),
                    "button": msg.get("button"),
                    "x_fb": msg.get("x_fb"),
                    "y_fb": msg.get("y_fb"),
                    "fb_w": msg.get("fb_w"),
                    "fb_h": msg.get("fb_h"),
                    "x_norm": msg.get("x_norm"),
                    "y_norm": msg.get("y_norm"),
                }
            if str(msg.get("type", "")).strip().lower() == "mouse_stop":
                # Store a normalized snapshot for future use; do not act on it yet.
                self._last_mouse_stop = {
                    "ts": int(msg.get("ts") or msg.get("t_ms") or _now_ms()),
                    "surface": msg.get("surface", "osd_main"),
                    "x_fb": msg.get("x_fb"),
                    "y_fb": msg.get("y_fb"),
                    "fb_w": msg.get("fb_w"),
                    "fb_h": msg.get("fb_h"),
                    "x_norm": msg.get("x_norm"),
                    "y_norm": msg.get("y_norm"),
                }
        except Exception:
            pass

        # ---- Telemetry out retarget ----
        if ("telemetry_ip" in msg) or ("telemetry_port" in msg):
            try:
                cur_host, cur_port = getattr(self.json_tx, "addr", (None, None))
            except Exception:
                cur_host, cur_port = (None, None)
            host = msg.get("telemetry_ip", cur_host) or "127.0.0.1"
            port = int(msg.get("telemetry_port", cur_port or 6021))
            try:
                self.json_tx.addr = (host, port)  # type: ignore[attr-defined]
                print(f"[control] telemetry_out {host}:{port}")
                changed["telemetry_out"] = f"{host}:{port}"
            except Exception as e:
                print(f"[control] telemetry_out error: {e}")

        # ---- Video controls ----
        if self.video is not None:
            v: Dict[str, Any] = {}
            sender_ip = (sender_addr[0] if sender_addr else None)

            if msg.get("video_to_sender") is True and sender_ip:
                sink = self.cfg.get("video.udp_sink", "127.0.0.1:5600")
                try:
                    from udp import parse_hostport
                    _, port = parse_hostport(sink)
                except Exception:
                    port = 5600
                v["udp_sink"] = f"{sender_ip}:{port}"
                print(f"[control] video_out {v['udp_sink']}"); changed["udp_sink"] = v["udp_sink"]

            if ("dest_ip" in msg) or ("dest_port" in msg):
                sink = self.cfg.get("video.udp_sink", "127.0.0.1:5600")
                try:
                    from udp import parse_hostport
                    cur_host, cur_port = parse_hostport(sink)
                except Exception:
                    cur_host, cur_port = ("127.0.0.1", 5600)
                host = msg.get("dest_ip", cur_host)
                port = int(msg.get("dest_port", cur_port))
                v["udp_sink"] = f"{host}:{port}"
                print(f"[control] video_out {v['udp_sink']}"); changed["udp_sink"] = v["udp_sink"]

            if "bitrate_kbps" in msg:
                v["bitrate_kbps"] = int(msg["bitrate_kbps"])
                print(f"[control] bitrate {v['bitrate_kbps']} kbps"); changed["bitrate_kbps"] = v["bitrate_kbps"]

            if "rc_mode" in msg:
                v["rc_mode"] = str(msg["rc_mode"])
                print(f"[control] rc_mode {v['rc_mode']}"); changed["rc_mode"] = v["rc_mode"]

            if "gop" in msg:
                v["iframeinterval"] = int(msg["gop"])
                print(f"[control] gop {v['iframeinterval']}"); changed["gop"] = v["iframeinterval"]

            if "idr" in msg and msg["idr"]:
                v["idr"] = True; print("[control] idr requested"); changed["idr"] = True

            if "stream_method" in msg:
                v["stream_method"] = str(msg["stream_method"])
                print(f"[control] stream_method {v['stream_method']}"); changed["stream_method"] = v["stream_method"]

            if v:
                try:
                    self.video.apply_control(v)
                    if "udp_sink" in v:
                        # update live cfg so future video_to_sender uses new port
                        try:
                            self.cfg.set("video.udp_sink", v["udp_sink"])
                        except Exception:
                            pass
                except Exception as e:
                    print(f"[control] video apply error: {e}")

        # ---- MAVLink mirror controls ----
        if self.mav is not None:
            if "mav_mirror_ip" in msg:
                try:
                    host = str(msg.get("mav_mirror_ip") or "").strip()
                    if host:
                        # Keep existing mirror port if possible.
                        port = None
                        try:
                            cur = getattr(self.mav, "_mirror_addr", None)
                            if cur is not None:
                                port = int(cur[1])
                        except Exception:
                            port = None
                        if port is None:
                            try:
                                from udp import parse_hostport
                                cur_udp = self.cfg.get("px4.mirror.udp", "")
                                _, port = parse_hostport(str(cur_udp))
                                port = int(port)
                            except Exception:
                                port = 14550

                        # Update in-memory cfg for visibility.
                        try:
                            self.cfg.set("px4.mirror.udp", f"{host}:{int(port)}")
                        except Exception:
                            pass

                        # Update running mirror destination if supported.
                        try:
                            if hasattr(self.mav, "set_mirror_destination"):
                                self.mav.set_mirror_destination(host, int(port))
                            else:
                                setattr(self.mav, "_mirror_addr", (host, int(port)))
                        except Exception:
                            pass

                        print(f"[control] mav_mirror udp://{host}:{int(port)}")
                        changed["mav_mirror"] = f"{host}:{int(port)}"
                except Exception:
                    pass

        # ---- Tracker controls (no ROI) ----
        if self.tracker is not None:
            if "track_select" in msg:
                val = msg.get("track_select")
                try:
                    tid = int(val["track_id"]) if isinstance(val, dict) else int(val)
                    if hasattr(self.tracker, "select"):
                        self.tracker.select(tid)
                    self._selected_id = tid
                    print(f"[TRACK] target_select id={tid}")
                    changed["track_selected"] = tid
                except Exception:
                    pass

            if "roi_select" in msg:
                val = msg.get("roi_select")
                bbox = None
                try:
                    if isinstance(val, dict) and "bbox" in val:
                        bbox = val.get("bbox")
                    else:
                        bbox = val
                    if hasattr(self.tracker, "roi_select"):
                        self.tracker.roi_select(bbox)
                    elif hasattr(self.tracker, "target_roi_select"):
                        self.tracker.target_roi_select(bbox)
                    print(f"[TRACK] roi_select bbox={bbox}")
                    self._selected_id = 0
                    changed["track_selected"] = 0
                except Exception:
                    pass

            if msg.get("track_cancel") is True:
                if hasattr(self.tracker, "cancel"):
                    self.tracker.cancel()
                self._selected_id = None
                print("[TRACK] target_lost_primary")
                changed["track_selected"] = None

        return changed
