#!/usr/bin/env python3
import time
import math
import json
import socket
import threading
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict, Any

from pymavlink import mavutil

# ---------------- Telemetry snapshot ----------------
@dataclass
class Telemetry:
    # Attitude (radians)
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0

    # Heading (deg 0..360) and basic flight data
    heading_deg: float = 0.0
    alt_m: float = 0.0
    groundspeed: float = 0.0
    climb_ms: float = 0.0
    battery_v: float = 0.0
    current_a: Optional[float] = None
    battery_remaining_pct: Optional[int] = None
    armed: bool = False
    mode: Optional[str] = None

    # Optional RC/throttle convenience (if present in your pipeline)
    throttle_norm: Optional[float] = None
    throttle: Optional[float] = None
    throttle_pct: Optional[float] = None

    # --- VIS odometry (from JSON "pose") ---
    vis_x: Optional[float] = None
    vis_y: Optional[float] = None
    vis_z: Optional[float] = None
    vis_ok: bool = False
    vis_err: Optional[str] = None

    # --- System stats (from JSON "sys") ---
    sys_cpu1: Optional[float] = None
    sys_cpu2: Optional[float] = None
    sys_cpu3: Optional[float] = None
    sys_cpu4: Optional[float] = None
    sys_mem: Optional[float] = None
    sys_swap: Optional[float] = None
    sys_ok: bool = False
    sys_err: Optional[str] = None

    # --- Cellular summary (from JSON "cellular") ---
    cell_state: Optional[str] = None           # e.g. "connected", "searching"
    cell_access_tech: Optional[str] = None     # e.g. "lte", "nr5g"
    cell_operator_name: Optional[str] = None   # e.g. "Cellcom"
    cell_dbm: Optional[int] = None             # RSSI dBm (negative)
    cell_quality: Optional[int] = None         # 0..100
    cell_ok: bool = False
    cell_err: Optional[str] = None

    # --- DEBUG counters for JSON feed ---
    json_rx_count: int = 0
    pose_rx_count: int = 0
    cell_rx_count: int = 0
    sys_rx_count: int = 0
    last_json_ts_ms: int = 0
    last_pose_ts_ms: int = 0
    last_cell_ts_ms: int = 0
    last_sys_ts_ms: int = 0
    last_json_type: Optional[str] = None

    # --- MAVLink GPS ---
    lat_deg: Optional[float] = None
    lon_deg: Optional[float] = None
    alt_amsl_m: Optional[float] = None
    rel_alt_m: Optional[float] = None
    gps_fix_type: Optional[int] = None
    gps_fix: Optional[str] = None
    gps_sats: Optional[int] = None
    gps_hdop: Optional[float] = None
    gps_vdop: Optional[float] = None
    gps_vel_ms: Optional[float] = None
    gps_cog_deg: Optional[float] = None

    # --- Back-compat aliases (used by existing UI/config) ---
    @property
    def lat(self) -> Optional[float]:
        return self.lat_deg

    @property
    def lon(self) -> Optional[float]:
        return self.lon_deg

    @property
    def hdop(self) -> Optional[float]:
        return self.gps_hdop

    @property
    def roll_deg(self) -> float:
        return float(self.roll) * (180.0 / math.pi)

    @property
    def pitch_deg(self) -> float:
        return float(self.pitch) * (180.0 / math.pi)

    @property
    def yaw_deg(self) -> float:
        return float(self.yaw) * (180.0 / math.pi)

    # --- DETECTIONS / TRACKING (legacy "track_status") ---
    detect_img_size: Optional[Tuple[int, int]] = None
    detect_seq: Optional[int] = None
    detections: List[Dict[str, Any]] = field(default_factory=list)  # raw dicts
    tracking_state: str = "idle"  # "idle" | "tracking" | "lost"
    tracking_id: Optional[int] = None
    # recent time-series for track_status (ts, ex, ey, yaw, pitch)
    tracking_hist: List[Tuple[int, float, float, float, float]] = field(default_factory=list)

    # --- PID tune live debug (from JSON "pid_debug") ---
    pid_state: str = "idle"
    pid_gate_rc: Optional[bool] = None
    pid_gate_mode: Optional[bool] = None
    pid_cx_px: Optional[float] = None
    pid_cy_px: Optional[float] = None
    pid_ex_px: Optional[float] = None
    pid_ey_px: Optional[float] = None
    pid_nx: Optional[float] = None
    pid_ny: Optional[float] = None
    pid_yaw: Optional[float] = None
    pid_pitch: Optional[float] = None
    pid_thrust: Optional[float] = None

    # --- 3D VO features overlay (from JSON "vo_features") ---
    vo_feat_img_size: Optional[Tuple[int, int]] = None
    vo_feat_pts: List[Tuple[int, int, int]] = field(default_factory=list)  # (u,v,group)
    # (ts_ms, cx, cy, yaw, pitch, thrust)
    pid_hist: List[Tuple[int, Optional[float], Optional[float], Optional[float], Optional[float], Optional[float]]] = field(default_factory=list)

    # --- 3D acquisition overlay (from JSON "acq_poly") ---
    acq_stage: Optional[int] = None  # 0|1|2 or None
    acq_img_size: Optional[Tuple[int, int]] = None
    acq_verts_uv: Optional[List[Tuple[int, int]]] = None
    acq_center_uv: Optional[Tuple[float, float]] = None

    # --- Active camera / stream selection (from JSON "camera") ---
    camera_active: Optional[str] = None  # "ir" | "depth" | "v4l"
    camera_source: Optional[str] = None  # e.g. "track3d" | "v4l"
    camera_img_size: Optional[Tuple[int, int]] = None
    camera_fps: Optional[float] = None
    camera_intrinsics: Optional[Dict[str, float]] = None  # fx,fy,cx,cy when present

    # --- 2D ROI tracker overlay (from JSON "track2d") ---
    track2d_rx_count: int = 0
    last_track2d_ts_ms: int = 0
    track2d_state: Optional[str] = None
    track2d_img_size: Optional[Tuple[int, int]] = None
    # (x0,y0,x1,y1) in normalized coords 0..1 over the active camera image.
    track2d_bbox_norm: Optional[Tuple[float, float, float, float]] = None
    # (cols, rows) for the DynMedianFlow internal grid (optional visualization aid).
    track2d_grid: Optional[Tuple[int, int]] = None


# ---------------- Helpers ----------------
def _safe_float(v):
    try:
        return float(v)
    except Exception:
        return None

def _safe_int(v):
    try:
        if v is None:
            return None
        return int(v)
    except Exception:
        try:
            return int(float(v))
        except Exception:
            return None

def _norm360(deg):
    if deg is None:
        return None
    d = deg % 360.0
    return d if d >= 0 else d + 360.0


# ---------------- JSON UDP listener / sender ----------------
class JsonTelemetryWorker:
    """
    Listens on UDP for JSON envelopes and updates a shared Telemetry snapshot.
    Sends control JSONs using the SAME bound socket/port as the listener
    (so servers can match on source port).
    Destination:
      - explicit override via set_control_dest(ip, port), or
      - last/first sender learned from incoming JSON.
    """
    def __init__(self, ip: str, port: int, snapshot: Telemetry, log_enabled: bool = True, log_cfg: Any = None):
        self.ip = str(ip)
        self.port = int(port)
        self._snap = snapshot
        self._stop = False
        self._th: Optional[threading.Thread] = None

        self._rxsock: Optional[socket.socket] = None  # single socket for RX+TX
        self._last_sender: Optional[Tuple[str, int]] = None
        self._ctrl_dest: Optional[Tuple[str, int]] = None

        # Debug logging
        self._log_cfg: Dict[str, Any] = self._normalize_log_cfg(log_enabled=log_enabled, log_cfg=log_cfg)
        self._clip = 240

    def set_logging(self, enabled: bool) -> None:
        self._log_cfg["enabled"] = bool(enabled)

    def set_log_cfg(self, cfg: Any) -> None:
        self._log_cfg = self._normalize_log_cfg(log_enabled=bool(self._log_cfg.get("enabled", False)), log_cfg=cfg)

    def _normalize_log_cfg(self, *, log_enabled: bool, log_cfg: Any) -> Dict[str, Any]:
        """
        Supported shapes:
          - None: use legacy log_enabled boolean (True prints everything, False prints nothing)
          - bool: same as legacy
          - dict:
              {
                "enabled": false,
                "print_lifecycle": false,
                "print_send_skip": false,
                "recv_raw": false,
                "send_raw": false,
                "type_summary": false,
                "types": {"pose": true, "*": false}
              }
        """
        # Defaults: everything off.
        cfg: Dict[str, Any] = {
            "enabled": False,
            "print_lifecycle": False,
            "print_send_skip": False,
            "recv_raw": False,
            "send_raw": False,
            "type_summary": False,
            "types": {},
        }

        # Legacy: bool enables all debug prints for all types.
        if log_cfg is None or isinstance(log_cfg, bool):
            enabled = bool(log_enabled if log_cfg is None else log_cfg)
            if enabled:
                cfg.update(
                    {
                        "enabled": True,
                        "print_lifecycle": True,
                        "print_send_skip": True,
                        "recv_raw": True,
                        "send_raw": True,
                        "type_summary": True,
                        "types": {"*": True},
                    }
                )
            return cfg

        if isinstance(log_cfg, dict):
            cfg["enabled"] = bool(log_cfg.get("enabled", False))
            cfg["print_lifecycle"] = bool(log_cfg.get("print_lifecycle", False))
            cfg["print_send_skip"] = bool(log_cfg.get("print_send_skip", False))
            cfg["recv_raw"] = bool(log_cfg.get("recv_raw", False))
            cfg["send_raw"] = bool(log_cfg.get("send_raw", False))
            cfg["type_summary"] = bool(log_cfg.get("type_summary", False))

            types = log_cfg.get("types", {})
            if isinstance(types, dict):
                out: Dict[str, bool] = {}
                for k, v in types.items():
                    try:
                        out[str(k).lower().strip()] = bool(v)
                    except Exception:
                        pass
                cfg["types"] = out
            elif isinstance(types, list):
                out = {}
                for k in types:
                    try:
                        out[str(k).lower().strip()] = True
                    except Exception:
                        pass
                cfg["types"] = out

            return cfg

        return cfg

    def _type_enabled(self, msg_type: Optional[str]) -> bool:
        types = self._log_cfg.get("types", {})
        if not isinstance(types, dict) or not types:
            return True
        t = (str(msg_type or "").lower().strip()) if msg_type is not None else ""
        if t in types:
            return bool(types.get(t, False))
        if "*" in types:
            return bool(types.get("*", False))
        return True

    def _log_allowed(self, category: str, msg_type: Optional[str] = None) -> bool:
        if not bool(self._log_cfg.get("enabled", False)):
            return False
        if not bool(self._log_cfg.get(category, False)):
            return False
        if msg_type is None:
            return True
        return self._type_enabled(msg_type)

    def _infer_outgoing_type(self, obj: Any) -> str:
        if isinstance(obj, dict):
            if "type" in obj:
                try:
                    return str(obj.get("type", "")).lower().strip()
                except Exception:
                    return ""
            if len(obj) == 1:
                try:
                    return str(next(iter(obj.keys()))).lower().strip()
                except Exception:
                    return ""
        return ""

    def start(self):
        if self._th and self._th.is_alive():
            return
        self._stop = False
        self._th = threading.Thread(target=self._run, daemon=True)
        self._th.start()
        if self._log_allowed("print_lifecycle"):
            print(f"[JSON] listening on udp://{self.ip}:{self.port}")

    def stop(self):
        self._stop = True
        try:
            if self._rxsock:
                self._rxsock.close()
        except Exception:
            pass

    def join(self, timeout: Optional[float] = None) -> None:
        th = self._th
        if th:
            th.join(timeout=timeout)

    def set_control_dest(self, ip: str, port: int):
        """Optionally fix the destination for commands (overrides auto-learn)."""
        self._ctrl_dest = (str(ip), int(port))
        if self._log_allowed("print_lifecycle"):
            print(f"[JSON] control destination override set to udp://{self._ctrl_dest[0]}:{self._ctrl_dest[1]}")

    def send_command(self, obj: dict):
        """
        Send a small control JSON using the same bound socket (preserves source port).
        Chooses destination: explicit override if set, otherwise the last learned sender.
        """
        try:
            if self._rxsock is None:
                if self._log_allowed("print_send_skip"):
                    print("[JSON] send_command ignored: listener socket not ready")
                return False
            dest = self._ctrl_dest or self._last_sender
            if not dest:
                if self._log_allowed("print_send_skip"):
                    print("[JSON] send_command skipped: no destination yet (no JSON received and no override)")
                return False
            data = (json.dumps(obj) + "\n").encode("utf-8")
            self._rxsock.sendto(data, dest)
            out_type = self._infer_outgoing_type(obj)
            if self._log_allowed("send_raw", out_type):
                try:
                    preview = data[:self._clip].decode("utf-8", "replace")
                    print(f"[JSON] SEND -> {dest[0]}:{dest[1]} {len(data)}B: {preview}{'…' if len(data) > self._clip else ''}")
                except Exception:
                    print(f"[JSON] SEND -> udp://{dest[0]}:{dest[1]} : <binary>")
            return True
        except Exception as e:
            print("[JSON] send_command failed:", e)
            return False

    def _run(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        # Windows quirk guard (ignore ICMP port unreachable)
        try:
            sock.ioctl(socket.SIO_UDP_CONNRESET, False)  # type: ignore[attr-defined]
        except Exception:
            pass
        sock.bind((self.ip, self.port))
        sock.settimeout(0.5)
        self._rxsock = sock

        while not self._stop:
            try:
                data, addr = sock.recvfrom(65536)
            except socket.timeout:
                continue
            except Exception as e:
                print("[JSON] recv error:", e)
                continue

            if False:
                try:
                    preview = data[:self._clip].decode("utf-8", errors="replace")
                except Exception:
                    preview = "<binary>"
                print(f"[JSON] RECV <- {addr[0]}:{addr[1]} {len(data)}B: {preview}{'…' if len(data) > self._clip else ''}")

            # Auto-learn dest if not overridden
            self._last_sender = addr
            if self._ctrl_dest is None:
                self._ctrl_dest = addr
                if self._log_allowed("print_lifecycle"):
                    print(f"[JSON] learned control destination: udp://{addr[0]}:{addr[1]}")

            # Try decode & parse
            try:
                msg = json.loads(data.decode("utf-8"))
            except Exception:
                # print a short form to avoid spam
                txt = data.decode("utf-8", errors="replace")
                if len(txt) > 300: txt = txt[:300] + "...(truncated)"
                print("[JSON] decode error from", addr, "payload:", txt)
                continue

            t = (msg.get("type") or "").lower().strip()
            if self._log_allowed("recv_raw", t):
                try:
                    preview = data[:self._clip].decode("utf-8", errors="replace")
                except Exception:
                    preview = "<binary>"
                print(f"[JSON] RECV <- {addr[0]}:{addr[1]} {len(data)}B: {preview}{'…' if len(data) > self._clip else ''}")

            # Apply (best-effort)
            try:
                self._apply_json(msg)
            except Exception as e:
                txt = json.dumps(msg)[:300] + "...(truncated)" if len(json.dumps(msg)) > 300 else json.dumps(msg)
                print("[JSON] apply error:", e, "payload:", txt)

    def _apply_json(self, m: dict):
        t = (m.get("type") or "").lower()
        ts = int(m.get("ts") or m.get("t_ms") or 0)

        if self._log_allowed("type_summary", t):
            try:
                hints = []
                if "vo_pose" in m:
                    hints.append("vo_pose")
                if "zed_pose" in m:
                    hints.append("zed_pose")
                if "signal" in m:
                    hints.append("signal")
                if "detections" in m:
                    try:
                        hints.append(f"dets={len(m['detections'] or [])}")
                    except Exception:
                        pass
                print(f"[JSON] type={t or '—'} ts={ts} {' '.join(hints)}")
            except Exception:
                pass

        self._snap.json_rx_count += 1
        self._snap.last_json_ts_ms = ts
        self._snap.last_json_type = t

        if t == "pose":
            self._snap.pose_rx_count += 1
            self._snap.last_pose_ts_ms = ts
            pose_obj = m.get("vo_pose", None)
            if pose_obj is None:
                pose_obj = m.get("zed_pose", None)
            if pose_obj is None:
                self._snap.vis_x = self._snap.vis_y = self._snap.vis_z = None
                self._snap.vis_ok = False
                self._snap.vis_err = m.get("reason") or "unavailable"
            else:
                # Accept either meters (position_m) or mm (x_mm/y_mm/z_mm)
                if "position_m" in pose_obj:
                    pos = (pose_obj.get("position_m") or {})
                    self._snap.vis_x = _safe_float(pos.get("x"))
                    self._snap.vis_y = _safe_float(pos.get("y"))
                    self._snap.vis_z = _safe_float(pos.get("z"))
                else:
                    def _fmt_mm(v):
                        try:
                            return round(float(v) / 1000.0, 1)
                        except Exception:
                            return None
                    self._snap.vis_x = _fmt_mm(pose_obj.get("x_mm"))
                    self._snap.vis_y = _fmt_mm(pose_obj.get("y_mm"))
                    self._snap.vis_z = _fmt_mm(pose_obj.get("z_mm"))
                self._snap.vis_ok = True if (pose_obj.get("ok", True)) else False
                self._snap.vis_err = pose_obj.get("err")

        elif t == "sys":
            self._snap.sys_rx_count += 1
            self._snap.last_sys_ts_ms = ts
            self._snap.sys_ok = True if bool(m.get("ok", False)) else False
            self._snap.sys_err = m.get("err")
            cpu = m.get("cpu_pct", None)
            if isinstance(cpu, list) and cpu:
                try:
                    self._snap.sys_cpu1 = float(cpu[0]) if len(cpu) > 0 else None
                    self._snap.sys_cpu2 = float(cpu[1]) if len(cpu) > 1 else None
                    self._snap.sys_cpu3 = float(cpu[2]) if len(cpu) > 2 else None
                    self._snap.sys_cpu4 = float(cpu[3]) if len(cpu) > 3 else None
                except Exception:
                    self._snap.sys_cpu1 = self._snap.sys_cpu2 = self._snap.sys_cpu3 = self._snap.sys_cpu4 = None
            else:
                self._snap.sys_cpu1 = self._snap.sys_cpu2 = self._snap.sys_cpu3 = self._snap.sys_cpu4 = None
            try:
                self._snap.sys_mem = float(m.get("mem_pct")) if m.get("mem_pct") is not None else None
            except Exception:
                self._snap.sys_mem = None
            try:
                self._snap.sys_swap = float(m.get("swap_pct")) if m.get("swap_pct") is not None else None
            except Exception:
                self._snap.sys_swap = None

        elif t == "cellular":
            self._snap.cell_rx_count += 1
            self._snap.last_cell_ts_ms = ts
            if not bool(m.get("available", False)):
                self._snap.cell_ok = False
                self._snap.cell_err = m.get("reason") or "unavailable"
                self._snap.cell_state = None
                self._snap.cell_access_tech = None
                self._snap.cell_operator_name = None
                self._snap.cell_dbm = None
                self._snap.cell_quality = None
                return

            self._snap.cell_ok = True
            self._snap.cell_err = None

            status = m.get("status") or {}
            reg    = m.get("registration") or {}
            sig    = m.get("signal") or {}

            self._snap.cell_state = status.get("state") or reg.get("state")
            self._snap.cell_access_tech = status.get("access_tech") or reg.get("access_tech")
            self._snap.cell_operator_name = reg.get("operator_name")
            self._snap.cell_dbm = _safe_int(sig.get("dbm"))
            self._snap.cell_quality = _safe_int(sig.get("quality", (status.get("signal_quality") if isinstance(status.get("signal_quality"), (int, float)) else None)))

        elif t == "detect":
            img_size = m.get("img_size") or [None, None]
            try:
                self._snap.detect_img_size = (int(img_size[0]), int(img_size[1]))
            except Exception:
                self._snap.detect_img_size = None
            self._snap.detect_seq = _safe_int(m.get("seq"))
            dets = m.get("detections") or []
            self._snap.detections = dets  # keep raw dicts

        elif t == "acq_poly":
            # 3D acquisition polygon overlay (preview/selection/confirmed).
            try:
                stage = m.get("stage", None)
                self._snap.acq_stage = int(stage) if stage is not None else None
            except Exception:
                self._snap.acq_stage = None
            img_size = m.get("img_size") or [None, None]
            try:
                self._snap.acq_img_size = (int(img_size[0]), int(img_size[1]))
            except Exception:
                self._snap.acq_img_size = None
            verts = m.get("verts_uv", None)
            if isinstance(verts, list):
                out: List[Tuple[int, int]] = []
                for p in verts:
                    if isinstance(p, (list, tuple)) and len(p) >= 2:
                        try:
                            out.append((int(p[0]), int(p[1])))
                        except Exception:
                            continue
                self._snap.acq_verts_uv = out if out else None
            else:
                self._snap.acq_verts_uv = None
            c = m.get("center_uv", None)
            if isinstance(c, (list, tuple)) and len(c) >= 2:
                try:
                    self._snap.acq_center_uv = (float(c[0]), float(c[1]))
                except Exception:
                    self._snap.acq_center_uv = None
            else:
                self._snap.acq_center_uv = None

        elif t == "vo_features":
            img_size = m.get("img_size") or [None, None]
            try:
                self._snap.vo_feat_img_size = (int(img_size[0]), int(img_size[1]))
            except Exception:
                self._snap.vo_feat_img_size = None
            pts = m.get("pts", None)
            if isinstance(pts, list):
                out2: List[Tuple[int, int, int]] = []
                for p in pts:
                    if isinstance(p, (list, tuple)) and len(p) >= 3:
                        try:
                            out2.append((int(p[0]), int(p[1]), int(p[2])))
                        except Exception:
                            continue
                self._snap.vo_feat_pts = out2
            else:
                self._snap.vo_feat_pts = []

        elif t == "camera":
            # Active camera / stream metadata (used by GUI to gate overlays/inputs).
            act = m.get("active")
            if act is None:
                self._snap.camera_active = None
            else:
                try:
                    self._snap.camera_active = str(act).lower().strip()
                except Exception:
                    self._snap.camera_active = None

            src = m.get("source")
            if src is None:
                self._snap.camera_source = None
            else:
                try:
                    self._snap.camera_source = str(src).lower().strip()
                except Exception:
                    self._snap.camera_source = None

            img_size = m.get("img_size") or [None, None]
            try:
                self._snap.camera_img_size = (int(img_size[0]), int(img_size[1]))
            except Exception:
                self._snap.camera_img_size = None

            self._snap.camera_fps = _safe_float(m.get("fps"))

            intr = m.get("intrinsics")
            if isinstance(intr, dict):
                out_intr: Dict[str, float] = {}
                for k in ("fx", "fy", "cx", "cy"):
                    v = _safe_float(intr.get(k))
                    if v is not None:
                        out_intr[str(k)] = float(v)
                self._snap.camera_intrinsics = out_intr if out_intr else None
            else:
                self._snap.camera_intrinsics = None

        elif t == "track2d":
            # 2D ROI tracker bbox overlay.
            self._snap.track2d_rx_count += 1
            self._snap.last_track2d_ts_ms = ts
            try:
                self._snap.track2d_state = str(m.get("state") or "").lower().strip() or None
            except Exception:
                self._snap.track2d_state = None

            img_size = m.get("img_size") or [None, None]
            try:
                self._snap.track2d_img_size = (int(img_size[0]), int(img_size[1]))
            except Exception:
                self._snap.track2d_img_size = None

            def _clamp01(v: float) -> float:
                return 0.0 if v < 0.0 else (1.0 if v > 1.0 else float(v))

            bbox_norm: Optional[Tuple[float, float, float, float]] = None
            if all(k in m for k in ("x0_norm", "y0_norm", "x1_norm", "y1_norm")):
                try:
                    x0 = float(m.get("x0_norm"))
                    y0 = float(m.get("y0_norm"))
                    x1 = float(m.get("x1_norm"))
                    y1 = float(m.get("y1_norm"))
                    bbox_norm = (_clamp01(min(x0, x1)), _clamp01(min(y0, y1)), _clamp01(max(x0, x1)), _clamp01(max(y0, y1)))
                except Exception:
                    bbox_norm = None
            else:
                bn = m.get("bbox_norm") or m.get("rect_norm") or m.get("bbox")
                if isinstance(bn, (list, tuple)) and len(bn) >= 4:
                    try:
                        x0, y0, x1, y1 = float(bn[0]), float(bn[1]), float(bn[2]), float(bn[3])
                        bbox_norm = (_clamp01(min(x0, x1)), _clamp01(min(y0, y1)), _clamp01(max(x0, x1)), _clamp01(max(y0, y1)))
                    except Exception:
                        bbox_norm = None
                elif isinstance(bn, dict):
                    if all(k in bn for k in ("x0", "y0", "x1", "y1")):
                        try:
                            x0, y0, x1, y1 = float(bn.get("x0")), float(bn.get("y0")), float(bn.get("x1")), float(bn.get("y1"))
                            bbox_norm = (_clamp01(min(x0, x1)), _clamp01(min(y0, y1)), _clamp01(max(x0, x1)), _clamp01(max(y0, y1)))
                        except Exception:
                            bbox_norm = None
                    elif all(k in bn for k in ("x", "y", "w", "h")):
                        try:
                            x = float(bn.get("x"))
                            y = float(bn.get("y"))
                            w = float(bn.get("w"))
                            h = float(bn.get("h"))
                            bbox_norm = (_clamp01(x), _clamp01(y), _clamp01(x + w), _clamp01(y + h))
                        except Exception:
                            bbox_norm = None

            # Fallback: bbox in pixels -> normalize if we know an image size.
            if bbox_norm is None:
                bp = m.get("bbox_px")
                img_sz = self._snap.track2d_img_size or self._snap.camera_img_size
                if isinstance(bp, dict) and img_sz and img_sz[0] and img_sz[1]:
                    try:
                        img_w = float(img_sz[0])
                        img_h = float(img_sz[1])
                        x = float(bp.get("x", 0.0))
                        y = float(bp.get("y", 0.0))
                        w = float(bp.get("w", 0.0))
                        h = float(bp.get("h", 0.0))
                        if img_w > 0 and img_h > 0:
                            x0 = x / img_w
                            y0 = y / img_h
                            x1 = (x + w) / img_w
                            y1 = (y + h) / img_h
                            bbox_norm = (_clamp01(min(x0, x1)), _clamp01(min(y0, y1)), _clamp01(max(x0, x1)), _clamp01(max(y0, y1)))
                    except Exception:
                        bbox_norm = None

            self._snap.track2d_bbox_norm = bbox_norm

            # Optional DynMedianFlow grid dims (used to draw the internal cell grid overlay).
            try:
                gc = m.get("grid_cols", None)
                gr = m.get("grid_rows", None)
                if isinstance(m.get("grid"), dict):
                    g = m.get("grid") or {}
                    gc = g.get("cols", gc)
                    gr = g.get("rows", gr)
                gc_i = int(gc) if gc is not None else None
                gr_i = int(gr) if gr is not None else None
                if gc_i is not None and gr_i is not None and gc_i > 0 and gr_i > 0:
                    self._snap.track2d_grid = (int(gc_i), int(gr_i))
                else:
                    self._snap.track2d_grid = None
            except Exception:
                self._snap.track2d_grid = None

        elif t == "track_status":
            state = str(m.get("state", "idle")).lower()
            self._snap.tracking_state = state
            self._snap.tracking_id = _safe_int(m.get("track_id"))
            err = m.get("error") or {}
            cmd = m.get("cmd") or {}
            ex = _safe_float(err.get("ex"))
            ey = _safe_float(err.get("ey"))
            yaw = _safe_float(cmd.get("yaw"))
            pitch = _safe_float(cmd.get("pitch"))
            ts_val = int(m.get("ts") or 0)
            self._snap.tracking_hist.append((ts_val, ex or 0.0, ey or 0.0, yaw or 0.0, pitch or 0.0))
            if len(self._snap.tracking_hist) > 300:
                self._snap.tracking_hist = self._snap.tracking_hist[-300:]

        elif t == "pid_debug":
            # New rich PID debug payload
            self._snap.pid_state = str(m.get("state", "idle")).lower()

            gate = m.get("gate") or {}
            self._snap.pid_gate_rc = bool(gate.get("rc")) if gate.get("rc") is not None else None
            self._snap.pid_gate_mode = bool(gate.get("mode")) if gate.get("mode") is not None else None

            meas = m.get("measurement") or {}
            self._snap.pid_cx_px = _safe_float(meas.get("cx_px"))
            self._snap.pid_cy_px = _safe_float(meas.get("cy_px"))
            self._snap.pid_ex_px = _safe_float(meas.get("ex_px"))
            self._snap.pid_ey_px = _safe_float(meas.get("ey_px"))
            self._snap.pid_nx = _safe_float(meas.get("nx"))
            self._snap.pid_ny = _safe_float(meas.get("ny"))

            res = m.get("result") or {}
            self._snap.pid_pitch = _safe_float(res.get("pitch"))
            self._snap.pid_yaw = _safe_float(res.get("yaw"))
            self._snap.pid_thrust = _safe_float(res.get("thrust"))

            ts_val = ts
            self._snap.pid_hist.append((
                ts_val,
                self._snap.pid_cx_px,
                self._snap.pid_cy_px,
                self._snap.pid_yaw,
                self._snap.pid_pitch,
                self._snap.pid_thrust
            ))
            if len(self._snap.pid_hist) > 300:
                self._snap.pid_hist = self._snap.pid_hist[-300:]


# ---------------- MAVLink worker ----------------
class TelemetryWorker:
    """
    Reads MAVLink and updates a shared Telemetry snapshot.
    Use .get() to obtain the live snapshot object (same instance each time).
    """
    def __init__(self, endpoint: str):
        self.endpoint = endpoint
        self._stop = False
        self._th: Optional[threading.Thread] = None
        self._snap = Telemetry()
        # MAVLink sources: some setups (routers, companions) emit multiple HEARTBEATs.
        # We "lock" onto the first autopilot-looking sysid to avoid mode flicker.
        self._selected_sysid: Optional[int] = None
        self._selected_compid: Optional[int] = None

    def get(self) -> Telemetry:
        return self._snap

    def start(self):
        if self._th and self._th.is_alive():
            return
        self._stop = False
        self._th = threading.Thread(target=self._run, daemon=True)
        self._th.start()
        print(f"[MAV] listening on {self.endpoint}")

    def stop(self):
        self._stop = True

    def join(self, timeout: Optional[float] = None) -> None:
        th = self._th
        if th:
            th.join(timeout=timeout)

    def _run(self):
        try:
            m = mavutil.mavlink_connection(self.endpoint)
        except Exception as e:
            print("[MAV] connection error:", e)
            return

        def _msg_src_system(msg) -> Optional[int]:
            try:
                return int(msg.get_srcSystem())
            except Exception:
                return None

        def _msg_src_component(msg) -> Optional[int]:
            try:
                return int(msg.get_srcComponent())
            except Exception:
                return None

        def _is_autopilot_heartbeat(msg) -> bool:
            # Heuristic: autopilot field must be valid, and message must not be from a GCS.
            try:
                ap = int(getattr(msg, "autopilot"))
                ap_invalid = int(getattr(mavutil.mavlink, "MAV_AUTOPILOT_INVALID", -1))
                if ap_invalid >= 0 and ap == ap_invalid:
                    return False
            except Exception:
                return False
            try:
                t = int(getattr(msg, "type"))
                t_gcs = int(getattr(mavutil.mavlink, "MAV_TYPE_GCS", -1))
                if t_gcs >= 0 and t == t_gcs:
                    return False
                t_onb = int(getattr(mavutil.mavlink, "MAV_TYPE_ONBOARD_CONTROLLER", -1))
                if t_onb >= 0 and t == t_onb:
                    return False
            except Exception:
                pass
            return True

        try:
            m.wait_heartbeat(timeout=3)
        except Exception:
            pass

        try:
            while not self._stop:
                try:
                    msg = m.recv_match(blocking=True, timeout=0.2)
                except Exception:
                    msg = None

                if msg is None:
                    continue

                mtype = msg.get_type()
                src_sysid = _msg_src_system(msg)
                src_compid = _msg_src_component(msg)

                # Select an autopilot sysid to stabilize mode/arming display.
                if self._selected_sysid is None and mtype == "HEARTBEAT" and _is_autopilot_heartbeat(msg):
                    if src_sysid is not None:
                        self._selected_sysid = int(src_sysid)
                        if src_compid is not None:
                            self._selected_compid = int(src_compid)

                # If we have a selected sysid, ignore messages from other systems.
                if self._selected_sysid is not None and src_sysid is not None and int(src_sysid) != int(self._selected_sysid):
                    continue
                # For HEARTBEAT only, also lock to the selected component (avoid flicker from camera/gimbal heartbeats).
                if (
                    mtype == "HEARTBEAT"
                    and self._selected_compid is not None
                    and src_compid is not None
                    and int(src_compid) != int(self._selected_compid)
                ):
                    continue

                if mtype == "ATTITUDE":
                    try:
                        self._snap.roll = float(msg.roll)
                        self._snap.pitch = float(msg.pitch)
                        self._snap.yaw = float(msg.yaw)
                    except Exception:
                        pass

                elif mtype == "VFR_HUD":
                    try:
                        self._snap.groundspeed = float(msg.groundspeed)
                    except Exception:
                        pass
                    try:
                        hdg = float(msg.heading)
                        self._snap.heading_deg = _norm360(hdg) or self._snap.heading_deg
                    except Exception:
                        pass
                    try:
                        self._snap.alt_m = float(msg.alt)
                    except Exception:
                        pass
                    try:
                        self._snap.climb_ms = float(msg.climb)
                    except Exception:
                        pass
                    try:
                        self._snap.throttle_pct = float(msg.throttle)
                        self._snap.throttle_norm = max(0.0, min(1.0, self._snap.throttle_pct / 100.0))
                    except Exception:
                        pass

                elif mtype == "SYS_STATUS":
                    try:
                        mv = float(msg.voltage_battery)
                        if mv > 0:
                            self._snap.battery_v = mv / 1000.0
                    except Exception:
                        pass
                    try:
                        cur = getattr(msg, "current_battery", None)
                        if cur is not None:
                            cur_i = int(cur)
                            # -1 / 255 are commonly used as "unknown"
                            if cur_i not in (-1, 255):
                                # MAVLink SYS_STATUS current_battery is in 10 mA (centiamp)
                                self._snap.current_a = float(cur_i) / 100.0
                    except Exception:
                        pass
                    try:
                        rem = getattr(msg, "battery_remaining", None)
                        if rem is not None:
                            rem_i = int(rem)
                            if rem_i not in (-1, 255):
                                self._snap.battery_remaining_pct = max(0, min(100, rem_i))
                    except Exception:
                        pass

                elif mtype == "HEARTBEAT":
                    try:
                        base_mode = int(msg.base_mode)
                        armed = (base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED) != 0
                        self._snap.armed = bool(armed)
                    except Exception:
                        pass
                    try:
                        mode = mavutil.mode_string_v10(msg)
                        if mode:
                            mode_s = str(mode).strip()
                            # Some non-autopilot components produce a "0x......" placeholder mode string.
                            # Don't clobber a real decoded mode with that placeholder.
                            if mode_s.startswith("0x") and all(c in "0123456789abcdefABCDEF" for c in mode_s[2:]):
                                prev = str(self._snap.mode or "").strip()
                                if prev and not (prev.startswith("0x") and all(c in "0123456789abcdefABCDEF" for c in prev[2:])):
                                    mode_s = ""
                            if mode_s:
                                self._snap.mode = mode_s
                    except Exception:
                        # Fallback: some dialects only expose custom_mode
                        try:
                            cm = getattr(msg, "custom_mode")
                            # Keep the previous decoded mode if the fallback is the all-zero default.
                            # This avoids flicker when multiple HEARTBEAT sources are present.
                            cm_i = int(cm) if cm is not None else 0
                            if cm_i != 0 or not self._snap.mode:
                                self._snap.mode = f"0x{cm_i:06X}"
                        except Exception:
                            pass

                elif mtype == "GPS_RAW_INT":
                    try:
                        lat = float(msg.lat) / 1e7 if msg.lat not in (None, 0) else None
                        lon = float(msg.lon) / 1e7 if msg.lon not in (None, 0) else None
                        if lat is not None and lon is not None:
                            self._snap.lat_deg = lat
                            self._snap.lon_deg = lon
                    except Exception:
                        pass
                    try:
                        if msg.alt is not None:
                            self._snap.alt_amsl_m = float(msg.alt) / 1000.0
                    except Exception:
                        pass
                    try:
                        self._snap.gps_sats = int(msg.satellites_visible) if msg.satellites_visible is not None else self._snap.gps_sats
                    except Exception:
                        pass
                    try:
                        fx = int(msg.fix_type)
                        self._snap.gps_fix_type = fx
                        self._snap.gps_fix = ["NO_GPS","NO_FIX","2D","3D","DGPS","RTK_FLOAT","RTK_FIXED"][fx] if 0 <= fx < 7 else None
                    except Exception:
                        pass
                    try:
                        if msg.eph is not None and int(msg.eph) != 0xFFFF:
                            self._snap.gps_hdop = float(msg.eph) / 100.0
                        if msg.epv is not None and int(msg.epv) != 0xFFFF:
                            self._snap.gps_vdop = float(msg.epv) / 100.0
                    except Exception:
                        pass
                    try:
                        if msg.vel is not None and int(msg.vel) != 0xFFFF:
                            self._snap.gps_vel_ms = float(msg.vel) / 100.0
                        if msg.cog is not None and int(msg.cog) != 0xFFFF:
                            self._snap.gps_cog_deg = _norm360(float(msg.cog) / 100.0)
                    except Exception:
                        pass
        finally:
            try:
                m.close()
            except Exception:
                pass
