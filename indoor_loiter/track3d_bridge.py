from __future__ import annotations

import json
import os
import queue
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


def _now_ms() -> int:
    return int(time.time() * 1000)


def _json_dumps_line(obj: dict) -> str:
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=True)


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return float(default)


def _safe_int(v: Any, default: int = 0) -> int:
    try:
        return int(v)
    except Exception:
        try:
            return int(float(v))
        except Exception:
            return int(default)


def _point_in_poly(x: float, y: float, poly: List[Tuple[float, float]]) -> bool:
    # Ray casting (odd-even rule). Assumes poly is in image pixel coords.
    inside = False
    n = len(poly)
    if n < 3:
        return False
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        intersect = ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi)
        if intersect:
            inside = not inside
        j = i
    return inside


def _simplify_verts(verts_uv: List[Tuple[float, float]], max_n: int = 64) -> List[Tuple[int, int]]:
    if not verts_uv:
        return []
    pts = [(int(round(float(u))), int(round(float(v)))) for (u, v) in verts_uv]
    if max_n <= 0 or len(pts) <= max_n:
        return pts
    step = max(1, len(pts) // max_n)
    return pts[::step][:max_n]


def _center_of_poly(verts_uv: List[Tuple[float, float]]) -> Optional[Tuple[float, float]]:
    if not verts_uv:
        return None
    # Area-weighted centroid (robust to non-uniform vertex sampling).
    # Falls back to bbox center if the polygon area is degenerate.
    try:
        if len(verts_uv) < 3:
            xs = [p[0] for p in verts_uv]
            ys = [p[1] for p in verts_uv]
            return (float(sum(xs)) / float(len(xs)), float(sum(ys)) / float(len(ys)))

        # Shoelace centroid formula for simple polygons.
        a2 = 0.0
        cx6 = 0.0
        cy6 = 0.0
        n = int(len(verts_uv))
        for i in range(n):
            x0, y0 = verts_uv[i]
            x1, y1 = verts_uv[(i + 1) % n]
            cross = float(x0) * float(y1) - float(x1) * float(y0)
            a2 += cross
            cx6 += (float(x0) + float(x1)) * cross
            cy6 += (float(y0) + float(y1)) * cross

        if abs(a2) > 1e-6:
            cx = cx6 / (3.0 * a2)
            cy = cy6 / (3.0 * a2)
            return (float(cx), float(cy))

        # Degenerate: bbox center.
        xs = [p[0] for p in verts_uv]
        ys = [p[1] for p in verts_uv]
        return (0.5 * (float(min(xs)) + float(max(xs))), 0.5 * (float(min(ys)) + float(max(ys))))
    except Exception:
        return None


@dataclass
class Track3dInit:
    out_w: int
    out_h: int
    fx: float
    fy: float
    cx: float
    cy: float
    ring_spec: dict


class Track3dBridge:
    """
    Spawns an external headless 3D tracker and bridges JSONL stdin/stdout.

    This module is intentionally minimal:
    - no GUI logic (server/client decide rendering)
    - no RealSense access (external tracker owns device)
    """

    def __init__(self, *, cfg: Any, print_fn=print):
        self.cfg = cfg
        self.print = print_fn

        self.proc: Optional[subprocess.Popen] = None
        self._stop = threading.Event()
        self._evt_q: "queue.Queue[dict]" = queue.Queue(maxsize=512)
        self._stderr_tail: "deque[str]" = deque(maxlen=80)
        self._use_pgroup = (os.name != "nt")
        self._pidfile: str = ""

        self._init: Optional[Track3dInit] = None
        self._lock = threading.Lock()

        # Non-blocking stdin writer:
        # - send_cmd() must never block the server control thread
        # - hover/mouse_move is high-rate and best-effort; keep only the latest hover
        self._cmd_q: "queue.Queue[dict]" = queue.Queue(maxsize=64)
        self._hover_cmd: Optional[dict] = None
        self._hover_dirty: bool = False
        self._stdin_th: Optional[threading.Thread] = None

        # Latest polygon state (from telemetry).
        self._poly_active = False
        self._pid_confirmed = False
        self._poly_sel_kind: str = "none"
        self._poly_verts_uv: List[Tuple[float, float]] = []
        self._last_stage_sent: Optional[int] = None
        # Latest stage-0 preview polygon (from "preview" events).
        self._preview_fi: int = -1
        self._preview_sel_kind: str = "none"
        self._preview_verts_uv: List[Tuple[float, float]] = []
        self._preview_last_t_mono: float = 0.0
        # Runtime enable state (so we can suppress hover traffic when acquisition is off).
        self._hole_enabled: bool = False
        self._plane_enabled: bool = False

    # ---------------- lifecycle ----------------
    def _pidfile_read(self, path: str) -> Optional[int]:
        try:
            s = str(open(str(path), "r", encoding="utf-8").read() or "").strip()
        except Exception:
            return None
        if not s:
            return None
        try:
            return int(s.split()[0])
        except Exception:
            return None

    def _pidfile_write(self, path: str, pid: int) -> None:
        try:
            p = str(path or "").strip()
        except Exception:
            p = ""
        if not p:
            return
        try:
            os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
        except Exception:
            pass
        try:
            with open(p, "w", encoding="utf-8") as f:
                f.write(f"{int(pid)}\n")
        except Exception:
            pass

    def _pidfile_remove(self, path: str) -> None:
        try:
            p = str(path or "").strip()
        except Exception:
            p = ""
        if not p:
            return
        try:
            os.remove(p)
        except Exception:
            pass

    def _pid_looks_like_track3d(self, pid: int, script_basename: str) -> bool:
        if int(pid) <= 0:
            return False
        if os.name == "nt":
            return True
        if not str(script_basename or "").strip():
            return False
        try:
            cmdline = open(f"/proc/{int(pid)}/cmdline", "rb").read().decode(errors="ignore").replace("\x00", " ")
        except Exception:
            return False
        return str(script_basename) in str(cmdline)

    def _kill_pid_best_effort(self, pid: int) -> None:
        if int(pid) <= 0:
            return
        # Prefer killing the process group (track3d is spawned with start_new_session so PGID==PID).
        if bool(self._use_pgroup):
            try:
                os.killpg(int(pid), signal.SIGTERM)
            except Exception:
                pass
            time.sleep(0.2)
            try:
                os.killpg(int(pid), signal.SIGKILL)
            except Exception:
                pass
            return
        try:
            os.kill(int(pid), signal.SIGTERM)
        except Exception:
            pass
        time.sleep(0.2)
        try:
            os.kill(int(pid), signal.SIGKILL)
        except Exception:
            pass

    def _cleanup_stale_pidfile(self, pidfile: str, script_basename: str) -> None:
        try:
            pf = str(pidfile or "").strip()
        except Exception:
            pf = ""
        if not pf or not os.path.exists(pf):
            return
        pid0 = self._pidfile_read(pf)
        if pid0 is None or int(pid0) <= 0:
            self._pidfile_remove(pf)
            return
        if not self._pid_looks_like_track3d(int(pid0), str(script_basename)):
            return
        try:
            self.print(f"[track3d] cleaning stale pid {int(pid0)} from {pf}", flush=True)
        except Exception:
            pass
        try:
            self._kill_pid_best_effort(int(pid0))
        except Exception:
            pass
        self._pidfile_remove(pf)

    def start(self) -> bool:
        if self.proc is not None:
            return True

        enabled = bool(self.cfg.get("track3d.enabled", False))
        if not enabled:
            return False

        workdir = str(self.cfg.get("track3d.workdir", "") or "").strip()
        script = str(
            self.cfg.get("track3d.script", "apps/realsense_orbslam3_rgbd_inertial_minimal.py")
            or "apps/realsense_orbslam3_rgbd_inertial_minimal.py"
        ).strip()
        cfg_path = str(self.cfg.get("track3d.config", "") or "").strip()

        if not workdir:
            workdir = os.getcwd()
        workdir_abs = os.path.abspath(workdir)
        script_path = script if os.path.isabs(script) else os.path.join(workdir_abs, script)
        if not os.path.exists(script_path):
            raise FileNotFoundError(f"track3d.script not found: {script_path}")
        if not cfg_path:
            raise ValueError("track3d.config is required when track3d.enabled=true")
        cfg_abs = cfg_path if os.path.isabs(cfg_path) else os.path.join(workdir_abs, cfg_path)
        if not os.path.exists(cfg_abs):
            raise FileNotFoundError(f"track3d.config not found: {cfg_abs}")

        py = str(self.cfg.get("track3d.python", "") or "").strip()
        if not py:
            py = sys.executable

        pidfile = str(self.cfg.get("track3d.pidfile", "./logs/track3d.pid") or "").strip()
        self._pidfile = str(pidfile)
        try:
            self._cleanup_stale_pidfile(str(pidfile), script_basename=os.path.basename(str(script_path)))
        except Exception:
            pass

        cmd = [py, script_path, "--config", cfg_abs, "--headless"]
        self.print(f"[track3d] spawn: {cmd} (cwd={workdir_abs})", flush=True)

        self._stop.clear()
        env = os.environ.copy()
        # Ensure the headless JSONL events flush immediately (avoid init timeouts due to stdio buffering).
        env.setdefault("PYTHONUNBUFFERED", "1")
        self.proc = subprocess.Popen(
            cmd,
            cwd=workdir_abs,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=env,
            start_new_session=bool(self._use_pgroup),
        )
        try:
            if bool(pidfile) and self.proc is not None:
                self._pidfile_write(str(pidfile), int(self.proc.pid))
        except Exception:
            pass

        self._stdin_th = threading.Thread(target=self._stdin_loop, name="track3d-stdin", daemon=True)
        self._stdin_th.start()
        threading.Thread(target=self._stdout_loop, name="track3d-stdout", daemon=True).start()
        threading.Thread(target=self._stderr_loop, name="track3d-stderr", daemon=True).start()
        return True

    def returncode(self) -> Optional[int]:
        p = self.proc
        if p is None:
            return None
        try:
            return p.poll()
        except Exception:
            return None

    def stderr_tail(self, max_lines: int = 30) -> List[str]:
        try:
            n = int(max(0, int(max_lines)))
        except Exception:
            n = 30
        if n <= 0:
            return []
        try:
            items = list(self._stderr_tail)
        except Exception:
            return []
        return items[-n:]

    def stop(self) -> None:
        self._stop.set()
        p = self.proc
        self.proc = None
        if p is None:
            return
        pid = 0
        try:
            pid = int(getattr(p, "pid", 0) or 0)
        except Exception:
            pid = 0
        try:
            # Prefer a graceful shutdown so the RealSense pipeline is closed cleanly.
            # The headless tracker treats EOF as shutdown and also accepts {"cmd":"shutdown"}.
            try:
                if p.stdin is not None:
                    try:
                        p.stdin.write(_json_dumps_line({"cmd": "shutdown"}) + "\n")
                        p.stdin.flush()
                    except Exception:
                        pass
                    try:
                        p.stdin.close()
                    except Exception:
                        pass
            except Exception:
                pass

            try:
                p.wait(timeout=4.0)
                return
            except Exception:
                pass

            # Escalate: terminate main process (SIGTERM) and wait a bit more.
            try:
                p.terminate()
            except Exception:
                pass
            try:
                p.wait(timeout=2.0)
                return
            except Exception:
                pass

            # Last resort: kill the whole process group so multiprocessing children don't get orphaned.
            if self._use_pgroup and pid > 0:
                try:
                    os.killpg(pid, signal.SIGTERM)
                except Exception:
                    pass
                try:
                    p.wait(timeout=2.0)
                    return
                except Exception:
                    pass
                try:
                    os.killpg(pid, signal.SIGKILL)
                except Exception:
                    pass
                try:
                    p.wait(timeout=1.0)
                except Exception:
                    pass
                return

            try:
                p.kill()
            except Exception:
                pass
        except Exception:
            try:
                p.kill()
            except Exception:
                pass
        finally:
            try:
                # Only remove pidfile once we're confident the subprocess is gone.
                # If we remove it too early and the process survives (SIGKILL, deadlocks, etc),
                # the next start() cannot reliably clean it up.
                alive = False
                try:
                    alive = (p.poll() is None)
                except Exception:
                    alive = False
                if (not bool(alive)) and str(self._pidfile or "").strip():
                    self._pidfile_remove(str(self._pidfile))
            except Exception:
                pass

    # ---------------- threads ----------------
    def _stdout_loop(self) -> None:
        p = self.proc
        if p is None or p.stdout is None:
            return
        while (not self._stop.is_set()) and (self.proc is not None):
            try:
                line = p.stdout.readline()
            except Exception:
                break
            if not line:
                break
            s = str(line).strip()
            if not s:
                continue
            try:
                msg = json.loads(s)
            except Exception:
                continue
            if not isinstance(msg, dict):
                continue
            try:
                self._evt_q.put_nowait(msg)
            except Exception:
                try:
                    _ = self._evt_q.get_nowait()
                except Exception:
                    pass
                try:
                    self._evt_q.put_nowait(msg)
                except Exception:
                    pass

        self.print("[track3d] stdout loop exit", flush=True)

    def _stderr_loop(self) -> None:
        p = self.proc
        if p is None or p.stderr is None:
            return
        while (not self._stop.is_set()) and (self.proc is not None):
            try:
                line = p.stderr.readline()
            except Exception:
                break
            if not line:
                break
            s = str(line).rstrip("\n")
            if s:
                try:
                    self._stderr_tail.append(str(s))
                except Exception:
                    pass
                self.print(f"[track3d:stderr] {s}", flush=True)
        self.print("[track3d] stderr loop exit", flush=True)

    def _stdin_loop(self) -> None:
        p = self.proc
        if p is None or p.stdin is None:
            return
        # Best-effort: keep sending commands without ever blocking the server control path.
        # If the tracker stops reading stdin (e.g. busy loop or deadlock), this thread may block,
        # but it is isolated from control/telemetry threads.
        while (not self._stop.is_set()) and (self.proc is not None):
            cmd = None
            try:
                cmd = self._cmd_q.get(timeout=0.02)
            except queue.Empty:
                cmd = None
            except Exception:
                cmd = None

            if cmd is None:
                try:
                    with self._lock:
                        if bool(self._hover_dirty):
                            cmd = dict(self._hover_cmd) if isinstance(self._hover_cmd, dict) else None
                            self._hover_dirty = False
                except Exception:
                    cmd = None

            if not isinstance(cmd, dict):
                continue

            try:
                line = _json_dumps_line(cmd) + "\n"
                p.stdin.write(line)
                p.stdin.flush()
            except Exception:
                break

        self.print("[track3d] stdin loop exit", flush=True)

    # ---------------- event handling ----------------
    def poll_events(self, max_n: int = 200) -> List[dict]:
        out: List[dict] = []
        for _ in range(int(max(0, max_n))):
            try:
                msg = self._evt_q.get_nowait()
            except Exception:
                break
            if isinstance(msg, dict):
                out.append(msg)
        return out

    def init_info(self) -> Optional[Track3dInit]:
        with self._lock:
            return self._init

    def snapshot_poly(self) -> dict:
        """Return the latest polygon selection state (thread-safe)."""
        with self._lock:
            return {
                "active": bool(self._poly_active),
                "pid_confirmed": bool(self._pid_confirmed),
                "sel_kind": str(self._poly_sel_kind),
                "n_verts": int(len(self._poly_verts_uv)),
            }

    def apply_event(self, evt: dict) -> None:
        t = str(evt.get("type", "") or "").strip().lower()
        payload = evt.get("payload", None)
        if t == "init" and isinstance(payload, dict):
            try:
                intr = payload.get("intr", {}) or {}
                out_wh = payload.get("out_wh", None)
                out_w = _safe_int(out_wh[0], 0) if isinstance(out_wh, (list, tuple)) and len(out_wh) >= 2 else _safe_int(intr.get("w", 0), 0)
                out_h = _safe_int(out_wh[1], 0) if isinstance(out_wh, (list, tuple)) and len(out_wh) >= 2 else _safe_int(intr.get("h", 0), 0)
                fx = _safe_float(intr.get("fx", 0.0), 0.0)
                fy = _safe_float(intr.get("fy", 0.0), 0.0)
                cx = _safe_float(intr.get("cx", 0.0), 0.0)
                cy = _safe_float(intr.get("cy", 0.0), 0.0)
                ring_spec = payload.get("ring_spec", {}) or {}
                with self._lock:
                    self._init = Track3dInit(out_w=out_w, out_h=out_h, fx=fx, fy=fy, cx=cx, cy=cy, ring_spec=dict(ring_spec))
            except Exception:
                pass
            return

        if t == "preview" and isinstance(payload, dict):
            try:
                fi = _safe_int(payload.get("fi", -1), -1)
                sk = payload.get("sel_kind", None)
                verts = payload.get("verts_uv", None)
                verts_uv: List[Tuple[float, float]] = []
                if isinstance(verts, list):
                    for p in verts:
                        if isinstance(p, (list, tuple)) and len(p) >= 2:
                            verts_uv.append((float(p[0]), float(p[1])))
                with self._lock:
                    self._preview_fi = int(fi)
                    if sk is not None:
                        self._preview_sel_kind = str(sk or "none").strip().lower() or "none"
                    self._preview_verts_uv = verts_uv
                    self._preview_last_t_mono = float(time.monotonic())
            except Exception:
                pass
            return

        if t == "telemetry" and isinstance(payload, dict):
            poly = payload.get("poly", None)
            if isinstance(poly, dict):
                active = bool(poly.get("active", False))
                pid_conf = bool(poly.get("pid_confirmed", False))
                sel_kind = str(poly.get("sel_kind", "none") or "none").strip().lower() or "none"
                verts = poly.get("verts_uv", None)
                verts_uv: List[Tuple[float, float]] = []
                if isinstance(verts, list):
                    for p in verts:
                        if isinstance(p, (list, tuple)) and len(p) >= 2:
                            verts_uv.append((float(p[0]), float(p[1])))
                with self._lock:
                    self._poly_active = bool(active)
                    self._pid_confirmed = bool(pid_conf)
                    self._poly_sel_kind = str(sel_kind)
                    self._poly_verts_uv = verts_uv
            return

    # ---------------- commands ----------------
    def send_cmd(self, cmd: dict) -> bool:
        p = self.proc
        if p is None:
            return False
        c = str(cmd.get("cmd", "") or "").strip().lower() if isinstance(cmd, dict) else ""
        if c in ("hole_enable", "plane_enable"):
            try:
                en = cmd.get("enable", True)
                try:
                    en_i = int(en)
                    en_b = bool(en_i != 0)
                except Exception:
                    en_b = bool(en)
            except Exception:
                en_b = True
            with self._lock:
                if c == "hole_enable":
                    self._hole_enabled = bool(en_b)
                    if bool(en_b):
                        self._plane_enabled = False
                else:
                    self._plane_enabled = bool(en_b)
                    if bool(en_b):
                        self._hole_enabled = False
        if c == "hover":
            try:
                with self._lock:
                    self._hover_cmd = dict(cmd)
                    self._hover_dirty = True
                return True
            except Exception:
                return False
        try:
            self._cmd_q.put_nowait(dict(cmd))
            return True
        except queue.Full:
            # Drop the oldest queued command and retry once.
            try:
                _ = self._cmd_q.get_nowait()
            except Exception:
                pass
            try:
                self._cmd_q.put_nowait(dict(cmd))
                return True
            except Exception:
                return False
        except Exception:
            return False

    def handle_mouse(self, msg: dict) -> None:
        # Expected message schema from indoor-osd-app:
        # {"type":"mouse_move"/"mouse_click"/"hole_enable"/"plane_enable","x_norm":0..1,"y_norm":0..1,"button":"left|right",...}
        t = str(msg.get("type", "") or "").strip().lower()
        if t in ("hole_enable", "plane_enable"):
            enable = msg.get("enable", True)
            try:
                enable_i = int(enable)
                enable_b = bool(enable_i != 0)
            except Exception:
                enable_b = bool(enable)
            # Control the expensive depth-based stage-0 preview detector / acquisition.
            if t == "hole_enable":
                with self._lock:
                    self._hole_enabled = bool(enable_b)
                    if bool(enable_b):
                        self._plane_enabled = False
                self.send_cmd({"cmd": "hole_enable", "enable": int(bool(enable_b)), "ts": _now_ms()})
                # Enforce mutual exclusion (best-effort).
                if bool(enable_b):
                    self.send_cmd({"cmd": "plane_enable", "enable": 0, "ts": _now_ms()})
            else:
                with self._lock:
                    self._plane_enabled = bool(enable_b)
                    if bool(enable_b):
                        self._hole_enabled = False
                self.send_cmd({"cmd": "plane_enable", "enable": int(bool(enable_b)), "ts": _now_ms()})
                # Enforce mutual exclusion (best-effort).
                if bool(enable_b):
                    self.send_cmd({"cmd": "hole_enable", "enable": 0, "ts": _now_ms()})
            return
        if t not in ("mouse_move", "mouse_click", "mouse_stop"):
            return
        if str(msg.get("surface", "") or "osd_main") != "osd_main":
            return

        init = self.init_info()
        if init is None or init.out_w <= 0 or init.out_h <= 0:
            return

        x_norm = _safe_float(msg.get("x_norm", 0.0), 0.0)
        y_norm = _safe_float(msg.get("y_norm", 0.0), 0.0)
        x = int(round(max(0.0, min(1.0, x_norm)) * float(max(1, init.out_w - 1))))
        y = int(round(max(0.0, min(1.0, y_norm)) * float(max(1, init.out_h - 1))))

        with self._lock:
            poly_active = bool(self._poly_active)
            pid_conf = bool(self._pid_confirmed)
            poly_sel_kind = str(self._poly_sel_kind or "none")
            poly_uv = list(self._poly_verts_uv)
            hole_enabled = bool(self._hole_enabled)
            plane_enabled = bool(self._plane_enabled)

        if t == "mouse_move":
            # Only drive hover preview when no polygon is active (Stage 0).
            if (not poly_active) and (hole_enabled or plane_enabled):
                self.send_cmd({"cmd": "hover", "x": int(x), "y": int(y), "ts": _now_ms()})
            return

        if t == "mouse_stop":
            # Not used by the current acquisition flow; ignore.
            return

        if t == "mouse_click":
            btn = str(msg.get("button", "") or "").strip().lower()
            if btn == "right":
                try:
                    self.print(f"[track3d] mouse_click right -> clear", flush=True)
                except Exception:
                    pass
                self.send_cmd({"cmd": "clear", "ts": _now_ms()})
                return
            if btn != "left":
                return

            # Stage transitions:
            # - Stage 0 (no selection): left click selects a hole at the click point.
            # - Stage 1 (selected, not confirmed): left click confirms only if click is inside polygon.
            if not poly_active:
                want_kind = "plane" if bool(plane_enabled) else ("hole" if bool(hole_enabled) else "")
                if not want_kind:
                    return

                # If the click is inside the latest stage-0 preview polygon, prefer selecting against the
                # same preview frame index. This avoids "clicking a cached polygon" mismatch when the
                # drone/video is moving and the preview overlay lags by a frame or two.
                try:
                    with self._lock:
                        prev_fi = int(self._preview_fi)
                        prev_kind = str(self._preview_sel_kind or "none")
                        prev_poly = list(self._preview_verts_uv)
                        prev_age_s = float(time.monotonic() - float(self._preview_last_t_mono))
                except Exception:
                    prev_fi = -1
                    prev_kind = "none"
                    prev_poly = []
                    prev_age_s = 1e9

                if prev_fi >= 0 and len(prev_poly) >= 3 and prev_age_s <= 0.75:
                    try:
                        if str(prev_kind or "none") == str(want_kind) and _point_in_poly(float(x), float(y), prev_poly):
                            try:
                                self.print(
                                    f"[track3d] mouse_click left -> select_{want_kind} (preview_fi={int(prev_fi)} age_ms={int(prev_age_s*1000.0)}) xy=({int(x)},{int(y)})",
                                    flush=True,
                                )
                            except Exception:
                                pass
                            self.send_cmd({"cmd": f"select_{str(want_kind)}", "x": int(x), "y": int(y), "fi": int(prev_fi)})
                            return
                    except Exception:
                        pass

                # Fallback: select on the latest frame.
                try:
                    self.print(f"[track3d] mouse_click left -> select_{want_kind} xy=({int(x)},{int(y)})", flush=True)
                except Exception:
                    pass
                self.send_cmd({"cmd": f"select_{str(want_kind)}", "x": int(x), "y": int(y), "ts": _now_ms()})
                return
            if poly_active and (not pid_conf):
                if _point_in_poly(float(x), float(y), poly_uv):
                    cmd = "confirm_plane" if str(poly_sel_kind) == "plane" else "confirm_hole"
                    try:
                        self.print(f"[track3d] mouse_click left -> {cmd} xy=({int(x)},{int(y)})", flush=True)
                    except Exception:
                        pass
                    self.send_cmd({"cmd": str(cmd), "x": int(x), "y": int(y), "ts": _now_ms()})
                else:
                    try:
                        self.print(
                            f"[track3d] mouse_click left ignored (outside poly) sel_kind={str(poly_sel_kind)} xy=({int(x)},{int(y)}) n_verts={int(len(poly_uv))}",
                            flush=True,
                        )
                    except Exception:
                        pass
                return

    # ---------------- overlay helpers ----------------
    def make_acq_poly_msg(self, *, stage: int, verts_uv: List[Tuple[float, float]], sel_kind: Optional[str] = None) -> Optional[dict]:
        init = self.init_info()
        if init is None or init.out_w <= 0 or init.out_h <= 0:
            return None
        verts_i = _simplify_verts(verts_uv, max_n=64)
        center = _center_of_poly(verts_uv) if verts_uv else None
        msg = {
            "type": "acq_poly",
            "ts": _now_ms(),
            "stage": int(stage),
            "img_size": [int(init.out_w), int(init.out_h)],
            "verts_uv": verts_i if verts_i else None,
            "center_uv": [float(center[0]), float(center[1])] if center is not None else None,
            "source": "orbslam",
        }
        if sel_kind is not None and str(sel_kind).strip():
            msg["sel_kind"] = str(sel_kind)
        return msg
