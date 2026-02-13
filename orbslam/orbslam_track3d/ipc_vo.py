from __future__ import annotations

import math
import os
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np

from ipc.shm_state import ShmOdomState, ShmStateSpec  # type: ignore

from .file_io import write_json_atomic
from .yaml_config import load_yaml_dict

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class IpcVoState:
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


class InterProcessVoCommunicator:
    """Publish latest VO pose to shared memory (Drone_client ipc/shm_state.py compatible)."""

    def __init__(
        self,
        *,
        state: IpcVoState,
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

        self._shm: Optional[ShmOdomState] = None
        self._spec: Optional[dict[str, str]] = None
        self._last_Twc_w: Optional[np.ndarray] = None
        self._unlink_on_exit: bool = True

    def start(self) -> None:
        self._thread.start()

    def spec_dict(self) -> Optional[dict[str, str]]:
        try:
            return dict(self._spec) if isinstance(self._spec, dict) else None
        except Exception:
            return None

    def wait_spec(self, *, timeout_s: float = 2.0) -> Optional[dict[str, str]]:
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

        shm = self._shm
        self._shm = None
        if shm is None:
            return
        try:
            for seg in (getattr(shm, "_shm_meta", None), getattr(shm, "_shm_R", None), getattr(shm, "_shm_t", None), getattr(shm, "_shm_Twc", None)):
                if seg is None:
                    continue
                try:
                    seg.close()
                except Exception:
                    pass
        except Exception:
            pass
        if bool(self._unlink_on_exit):
            try:
                for seg in (getattr(shm, "_shm_meta", None), getattr(shm, "_shm_R", None), getattr(shm, "_shm_t", None), getattr(shm, "_shm_Twc", None)):
                    if seg is None:
                        continue
                    try:
                        seg.unlink()
                    except Exception:
                        pass
            except Exception:
                pass

    def _resolve_name_prefix(self, cfg: dict) -> str:
        env_prefix = str(os.environ.get("ORB_VO_IPC_PREFIX", "") or "").strip()
        if env_prefix:
            return env_prefix

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
        cfg_path = self.cfg_path or (PROJECT_ROOT / "apps" / "px4_mavlink.yaml")
        cfg = load_yaml_dict(cfg_path) if cfg_path is not None else {}
        ipc_cfg = cfg.get("ipc_vo", {}) if isinstance(cfg, dict) else {}
        enabled = True
        if isinstance(ipc_cfg, dict) and ("enabled" in ipc_cfg):
            enabled = bool(ipc_cfg.get("enabled", True))
        if not bool(enabled):
            try:
                self.print("[ipc_vo] disabled by config", flush=True)
            except Exception:
                pass
            return

        if isinstance(ipc_cfg, dict) and ("unlink_on_exit" in ipc_cfg):
            self._unlink_on_exit = bool(ipc_cfg.get("unlink_on_exit", True))

        name_prefix = self._resolve_name_prefix(cfg)
        shm = ShmOdomState.create(name_prefix=str(name_prefix))
        self._shm = shm
        try:
            shm.write(
                frame_idx=-1,
                timestamp=0.0,
                ok=False,
                odom_stable=False,
                n_corr=0,
                n_inliers=0,
                rmse_m=0.0,
                rot_deg=0.0,
                status_code=0,
                est_code=0,
                R=np.eye(3, dtype=np.float32),
                t=np.zeros(3, dtype=np.float32),
                Twc=np.eye(4, dtype=np.float32),
            )
        except Exception:
            pass

        self._spec = dict(shm.spec.to_dict()) if hasattr(shm, "spec") else None
        try:
            spec = dict(self._spec or {})
            out_path = None
            if isinstance(ipc_cfg, dict):
                out_path = ipc_cfg.get("out_spec_path", None) or ipc_cfg.get("spec_out_path", None)
            out_path = self.out_spec_path or (Path(out_path) if out_path else None) or (PROJECT_ROOT / ".tmp" / "orb_vo_ipc_state_spec.json")
            if not Path(out_path).is_absolute():
                out_path = PROJECT_ROOT / str(out_path)
            write_json_atomic(Path(out_path), spec)
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

    def _write_state(self, *, frame_idx: int, timestamp: float, ok: bool, Twc: Optional[np.ndarray]) -> None:
        shm = self._shm
        if shm is None:
            return

        Rstep = np.eye(3, dtype=np.float32)
        tstep = np.zeros(3, dtype=np.float32)
        rot_deg = 0.0

        if Twc is not None:
            T = np.asarray(Twc, dtype=np.float32).reshape(4, 4)
            try:
                if self._last_Twc_w is not None:
                    Tprev = np.asarray(self._last_Twc_w, dtype=np.float32).reshape(4, 4)
                    Rprev = Tprev[:3, :3]
                    Rcur = T[:3, :3]
                    Rstep = (Rprev.T @ Rcur).astype(np.float32, copy=False)
                    tstep = (T[:3, 3] - Tprev[:3, 3]).astype(np.float32, copy=False)
                    tr = float(Rstep[0, 0] + Rstep[1, 1] + Rstep[2, 2])
                    c = max(-1.0, min(1.0, (tr - 1.0) * 0.5))
                    rot_deg = float(math.degrees(math.acos(c)))
            except Exception:
                Rstep = np.eye(3, dtype=np.float32)
                tstep = np.zeros(3, dtype=np.float32)
                rot_deg = 0.0
            self._last_Twc_w = T.copy()
        else:
            if self._last_Twc_w is not None:
                T = np.asarray(self._last_Twc_w, dtype=np.float32).reshape(4, 4)
            else:
                T = np.eye(4, dtype=np.float32)

        try:
            shm.write(
                frame_idx=int(frame_idx),
                timestamp=float(timestamp),
                ok=bool(Twc is not None and ok),
                odom_stable=bool(Twc is not None and ok),
                n_corr=0,
                n_inliers=0,
                rmse_m=0.0,
                rot_deg=float(rot_deg),
                status_code=0 if bool(Twc is not None and ok) else 1,
                est_code=0,
                R=Rstep,
                t=tstep,
                Twc=T,
                ms_est=0.0,
                ms_total=0.0,
                ms_prefilter=0.0,
                ms_pg=0.0,
                ms_jitter=0.0,
                ms_imu=0.0,
                ms_weights=0.0,
                ms_gate=0.0,
            )
        except Exception:
            pass

