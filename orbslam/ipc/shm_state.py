from __future__ import annotations

import uuid
from dataclasses import dataclass
from multiprocessing.shared_memory import SharedMemory
from typing import Optional, Tuple

import numpy as np


def _attach_shm(name: str) -> SharedMemory:
    try:
        # Python 3.13+: avoid registering attached SHM with resource_tracker.
        return SharedMemory(name=str(name), create=False, track=False)  # type: ignore[call-arg]
    except TypeError:
        # Older Python: `track` kwarg not supported.
        return SharedMemory(name=str(name), create=False)


def unlink_state_prefix_best_effort(prefix: str) -> None:
    """Best-effort unlink of shm_state segments for a given prefix.

    On Linux, these live under `/dev/shm/` as `/<name>`; on Windows they are named segments.
    Unlinking helps recover from unclean exits where the OS-level shm name still exists.
    """
    try:
        prefix_s = str(prefix or "").strip()
    except Exception:
        prefix_s = ""
    if not prefix_s:
        return
    for suffix in ("meta", "R", "t", "Twc"):
        name = f"{prefix_s}_{suffix}"
        try:
            shm = _attach_shm(name)
        except Exception:
            shm = None
        if shm is None:
            continue
        try:
            shm.unlink()
        except Exception:
            pass
        try:
            shm.close()
        except Exception:
            pass

_META_DTYPE = np.dtype(
    [
        ("seq", np.uint32),  # even=stable, odd=writer in progress
        ("frame_idx", np.int64),
        ("timestamp", np.float64),
        ("ok", np.uint8),
        ("odom_stable", np.uint8),
        ("n_corr", np.int32),
        ("n_inliers", np.int32),
        ("rmse_m", np.float32),
        ("rot_deg", np.float32),
        ("status_code", np.int32),
        # Estimator backend tag (small int code). Used for diagnostics in external loggers.
        ("est_code", np.int32),
        # Timing (milliseconds) for the odom process.
        ("ms_est", np.float32),  # time spent in RANSAC/estimator calls
        ("ms_total", np.float32),  # total per-frame time for the odom process iteration
        ("ms_prefilter", np.float32),  # time spent in prefilters (jitter-bins, MAD3D)
        ("ms_pg", np.float32),  # time spent in local pose-graph block (push/trim/solve)
        ("ms_jitter", np.float32),  # time spent reading jitter SHM + building group maps
        ("ms_imu", np.float32),  # time spent in per-frame IMU sync/integration logic
        ("ms_weights", np.float32),  # time spent building correspondence weights
        ("ms_gate", np.float32),  # time spent in accept-gate computations
    ]
)


@dataclass(frozen=True)
class ShmStateSpec:
    name_prefix: str
    shm_meta: str
    shm_R: str
    shm_t: str
    shm_Twc: str

    def to_dict(self) -> dict:
        return {
            "name_prefix": str(self.name_prefix),
            "shm_meta": str(self.shm_meta),
            "shm_R": str(self.shm_R),
            "shm_t": str(self.shm_t),
            "shm_Twc": str(self.shm_Twc),
        }

    @staticmethod
    def from_dict(d: dict) -> "ShmStateSpec":
        return ShmStateSpec(
            name_prefix=str(d.get("name_prefix", "")),
            shm_meta=str(d["shm_meta"]),
            shm_R=str(d["shm_R"]),
            shm_t=str(d["shm_t"]),
            shm_Twc=str(d["shm_Twc"]),
        )


class ShmOdomState:
    """
    Shared-memory state for latest odometry estimate.
    """

    def __init__(
        self,
        *,
        spec: ShmStateSpec,
        shm_meta: SharedMemory,
        shm_R: SharedMemory,
        shm_t: SharedMemory,
        shm_Twc: SharedMemory,
    ) -> None:
        self.spec = spec
        self._shm_meta = shm_meta
        self._shm_R = shm_R
        self._shm_t = shm_t
        self._shm_Twc = shm_Twc

        self.meta = np.ndarray((1,), dtype=_META_DTYPE, buffer=shm_meta.buf)
        self.R = np.ndarray((3, 3), dtype=np.float32, buffer=shm_R.buf)
        self.t = np.ndarray((3,), dtype=np.float32, buffer=shm_t.buf)
        self.Twc = np.ndarray((4, 4), dtype=np.float32, buffer=shm_Twc.buf)

    @staticmethod
    def create(*, name_prefix: Optional[str] = None, force_unlink: bool = True) -> "ShmOdomState":
        prefix = str(name_prefix or f"mf_state_{uuid.uuid4().hex[:8]}")
        meta_nbytes = int(_META_DTYPE.itemsize)
        R_nbytes = int(3 * 3 * np.dtype(np.float32).itemsize)
        t_nbytes = int(3 * np.dtype(np.float32).itemsize)
        Twc_nbytes = int(4 * 4 * np.dtype(np.float32).itemsize)

        if bool(force_unlink) and str(prefix).strip():
            unlink_state_prefix_best_effort(str(prefix))

        try:
            shm_meta = SharedMemory(create=True, size=meta_nbytes, name=f"{prefix}_meta")
            shm_R = SharedMemory(create=True, size=R_nbytes, name=f"{prefix}_R")
            shm_t = SharedMemory(create=True, size=t_nbytes, name=f"{prefix}_t")
            shm_Twc = SharedMemory(create=True, size=Twc_nbytes, name=f"{prefix}_Twc")
        except FileExistsError:
            # Recover from stale segments (e.g. previous unclean exit). If they still exist after our
            # best-effort unlink, attach and reuse them rather than crashing the writer thread.
            unlink_state_prefix_best_effort(str(prefix))
            try:
                shm_meta = SharedMemory(create=True, size=meta_nbytes, name=f"{prefix}_meta")
                shm_R = SharedMemory(create=True, size=R_nbytes, name=f"{prefix}_R")
                shm_t = SharedMemory(create=True, size=t_nbytes, name=f"{prefix}_t")
                shm_Twc = SharedMemory(create=True, size=Twc_nbytes, name=f"{prefix}_Twc")
            except FileExistsError:
                spec = ShmStateSpec(
                    name_prefix=str(prefix),
                    shm_meta=f"{prefix}_meta",
                    shm_R=f"{prefix}_R",
                    shm_t=f"{prefix}_t",
                    shm_Twc=f"{prefix}_Twc",
                )
                return ShmOdomState.attach(spec)

        spec = ShmStateSpec(
            name_prefix=prefix,
            shm_meta=shm_meta.name,
            shm_R=shm_R.name,
            shm_t=shm_t.name,
            shm_Twc=shm_Twc.name,
        )
        st = ShmOdomState(spec=spec, shm_meta=shm_meta, shm_R=shm_R, shm_t=shm_t, shm_Twc=shm_Twc)
        st.meta["seq"][0] = np.uint32(0)
        st.meta["frame_idx"][0] = np.int64(-1)
        st.meta["timestamp"][0] = float(0.0)
        st.meta["ok"][0] = np.uint8(0)
        st.meta["odom_stable"][0] = np.uint8(0)
        st.meta["n_corr"][0] = np.int32(0)
        st.meta["n_inliers"][0] = np.int32(0)
        st.meta["rmse_m"][0] = np.float32(0.0)
        st.meta["rot_deg"][0] = np.float32(0.0)
        st.meta["status_code"][0] = np.int32(0)
        st.meta["est_code"][0] = np.int32(0)
        st.meta["ms_est"][0] = np.float32(0.0)
        st.meta["ms_total"][0] = np.float32(0.0)
        st.meta["ms_prefilter"][0] = np.float32(0.0)
        st.meta["ms_pg"][0] = np.float32(0.0)
        st.meta["ms_jitter"][0] = np.float32(0.0)
        st.meta["ms_imu"][0] = np.float32(0.0)
        st.meta["ms_weights"][0] = np.float32(0.0)
        st.meta["ms_gate"][0] = np.float32(0.0)
        st.R[:] = np.eye(3, dtype=np.float32)
        st.t[:] = 0.0
        st.Twc[:] = np.eye(4, dtype=np.float32)
        return st

    @staticmethod
    def attach(spec: ShmStateSpec) -> "ShmOdomState":
        shm_meta = _attach_shm(spec.shm_meta)
        shm_R = _attach_shm(spec.shm_R)
        shm_t = _attach_shm(spec.shm_t)
        shm_Twc = _attach_shm(spec.shm_Twc)
        return ShmOdomState(spec=spec, shm_meta=shm_meta, shm_R=shm_R, shm_t=shm_t, shm_Twc=shm_Twc)

    def close(self) -> None:
        for shm in (self._shm_meta, self._shm_R, self._shm_t, self._shm_Twc):
            try:
                shm.close()
            except Exception:
                pass

    def unlink(self) -> None:
        for shm in (self._shm_meta, self._shm_R, self._shm_t, self._shm_Twc):
            try:
                shm.unlink()
            except Exception:
                pass

    def write(
        self,
        *,
        frame_idx: int,
        timestamp: float,
        ok: bool,
        odom_stable: bool,
        n_corr: int,
        n_inliers: int,
        rmse_m: float,
        rot_deg: float,
        status_code: int,
        est_code: int = 0,
        R: np.ndarray,
        t: np.ndarray,
        Twc: np.ndarray,
        ms_est: float = 0.0,
        ms_total: float = 0.0,
        ms_prefilter: float = 0.0,
        ms_pg: float = 0.0,
        ms_jitter: float = 0.0,
        ms_imu: float = 0.0,
        ms_weights: float = 0.0,
        ms_gate: float = 0.0,
    ) -> None:
        seq0 = int(self.meta["seq"][0])
        self.meta["seq"][0] = np.uint32(seq0 + 1)

        self.meta["frame_idx"][0] = np.int64(int(frame_idx))
        self.meta["timestamp"][0] = float(timestamp)
        self.meta["ok"][0] = np.uint8(1 if bool(ok) else 0)
        self.meta["odom_stable"][0] = np.uint8(1 if bool(odom_stable) else 0)
        self.meta["n_corr"][0] = np.int32(int(n_corr))
        self.meta["n_inliers"][0] = np.int32(int(n_inliers))
        self.meta["rmse_m"][0] = np.float32(float(rmse_m))
        self.meta["rot_deg"][0] = np.float32(float(rot_deg))
        self.meta["status_code"][0] = np.int32(int(status_code))
        self.meta["est_code"][0] = np.int32(int(est_code))
        self.meta["ms_est"][0] = np.float32(float(ms_est))
        self.meta["ms_total"][0] = np.float32(float(ms_total))
        self.meta["ms_prefilter"][0] = np.float32(float(ms_prefilter))
        self.meta["ms_pg"][0] = np.float32(float(ms_pg))
        self.meta["ms_jitter"][0] = np.float32(float(ms_jitter))
        self.meta["ms_imu"][0] = np.float32(float(ms_imu))
        self.meta["ms_weights"][0] = np.float32(float(ms_weights))
        self.meta["ms_gate"][0] = np.float32(float(ms_gate))

        self.R[:, :] = np.asarray(R, dtype=np.float32).reshape(3, 3)
        self.t[:] = np.asarray(t, dtype=np.float32).reshape(3)
        self.Twc[:, :] = np.asarray(Twc, dtype=np.float32).reshape(4, 4)

        self.meta["seq"][0] = np.uint32(seq0 + 2)

    def read(
        self, *, copy: bool = True
    ) -> Optional[
        Tuple[int, float, bool, bool, int, int, float, float, int, np.ndarray, np.ndarray, np.ndarray, float, float, float, float, float, float, float, float, int]
    ]:
        s1 = int(self.meta["seq"][0])
        if s1 & 1:
            return None
        frame_idx = int(self.meta["frame_idx"][0])
        if frame_idx < 0:
            return None
        timestamp = float(self.meta["timestamp"][0])
        ok = bool(int(self.meta["ok"][0]) != 0)
        odom_stable = bool(int(self.meta["odom_stable"][0]) != 0)
        n_corr = int(self.meta["n_corr"][0])
        n_inliers = int(self.meta["n_inliers"][0])
        rmse_m = float(self.meta["rmse_m"][0])
        rot_deg = float(self.meta["rot_deg"][0])
        status_code = int(self.meta["status_code"][0])
        est_code = int(self.meta["est_code"][0])
        ms_est = float(self.meta["ms_est"][0])
        ms_total = float(self.meta["ms_total"][0])
        ms_prefilter = float(self.meta["ms_prefilter"][0])
        ms_pg = float(self.meta["ms_pg"][0])
        ms_jitter = float(self.meta["ms_jitter"][0])
        ms_imu = float(self.meta["ms_imu"][0])
        ms_weights = float(self.meta["ms_weights"][0])
        ms_gate = float(self.meta["ms_gate"][0])
        s2 = int(self.meta["seq"][0])
        if s1 != s2 or (s2 & 1):
            return None

        R = self.R
        t = self.t
        Twc = self.Twc
        if copy:
            R = np.asarray(R, dtype=np.float32).copy()
            t = np.asarray(t, dtype=np.float32).copy()
            Twc = np.asarray(Twc, dtype=np.float32).copy()
        return (
            frame_idx,
            timestamp,
            ok,
            odom_stable,
            n_corr,
            n_inliers,
            rmse_m,
            rot_deg,
            status_code,
            R,
            t,
            Twc,
            float(ms_est),
            float(ms_total),
            float(ms_prefilter),
            float(ms_pg),
            float(ms_jitter),
            float(ms_imu),
            float(ms_weights),
            float(ms_gate),
            int(est_code),
        )
