from __future__ import annotations

import re
from dataclasses import dataclass
from multiprocessing.shared_memory import SharedMemory
from typing import Any, Optional, Tuple

import numpy as np

_SHM_NAME_SAFE_RE = re.compile(r"[^0-9A-Za-z_\-]+")


def sanitize_shm_prefix(v: Any) -> Optional[str]:
    if v is None:
        return None
    try:
        s = str(v).strip()
    except Exception:
        return None
    if not s:
        return None
    s = _SHM_NAME_SAFE_RE.sub("_", s).strip("_")
    if not s:
        return None
    if len(s) > 48:
        s = s[:48]
    return s


def _attach_shm(name: str) -> SharedMemory:
    try:
        return SharedMemory(name=str(name), create=False, track=False)  # type: ignore[call-arg]
    except TypeError:
        return SharedMemory(name=str(name), create=False)


def unlink_shm_best_effort(name: str) -> None:
    try:
        shm = _attach_shm(str(name))
    except Exception:
        return
    try:
        shm.unlink()
    except Exception:
        pass
    try:
        shm.close()
    except Exception:
        pass


def unlink_ring_prefix_best_effort(prefix: str) -> None:
    for suf in ("_meta", "_bgr", "_gray", "_depthraw", "_depthbgr"):
        unlink_shm_best_effort(f"{prefix}{suf}")


def unlink_imu_prefix_best_effort(prefix: str) -> None:
    for suf in ("_meta", "_slots"):
        unlink_shm_best_effort(f"{prefix}{suf}")


def parse_shm_slots(v: Any, *, fps: int, min_slots: int = 3) -> int:
    try:
        if v is None:
            raise ValueError("none")
        if isinstance(v, str):
            s = str(v).strip().lower()
            if s in ("auto", "fps/2", "halfsec", "half_sec", "half-second", "half_second"):
                return int(max(int(min_slots), int(fps) // 2))
        return int(v)
    except Exception:
        return int(max(int(min_slots), int(fps) // 2))


_RING_META_DTYPE = np.dtype([("seq", np.uint32), ("frame_idx", np.int64), ("timestamp", np.float64)])


@dataclass(frozen=True)
class ShmRingSpec:
    slots: int
    h: int
    w: int
    name_prefix: str
    shm_meta: str
    shm_bgr: str
    shm_gray: str
    shm_depth_raw: str
    shm_depth_bgr: str

    def to_dict(self) -> dict:
        return {
            "slots": int(self.slots),
            "h": int(self.h),
            "w": int(self.w),
            "name_prefix": str(self.name_prefix),
            "shm_meta": str(self.shm_meta),
            "shm_bgr": str(self.shm_bgr),
            "shm_gray": str(self.shm_gray),
            "shm_depth_raw": str(self.shm_depth_raw),
            "shm_depth_bgr": str(self.shm_depth_bgr),
        }


class ShmRingWriter:
    """
    Shared-memory ring buffer for latest frames, compatible with:
      `D:\\DroneServer\\Drone_client\\New folder\\ipc\\shm_ring.py`
    """

    def __init__(
        self,
        *,
        spec: ShmRingSpec,
        shm_meta: SharedMemory,
        shm_bgr: SharedMemory,
        shm_gray: SharedMemory,
        shm_depth_raw: SharedMemory,
        shm_depth_bgr: SharedMemory,
    ) -> None:
        self.spec = spec
        self._shm_meta = shm_meta
        self._shm_bgr = shm_bgr
        self._shm_gray = shm_gray
        self._shm_depth_raw = shm_depth_raw
        self._shm_depth_bgr = shm_depth_bgr

        s = int(spec.slots)
        h = int(spec.h)
        w = int(spec.w)
        self.meta = np.ndarray((s,), dtype=_RING_META_DTYPE, buffer=shm_meta.buf)
        self.bgr = np.ndarray((s, h, w, 3), dtype=np.uint8, buffer=shm_bgr.buf)
        self.gray = np.ndarray((s, h, w), dtype=np.uint8, buffer=shm_gray.buf)
        self.depth_raw = np.ndarray((s, h, w), dtype=np.uint16, buffer=shm_depth_raw.buf)
        self.depth_bgr = np.ndarray((s, h, w, 3), dtype=np.uint8, buffer=shm_depth_bgr.buf)

        self._write_slot: int = 0

    @staticmethod
    def create(
        *,
        h: int,
        w: int,
        slots: int,
        name_prefix: str,
        force_unlink: bool = True,
    ) -> "ShmRingWriter":
        slots_i = int(max(2, int(slots)))
        h_i = int(max(1, int(h)))
        w_i = int(max(1, int(w)))
        prefix = str(name_prefix)

        if force_unlink and prefix:
            unlink_ring_prefix_best_effort(prefix)

        meta_nbytes = int(_RING_META_DTYPE.itemsize * slots_i)
        bgr_nbytes = int(slots_i * h_i * w_i * 3)
        gray_nbytes = int(slots_i * h_i * w_i)
        depth_raw_nbytes = int(slots_i * h_i * w_i * np.dtype(np.uint16).itemsize)
        depth_bgr_nbytes = int(slots_i * h_i * w_i * 3)

        shm_meta = SharedMemory(create=True, size=meta_nbytes, name=f"{prefix}_meta")
        shm_bgr = SharedMemory(create=True, size=bgr_nbytes, name=f"{prefix}_bgr")
        shm_gray = SharedMemory(create=True, size=gray_nbytes, name=f"{prefix}_gray")
        shm_depth_raw = SharedMemory(create=True, size=depth_raw_nbytes, name=f"{prefix}_depthraw")
        shm_depth_bgr = SharedMemory(create=True, size=depth_bgr_nbytes, name=f"{prefix}_depthbgr")

        spec = ShmRingSpec(
            slots=slots_i,
            h=h_i,
            w=w_i,
            name_prefix=prefix,
            shm_meta=shm_meta.name,
            shm_bgr=shm_bgr.name,
            shm_gray=shm_gray.name,
            shm_depth_raw=shm_depth_raw.name,
            shm_depth_bgr=shm_depth_bgr.name,
        )
        ring = ShmRingWriter(
            spec=spec,
            shm_meta=shm_meta,
            shm_bgr=shm_bgr,
            shm_gray=shm_gray,
            shm_depth_raw=shm_depth_raw,
            shm_depth_bgr=shm_depth_bgr,
        )

        ring.meta["seq"][:] = 0
        ring.meta["frame_idx"][:] = -1
        ring.meta["timestamp"][:] = 0.0
        return ring

    def close(self) -> None:
        for shm in (self._shm_meta, self._shm_bgr, self._shm_gray, self._shm_depth_raw, self._shm_depth_bgr):
            try:
                shm.close()
            except Exception:
                pass

    def unlink(self) -> None:
        for shm in (self._shm_meta, self._shm_bgr, self._shm_gray, self._shm_depth_raw, self._shm_depth_bgr):
            try:
                shm.unlink()
            except Exception:
                pass

    def write(
        self,
        *,
        frame_idx: int,
        timestamp: float,
        bgr: Optional[np.ndarray],
        gray: np.ndarray,
        depth_raw: Optional[np.ndarray],
        depth_bgr: Optional[np.ndarray] = None,
        fill_bgr_from_gray: bool = False,
    ) -> int:
        s = int(self.spec.slots)
        slot = int(self._write_slot % s)
        self._write_slot = (slot + 1) % s

        seq0 = int(self.meta["seq"][slot])
        self.meta["seq"][slot] = np.uint32(seq0 + 1)  # odd => writing

        if bgr is not None:
            self.bgr[slot][:] = bgr
        elif bool(fill_bgr_from_gray):
            try:
                self.bgr[slot][:] = np.asarray(gray, dtype=np.uint8)[:, :, None]
            except Exception:
                pass
        self.gray[slot][:] = gray
        if depth_raw is not None:
            self.depth_raw[slot][:] = depth_raw
        else:
            self.depth_raw[slot][:] = 0
        if depth_bgr is not None:
            self.depth_bgr[slot][:] = depth_bgr

        self.meta["frame_idx"][slot] = np.int64(int(frame_idx))
        self.meta["timestamp"][slot] = float(timestamp)
        self.meta["seq"][slot] = np.uint32(seq0 + 2)  # even => stable
        return slot


_IMU_META_DTYPE = np.dtype([("idx", np.uint64)])
IMU_KIND_GYRO = 0
IMU_KIND_ACCEL = 1
_IMU_SLOT_DTYPE = np.dtype(
    [
        ("seq", np.uint32),
        ("idx", np.int64),
        ("timestamp", np.float64),
        ("kind", np.uint8),
        ("gx", np.float32),
        ("gy", np.float32),
        ("gz", np.float32),
        ("ax", np.float32),
        ("ay", np.float32),
        ("az", np.float32),
    ]
)


@dataclass(frozen=True)
class ShmImuRingSpec:
    slots: int
    name_prefix: str
    shm_meta: str
    shm_slots: str

    def to_dict(self) -> dict:
        return {
            "slots": int(self.slots),
            "name_prefix": str(self.name_prefix),
            "shm_meta": str(self.shm_meta),
            "shm_slots": str(self.shm_slots),
        }


class ShmImuRingWriter:
    """
    Shared-memory IMU ring buffer compatible with:
      `D:\\DroneServer\\Drone_client\\New folder\\ipc\\shm_imu_ring.py`
    """

    def __init__(self, *, spec: ShmImuRingSpec, shm_meta: SharedMemory, shm_slots: SharedMemory) -> None:
        self.spec = spec
        self._shm_meta = shm_meta
        self._shm_slots = shm_slots

        s = int(spec.slots)
        self.meta = np.ndarray((1,), dtype=_IMU_META_DTYPE, buffer=shm_meta.buf)
        self.slots = np.ndarray((s,), dtype=_IMU_SLOT_DTYPE, buffer=shm_slots.buf)
        self._write_slot: int = 0

    @staticmethod
    def create(*, slots: int, name_prefix: str, force_unlink: bool = True) -> "ShmImuRingWriter":
        slots_i = int(max(128, int(slots)))
        prefix = str(name_prefix)

        if force_unlink and prefix:
            unlink_imu_prefix_best_effort(prefix)

        meta_nbytes = int(_IMU_META_DTYPE.itemsize)
        slot_nbytes = int(slots_i * _IMU_SLOT_DTYPE.itemsize)

        shm_meta = SharedMemory(create=True, size=meta_nbytes, name=f"{prefix}_meta")
        shm_slots = SharedMemory(create=True, size=slot_nbytes, name=f"{prefix}_slots")

        spec = ShmImuRingSpec(slots=slots_i, name_prefix=prefix, shm_meta=shm_meta.name, shm_slots=shm_slots.name)
        ring = ShmImuRingWriter(spec=spec, shm_meta=shm_meta, shm_slots=shm_slots)
        ring.meta["idx"][0] = np.uint64(0)
        ring.slots["seq"][:] = 0
        ring.slots["idx"][:] = -1
        ring.slots["timestamp"][:] = 0.0
        ring.slots["kind"][:] = np.uint8(0)
        ring.slots["gx"][:] = np.float32(0.0)
        ring.slots["gy"][:] = np.float32(0.0)
        ring.slots["gz"][:] = np.float32(0.0)
        ring.slots["ax"][:] = np.float32(0.0)
        ring.slots["ay"][:] = np.float32(0.0)
        ring.slots["az"][:] = np.float32(0.0)
        return ring

    def close(self) -> None:
        for shm in (self._shm_meta, self._shm_slots):
            try:
                shm.close()
            except Exception:
                pass

    def unlink(self) -> None:
        for shm in (self._shm_meta, self._shm_slots):
            try:
                shm.unlink()
            except Exception:
                pass

    def write(
        self,
        *,
        timestamp: float,
        kind: int,
        gyro_xyz_rad_s: Optional[Tuple[float, float, float]] = None,
        accel_xyz_m_s2: Optional[Tuple[float, float, float]] = None,
    ) -> int:
        s = int(self.spec.slots)
        cur_idx = int(self.meta["idx"][0])
        next_idx = int(cur_idx + 1)
        slot = int(self._write_slot % s)
        self._write_slot = (slot + 1) % s

        seq0 = int(self.slots["seq"][slot])
        self.slots["seq"][slot] = np.uint32(seq0 + 1)  # odd => writing

        if gyro_xyz_rad_s is None:
            gx, gy, gz = 0.0, 0.0, 0.0
        else:
            gx, gy, gz = gyro_xyz_rad_s
        if accel_xyz_m_s2 is None:
            ax, ay, az = 0.0, 0.0, 0.0
        else:
            ax, ay, az = accel_xyz_m_s2

        self.slots["idx"][slot] = np.int64(int(next_idx))
        self.slots["timestamp"][slot] = float(timestamp)
        self.slots["kind"][slot] = np.uint8(int(kind))
        self.slots["gx"][slot] = np.float32(gx)
        self.slots["gy"][slot] = np.float32(gy)
        self.slots["gz"][slot] = np.float32(gz)
        self.slots["ax"][slot] = np.float32(ax)
        self.slots["ay"][slot] = np.float32(ay)
        self.slots["az"][slot] = np.float32(az)

        self.slots["seq"][slot] = np.uint32(seq0 + 2)  # even => stable
        self.meta["idx"][0] = np.uint64(int(next_idx))
        return int(next_idx)

