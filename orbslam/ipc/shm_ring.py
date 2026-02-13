from __future__ import annotations

import time
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


_META_DTYPE = np.dtype(
    [
        ("seq", np.uint32),  # even=stable, odd=writer in progress
        ("frame_idx", np.int64),
        ("timestamp", np.float64),
    ]
)


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

    @staticmethod
    def from_dict(d: dict) -> "ShmRingSpec":
        return ShmRingSpec(
            slots=int(d["slots"]),
            h=int(d["h"]),
            w=int(d["w"]),
            name_prefix=str(d.get("name_prefix", "")),
            shm_meta=str(d["shm_meta"]),
            shm_bgr=str(d["shm_bgr"]),
            shm_gray=str(d["shm_gray"]),
            shm_depth_raw=str(d["shm_depth_raw"]),
            shm_depth_bgr=str(d["shm_depth_bgr"]),
        )


class ShmRing:
    """
    Shared-memory ring buffer for latest frames.

    - Fixed resolution, fixed dtype.
    - Writer does a 2-phase commit using a per-slot sequence counter:
        seq odd   -> writing
        seq even  -> stable
      Reader checks seq before choosing a slot.
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

        self.meta = np.ndarray((s,), dtype=_META_DTYPE, buffer=shm_meta.buf)
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
        slots: int = 10,
        name_prefix: Optional[str] = None,
    ) -> "ShmRing":
        slots_i = int(max(2, slots))
        h_i = int(max(1, h))
        w_i = int(max(1, w))
        prefix = str(name_prefix or f"mf_{uuid.uuid4().hex[:8]}")

        meta_nbytes = int(_META_DTYPE.itemsize * slots_i)
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

        ring = ShmRing(
            spec=spec,
            shm_meta=shm_meta,
            shm_bgr=shm_bgr,
            shm_gray=shm_gray,
            shm_depth_raw=shm_depth_raw,
            shm_depth_bgr=shm_depth_bgr,
        )

        # Init metadata to "empty"
        ring.meta["seq"][:] = 0
        ring.meta["frame_idx"][:] = -1
        ring.meta["timestamp"][:] = 0.0

        return ring

    @staticmethod
    def attach(spec: ShmRingSpec) -> "ShmRing":
        shm_meta = _attach_shm(spec.shm_meta)
        shm_bgr = _attach_shm(spec.shm_bgr)
        shm_gray = _attach_shm(spec.shm_gray)
        shm_depth_raw = _attach_shm(spec.shm_depth_raw)
        shm_depth_bgr = _attach_shm(spec.shm_depth_bgr)
        return ShmRing(
            spec=spec,
            shm_meta=shm_meta,
            shm_bgr=shm_bgr,
            shm_gray=shm_gray,
            shm_depth_raw=shm_depth_raw,
            shm_depth_bgr=shm_depth_bgr,
        )

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
        depth_bgr: Optional[np.ndarray],
        fill_bgr_from_gray: bool = False,
    ) -> int:
        """Write into next slot and return slot index."""
        s = int(self.spec.slots)
        slot = int(self._write_slot % s)
        self._write_slot = (slot + 1) % s

        seq0 = int(self.meta["seq"][slot])
        self.meta["seq"][slot] = np.uint32(seq0 + 1)  # odd => writing

        if bgr is not None:
            self.bgr[slot][:] = bgr
        elif bool(fill_bgr_from_gray):
            # Avoid leaving stale BGR contents when the source stream is mono.
            # Broadcasting copies gray into all 3 channels without allocating a temporary BGR image.
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
        # If depth_bgr is None, leave the existing buffer contents untouched (saves a full-frame memset/copy).

        self.meta["frame_idx"][slot] = np.int64(int(frame_idx))
        self.meta["timestamp"][slot] = float(timestamp)
        self.meta["seq"][slot] = np.uint32(seq0 + 2)  # even => stable

        return slot

    def _stable_slot_info(self, slot: int) -> Optional[Tuple[int, float]]:
        s1 = int(self.meta["seq"][slot])
        if s1 & 1:
            return None
        fi = int(self.meta["frame_idx"][slot])
        if fi < 0:
            return None
        ts = float(self.meta["timestamp"][slot])
        s2 = int(self.meta["seq"][slot])
        if s1 != s2 or (s2 & 1):
            return None
        return fi, ts

    def read_latest(self) -> Optional[Tuple[int, int, float]]:
        """Return (slot, frame_idx, timestamp) for latest stable slot."""
        best_slot = -1
        best_fi = -1
        best_ts = 0.0
        for slot in range(int(self.spec.slots)):
            info = self._stable_slot_info(slot)
            if info is None:
                continue
            fi, ts = info
            if fi > best_fi:
                best_slot = slot
                best_fi = fi
                best_ts = ts
        if best_slot < 0:
            return None
        return best_slot, best_fi, best_ts

    def read_next(self, *, after_frame_idx: int, timeout: float = 0.05) -> Optional[Tuple[int, int, float]]:
        """Return (slot, frame_idx, timestamp) for the oldest stable slot with frame_idx > after."""
        after = int(after_frame_idx)
        end = time.time() + float(max(0.0, timeout))
        while True:
            best_slot = -1
            best_fi = None
            best_ts = 0.0
            for slot in range(int(self.spec.slots)):
                info = self._stable_slot_info(slot)
                if info is None:
                    continue
                fi, ts = info
                if fi <= after:
                    continue
                if best_fi is None or fi < best_fi:
                    best_slot = slot
                    best_fi = fi
                    best_ts = ts
            if best_slot >= 0 and best_fi is not None:
                return best_slot, int(best_fi), float(best_ts)
            if time.time() >= end:
                return None
            time.sleep(0.001)

    def find_slot_by_frame_idx(self, frame_idx: int) -> Optional[int]:
        """Return slot index that contains frame_idx if still present and stable."""
        target = int(frame_idx)
        for slot in range(int(self.spec.slots)):
            info = self._stable_slot_info(slot)
            if info is None:
                continue
            fi, _ts = info
            if fi == target:
                return slot
        return None
