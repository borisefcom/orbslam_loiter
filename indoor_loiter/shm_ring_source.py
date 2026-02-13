from __future__ import annotations

from dataclasses import dataclass
from multiprocessing.shared_memory import SharedMemory
from typing import Any, Optional, Tuple

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
        ("timestamp", np.float64),  # seconds
    ]
)


@dataclass(frozen=True)
class ShmRingSpec:
    slots: int
    h: int
    w: int
    shm_meta: str
    shm_bgr: str
    shm_gray: str
    shm_depth_raw: str
    shm_depth_bgr: str

    @staticmethod
    def from_dict(d: dict) -> "ShmRingSpec":
        return ShmRingSpec(
            slots=int(d["slots"]),
            h=int(d["h"]),
            w=int(d["w"]),
            shm_meta=str(d["shm_meta"]),
            shm_bgr=str(d["shm_bgr"]),
            shm_gray=str(d["shm_gray"]),
            shm_depth_raw=str(d["shm_depth_raw"]),
            shm_depth_bgr=str(d["shm_depth_bgr"]),
        )


class ShmRingSource:
    """
    Read-only attachment to the 3D tracker's shared-memory ring.

    Provides a FrameBus-compatible `.latest()` method:
      (frame, ts_ms, W, H, cam_name, seq)
    """

    def __init__(self, *, spec: ShmRingSpec, cam_name: str = "realsense", frame_kind: str = "bgr") -> None:
        self.spec = spec
        self.cam_name = str(cam_name or "realsense")
        self.frame_kind = str(frame_kind or "bgr").strip().lower()

        self._shm_meta = _attach_shm(spec.shm_meta)
        self._shm_bgr = _attach_shm(spec.shm_bgr)
        self._shm_gray = _attach_shm(spec.shm_gray)
        self._shm_depth_raw = _attach_shm(spec.shm_depth_raw)
        self._shm_depth_bgr = _attach_shm(spec.shm_depth_bgr)

        s = int(spec.slots)
        h = int(spec.h)
        w = int(spec.w)
        self.meta = np.ndarray((s,), dtype=_META_DTYPE, buffer=self._shm_meta.buf)
        self.bgr = np.ndarray((s, h, w, 3), dtype=np.uint8, buffer=self._shm_bgr.buf)
        self.gray = np.ndarray((s, h, w), dtype=np.uint8, buffer=self._shm_gray.buf)
        self.depth_raw = np.ndarray((s, h, w), dtype=np.uint16, buffer=self._shm_depth_raw.buf)
        self.depth_bgr = np.ndarray((s, h, w, 3), dtype=np.uint8, buffer=self._shm_depth_bgr.buf)

        self._last_frame_idx = -1

    def close(self) -> None:
        for shm in (self._shm_meta, self._shm_bgr, self._shm_gray, self._shm_depth_raw, self._shm_depth_bgr):
            try:
                shm.close()
            except Exception:
                pass

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

    def latest(self) -> Optional[Tuple[Any, int, int, int, str, int]]:
        info = self.read_latest()
        if info is None:
            return None
        slot, fi, ts_s = info
        ts_ms = int(float(ts_s) * 1000.0)
        if self.frame_kind in ("gray", "grey", "gray8", "mono"):
            frame = self.gray[int(slot)]
        elif self.frame_kind in ("depth", "depth_bgr", "depthbgr", "depth_bw", "depthbw"):
            frame = self.depth_bgr[int(slot)]
        elif self.frame_kind in ("depth_raw", "depthraw", "z16", "depth16"):
            frame = self.depth_raw[int(slot)]
        else:
            frame = self.bgr[int(slot)]
        return (frame, ts_ms, int(self.spec.w), int(self.spec.h), self.cam_name, int(fi))
