"""
Frame + IMU utilities.

- FrameBus: shared-memory ring buffer for BGR frames (zero-copy reads across processes).
- IMU packets: process-local queue (used by IMU/VO code running in the same process as capture).
- calibrate_imu(): simple startup bias calibration (ported from Drone_client/world_cord_persuit).
"""

from __future__ import annotations

import atexit
import multiprocessing as mp
import threading
import time
from collections import deque
from dataclasses import dataclass
from multiprocessing import shared_memory
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

__all__ = [
    "FrameBus",
    "IMUPacket",
    "IMUCalibration",
    "calibrate_imu",
]


@dataclass(frozen=True)
class IMUPacket:
    ts: float
    dev_ts: float
    kind: str  # "gyro" | "accel"
    x: float
    y: float
    z: float


@dataclass(frozen=True)
class IMUCalibration:
    gyro_bias: np.ndarray
    accel_bias: np.ndarray
    accel_mean: np.ndarray


class FrameBus:
    """
    Shared-memory ring buffer for frames (BGR uint8) + optional IMU queue.

    One writer (camera thread) copies frames into shared slots; readers in other
    processes call latest() and receive a zero-copy numpy view.

    NOTE: IMU packets are process-local by design (not shared across processes).
    """

    def __init__(self, width: int = 1280, height: int = 720, channels: int = 3, slots: int = 32):
        self.width = int(width)
        self.height = int(height)
        self.channels = int(channels)
        self.shape = (self.height, self.width, self.channels)
        self.dtype = np.uint8
        self.slots = max(2, int(slots))

        self._frame_bytes = int(np.prod(self.shape) * np.dtype(self.dtype).itemsize)
        self._owner = True
        self._frame_shm: List[shared_memory.SharedMemory] = [
            shared_memory.SharedMemory(create=True, size=self._frame_bytes) for _ in range(self.slots)
        ]
        self._frames = [np.ndarray(self.shape, dtype=self.dtype, buffer=shm.buf) for shm in self._frame_shm]

        # meta[slot] = [ts_ms, W, H, cam_id, seq]
        self._meta_shm = shared_memory.SharedMemory(
            create=True, size=self.slots * 5 * np.dtype(np.int64).itemsize
        )
        self._meta = np.ndarray((self.slots, 5), dtype=np.int64, buffer=self._meta_shm.buf)
        self._meta.fill(-1)

        # header = [idx, seq, update_count]
        self._hdr_shm = shared_memory.SharedMemory(create=True, size=3 * np.dtype(np.int64).itemsize)
        self._hdr = np.ndarray((3,), dtype=np.int64, buffer=self._hdr_shm.buf)
        self._hdr[:] = 0

        self._write_lock = mp.Lock()
        self._last_ts_ms: Optional[float] = None
        self._cap_dt_hist: deque = deque(maxlen=240)
        self._cam_name = "realsense"

        # IMU queue (process-local; not shared)
        self._imu_lock = threading.Lock()
        self._imu_ring: deque = deque(maxlen=8192)

        self._closed = False
        atexit.register(self.close)

    # ------------------------------------------------------------------
    # Shared-memory handles
    # ------------------------------------------------------------------
    def export_handles(self) -> Dict[str, Any]:
        """Return shared-memory names + metadata so another process can attach."""
        return {
            "frame_names": [shm.name for shm in self._frame_shm],
            "meta_name": self._meta_shm.name,
            "header_name": self._hdr_shm.name,
            "shape": tuple(self.shape),
            "dtype": str(self.dtype),
            "slots": self.slots,
            "cam": self._cam_name,
        }

    @classmethod
    def attach(cls, handles: Dict[str, Any]) -> "FrameBus":
        """Attach to an existing shared ring created by export_handles()."""
        obj = cls.__new__(cls)
        obj.width = int(handles["shape"][1])
        obj.height = int(handles["shape"][0])
        obj.channels = int(handles["shape"][2]) if len(handles["shape"]) > 2 else 1
        obj.shape = tuple(handles["shape"])

        dtype_field = handles.get("dtype", "uint8")
        if not isinstance(dtype_field, str):
            dtype_field = str(dtype_field)
        dtype_field = dtype_field.replace("<class '", "").replace("'>", "")
        try:
            obj.dtype = np.dtype(dtype_field)
        except TypeError:
            obj.dtype = np.dtype(dtype_field.replace("numpy.", ""))

        obj.slots = int(handles.get("slots", 8))
        obj._frame_bytes = int(np.prod(obj.shape) * np.dtype(obj.dtype).itemsize)
        obj._owner = False

        obj._frame_shm = []
        for name in handles.get("frame_names", []):
            shm = shared_memory.SharedMemory(name=name)
            obj._frame_shm.append(shm)

        obj._frames = [np.ndarray(obj.shape, dtype=obj.dtype, buffer=shm.buf) for shm in obj._frame_shm]

        obj._meta_shm = shared_memory.SharedMemory(name=handles["meta_name"])
        obj._meta = np.ndarray((obj.slots, 5), dtype=np.int64, buffer=obj._meta_shm.buf)

        obj._hdr_shm = shared_memory.SharedMemory(name=handles["header_name"])
        obj._hdr = np.ndarray((3,), dtype=np.int64, buffer=obj._hdr_shm.buf)

        obj._write_lock = None  # reader-only in attached processes
        obj._last_ts_ms = None
        obj._cap_dt_hist = deque(maxlen=240)
        obj._cam_name = str(handles.get("cam", "realsense"))

        obj._imu_lock = threading.Lock()
        obj._imu_ring = deque(maxlen=8192)

        obj._closed = False
        atexit.register(obj.close)
        return obj

    # ------------------------------------------------------------------
    # Frame path
    # ------------------------------------------------------------------
    def update(self, frame, ts_ms: int, w: int, h: int, cam: str = "realsense", owner=None, cleanup=None):
        """
        Copy the input frame into the shared ring and publish as latest.
        Frame is treated as read-only by all consumers.
        """
        if frame is None:
            return
        self._cam_name = str(cam) if cam else "cam"

        try:
            arr = frame
            if hasattr(arr, "shape") and arr.ndim == 3 and arr.shape[2] > 3:
                arr = arr[:, :, :3]
            if hasattr(arr, "shape") and arr.ndim == 2:
                arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
            if arr.shape[:2] != (self.height, self.width):
                arr = cv2.resize(arr, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        except Exception:
            arr = frame

        if self._write_lock is not None:
            with self._write_lock:
                self._publish(arr, ts_ms, w, h)
        else:
            self._publish(arr, ts_ms, w, h)

    def _publish(self, arr, ts_ms: int, w: int, h: int):
        next_idx = (int(self._hdr[0]) + 1) % self.slots
        try:
            np.copyto(self._frames[next_idx], arr, casting="unsafe")
        except Exception:
            try:
                self._frames[next_idx][:] = arr
            except Exception:
                return

        self._meta[next_idx, 0] = int(ts_ms)
        self._meta[next_idx, 1] = int(w)
        self._meta[next_idx, 2] = int(h)
        self._meta[next_idx, 3] = 0  # reserved cam id; recover string from _cam_name
        self._meta[next_idx, 4] = int(self._hdr[1]) + 1

        self._hdr[0] = next_idx
        self._hdr[1] = self._meta[next_idx, 4]
        self._hdr[2] = self._hdr[2] + 1

        if self._last_ts_ms is not None:
            dt_s = max(0.0, (float(ts_ms) - float(self._last_ts_ms)) / 1000.0)
            self._cap_dt_hist.append(dt_s)
        self._last_ts_ms = float(ts_ms)

    def latest(self):
        idx = int(self._hdr[0])
        if idx < 0 or idx >= self.slots:
            return None
        ts_ms, W, H, cam_id, seq = map(int, self._meta[idx])
        return (self._frames[idx], ts_ms, W, H, self._cam_name, seq)

    def get_capture_dt_avg(self) -> float:
        if not self._cap_dt_hist:
            return 0.0
        try:
            return float(sum(self._cap_dt_hist) / len(self._cap_dt_hist))
        except Exception:
            return 0.0

    def get_capture_fps(self) -> float:
        dt = self.get_capture_dt_avg()
        return 0.0 if dt <= 1e-9 else (1.0 / dt)

    def get_update_count(self) -> int:
        return int(self._hdr[2])

    # ------------------------------------------------------------------
    # IMU path (process-local)
    # ------------------------------------------------------------------
    def push_imu(self, kind: str, ts_host_s: float, ts_dev_s: float, x: float, y: float, z: float):
        if kind not in ("gyro", "accel"):
            return
        pkt = IMUPacket(
            ts=float(ts_host_s),
            dev_ts=float(ts_dev_s),
            kind=str(kind),
            x=float(x),
            y=float(y),
            z=float(z),
        )
        with self._imu_lock:
            self._imu_ring.append(pkt)

    def drain_imu(self, max_items: int = 4096) -> List[IMUPacket]:
        out: List[IMUPacket] = []
        with self._imu_lock:
            for _ in range(min(int(max_items), len(self._imu_ring))):
                try:
                    out.append(self._imu_ring.popleft())
                except Exception:
                    break
        return out

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
    def close(self):
        if getattr(self, "_closed", False):
            return
        self._closed = True
        try:
            for shm in getattr(self, "_frame_shm", []) or []:
                try:
                    shm.close()
                except Exception:
                    pass
            try:
                self._meta_shm.close()
            except Exception:
                pass
            try:
                self._hdr_shm.close()
            except Exception:
                pass

            if getattr(self, "_owner", False):
                for shm in getattr(self, "_frame_shm", []) or []:
                    try:
                        shm.unlink()
                    except FileNotFoundError:
                        pass
                    except Exception:
                        pass
                try:
                    self._meta_shm.unlink()
                except FileNotFoundError:
                    pass
                except Exception:
                    pass
                try:
                    self._hdr_shm.unlink()
                except FileNotFoundError:
                    pass
                except Exception:
                    pass
        except Exception:
            pass
        try:
            self._owner = False
        except Exception:
            pass

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def calibrate_imu(imu_drain_fn, seconds: float = 5.0) -> IMUCalibration:
    """
    Estimate gyro bias (mean gyro while stationary) and accel bias (align mean accel to 9.81 m/s^2).

    imu_drain_fn: callable(max_items:int)->List[IMUPacket]
    """
    seconds = float(max(0.0, seconds))
    t0 = time.monotonic()

    gyro: List[List[float]] = []
    accel: List[List[float]] = []

    while (time.monotonic() - t0) < seconds:
        pkts = imu_drain_fn(8192)
        for p in pkts:
            if p.kind == "gyro":
                gyro.append([p.x, p.y, p.z])
            elif p.kind == "accel":
                accel.append([p.x, p.y, p.z])
        time.sleep(0.01)

    gyro_bias = np.mean(np.array(gyro, dtype=float), axis=0) if gyro else np.zeros(3, dtype=float)

    if accel:
        accel_mean = np.mean(np.array(accel, dtype=float), axis=0)
        mag = float(np.linalg.norm(accel_mean))
        if mag < 1e-9:
            accel_bias = np.zeros(3, dtype=float)
        else:
            g = 9.81
            g_hat = accel_mean / mag
            accel_bias = accel_mean - g_hat * g
    else:
        accel_mean = np.zeros(3, dtype=float)
        accel_bias = np.zeros(3, dtype=float)

    return IMUCalibration(gyro_bias=gyro_bias, accel_bias=accel_bias, accel_mean=accel_mean)
