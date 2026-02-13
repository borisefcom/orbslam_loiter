from __future__ import annotations

from dataclasses import dataclass
from multiprocessing import shared_memory
from typing import Optional, Tuple

import numpy as np

FRAMEBUS_VERSION = 1
DEFAULT_FRAMEBUS_NAME = "orbslam_framebus"

FLAG_RGB = 1 << 0
FLAG_DEPTH = 1 << 1
FLAG_IR_LEFT = 1 << 2
FLAG_IR_RIGHT = 1 << 3

CTRL_VERSION = 0
CTRL_WRITE_IDX = 1
CTRL_FRAME_SEQ = 2
CTRL_GYRO_COUNT = 3
CTRL_ACCEL_COUNT = 4
CTRL_ACTIVE_MODE = 5
CTRL_REQUEST_MODE = 6
CTRL_STATUS = 7
CTRL_LEN = 8

STATUS_RUNNING = 0
STATUS_STOP_REQUESTED = 1

META_DTYPE = np.dtype(
    [
        ("seq", "<u8"),
        ("ts", "<f8"),
        ("flags", "<u4"),
        ("mode", "<u2"),
        ("pad", "<u2"),
    ],
    align=False,
)


class FrameBusError(RuntimeError):
    pass


@dataclass(frozen=True)
class FrameBusConfig:
    name: str = DEFAULT_FRAMEBUS_NAME
    width: int = 640
    height: int = 480
    slots: int = 6
    imu_slots: int = 8192


def _shm_name(base: str, suffix: str) -> str:
    return f"{base}_{suffix}"


def _create_shm(name: str, size: int, *, force: bool) -> shared_memory.SharedMemory:
    try:
        return shared_memory.SharedMemory(name=name, create=True, size=int(size))
    except FileExistsError:
        if not force:
            raise FrameBusError(f"Shared memory '{name}' already exists.")
        existing = shared_memory.SharedMemory(name=name, create=False)
        existing.close()
        existing.unlink()
        return shared_memory.SharedMemory(name=name, create=True, size=int(size))


def _attach_shm(name: str) -> shared_memory.SharedMemory:
    try:
        return shared_memory.SharedMemory(name=name, create=False)
    except FileNotFoundError as exc:
        raise FrameBusError(f"Shared memory '{name}' not found. Start the framebus process first.") from exc


@dataclass(frozen=True)
class FrameBusFrame:
    seq: int
    ts_s: float
    flags: int
    mode_id: int
    rgb: Optional[np.ndarray]
    depth: Optional[np.ndarray]
    ir_left: Optional[np.ndarray]
    ir_right: Optional[np.ndarray]


class FrameBusWriter:
    def __init__(self, config: FrameBusConfig, *, force: bool = False) -> None:
        self.config = config
        self._shms: list[shared_memory.SharedMemory] = []
        try:
            self.ctrl_shm = _create_shm(_shm_name(config.name, "ctrl"), CTRL_LEN * 8, force=force)
            self._shms.append(self.ctrl_shm)
            self.meta_shm = _create_shm(
                _shm_name(config.name, "meta"), config.slots * META_DTYPE.itemsize, force=force
            )
            self._shms.append(self.meta_shm)

            rgb_bytes = int(config.slots * config.height * config.width * 3)
            depth_bytes = int(config.slots * config.height * config.width * 2)
            ir_bytes = int(config.slots * config.height * config.width)
            imu_bytes = int(config.imu_slots * 4 * 8)

            self.rgb_shm = _create_shm(_shm_name(config.name, "rgb"), rgb_bytes, force=force)
            self.depth_shm = _create_shm(_shm_name(config.name, "depth"), depth_bytes, force=force)
            self.ir_left_shm = _create_shm(_shm_name(config.name, "ir_left"), ir_bytes, force=force)
            self.ir_right_shm = _create_shm(_shm_name(config.name, "ir_right"), ir_bytes, force=force)
            self.gyro_shm = _create_shm(_shm_name(config.name, "gyro"), imu_bytes, force=force)
            self.accel_shm = _create_shm(_shm_name(config.name, "accel"), imu_bytes, force=force)
            self._shms.extend(
                [
                    self.rgb_shm,
                    self.depth_shm,
                    self.ir_left_shm,
                    self.ir_right_shm,
                    self.gyro_shm,
                    self.accel_shm,
                ]
            )
        except Exception:
            self.close(unlink=True)
            raise

        self.ctrl = np.ndarray((CTRL_LEN,), dtype=np.uint64, buffer=self.ctrl_shm.buf)
        self.ctrl[:] = 0
        self.ctrl[CTRL_VERSION] = np.uint64(FRAMEBUS_VERSION)

        self.meta = np.ndarray((config.slots,), dtype=META_DTYPE, buffer=self.meta_shm.buf)
        self.meta[:] = 0

        self.rgb = np.ndarray(
            (config.slots, config.height, config.width, 3), dtype=np.uint8, buffer=self.rgb_shm.buf
        )
        self.depth = np.ndarray(
            (config.slots, config.height, config.width), dtype=np.uint16, buffer=self.depth_shm.buf
        )
        self.ir_left = np.ndarray(
            (config.slots, config.height, config.width), dtype=np.uint8, buffer=self.ir_left_shm.buf
        )
        self.ir_right = np.ndarray(
            (config.slots, config.height, config.width), dtype=np.uint8, buffer=self.ir_right_shm.buf
        )
        self.gyro = np.ndarray((config.imu_slots, 4), dtype=np.float64, buffer=self.gyro_shm.buf)
        self.accel = np.ndarray((config.imu_slots, 4), dtype=np.float64, buffer=self.accel_shm.buf)

    def close(self, *, unlink: bool = False) -> None:
        for shm in self._shms:
            try:
                shm.close()
            except Exception:
                pass
        if unlink:
            for shm in self._shms:
                try:
                    shm.unlink()
                except Exception:
                    pass

    def set_active_mode(self, mode_id: int) -> None:
        self.ctrl[CTRL_ACTIVE_MODE] = np.uint64(mode_id)

    def set_request_mode(self, mode_id: int) -> None:
        self.ctrl[CTRL_REQUEST_MODE] = np.uint64(mode_id)

    def request_mode_id(self) -> int:
        return int(self.ctrl[CTRL_REQUEST_MODE])

    def stop_requested(self) -> bool:
        return int(self.ctrl[CTRL_STATUS]) == STATUS_STOP_REQUESTED

    def set_status(self, status: int) -> None:
        self.ctrl[CTRL_STATUS] = np.uint64(status)

    def write_frame(
        self,
        *,
        mode_id: int,
        ts_s: float,
        rgb: Optional[np.ndarray],
        depth: Optional[np.ndarray],
        ir_left: Optional[np.ndarray],
        ir_right: Optional[np.ndarray],
    ) -> None:
        idx = int(self.ctrl[CTRL_WRITE_IDX] % np.uint64(self.config.slots))
        flags = 0
        if rgb is not None:
            np.copyto(self.rgb[idx], rgb)
            flags |= FLAG_RGB
        if depth is not None:
            np.copyto(self.depth[idx], depth)
            flags |= FLAG_DEPTH
        if ir_left is not None:
            np.copyto(self.ir_left[idx], ir_left)
            flags |= FLAG_IR_LEFT
        if ir_right is not None:
            np.copyto(self.ir_right[idx], ir_right)
            flags |= FLAG_IR_RIGHT

        seq = int(self.ctrl[CTRL_FRAME_SEQ] + np.uint64(1))
        self.meta[idx]["ts"] = float(ts_s)
        self.meta[idx]["flags"] = np.uint32(flags)
        self.meta[idx]["mode"] = np.uint16(mode_id)
        self.meta[idx]["seq"] = np.uint64(seq)
        self.ctrl[CTRL_FRAME_SEQ] = np.uint64(seq)
        self.ctrl[CTRL_WRITE_IDX] = np.uint64(self.ctrl[CTRL_WRITE_IDX] + np.uint64(1))

    def write_gyro(self, *, t_s: float, gx: float, gy: float, gz: float) -> None:
        count = int(self.ctrl[CTRL_GYRO_COUNT])
        idx = int(count % self.config.imu_slots)
        self.gyro[idx] = (float(t_s), float(gx), float(gy), float(gz))
        self.ctrl[CTRL_GYRO_COUNT] = np.uint64(count + 1)

    def write_accel(self, *, t_s: float, ax: float, ay: float, az: float) -> None:
        count = int(self.ctrl[CTRL_ACCEL_COUNT])
        idx = int(count % self.config.imu_slots)
        self.accel[idx] = (float(t_s), float(ax), float(ay), float(az))
        self.ctrl[CTRL_ACCEL_COUNT] = np.uint64(count + 1)


class FrameBusReader:
    def __init__(self, config: FrameBusConfig) -> None:
        self.config = config
        self.ctrl_shm = _attach_shm(_shm_name(config.name, "ctrl"))
        self.meta_shm = _attach_shm(_shm_name(config.name, "meta"))
        self.rgb_shm = _attach_shm(_shm_name(config.name, "rgb"))
        self.depth_shm = _attach_shm(_shm_name(config.name, "depth"))
        self.ir_left_shm = _attach_shm(_shm_name(config.name, "ir_left"))
        self.ir_right_shm = _attach_shm(_shm_name(config.name, "ir_right"))
        self.gyro_shm = _attach_shm(_shm_name(config.name, "gyro"))
        self.accel_shm = _attach_shm(_shm_name(config.name, "accel"))

        self.ctrl = np.ndarray((CTRL_LEN,), dtype=np.uint64, buffer=self.ctrl_shm.buf)
        if int(self.ctrl[CTRL_VERSION]) != FRAMEBUS_VERSION:
            raise FrameBusError("Framebus version mismatch.")

        self.meta = np.ndarray((config.slots,), dtype=META_DTYPE, buffer=self.meta_shm.buf)
        self.rgb = np.ndarray(
            (config.slots, config.height, config.width, 3), dtype=np.uint8, buffer=self.rgb_shm.buf
        )
        self.depth = np.ndarray(
            (config.slots, config.height, config.width), dtype=np.uint16, buffer=self.depth_shm.buf
        )
        self.ir_left = np.ndarray(
            (config.slots, config.height, config.width), dtype=np.uint8, buffer=self.ir_left_shm.buf
        )
        self.ir_right = np.ndarray(
            (config.slots, config.height, config.width), dtype=np.uint8, buffer=self.ir_right_shm.buf
        )
        self.gyro = np.ndarray((config.imu_slots, 4), dtype=np.float64, buffer=self.gyro_shm.buf)
        self.accel = np.ndarray((config.imu_slots, 4), dtype=np.float64, buffer=self.accel_shm.buf)

        self._gyro_read = 0
        self._accel_read = 0

    def close(self) -> None:
        for shm in (
            self.ctrl_shm,
            self.meta_shm,
            self.rgb_shm,
            self.depth_shm,
            self.ir_left_shm,
            self.ir_right_shm,
            self.gyro_shm,
            self.accel_shm,
        ):
            try:
                shm.close()
            except Exception:
                pass

    def request_mode(self, mode_id: int) -> None:
        self.ctrl[CTRL_REQUEST_MODE] = np.uint64(mode_id)

    def active_mode(self) -> int:
        return int(self.ctrl[CTRL_ACTIVE_MODE])

    def status(self) -> dict[str, int]:
        return {
            "frame_seq": int(self.ctrl[CTRL_FRAME_SEQ]),
            "write_idx": int(self.ctrl[CTRL_WRITE_IDX]),
            "active_mode": int(self.ctrl[CTRL_ACTIVE_MODE]),
            "request_mode": int(self.ctrl[CTRL_REQUEST_MODE]),
            "gyro_count": int(self.ctrl[CTRL_GYRO_COUNT]),
            "accel_count": int(self.ctrl[CTRL_ACCEL_COUNT]),
            "status": int(self.ctrl[CTRL_STATUS]),
        }

    def request_stop(self) -> None:
        self.ctrl[CTRL_STATUS] = np.uint64(STATUS_STOP_REQUESTED)

    def read_latest(
        self,
        *,
        last_seq: int,
        required_flags: int = 0,
        mode_id: Optional[int] = None,
        copy: bool = True,
    ) -> Optional[FrameBusFrame]:
        write_idx = int(self.ctrl[CTRL_WRITE_IDX])
        if write_idx <= 0:
            return None
        idx = int((write_idx - 1) % self.config.slots)
        meta = self.meta[idx]
        seq = int(meta["seq"])
        if seq <= int(last_seq):
            return None
        if mode_id is not None and int(meta["mode"]) != int(mode_id):
            return None
        flags = int(meta["flags"])
        if required_flags and (flags & int(required_flags)) != int(required_flags):
            return None
        ts = float(meta["ts"])

        rgb = None
        depth = None
        ir_left = None
        ir_right = None
        if flags & FLAG_RGB:
            rgb = np.array(self.rgb[idx], copy=copy)
        if flags & FLAG_DEPTH:
            depth = np.array(self.depth[idx], copy=copy)
        if flags & FLAG_IR_LEFT:
            ir_left = np.array(self.ir_left[idx], copy=copy)
        if flags & FLAG_IR_RIGHT:
            ir_right = np.array(self.ir_right[idx], copy=copy)

        return FrameBusFrame(
            seq=seq,
            ts_s=ts,
            flags=flags,
            mode_id=int(meta["mode"]),
            rgb=rgb,
            depth=depth,
            ir_left=ir_left,
            ir_right=ir_right,
        )

    def _read_ring(self, buf: np.ndarray, total: int, last: int) -> Tuple[np.ndarray, int]:
        count = int(total - last)
        if count <= 0:
            return np.empty((0, 4), dtype=np.float64), int(last)
        if count > int(buf.shape[0]):
            last = int(total - buf.shape[0])
            count = int(buf.shape[0])
        start = int(last % buf.shape[0])
        end = int(start + count)
        if end <= int(buf.shape[0]):
            data = np.array(buf[start:end], copy=True)
        else:
            part1 = np.array(buf[start:], copy=True)
            part2 = np.array(buf[: end - buf.shape[0]], copy=True)
            data = np.concatenate([part1, part2], axis=0)
        return data, int(last + count)

    def drain_imu(self) -> Tuple[np.ndarray, np.ndarray]:
        gyro_total = int(self.ctrl[CTRL_GYRO_COUNT])
        accel_total = int(self.ctrl[CTRL_ACCEL_COUNT])
        gyro, new_gyro = self._read_ring(self.gyro, gyro_total, self._gyro_read)
        accel, new_accel = self._read_ring(self.accel, accel_total, self._accel_read)
        self._gyro_read = int(new_gyro)
        self._accel_read = int(new_accel)
        return gyro, accel
