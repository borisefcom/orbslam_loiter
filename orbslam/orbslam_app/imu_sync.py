from __future__ import annotations

from collections import deque
from typing import Deque, Optional, Tuple

import numpy as np


class ImuSynchronizer:
    """
    Collects RealSense IMU samples (gyro+accel) and produces ORB-SLAM3 IMU batches:
    Nx7 float64: [ax, ay, az, gx, gy, gz, timestamp_s]
    """

    def __init__(self, *, max_gyro: int = 20000, max_accel: int = 6000) -> None:
        self._gyro: Deque[Tuple[float, float, float, float]] = deque(maxlen=int(max_gyro))
        self._accel: Deque[Tuple[float, float, float, float]] = deque(maxlen=int(max_accel))

    def reset(self) -> None:
        self._gyro.clear()
        self._accel.clear()

    def gyro_time_range(self) -> Tuple[Optional[float], Optional[float]]:
        if not self._gyro:
            return None, None
        return float(self._gyro[0][0]), float(self._gyro[-1][0])

    def add_gyro(self, *, t_s: float, gx: float, gy: float, gz: float) -> None:
        t_s = float(t_s)
        # ORB-SLAM3 IMU preintegration expects strictly increasing timestamps.
        if self._gyro:
            last_t = float(self._gyro[-1][0])
            if t_s <= last_t:
                # If the clock jumps backwards significantly, treat it as a reset.
                if t_s < last_t - 0.5:
                    self._gyro.clear()
                else:
                    return
        self._gyro.append((t_s, float(gx), float(gy), float(gz)))

    def add_accel(self, *, t_s: float, ax: float, ay: float, az: float) -> None:
        t_s = float(t_s)
        if self._accel:
            last_t = float(self._accel[-1][0])
            if t_s <= last_t:
                if t_s < last_t - 0.5:
                    self._accel.clear()
                else:
                    return
        self._accel.append((t_s, float(ax), float(ay), float(az)))

    def _interp_accel(self, t_s: float) -> Optional[Tuple[float, float, float]]:
        if not self._accel:
            return None

        # Keep the accel deque centered around the query time (t between first and second).
        while len(self._accel) >= 2 and float(self._accel[1][0]) <= float(t_s):
            self._accel.popleft()

        if len(self._accel) == 1:
            _t0, ax, ay, az = self._accel[0]
            return float(ax), float(ay), float(az)

        t0, ax0, ay0, az0 = self._accel[0]
        t1, ax1, ay1, az1 = self._accel[1]
        if float(t1) <= float(t0) + 1e-12:
            return float(ax1), float(ay1), float(az1)

        alpha = (float(t_s) - float(t0)) / (float(t1) - float(t0))
        alpha = float(np.clip(alpha, 0.0, 1.0))
        ax = float(ax0 + alpha * (ax1 - ax0))
        ay = float(ay0 + alpha * (ay1 - ay0))
        az = float(az0 + alpha * (az1 - az0))
        return ax, ay, az

    def pop_batch(self, *, t0_s: float, t1_s: float, min_samples: int = 2) -> np.ndarray:
        """
        Return a batch of IMU samples for ORB-SLAM3:
          - all gyro samples with t in (t0_s, t1_s], paired with interpolated accel at gyro timestamps
          - PLUS (when available) the first gyro sample strictly after t1_s.

        Rationale:
        ORB-SLAM3's IMU preintegration expects that, when processing a frame at time t1_s, its internal IMU
        queue contains at least one measurement at/after the frame timestamp so it can integrate up to t1_s
        (interpolating the last segment). If we only push samples <= t1_s, ORB-SLAM3 may log "not IMU meas"
        and can become unstable.

        Gyro samples are only consumed if at least `min_samples` are available.
        """
        t0_s = float(t0_s)
        t1_s = float(t1_s)
        min_samples = int(min_samples)
        if t1_s <= t0_s:
            return np.empty((0, 7), dtype=np.float64)

        # Drop old gyro samples (<= t0_s).
        while self._gyro and float(self._gyro[0][0]) <= float(t0_s):
            self._gyro.popleft()

        rows = []
        i = 0
        added_post = False
        while i < len(self._gyro):
            tg, gx, gy, gz = self._gyro[i]
            tg_f = float(tg)
            if tg_f > float(t1_s):
                # Include exactly one sample after the frame timestamp (if possible).
                if not added_post:
                    a = self._interp_accel(tg_f)
                    if a is not None:
                        ax, ay, az = a
                        rows.append([ax, ay, az, float(gx), float(gy), float(gz), tg_f])
                        i += 1
                    added_post = True
                break
            a = self._interp_accel(tg_f)
            if a is not None:
                ax, ay, az = a
                rows.append([ax, ay, az, float(gx), float(gy), float(gz), tg_f])
            i += 1

        if len(rows) < max(1, min_samples):
            return np.empty((0, 7), dtype=np.float64)

        # Consume gyro samples used in this batch (including the optional post-frame sample).
        for _ in range(i):
            if not self._gyro:
                break
            self._gyro.popleft()

        return np.asarray(rows, dtype=np.float64).reshape(-1, 7)
