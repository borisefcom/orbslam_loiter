#!/usr/bin/env python3
"""
imu_fusion.py

IMU + (optional) VO fusion thread to estimate camera orientation and stabilize yaw drift.

Ported from: D:/DroneServer/Drone_client/world_cord_persuit/imu.py
Adaptations for loiter:
  - Consumes IMUPacket objects from loiter FrameBus (ts/dev_ts, kind, x/y/z).
  - Consumes VO messages produced by loiter egomotion worker:
      {"type":"vo", "q_ck_c0":[w,x,y,z], "quality":..., "inliers":..., "tracks":...}
      {"type":"vo_stats", "fps":...}
  - Optionally calls a callback to publish measured VO FPS into tracker perf line.
"""

from __future__ import annotations

import math
import threading
import time
from typing import Any, Callable, Optional, Tuple

import numpy as np

from imu import IMUPacket, IMUCalibration, calibrate_imu


def wrap_pi(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


def skew(w: np.ndarray) -> np.ndarray:
    wx, wy, wz = float(w[0]), float(w[1]), float(w[2])
    return np.array([[0.0, -wz, wy], [wz, 0.0, -wx], [-wy, wx, 0.0]], dtype=float)


def so3_exp(phi: np.ndarray) -> np.ndarray:
    angle = float(np.linalg.norm(phi))
    if angle < 1e-10:
        return np.eye(3, dtype=float) + skew(phi)
    axis = phi / angle
    K = skew(axis)
    return np.eye(3, dtype=float) + math.sin(angle) * K + (1.0 - math.cos(angle)) * (K @ K)


def so3_log(R: np.ndarray) -> np.ndarray:
    tr = float(np.trace(R))
    c = (tr - 1.0) * 0.5
    c = max(-1.0, min(1.0, c))
    angle = math.acos(c)
    if angle < 1e-10:
        return np.zeros(3, dtype=float)
    w = (
        np.array(
            [R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]],
            dtype=float,
        )
        * (0.5 / math.sin(angle))
    )
    return w * angle


def quat_norm(q: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(q))
    if n < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    return q / n


def R_from_quat(q: np.ndarray) -> np.ndarray:
    """(w,x,y,z) -> R"""
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    n = w * w + x * x + y * y + z * z
    if n < 1e-12:
        return np.eye(3, dtype=float)
    s = 2.0 / n
    wx, wy, wz = s * w * x, s * w * y, s * w * z
    xx, xy, xz = s * x * x, s * x * y, s * x * z
    yy, yz, zz = s * y * y, s * y * z, s * z * z
    return np.array(
        [
            [1.0 - (yy + zz), xy - wz, xz + wy],
            [xy + wz, 1.0 - (xx + zz), yz - wx],
            [xz - wy, yz + wx, 1.0 - (xx + yy)],
        ],
        dtype=float,
    )


def quat_from_R(R: np.ndarray) -> np.ndarray:
    """R -> (w,x,y,z)"""
    tr = float(np.trace(R))
    if tr > 0.0:
        S = math.sqrt(tr + 1.0) * 2.0
        w = 0.25 * S
        x = (R[2, 1] - R[1, 2]) / S
        y = (R[0, 2] - R[2, 0]) / S
        z = (R[1, 0] - R[0, 1]) / S
    else:
        if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            S = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
            w = (R[2, 1] - R[1, 2]) / S
            x = 0.25 * S
            y = (R[0, 1] + R[1, 0]) / S
            z = (R[0, 2] + R[2, 0]) / S
        elif R[1, 1] > R[2, 2]:
            S = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
            w = (R[0, 2] - R[2, 0]) / S
            x = (R[0, 1] + R[1, 0]) / S
            y = 0.25 * S
            z = (R[1, 2] + R[2, 1]) / S
        else:
            S = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
            w = (R[1, 0] - R[0, 1]) / S
            x = (R[0, 2] + R[2, 0]) / S
            y = (R[1, 2] + R[2, 1]) / S
            z = 0.25 * S
    return quat_norm(np.array([w, x, y, z], dtype=float))


def R_imu_to_cam_from_yaw(yaw_rad: float) -> np.ndarray:
    """Simple mount model: IMU->camera is yaw about +Z."""
    c = math.cos(float(yaw_rad))
    s = math.sin(float(yaw_rad))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)


class IMUProcessor(threading.Thread):
    """
    IMU fusion + VO correction.
    """

    def __init__(
        self,
        *,
        imu_drain_fn: Callable[[int], list[IMUPacket]],
        vo_queue: Optional[Any],
        get_mount_yaw_rad: Callable[[], float],
        imu_calib_seconds: float = 5.0,
        gyro_scale: float = 1.0,
        accel_scale: float = 1.0,
        set_vo_fps_cb: Optional[Callable[[float], None]] = None,
    ):
        super().__init__(name="IMUProcessor", daemon=True)

        self.imu_drain_fn = imu_drain_fn
        self.vo_queue = vo_queue
        self.get_mount_yaw_rad = get_mount_yaw_rad
        self.imu_calib_seconds = float(imu_calib_seconds)

        self.gyro_scale = float(gyro_scale)
        self.accel_scale = float(accel_scale)

        # nominal state: world<-imu
        self.q_WI = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
        self.b_g = np.zeros(3, dtype=float)
        self.accel_bias = np.zeros(3, dtype=float)

        # covariance on [dtheta(3), dbg(3)]
        self.P = np.eye(6, dtype=float) * 1e-2

        # noise params
        self.gyro_noise_std = 0.02
        self.gyro_bias_rw_std = 0.001
        self.accel_dir_std = 0.08
        self.accel_gate_tol = 1.2

        # VO gating
        self.vo_min_quality = 0.25
        self.vo_min_inliers = 30

        # online bias learning (stationary)
        self.stationary_gyro_thresh = 0.02  # rad/s
        self.stationary_acc_tol = 0.25  # m/s^2 around 9.81
        self.bias_tau = 8.0  # seconds
        self._stationary_time = 0.0

        # accel LPF
        self.accel_lpf_tau = 0.15  # seconds
        self._acc_lp: Optional[np.ndarray] = None
        self._last_acc_dev_s: Optional[float] = None
        self._last_acc_host_s: Optional[float] = None
        self._last_acc_meas: Optional[np.ndarray] = None

        # timing for gyro
        self._last_gyro_dev_s: Optional[float] = None
        self._last_gyro_host_s: Optional[float] = None

        # VO reference
        self._R_WC0: Optional[np.ndarray] = None

        # Latest IMU values (bias-corrected)
        self._latest_gyro_corr = np.zeros(3, dtype=float)
        self._latest_acc_corr = np.zeros(3, dtype=float)

        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._est_initialized = False

        # boresight smoothing
        self.boresight_tau = 0.25
        self._bore_init = False
        self._bore_yaw_s = 0.0
        self._bore_pitch_s = 0.0
        self._bore_last_t = time.monotonic()

        # VO perf callback into tracker perf line
        self._set_vo_fps_cb = set_vo_fps_cb
        self._last_vo_fps = 0.0
        self._last_vo_fps_push_t = 0.0

    def stop(self) -> None:
        self._stop_event.set()

    def get_biases(self) -> Tuple[np.ndarray, np.ndarray]:
        with self._lock:
            return self.b_g.copy(), self.accel_bias.copy()

    def get_latest_imu(self) -> Tuple[np.ndarray, np.ndarray]:
        with self._lock:
            return self._latest_gyro_corr.copy(), self._latest_acc_corr.copy()

    def get_camera_orientation_R(self) -> Optional[np.ndarray]:
        if not self._est_initialized:
            return None
        yaw = float(self.get_mount_yaw_rad())
        R_IC = R_imu_to_cam_from_yaw(yaw)
        R_CI = R_IC.T
        R_WI = R_from_quat(self.q_WI)
        return R_WI @ R_CI

    def ray_world_angles(self, x_norm: float, y_norm: float) -> Optional[Tuple[float, float]]:
        """
        Convert a camera-frame ray (x_norm, y_norm, 1) to world azimuth/elevation using the
        current fused camera orientation.
        """
        R_WC = self.get_camera_orientation_R()
        if R_WC is None:
            return None

        v_c = np.array([float(x_norm), float(y_norm), 1.0], dtype=float)
        v_c /= max(1e-12, float(np.linalg.norm(v_c)))

        v_w = R_WC @ v_c
        horiz = v_w.copy()
        horiz[2] = 0.0
        hnorm = float(np.linalg.norm(horiz))
        az = 0.0 if hnorm < 1e-9 else math.atan2(horiz[1], horiz[0])
        el = math.atan2(v_w[2], hnorm)
        return wrap_pi(az), el

    def boresight_yaw_pitch_raw(self) -> Optional[Tuple[float, float]]:
        return self.ray_world_angles(0.0, 0.0)

    def boresight_yaw_pitch(self) -> Optional[Tuple[float, float]]:
        raw = self.boresight_yaw_pitch_raw()
        if raw is None:
            return None
        yaw, pitch = raw

        now = time.monotonic()
        dt = max(1e-3, now - self._bore_last_t)
        self._bore_last_t = now

        alpha = 1.0 - math.exp(-dt / max(1e-3, self.boresight_tau))

        if not self._bore_init:
            self._bore_init = True
            self._bore_yaw_s = yaw
            self._bore_pitch_s = pitch
            return yaw, pitch

        dy = wrap_pi(yaw - self._bore_yaw_s)
        self._bore_yaw_s = wrap_pi(self._bore_yaw_s + alpha * dy)
        self._bore_pitch_s = self._bore_pitch_s + alpha * (pitch - self._bore_pitch_s)
        return self._bore_yaw_s, self._bore_pitch_s

    # ---------- init from gravity ----------

    def _initialize_from_gravity(self, accel_mean: np.ndarray) -> None:
        a = accel_mean - self.accel_bias
        mag = float(np.linalg.norm(a))
        if mag < 1e-9:
            self.q_WI = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
            return

        down_I = a / mag
        up_I = -down_I

        f_I = np.array([0.0, 0.0, 1.0], dtype=float)
        f_h = f_I - np.dot(f_I, up_I) * up_I
        if float(np.linalg.norm(f_h)) < 1e-9:
            f_h = np.array([1.0, 0.0, 0.0], dtype=float)

        x_w_in_I = f_h / float(np.linalg.norm(f_h))
        z_w_in_I = up_I
        y_w_in_I = np.cross(z_w_in_I, x_w_in_I)
        y_w_in_I = y_w_in_I / max(1e-9, float(np.linalg.norm(y_w_in_I)))

        M_IW = np.column_stack((x_w_in_I, y_w_in_I, z_w_in_I))
        R_WI = M_IW.T
        self.q_WI = quat_from_R(R_WI)

    # ---------- propagate ----------

    def _propagate(self, w_meas: np.ndarray, dt: float) -> None:
        w = w_meas - self.b_g
        dR = so3_exp(w * dt)

        R_WI = R_from_quat(self.q_WI)
        R_WI = R_WI @ dR
        self.q_WI = quat_from_R(R_WI)

        F = np.eye(6, dtype=float)
        F[0:3, 0:3] -= skew(w) * dt
        F[0:3, 3:6] = -np.eye(3, dtype=float) * dt

        Q = np.zeros((6, 6), dtype=float)
        Q[0:3, 0:3] = (self.gyro_noise_std**2) * dt * np.eye(3, dtype=float)
        Q[3:6, 3:6] = (self.gyro_bias_rw_std**2) * dt * np.eye(3, dtype=float)

        self.P = F @ self.P @ F.T + Q

    # ---------- accel update ----------

    def _update_accel_dir(self, a_meas: np.ndarray) -> None:
        a = a_meas - self.accel_bias
        mag = float(np.linalg.norm(a))
        if mag < 1e-9:
            return
        if abs(mag - 9.81) > self.accel_gate_tol:
            return

        d_meas = a / mag

        R_WI = R_from_quat(self.q_WI)
        R_IW = R_WI.T
        down_w = np.array([0.0, 0.0, -1.0], dtype=float)
        d_pred = R_IW @ down_w

        y = (d_meas - d_pred).reshape(3, 1)

        H = np.zeros((3, 6), dtype=float)
        H[:, 0:3] = -skew(d_pred)

        Rm = (self.accel_dir_std**2) * np.eye(3, dtype=float)

        S = H @ self.P @ H.T + Rm
        K = self.P @ H.T @ np.linalg.inv(S)
        dx = (K @ y).reshape(-1)

        dtheta = dx[0:3]
        dbg = dx[3:6]

        R_WI = R_from_quat(self.q_WI)
        R_WI = R_WI @ so3_exp(dtheta)
        self.q_WI = quat_from_R(R_WI)

        self.b_g = self.b_g + dbg

        I = np.eye(6, dtype=float)
        self.P = (I - K @ H) @ self.P

    # ---------- VO update ----------

    def _update_vo(self, R_ck_c0: np.ndarray, quality: float) -> None:
        yaw = float(self.get_mount_yaw_rad())
        R_IC = R_imu_to_cam_from_yaw(yaw)
        R_CI = R_IC.T

        R_WI = R_from_quat(self.q_WI)
        R_WC_pred = R_WI @ R_CI

        if self._R_WC0 is None:
            self._R_WC0 = R_WC_pred.copy()

        # camera world measurement:
        R_WC_meas = self._R_WC0 @ (R_ck_c0.T)

        R_res = R_WC_pred.T @ R_WC_meas
        r = so3_log(R_res).reshape(3, 1)

        H = np.zeros((3, 6), dtype=float)
        H[:, 0:3] = R_IC

        q = max(0.05, min(1.0, float(quality)))
        sigma = math.radians(6.0) / q
        Rm = (sigma**2) * np.eye(3, dtype=float)

        S = H @ self.P @ H.T + Rm
        K = self.P @ H.T @ np.linalg.inv(S)
        dx = (K @ r).reshape(-1)

        dtheta_I = dx[0:3]
        dbg = dx[3:6]

        R_WI = R_from_quat(self.q_WI)
        R_WI = R_WI @ so3_exp(dtheta_I)
        self.q_WI = quat_from_R(R_WI)

        self.b_g = self.b_g + dbg

        I = np.eye(6, dtype=float)
        self.P = (I - K @ H) @ self.P

    def _dt(self, p: IMUPacket, last_dev_s: Optional[float], last_host_s: Optional[float], max_dt: float) -> Tuple[float, Optional[float], Optional[float]]:
        dt = None
        if p.dev_ts and p.dev_ts > 0.0:
            if last_dev_s is not None:
                dt = float(p.dev_ts - last_dev_s)
            last_dev_s = float(p.dev_ts)
        else:
            if last_host_s is not None:
                dt = float(p.ts - last_host_s)
            last_host_s = float(p.ts)

        if dt is None or dt <= 0.0 or dt > max_dt:
            dt = 0.01
        return float(dt), last_dev_s, last_host_s

    def _push_vo_fps(self, fps: float) -> None:
        self._last_vo_fps = float(fps)
        cb = self._set_vo_fps_cb
        if cb is None:
            return
        now = time.monotonic()
        if (now - self._last_vo_fps_push_t) < 0.5:
            return
        self._last_vo_fps_push_t = now
        try:
            cb(float(fps))
        except Exception:
            pass

    # ---------- thread loop ----------

    def run(self) -> None:
        calib: Optional[IMUCalibration] = None
        if self.imu_calib_seconds > 0.0:
            calib = calibrate_imu(self.imu_drain_fn, seconds=self.imu_calib_seconds)
            self.b_g = calib.gyro_bias.copy() * self.gyro_scale
            self.accel_bias = calib.accel_bias.copy() * self.accel_scale
            self._initialize_from_gravity(calib.accel_mean * self.accel_scale)
            self._est_initialized = True
            print("[IMU] gyro_bias:", self.b_g.tolist(), flush=True)
            print("[IMU] accel_bias:", self.accel_bias.tolist(), flush=True)
        else:
            print("[IMU] calibration skipped", flush=True)

        print("[IMU] started", flush=True)

        while not self._stop_event.is_set():
            pkts = self.imu_drain_fn(8192)

            for p in pkts:
                if p.kind == "accel":
                    a = np.array([p.x, p.y, p.z], dtype=float) * self.accel_scale

                    dt, self._last_acc_dev_s, self._last_acc_host_s = self._dt(
                        p, self._last_acc_dev_s, self._last_acc_host_s, max_dt=0.1
                    )

                    alpha = 1.0 - math.exp(-dt / max(1e-3, self.accel_lpf_tau))
                    if self._acc_lp is None:
                        self._acc_lp = a.copy()
                    else:
                        self._acc_lp = (1.0 - alpha) * self._acc_lp + alpha * a

                    self._last_acc_meas = a.copy()

                    if not self._est_initialized:
                        self._initialize_from_gravity(self._acc_lp)
                        self._est_initialized = True

                    self._update_accel_dir(self._acc_lp)

                    with self._lock:
                        self._latest_acc_corr = (a - self.accel_bias)

                elif p.kind == "gyro":
                    w_meas = np.array([p.x, p.y, p.z], dtype=float) * self.gyro_scale

                    dt, self._last_gyro_dev_s, self._last_gyro_host_s = self._dt(
                        p, self._last_gyro_dev_s, self._last_gyro_host_s, max_dt=0.05
                    )

                    if self._est_initialized:
                        self._propagate(w_meas, dt)

                    if self._last_acc_meas is not None:
                        acc_mag = float(np.linalg.norm(self._last_acc_meas - self.accel_bias))
                        gyro_mag = float(np.linalg.norm(w_meas - self.b_g))
                        stationary = (gyro_mag < self.stationary_gyro_thresh) and (
                            abs(acc_mag - 9.81) < self.stationary_acc_tol
                        )
                        if stationary:
                            self._stationary_time += dt
                            if self._stationary_time > 0.5:
                                beta = min(1.0, dt / max(1e-3, self.bias_tau))
                                self.b_g = (1.0 - beta) * self.b_g + beta * w_meas
                        else:
                            self._stationary_time = 0.0

                    with self._lock:
                        self._latest_gyro_corr = (w_meas - self.b_g)

            # VO fusion (non-blocking)
            if self.vo_queue is not None:
                try:
                    while True:
                        msg = self.vo_queue.get_nowait()
                        if not isinstance(msg, dict):
                            continue
                        if msg.get("type") == "vo_stats":
                            try:
                                self._push_vo_fps(float(msg.get("fps", 0.0)))
                            except Exception:
                                pass
                            continue

                        if msg.get("type") != "vo":
                            continue
                        quality = float(msg.get("quality", 0.0))
                        inliers = int(msg.get("inliers", 0))
                        if quality < self.vo_min_quality or inliers < self.vo_min_inliers:
                            continue
                        q = np.array(msg["q_ck_c0"], dtype=float).reshape(4)
                        R_ck_c0 = R_from_quat(q)
                        if self._est_initialized:
                            self._update_vo(R_ck_c0, quality)
                except Exception:
                    pass

            time.sleep(0.002)

        print("[IMU] stopped", flush=True)
