"""Configuration helpers and defaults (YAML + CLI overrides)."""

from __future__ import annotations

import os
from typing import Any, Optional

import cv2

try:
    import yaml  # type: ignore

    _HAVE_YAML = True
except Exception:
    yaml = None  # type: ignore
    _HAVE_YAML = False

# === SECTION:CONFIG ===


def _set_cv_threads_1() -> None:
    _set_cv_threads(cfg=None, role="(legacy)", n_threads=1)


def _set_cv_threads(*, cfg: Optional[dict], role: str, n_threads: Optional[int] = None) -> None:
    opencv_cfg = {}
    try:
        opencv_cfg = dict((cfg or {}).get("opencv") or {})
    except Exception:
        opencv_cfg = {}

    if n_threads is None:
        n_threads = opencv_cfg.get("num_threads", 1)
        role_threads = opencv_cfg.get("role_threads", None)
        if isinstance(role_threads, dict):
            if role in role_threads:
                n_threads = role_threads.get(role)
            elif "default" in role_threads:
                n_threads = role_threads.get("default")

    n = 1
    try:
        if isinstance(n_threads, str):
            s = str(n_threads).strip().lower()
            if s in ("auto", "-1"):
                n_threads = int(os.cpu_count() or 1)
            elif s in ("default", "opencv_default", "0"):
                n_threads = 0
            else:
                n_threads = int(float(s))
        n = int(n_threads)
    except Exception:
        n = 1

    use_optimized = True
    try:
        use_optimized = bool(opencv_cfg.get("use_optimized", True))
    except Exception:
        use_optimized = True
    disable_opencl = True
    try:
        disable_opencl = bool(opencv_cfg.get("disable_opencl", True))
    except Exception:
        disable_opencl = True

    try:
        # n==0 means "OpenCV default" in OpenCV (restores default threads).
        cv2.setNumThreads(int(n))
    except Exception:
        pass
    try:
        cv2.setUseOptimized(bool(use_optimized))
    except Exception:
        pass
    if bool(disable_opencl):
        # Avoid OpenCL overhead / surprises on small devices.
        try:
            cv2.ocl.setUseOpenCL(False)
        except Exception:
            pass

    # Avoid polluting stdout in headless-json integrations; only log if log redirection is enabled.
    try:
        log_enabled = bool(((cfg or {}).get("_log_spec") or {}).get("enabled", False))
    except Exception:
        log_enabled = False
    if bool(log_enabled):
        try:
            try:
                gn = int(cv2.getNumThreads())
            except Exception:
                gn = -1
            try:
                opt = bool(cv2.useOptimized())
            except Exception:
                opt = bool(use_optimized)
            try:
                ocl = int(bool(cv2.ocl.useOpenCL()))
            except Exception:
                ocl = 0
            print(
                f"[OpenCV] role={str(role)} threads_req={int(n)} threads_get={int(gn)} opt={int(bool(opt))} opencl={int(ocl)}",
                flush=True,
            )
        except Exception:
            pass


def _deep_update(dst: dict, src: dict) -> dict:
    for k, v in (src or {}).items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_update(dst[k], v)
        else:
            dst[k] = v
    return dst


def _load_yaml_config(path: Optional[str]) -> dict:
    if not path:
        return {}
    try:
        if not os.path.exists(str(path)):
            return {}
    except Exception:
        return {}
    if not _HAVE_YAML:
        raise RuntimeError("PyYAML not available; install `pyyaml` or pass config via CLI.")
    try:
        with open(str(path), "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return dict(data or {})
    except Exception as exc:
        raise RuntimeError(f"Failed to parse YAML config: {path} ({type(exc).__name__}: {exc})") from exc


def _default_config() -> dict:
    return {
        "opencv": {
            # OpenCV is internally multithreaded. Because this app is multiprocess, be conservative by default.
            # Set num_threads=0 to restore OpenCV's default (often equals CPU core count).
            "num_threads": 1,
            # Per-process role overrides (optional). Keep empty by default to avoid oversubscribing CPU cores.
            "role_threads": {},
            "use_optimized": True,
            "disable_opencl": True,
        },
        "app": {
            "headless": False,
            # If true: run a small Numba warmup at startup to pre-compile cached @njit kernels.
            # This avoids a first-interaction stall when the operator first hovers/clicks for hole detection.
            "jit_warmup": {
                "enabled": True,
            },
            # Headless JSON I/O (integration shim).
            # When enabled, the app can accept commands and emit telemetry as JSON objects.
            # The initial transport is newline-delimited JSON over stdin/stdout ("stdio_jsonl").
            "headless_io": {
                "enabled": False,
                "transport": "stdio_jsonl",  # stdio_jsonl|none
                "stdin_enabled": True,
                "stdout_enabled": True,
                # Safety: max number of input JSON messages applied per tick.
                "max_in_per_tick": 50,
                # Output rates (Hz). 0 disables that stream.
                "telemetry_rate_hz": 10.0,
                "features_rate_hz": 5.0,
                # If true, emit compact feature payloads (uv+group only). This reduces CPU and bandwidth.
                "features_compact": True,
                # Caps to keep messages bounded.
                "max_tracks": 250,
            },
            "display": "color",  # color|gray
            "duration_s": 0.0,
            "drop_frames": True,
            "max_gap_frames": 8,
            "log_rate_hz": 1.0,
            "path_maxlen": 0,  # 0 = unlimited
            "debug_log": {
                "enabled": False,
                "latest_path": "3d_track_latest.log",
                "run_dir": "logs",
                "mirror_to_stdout": True,
            },
            "window_name": "3D Track",
        },
        "capture": {
            "source": "realsense",  # realsense|opencv
            "camera_index": 0,
            "fps": 60,
            "depth": True,
            "shm_slots": "auto",
            "out_width": 640,
            "out_height": 480,
            "undistort": True,
            "rs_global_time_enabled": "auto",  # auto|true|false
            "rs_depth_preset": "high_accuracy",  # off|default|high_accuracy|high_density|medium_density|hand|remove_ir_pattern|custom
            # IR projector / emitter (depth sensor). Keep disabled by default for VO (avoids projected pattern).
            "rs_emitter_enabled": False,  # auto|true|false
            "rs_color_format": "bgr8",  # bgr8|rgb8|yuyv|auto
            # Prefer the left IR imager by default (more stable features than RGB in many indoor scenes).
            "rs_color_stream": "infra1",  # color|infra1|infra2
            "rs_depth_filter": "off",  # off|spatial|temporal|both (optional)
            "enable_imu": True,
            "imu_slots": 8192,
            "clahe": True,
            "clahe_clip": 2.0,
            "clahe_grid": 8,
        },
        "features": {
            "max_points": 250,
            "min_track_points": 80,
            "grid_cell_px": 32,
            "features_per_cell": 2,
            "min_distance_px": 10,
            "subpix_win": 5,
            "detect_interval": 10,
        },
        "lk": {
            "win": 31,
            "levels": 4,
            "min_eig": 1e-4,
        },
        "depth": {
            "sample_r": 1,
            "edge_r": 2,
            "edge_mad_m": 0.05,
            "edge_min_valid": 6,
            "min_depth_m": 0.25,
            "viz_max_m": 6.0,
        },
        "imu": {
            "use_gyro_prior": True,
            "use_rs_extrinsics": True,
            "rs_extrinsics_transpose": False,
            # If true: estimate a constant offset between frame timestamps and IMU timestamps and apply it when integrating gyro / reading accel windows.
            "auto_time_offset": True,
            # Optional fixed time offset (seconds). Convention matches odom:
            #   t_imu ~= t_frame - time_offset_s
            # If set, it seeds the offset; set auto_time_offset=false to keep it fixed.
            "time_offset_s": None,
            # Warn (and optionally recompute) if |t_frame - offset - t_imu_last| exceeds this many seconds.
            "time_offset_warn_s": 0.5,
            # If true: recompute offset on large mismatch (useful across device reconnects/timebase changes).
            "time_offset_recompute": True,
            "gyro_invert": True,
            "gyro_sign_x": 1.0,
            "gyro_sign_y": 1.0,
            "gyro_sign_z": 1.0,
            "accel_invert": False,
            "accel_sign_x": 1.0,
            "accel_sign_y": 1.0,
            "accel_sign_z": 1.0,
            "gyro_bias": {
                "enabled": True,
                "duration_s": 2.0,
                "min_samples": 200,
                "max_std_rad_s": 0.05,
                "max_wait_s": 6.0,
                "block_vo_until_ok": True,
            },
            "accel_bias": {
                "enabled": True,
                "duration_s": 2.0,
                "min_samples": 200,
                "max_std_m_s2": 0.20,
                "max_wait_s": 6.0,
                "block_vo_until_ok": True,
            },
            "gravity": {
                "enabled": True,
                "window_s": 0.25,
                "g_m_s2": 9.80665,
                "max_dev_m_s2": 1.5,
                "min_samples": 10,
            },
        },
        "odometry": {
            "enabled": True,
            "method": "gyro_prior",  # gyro_prior|gyro_translation|kabsch
            "min_inliers": 12,
            "min_dt_s": 0.001,
            "ransac": {"iters": 80, "inlier_thresh_m": 0.03, "input_bins": "auto"},
            "local_pose_graph": {
                "enabled": False,
                "window_s": 0.5,
                "max_nodes": 20,
                "min_nodes": 5,
                "solve_every_n": 1,
                # Base weight applied to each accepted translation step; scaled by the quality model below.
                "meas_weight": 1.0,
                # Graph regularization weights.
                "anchor_w": 100.0,
                "raw_w": 0.05,
                "smooth_w": 0.25,
                # Robust reweighting on graph edges (Huber).
                "robust_delta_m": 0.10,
                "robust_iters": 2,
                "damping": 1e-6,
                "quality": {
                    # Reference inlier count where q_inliers saturates to 1.0.
                    "inliers_ref": 30,
                    # RMSE at which q_rmse decays to 0.0 (often set near accept_gate.max_rmse_m).
                    "rmse_ref_m": 0.05,
                    # Soft RMSE threshold below which q_rmse is 1.0.
                    "rmse_soft_m": 0.025,
                    # Prefer edges whose inlier ratio is well above accept_gate.min_inlier_ratio.
                    "ratio_soft": 0.60,
                    # Optional speed soft penalty (m/s). If speed_ref_mps==0, speed is ignored.
                    "speed_ref_mps": 0.0,
                    "speed_soft_mps": 0.0,
                    # Downweight gap-filled edges by 1/(gap_frames^gap_power).
                    "gap_power": 0.5,
                    # Spatial coverage score (cheap conditioning proxy) based on inlier spread over an image grid.
                    "coverage_grid": [3, 3],
                    "coverage_min": 0.25,
                    "coverage_soft": 0.60,
                    # Minimum overall quality required to add an edge.
                    "min_quality": 0.15,
                },
            },
            "accept_gate": {
                "min_inlier_ratio": 0.25,
                "max_rmse_m": 0.05,
                "max_step_m": 0.0,
                "max_step_rot_deg": 0.0,
                "max_step_speed_mps": 0.0,
            },
            "prefilter_mad3d": {"enabled": True, "mad_k": 3.0, "min_ref": 6, "thr_min_m": 0.03, "thr_max_m": 0.30},
            "priors": {
                "enabled": True,
                "gyro_sigma_deg": 0.8,
                "gravity_sigma_deg": 3.0,
            },
        },
        "jitter": {
            "enabled": True,
            "window_s": 1.5,
            "expire_s": 3.0,
            "max_samples": 60,
            "min_samples": 3,
            "promote_frames": 3,
            "hold_frames": 5,
            "score_ema_alpha": 0.4,
        },
        "zupt": {"enabled": False, "hold_s": 0.15, "trans_thresh_m": 0.008, "rot_thresh_deg": 0.5},
    }


