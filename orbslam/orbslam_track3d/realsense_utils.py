from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .file_io import write_text_atomic


def import_realsense():
    try:
        import pyrealsense2 as rs  # type: ignore
    except Exception as exc:
        raise RuntimeError(f"pyrealsense2 is required: {exc}") from exc
    return rs


def build_undistort_maps(rs, intr) -> tuple[bool, Optional[np.ndarray], Optional[np.ndarray], str]:
    """
    Return (enabled, map1, map2, msg).
    Maps are for cv2.remap: dst(u,v) = src(map1(u,v), map2(u,v)).
    """
    model = intr.model
    if model == rs.distortion.none:
        return False, None, None, "none"

    if model in (rs.distortion.brown_conrady, rs.distortion.modified_brown_conrady):
        K = np.array(
            [
                [float(intr.fx), 0.0, float(intr.ppx)],
                [0.0, float(intr.fy), float(intr.ppy)],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        dist = np.array(intr.coeffs[:5], dtype=np.float32)
        map1, map2 = cv2.initUndistortRectifyMap(K, dist, None, K, (int(intr.width), int(intr.height)), cv2.CV_32FC1)
        return True, map1, map2, "brown_conrady"

    if model == rs.distortion.inverse_brown_conrady:
        H, W = int(intr.height), int(intr.width)
        map1 = np.empty((H, W), dtype=np.float32)
        map2 = np.empty((H, W), dtype=np.float32)
        xs = (np.arange(W, dtype=np.float32) - float(intr.ppx)) / float(intr.fx)
        for v in range(H):
            y = (float(v) - float(intr.ppy)) / float(intr.fy)
            for u in range(W):
                pt = [float(xs[u]), float(y), 1.0]
                pix = rs.rs2_project_point_to_pixel(intr, pt)
                map1[v, u] = pix[0]
                map2[v, u] = pix[1]
        return True, map1, map2, "inverse_brown_conrady"

    return False, None, None, f"unsupported({int(model)})"


def write_orbslam3_settings_from_realsense(
    *,
    intr,
    template_path: Path,
    out_path: Path,
    camera_rgb: int,
    depth_map_factor: float,
    camera_fps: Optional[int],
    imu_frequency_hz: Optional[int],
    imu_T_b_c1: Optional[np.ndarray],
) -> Path:
    """
    Patch a template ORB-SLAM3 YAML with:
    - Camera intrinsics
    - Optional IMU frequency
    - Optional IMU.T_b_c1 (IMU/body -> camera)

    The templates in this repo already use distortion=0, so the caller should feed undistorted frames.
    Writes atomically (tmp + replace) to avoid corrupted YAML reads.
    """
    text = template_path.read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()
    out: list[str] = []

    def _fmt(v: float) -> str:
        return f"{float(v):.6f}"

    imu_block = False
    imu_data_line: Optional[str] = None
    imu_skip_multiline_data = False
    if imu_T_b_c1 is not None:
        T = np.asarray(imu_T_b_c1, dtype=np.float64).reshape(4, 4)
        flat = T.reshape(-1).tolist()
        flat_s = ",".join([f"{float(v):.6f}" for v in flat])
        imu_data_line = f"data: [{flat_s}]"

    for line in lines:
        if imu_skip_multiline_data:
            if "]" in line:
                imu_skip_multiline_data = False
            continue

        s = line.strip()
        if s.startswith("Camera1.fx:"):
            out.append(f"Camera1.fx: {_fmt(intr.fx)}")
            continue
        if s.startswith("Camera1.fy:"):
            out.append(f"Camera1.fy: {_fmt(intr.fy)}")
            continue
        if s.startswith("Camera1.cx:"):
            out.append(f"Camera1.cx: {_fmt(intr.ppx)}")
            continue
        if s.startswith("Camera1.cy:"):
            out.append(f"Camera1.cy: {_fmt(intr.ppy)}")
            continue
        if s.startswith("Camera.width:"):
            out.append(f"Camera.width: {int(intr.width)}")
            continue
        if s.startswith("Camera.height:"):
            out.append(f"Camera.height: {int(intr.height)}")
            continue
        if camera_fps is not None and s.startswith("Camera.fps:"):
            out.append(f"Camera.fps: {int(camera_fps)}")
            continue
        if s.startswith("Camera.RGB:"):
            out.append(f"Camera.RGB: {int(camera_rgb)}")
            continue
        if s.startswith("RGBD.DepthMapFactor:"):
            out.append(f"RGBD.DepthMapFactor: {float(depth_map_factor):.6f}")
            continue
        if imu_frequency_hz is not None and s.startswith("IMU.Frequency:"):
            out.append(f"IMU.Frequency: {float(int(imu_frequency_hz)):.1f}")
            continue
        if s.startswith("IMU.T_b_c1:"):
            imu_block = True
            out.append(line)
            continue
        if imu_block and s.startswith("data:"):
            if imu_data_line is not None:
                prefix = line[: len(line) - len(line.lstrip())]
                out.append(prefix + imu_data_line)
            else:
                out.append(line)
            if "]" not in line:
                imu_skip_multiline_data = True
            imu_block = False
            continue
        out.append(line)

    write_text_atomic(Path(out_path), "\n".join(out) + "\n", encoding="utf-8")
    return Path(out_path)


def compute_T_b_c1_from_realsense(rs, profile, video_stream) -> Optional[np.ndarray]:
    """
    RealSense gives extrinsics from video stream -> gyro stream (cam -> imu).
    ORB-SLAM3 expects IMU.T_b_c1 = (IMU/body -> camera), so we invert.
    """
    try:
        gyro_sp = profile.get_stream(rs.stream.gyro).as_motion_stream_profile()
    except Exception:
        return None
    try:
        ext = video_stream.get_extrinsics_to(gyro_sp)  # cam(video) -> gyro(IMU)
        R = np.array(ext.rotation, dtype=np.float64).reshape(3, 3)
        t = np.array(ext.translation, dtype=np.float64).reshape(3)
        T_cam_gyro = np.eye(4, dtype=np.float64)
        T_cam_gyro[:3, :3] = R
        T_cam_gyro[:3, 3] = t
        return np.linalg.inv(T_cam_gyro)
    except Exception:
        return None


def configure_realsense_depth_sensor(rs, depth_sensor) -> tuple[str, str]:
    """
    Configure RealSense depth sensor options for stable SLAM capture.

    - Set visual preset to High Accuracy (when supported).
    - Turn off IR emitter / illuminator (when supported).

    Returns (preset_msg, emitter_msg) for logging.
    """

    preset_msg = "n/a"
    emitter_msg = "n/a"

    # 1) Preset (can override other depth options).
    try:
        if depth_sensor.supports(rs.option.visual_preset):
            wanted_val: Optional[float] = None
            try:
                r = depth_sensor.get_option_range(rs.option.visual_preset)
                v_min = int(round(float(getattr(r, "min", 0.0))))
                v_max = int(round(float(getattr(r, "max", 0.0))))
                for v in range(v_min, v_max + 1):
                    try:
                        desc = str(depth_sensor.get_option_value_description(rs.option.visual_preset, v) or "")
                    except Exception:
                        continue
                    if "high accuracy" in desc.lower():
                        wanted_val = float(v)
                        break
            except Exception:
                wanted_val = None

            if wanted_val is None:
                # Common for D4xx: 3 == High Accuracy.
                wanted_val = 3.0

            depth_sensor.set_option(rs.option.visual_preset, float(wanted_val))

            try:
                cur = int(round(float(depth_sensor.get_option(rs.option.visual_preset))))
                desc = str(depth_sensor.get_option_value_description(rs.option.visual_preset, cur) or "")
                preset_msg = f"{desc}({cur})" if desc else str(cur)
            except Exception:
                preset_msg = f"set({wanted_val:g})"
    except Exception as exc:
        preset_msg = f"err({exc})"

    # 2) IR emitter / illuminator OFF (do after preset).
    try:
        if hasattr(rs.option, "emitter_enabled") and depth_sensor.supports(rs.option.emitter_enabled):
            depth_sensor.set_option(rs.option.emitter_enabled, 0.0)
            emitter_msg = "off"
        elif hasattr(rs.option, "laser_power") and depth_sensor.supports(rs.option.laser_power):
            depth_sensor.set_option(rs.option.laser_power, 0.0)
            emitter_msg = "laser_power=0"
    except Exception as exc:
        emitter_msg = f"err({exc})"

    return preset_msg, emitter_msg

