from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np


def tracking_state_name(state: int) -> str:
    return {
        -1: "SYSTEM_NOT_READY",
        0: "NO_IMAGES_YET",
        1: "NOT_INITIALIZED",
        2: "OK",
        3: "RECENTLY_LOST",
        4: "LOST",
        5: "OK_KLT",
    }.get(int(state), f"STATE_{int(state)}")


ButtonSpec = Tuple[str, str, Tuple[int, int, int, int]]


def layout_mode_buttons(
    *,
    size: int,
    labels: Sequence[Tuple[str, str]],
    margin: int = 10,
    button_w: int = 170,
    button_h: int = 26,
    gap: int = 6,
) -> list[ButtonSpec]:
    size = int(max(200, int(size)))
    x2 = int(size - margin)
    x1 = int(max(margin, x2 - button_w))
    y = int(margin)
    buttons: list[ButtonSpec] = []
    for label, display_label in labels:
        y1 = int(y)
        y2 = int(y + button_h)
        buttons.append((str(label), str(display_label), (x1, y1, x2, y2)))
        y = int(y2 + gap)
    return buttons


def layout_bottom_right_buttons(
    *,
    size: int,
    labels: Sequence[Tuple[str, str]],
    margin: int = 10,
    button_w: int = 170,
    button_h: int = 26,
    gap: int = 6,
) -> list[ButtonSpec]:
    size = int(max(200, int(size)))
    x2 = int(size - margin)
    x1 = int(max(margin, x2 - button_w))
    y = int(size - margin)
    buttons: list[ButtonSpec] = []
    for label, display_label in labels:
        y2 = int(y)
        y1 = int(y - button_h)
        buttons.append((str(label), str(display_label), (x1, y1, x2, y2)))
        y = int(y1 - gap)
    return buttons


def _draw_mode_buttons(img: np.ndarray, buttons: Sequence[ButtonSpec], active_label: Optional[str]) -> None:
    if not buttons:
        return
    font = cv2.FONT_HERSHEY_SIMPLEX
    for label, display_label, (x1, y1, x2, y2) in buttons:
        is_active = bool(active_label == label)
        fill = (40, 120, 40) if is_active else (30, 30, 30)
        border = (0, 255, 0) if is_active else (160, 160, 160)
        text_color = (255, 255, 255) if is_active else (220, 220, 220)
        cv2.rectangle(img, (x1, y1), (x2, y2), fill, -1)
        cv2.rectangle(img, (x1, y1), (x2, y2), border, 1)
        (tw, th), baseline = cv2.getTextSize(display_label, font, 0.5, 1)
        tx = int(x1 + ((x2 - x1) - tw) * 0.5)
        ty = int(y1 + ((y2 - y1) + th) * 0.5 - baseline)
        cv2.putText(img, display_label, (tx, ty), font, 0.5, text_color, 1, cv2.LINE_AA)


def render_top_view(
    *,
    traj_xyz: np.ndarray,
    map_xyz: np.ndarray,
    Twc: Optional[np.ndarray],
    tracking_state: int,
    prev_tracking_state: Optional[int] = None,
    camera_fps: Optional[float] = None,
    slam_fps: Optional[float] = None,
    odom_delta_xyz: Optional[Tuple[float, float, float]] = None,
    odom_rpy_deg: Optional[Tuple[float, float, float]] = None,
    odom_frame: Optional[str] = None,
    px4_odom_xyz: Optional[Tuple[float, float, float]] = None,
    px4_odom_vel_xyz: Optional[Tuple[float, float, float]] = None,
    px4_odom_rpy_deg: Optional[Tuple[float, float, float]] = None,
    px4_odom_age_s: Optional[float] = None,
    px4_odom_status: Optional[str] = None,
    map_points_enabled: bool = False,
    map_points_count: Optional[int] = None,
    map_points_hz: Optional[float] = None,
    size: int = 700,
    max_draw_points: int = 25000,
    buttons: Optional[Sequence[ButtonSpec]] = None,
    active_button: Optional[str] = None,
    map_buttons: Optional[Sequence[ButtonSpec]] = None,
    map_active_button: Optional[str] = None,
    status_text: Optional[str] = None,
) -> np.ndarray:
    """
    Render a simple X/Z top-down map:
    - map points: gray
    - trajectory: red line
    - current pose: green dot + heading + FOV rays
    - optional mode buttons: top-right overlay
    """
    size = int(max(200, int(size)))
    img = np.zeros((size, size, 3), dtype=np.uint8)

    def _filter_finite_xyz(points: np.ndarray) -> np.ndarray:
        if points.size == 0:
            return points
        mask = np.isfinite(points).all(axis=1)
        if not np.any(mask):
            return np.empty((0, 3), dtype=points.dtype)
        if not np.all(mask):
            return points[mask]
        return points

    traj_xyz = np.asarray(traj_xyz, dtype=np.float32).reshape(-1, 3) if traj_xyz is not None else np.empty((0, 3))
    map_xyz = np.asarray(map_xyz, dtype=np.float32).reshape(-1, 3) if map_xyz is not None else np.empty((0, 3))
    traj_xyz = _filter_finite_xyz(traj_xyz)
    map_xyz = _filter_finite_xyz(map_xyz)

    Twc_mat: Optional[np.ndarray] = None
    if Twc is not None:
        try:
            cand = np.asarray(Twc, dtype=np.float64).reshape(4, 4)
            if np.all(np.isfinite(cand)):
                Twc_mat = cand
        except Exception:
            Twc_mat = None

    xs = []
    zs = []
    if map_xyz.size > 0:
        xs.append(map_xyz[:, 0])
        zs.append(map_xyz[:, 2])
    if traj_xyz.size > 0:
        xs.append(traj_xyz[:, 0])
        zs.append(traj_xyz[:, 2])
    if Twc_mat is not None:
        xs.append(np.array([float(Twc_mat[0, 3])], dtype=np.float32))
        zs.append(np.array([float(Twc_mat[2, 3])], dtype=np.float32))

    def _draw_status(img: np.ndarray, text: str, y: int) -> None:
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.7
        thickness = 2
        (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
        x = 15
        y = int(y)
        pad = 6
        x1 = int(x - pad)
        y1 = int(y - th - pad)
        x2 = int(x + tw + pad)
        y2 = int(y + baseline + pad)
        cv2.rectangle(img, (x1, y1), (x2, y2), (15, 15, 15), -1)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 180, 255), 1)
        cv2.putText(img, text, (x, y), font, scale, (0, 220, 255), thickness, cv2.LINE_AA)

    if not xs or not zs:
        cv2.putText(img, "Waiting for SLAM...", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2)
        if status_text:
            _draw_status(img, status_text, 56)
        return img

    x_all = np.concatenate(xs).astype(np.float64, copy=False)
    z_all = np.concatenate(zs).astype(np.float64, copy=False)
    x_all = x_all[np.isfinite(x_all)]
    z_all = z_all[np.isfinite(z_all)]
    if x_all.size == 0 or z_all.size == 0:
        cv2.putText(img, "Waiting for SLAM...", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2)
        if status_text:
            _draw_status(img, status_text, 56)
        return img

    min_x = float(np.min(x_all))
    max_x = float(np.max(x_all))
    min_z = float(np.min(z_all))
    max_z = float(np.max(z_all))
    span_x = max(0.1, max_x - min_x)
    span_z = max(0.1, max_z - min_z)
    span = float(max(span_x, span_z))

    cx = 0.5 * (min_x + max_x)
    cz = 0.5 * (min_z + max_z)
    scale = 0.45 * float(size) / span

    def w2p(x: float, z: float) -> tuple[int, int]:
        px = int(round(0.5 * float(size) + (float(x) - cx) * scale))
        py = int(round(0.5 * float(size) - (float(z) - cz) * scale))
        return int(px), int(py)

    map_delta_text = None
    if traj_xyz.size > 0:
        start = traj_xyz[0].astype(np.float64, copy=False)
        end = traj_xyz[-1].astype(np.float64, copy=False)
        if Twc_mat is not None:
            end = Twc_mat[:3, 3]
        if np.isfinite(start).all() and np.isfinite(end).all():
            dx = float(end[0] - start[0])
            dy = float(end[1] - start[1])
            dz = float(end[2] - start[2])
            map_delta_text = f"dX={dx:+.2f}m dY={dy:+.2f}m dZ={dz:+.2f}m"
    elif Twc_mat is not None:
        start = Twc_mat[:3, 3]
        if np.isfinite(start).all():
            map_delta_text = "dX=+0.00m dY=+0.00m dZ=+0.00m"

    odom_label = "Odom"
    if odom_frame:
        odom_label = f"Odom({odom_frame})"
    odom_delta_text = None
    if odom_delta_xyz is not None:
        try:
            dx, dy, dz = odom_delta_xyz
            if all(math.isfinite(float(v)) for v in (dx, dy, dz)):
                odom_delta_text = f"{odom_label} dX={float(dx):+.2f}m dY={float(dy):+.2f}m dZ={float(dz):+.2f}m"
        except Exception:
            pass
    map_line = f"Map {map_delta_text}" if map_delta_text else None
    rpy_line = None
    if odom_rpy_deg is not None:
        try:
            roll, pitch, yaw = odom_rpy_deg
            if all(math.isfinite(float(v)) for v in (roll, pitch, yaw)):
                rpy_line = f"{odom_label} RPY={float(roll):+.1f},{float(pitch):+.1f},{float(yaw):+.1f}deg"
        except Exception:
            pass
    px4_lines = []
    if px4_odom_xyz is not None:
        try:
            x, y, z = px4_odom_xyz
            if all(math.isfinite(float(v)) for v in (x, y, z)):
                line = f"PX4 ODOM NED x={float(x):+.2f} y={float(y):+.2f} z={float(z):+.2f} m"
                if px4_odom_age_s is not None and math.isfinite(float(px4_odom_age_s)):
                    line = f"{line} age={float(px4_odom_age_s):.2f}s"
                px4_lines.append(line)
        except Exception:
            pass
    if px4_odom_vel_xyz is not None:
        try:
            vx, vy, vz = px4_odom_vel_xyz
            if all(math.isfinite(float(v)) for v in (vx, vy, vz)):
                px4_lines.append(f"PX4 VEL vx={float(vx):+.2f} vy={float(vy):+.2f} vz={float(vz):+.2f} m/s")
        except Exception:
            pass
    if px4_odom_rpy_deg is not None:
        try:
            roll, pitch, yaw = px4_odom_rpy_deg
            if all(math.isfinite(float(v)) for v in (roll, pitch, yaw)):
                px4_lines.append(f"PX4 RPY={float(roll):+.1f},{float(pitch):+.1f},{float(yaw):+.1f}deg")
        except Exception:
            pass
    if not px4_lines and px4_odom_status:
        px4_lines.append(str(px4_odom_status))

    # Draw map points (optionally downsample for speed).
    pts = map_xyz
    if int(pts.shape[0]) > int(max_draw_points):
        idx = np.random.choice(int(pts.shape[0]), int(max_draw_points), replace=False)
        pts = pts[idx]
    if pts.size > 0:
        px = np.rint(0.5 * float(size) + (pts[:, 0] - cx) * scale).astype(np.int32)
        py = np.rint(0.5 * float(size) - (pts[:, 2] - cz) * scale).astype(np.int32)
        mask = (px >= 0) & (px < size) & (py >= 0) & (py < size)
        if np.any(mask):
            img[py[mask], px[mask]] = (120, 120, 120)

    # Draw trajectory polyline.
    if traj_xyz.shape[0] >= 2:
        poly = np.array([w2p(float(p[0]), float(p[2])) for p in traj_xyz], dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(img, [poly], False, (0, 0, 255), 2, cv2.LINE_AA)

    # Draw current pose (camera).
    if Twc_mat is not None:
        try:
            x = float(Twc_mat[0, 3])
            z = float(Twc_mat[2, 3])
            px, py = w2p(x, z)
            cv2.circle(img, (px, py), 5, (0, 255, 0), -1, cv2.LINE_AA)

            R = Twc_mat[:3, :3]
            fwd = R @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
            yaw = math.atan2(float(fwd[0]), float(fwd[2]))
            fov = math.radians(70.0)
            for sign in (-1.0, 1.0):
                a = yaw + sign * 0.5 * fov
                dx = math.sin(a)
                dz = math.cos(a)
                px2, py2 = w2p(x + 1.0 * dx, z + 1.0 * dz)
                cv2.line(img, (px, py), (px2, py2), (0, 180, 0), 1, cv2.LINE_AA)

            # Heading arrow.
            pxh, pyh = w2p(x + 0.7 * math.sin(yaw), z + 0.7 * math.cos(yaw))
            cv2.arrowedLine(img, (px, py), (pxh, pyh), (0, 255, 0), 2, cv2.LINE_AA, tipLength=0.2)
        except Exception:
            pass

    # HUD text
    st = tracking_state_name(int(tracking_state))
    hud_lines = [line for line in (map_line, odom_delta_text, rpy_line) if line]
    if px4_lines:
        hud_lines.extend(px4_lines)
    hud_y = 28
    for line in hud_lines:
        cv2.putText(img, line, (15, int(hud_y)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 2)
        hud_y += 22
    if status_text:
        status_y = int(28 + (len(hud_lines) * 22) + 18)
        _draw_status(img, status_text, status_y)
    if buttons is not None:
        _draw_mode_buttons(img, buttons, active_button)
    if map_buttons is not None:
        _draw_mode_buttons(img, map_buttons, map_active_button)

    line_h = 22
    hud_y = int(size - 18)
    cam_text = "n/a"
    if camera_fps is not None and math.isfinite(float(camera_fps)):
        cam_text = f"{float(camera_fps):.1f}"
    slam_text = "n/a"
    if slam_fps is not None and math.isfinite(float(slam_fps)):
        slam_text = f"{float(slam_fps):.1f}"
    fps_line = f"Cam FPS: {cam_text} | SLAM FPS: {slam_text}"
    cv2.putText(img, fps_line, (15, hud_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 2)
    hud_y -= line_h

    mp_status = "ON" if map_points_enabled else "OFF"
    mp_line = f"Map points: {mp_status}"
    if map_points_enabled:
        count = int(map_points_count) if map_points_count is not None else int(map_xyz.shape[0])
        mp_line = f"{mp_line} ({count})"
        if map_points_hz is not None and math.isfinite(float(map_points_hz)):
            mp_line = f"{mp_line} @ {float(map_points_hz):.1f} Hz"
    cv2.putText(img, mp_line, (15, hud_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 2)
    hud_y -= line_h

    state_line = f"ORB-SLAM3: {st}"
    if prev_tracking_state is not None:
        prev = tracking_state_name(int(prev_tracking_state))
        state_line = f"{state_line} (prev {prev})"
    cv2.putText(img, state_line, (15, hud_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 2)
    return img
