#!/usr/bin/env python3
# tracker_types.py - shared datatypes and helpers for VO/IMU/tracker.

from typing import NamedTuple, Tuple, Optional
import numpy as np


class FramePacket(NamedTuple):
    """
    Minimal frame container used by VO/object tracker/IMU calibrator.
    """
    bgr: np.ndarray
    gray: np.ndarray
    ts: float     # seconds (host time)
    W: int
    H: int


class Det(NamedTuple):
    """Detection tuple used by the YOLO thread."""
    bbox: Tuple[int, int, int, int]
    conf: float
    cls: int
    label: str


def quad_from_bbox(x: float, y: float, w: float, h: float) -> np.ndarray:
    """Create a 4x2 quad from an axis-aligned bbox."""
    q = np.array(
        [
            [x,     y    ],
            [x + w, y    ],
            [x + w, y + h],
            [x,     y + h],
        ],
        dtype=np.float32,
    )
    return q


def bbox_from_quad(q: np.ndarray) -> Tuple[float, float, float, float]:
    """Axis-aligned bbox that encloses the quad."""
    q = np.asarray(q)
    xs = q[:, 0]
    ys = q[:, 1]
    x = float(xs.min())
    y = float(ys.min())
    w = float(xs.max() - x)
    h = float(ys.max() - y)
    return x, y, w, h


def clamp_rect_f(
    x: float, y: float, w: float, h: float,
    W: int, H: int
) -> Tuple[float, float, float, float]:
    """Clamp a floating rectangle to image bounds."""
    x = max(0.0, min(x, W - 1.0))
    y = max(0.0, min(y, H - 1.0))
    w = max(1.0, min(w, W - x))
    h = max(1.0, min(h, H - y))
    return x, y, w, h


def translate_quad(q: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """Translate a quad by (dx, dy)."""
    return q + np.array([dx, dy], dtype=np.float32)


def apply_affine_local_to_quad(
    M: np.ndarray,
    q: np.ndarray,
    origin_xy: Tuple[float, float],
) -> np.ndarray:
    """
    Apply 2x3 affine M to quad q, with origin_xy as local origin
    (so M's translation is relative to that origin).
    """
    ox, oy = origin_xy
    q_local = q - np.array([ox, oy], dtype=np.float32)
    qh = np.c_[q_local, np.ones((q_local.shape[0], 1), dtype=np.float32)]
    qq = (qh @ M.T)[:, :2]
    return qq + np.array([ox, oy], dtype=np.float32)


def is_valid_quad(q: Optional[np.ndarray], W: int, H: int) -> bool:
    """Basic sanity check that the quad is finite and not totally outside."""
    if q is None:
        return False
    q = np.asarray(q)
    if q.shape != (4, 2):
        return False
    if not np.all(np.isfinite(q)):
        return False
    xs = q[:, 0]
    ys = q[:, 1]
    if xs.min() >= W or ys.min() >= H or xs.max() < 0 or ys.max() < 0:
        return False
    return True


def delta_from_H_at_ref(H: Optional[np.ndarray],
                        ref_xy: Tuple[float, float]) -> Tuple[float, float]:
    """
    Given homography H and a reference point (x,y), return pixel shift
    (dx, dy) = warped_point - original_point.
    """
    if H is None:
        return 0.0, 0.0
    x, y = float(ref_xy[0]), float(ref_xy[1])
    p = np.array([x, y, 1.0], dtype=np.float64)
    q = H @ p
    if abs(q[2]) < 1e-9 or not np.isfinite(q).all():
        return 0.0, 0.0
    x2 = float(q[0] / q[2])
    y2 = float(q[1] / q[2])
    return x2 - x, y2 - y


def make_K(
    W: int,
    H: int,
    fx_px: Optional[float] = None,
    fy_px: Optional[float] = None,
    cx: Optional[float] = None,
    cy: Optional[float] = None,
) -> np.ndarray:
    """
    Build a simple pinhole intrinsic matrix K. If fx/fy/cx/cy are not
    provided, sensible defaults based on image size are used.
    """
    fx = float(fx_px) if fx_px is not None else float(W) * 0.9
    fy = float(fy_px) if fy_px is not None else float(H) * 0.9
    cx = float(cx) if cx is not None else float(W) * 0.5
    cy = float(cy) if cy is not None else float(H) * 0.5
    K = np.array(
        [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return K
