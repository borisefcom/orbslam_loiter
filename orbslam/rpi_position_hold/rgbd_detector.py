#!/usr/bin/env python3
"""
rgbd_detector.py

Headless RGB-D "window" detectors for:
  - Planes: mouse-seeded plane flood fill + depth-derived normals.
  - Holes:  mouse-seeded free-space flood fill + max-inscribed-circle radius gate.

This module is designed for headless use (e.g. control loops) but also includes an
optional interactive RealSense demo when run as a script.

Headless usage
--------------
The detectors operate on a *window* (crop) around a seed pixel (x,y) in the full frame.

Inputs:
  - `rgbd`: `RgbdWindow(...)` or a dict with:
      - `bgr`        : (H,W,3) uint8 BGR image (aligned to depth)
      - `depth`      : (H,W) depth image, either uint16 Z16 (0=invalid) or float32 meters (<=0 invalid)
      - `depth_units`: meters per unit (used only if depth is uint16)
      - `intr`       : camera intrinsics (`RgbdIntrinsics`-like object or dict with fx/fy/ppx/ppy)
  - `window_wh`: (win_w, win_h) crop size
  - `x, y`: seed pixel in the *full* image (cursor coordinates when testing)

Return:
  - uint8 mask (win_h, win_w) with values {0,255} in window coordinates.

Example:

    from rpi_position_hold.rgbd_detector import PlaneDetector, HoleDetector, RgbdWindow

    rgbd = RgbdWindow(bgr=bgr, depth=depth_raw_u16, depth_units=0.001, intr=intr_dict)
    plane = PlaneDetector()
    hole = HoleDetector()

    plane_mask = plane.detect(rgbd, window_wh=(320, 240), x=320, y=240)
    hole_mask  = hole.detect(rgbd, window_wh=(320, 240), x=320, y=240)

Interactive demo (requires pyrealsense2)
---------------------------------------
    python -m rpi_position_hold.rgbd_detector
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union

import numpy as np
import cv2

try:
    from numba import njit  # type: ignore

    _HAVE_NUMBA = True
except Exception:
    njit = None  # type: ignore
    _HAVE_NUMBA = False


# ---------------------------------------------------------------------------
# Data model (lightweight; accepts dicts / "RgbdIntrinsics-like" objects too)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RgbdIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    w: int
    h: int


@dataclass(frozen=True)
class RgbdWindow:
    bgr: np.ndarray
    depth: np.ndarray
    depth_units: float = 0.001  # meters per unit if depth is uint16
    intr: Optional[Any] = None  # RgbdIntrinsics-like or dict with fx/fy/ppx/ppy


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SpatialFilterSettings:
    """
    "Strong" edge-preserving smoothing applied to the window depth before detection.

    This is a lightweight approximation of the RealSense spatial filter behavior
    operating directly on the depth-in-meters array.
    """

    iterations: int = 4
    radius_px: int = 1
    delta_m: float = 0.02
    alpha: float = 0.80


@dataclass(frozen=True)
class PlaneSettings:
    max_normal_angle_deg: float = 30.0
    neighbor_delta_along_normal_mm: float = 100.0
    normal_grad_step_px: int = 15
    min_normal_mag: float = 0.45


@dataclass(frozen=True)
class HoleSettings:
    # Max depth the detector searches up to (meters). Keep this at/above your expected indoor range.
    # RealSense depth is typically usable up to ~15m in indoor scenes (device/model dependent).
    search_max_depth_m: float = 15.0
    # Optional gate used by the "column" (free-space) detector when searching for a deepest passing depth.
    # Keeping this >0 makes the depth search well-posed when the seed has unknown depth (RealSense often reports 0 inside openings).
    min_inscribed_radius_m: float = 0.20

    # Unknown depth handling (depth <= 0)
    unknown_as_full: bool = False
    unknown_one_way: bool = True
    unknown_guard_px: int = 5
    # Leak guard for the free-space flood fill (hole mask):
    # If the flood fill leaks through a tiny gap and touches the ROI border, we try to "break the neck" by applying
    # a small morphological opening and then keeping only the component nearest to the seed. This is inexpensive and
    # mitigates occasional depth/edge dropouts that can connect the opening to large background regions.
    leak_guard_enabled: bool = True
    leak_open_max_px: int = 6
    # If the hole mask touches the ROI border, treat it as a leak when it covers at least this fraction of the ROI.
    # This catches the common "firehose to whole frame" failure mode where the flood fill escapes through a thin gap.
    # Set to 1.0 to effectively disable (only reject when completely full).
    leak_border_fill_frac: float = 0.25
    # Legacy/strong leak threshold: reject only when the mask covers (almost) the entire ROI.
    leak_full_fill_frac: float = 0.95

    # Plane-guided hole mode (recommended for clean hole outlines):
    # When enabled, the detector first estimates a local wall plane around the cursor (RANSAC on an annulus),
    # then classifies "hole" pixels as those sufficiently behind that plane.
    plane_guided_enabled: bool = True
    plane_ransac_iters: int = 120
    # If true and numba is available: use a no-allocation Numba RANSAC core (faster on small CPUs).
    plane_ransac_numba: bool = True
    plane_inlier_thresh_m: float = 0.025
    plane_min_inliers: int = 120
    plane_sector_count: int = 12
    plane_sector_min_frac: float = 0.70
    plane_ring_r_in_px: int = 14
    plane_ring_r_out_px: int = 150
    plane_ring_step_px: int = 3
    plane_ring_max_samples: int = 2500
    # If true: select annulus samples in a sector-balanced way (prevents a single region like the floor from dominating).
    plane_ring_balance: bool = True
    # Annulus normals (recommended): estimate normals on the plane annulus and require inliers to agree with the plane
    # normal. This stabilizes plane selection in cluttered scenes and reduces "snapping" to unrelated surfaces.
    plane_normals_enabled: bool = True
    plane_normals_step_px: int = 5
    plane_normals_min_mag: float = 0.45
    plane_normals_max_angle_deg: float = 30.0
    plane_normals_min_valid_frac: float = 0.15
    # Hole classification margin (depth must be behind the plane by this much):
    hole_margin_abs_m: float = 0.06
    hole_margin_rel: float = 0.02
    # Additional hole margin based on plane fit RMS (margin >= k * plane_rms).
    hole_margin_rms_k: float = 3.0
    # If the clicked pixel isn't inside the hole mask, allow snapping the seed to the nearest hole pixel within this radius.
    hole_seed_snap_px: int = 32
    # Optional additional snap guard: only snap to components whose *inscribed-circle center* is within this radius (px).
    # This avoids "jumping" to large background regions when the seed is near an unrelated edge.
    # Set 0 to disable.
    hole_seed_snap_max_center_px: int = 0
    # When seed snapping is used and multiple hole components are within `hole_seed_snap_px`, choose the best component
    # by maximizing: score = inscribed_radius_px - w * distance_to_seed_px. Higher w biases toward closer components.
    hole_seed_snap_score_w: float = 0.60
    # Optional guard: only consider components where distance_to_seed_px <= ratio * inscribed_radius_px.
    # This avoids snapping to far-away tiny components (or unrelated regions) when a big snap radius is enabled.
    hole_seed_snap_max_dist_ratio: float = 1.40
    # Plane mask for visualization: pixel is "plane" if abs(point-to-plane) <= this.
    plane_mask_dist_m: float = 0.02
    # Post-processing for the plane-guided hole mask (improves stability when depth inside the opening is missing/noisy).
    hole_mask_close_px: int = 2
    hole_mask_erode_px: int = 0
    hole_fill_holes: bool = True

    # Image edge barrier ("virtual walls" for the depth flood fill):
    # This is the same concept used in `D:\DroneServer\realsence_2d.py` for the column/hole detector.
    # It prevents the free-space flood fill from leaking across strong image edges, which is especially
    # important when RealSense depth has missing/invalid pixels (common when the IR emitter is off).
    edge_limit_enabled: bool = True
    # Apply the edge barrier:
    #  - unknown (default): apply only on pixels with unknown depth (<=0); this keeps "virtual walls" only where depth is missing.
    #  - seed: apply only when the seed pixel has unknown depth (<=0). Matches `realsence_2d.py` behavior.
    #  - always: apply regardless of seed depth and pixel depth (most restrictive).
    edge_limit_mode: str = "unknown"  # unknown|seed|always
    # Edge detector used to build the barrier mask.
    edge_method: str = "laplace"  # laplace|canny
    # Canny thresholds (used when edge_method == "canny").
    edge_canny_lo: int = 60
    edge_canny_hi: int = 140
    edge_canny_L2gradient: bool = True
    # Laplacian thresholding (used when edge_method == "laplace").
    edge_laplace_ksize: int = 3
    edge_threshold: int = 25
    edge_morph_ksize: int = 2
    edge_morph_iter: int = 1
    edge_dilate_px: int = 2

    # Binary-search refinement for deepest passing depth
    avail_search_iters: int = 10
    avail_search_min_range_m: float = 0.05


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _clamp_int(v: int, lo: int, hi: int) -> int:
    if v < lo:
        return lo
    if v > hi:
        return hi
    return v


def _as_intrinsics(intr: Any, *, fallback_w: int, fallback_h: int) -> Optional[RgbdIntrinsics]:
    if intr is None:
        return None

    if isinstance(intr, RgbdIntrinsics):
        return intr

    # "RgbdIntrinsics-like" object
    try:
        fx = float(getattr(intr, "fx"))
        fy = float(getattr(intr, "fy"))
        cx = float(getattr(intr, "cx", getattr(intr, "ppx", 0.0)))
        cy = float(getattr(intr, "cy", getattr(intr, "ppy", 0.0)))
        w = int(getattr(intr, "w", getattr(intr, "width", fallback_w)))
        h = int(getattr(intr, "h", getattr(intr, "height", fallback_h)))
        if fx > 0.0 and fy > 0.0:
            return RgbdIntrinsics(fx=fx, fy=fy, cx=cx, cy=cy, w=w, h=h)
    except Exception:
        pass

    # Dict
    if isinstance(intr, dict):
        try:
            fx = float(intr.get("fx", 0.0))
            fy = float(intr.get("fy", 0.0))
            cx = float(intr.get("ppx", intr.get("cx", 0.0)))
            cy = float(intr.get("ppy", intr.get("cy", 0.0)))
            w = int(intr.get("width", fallback_w))
            h = int(intr.get("height", fallback_h))
            if fx > 0.0 and fy > 0.0:
                return RgbdIntrinsics(fx=fx, fy=fy, cx=cx, cy=cy, w=w, h=h)
        except Exception:
            return None

    return None


def _depth_to_meters(depth: np.ndarray, *, depth_units: float) -> np.ndarray:
    if depth is None or not isinstance(depth, np.ndarray) or depth.ndim != 2:
        return np.zeros((1, 1), dtype=np.float32)

    if np.issubdtype(depth.dtype, np.floating):
        return np.asarray(depth, dtype=np.float32)

    du = float(depth_units if np.isfinite(float(depth_units)) else 0.001)
    if du <= 0.0:
        du = 0.001
    return depth.astype(np.float32, copy=False) * du


def _window_bbox(frame_w: int, frame_h: int, *, x: int, y: int, win_w: int, win_h: int) -> Tuple[int, int, int, int]:
    frame_w = int(max(1, frame_w))
    frame_h = int(max(1, frame_h))
    # Special case: non-positive window means "use full frame".
    if int(win_w) <= 0 or int(win_h) <= 0:
        win_w = int(frame_w)
        win_h = int(frame_h)
    else:
        win_w = int(max(1, min(int(win_w), frame_w)))
        win_h = int(max(1, min(int(win_h), frame_h)))

    cx = _clamp_int(int(x), 0, frame_w - 1)
    cy = _clamp_int(int(y), 0, frame_h - 1)

    x0 = cx - win_w // 2
    y0 = cy - win_h // 2
    x0 = _clamp_int(x0, 0, frame_w - win_w)
    y0 = _clamp_int(y0, 0, frame_h - win_h)
    return int(x0), int(y0), int(win_w), int(win_h)


def _mask01_to_u8(mask01: np.ndarray, *, out: Optional[np.ndarray] = None) -> np.ndarray:
    if out is None:
        out = np.empty(mask01.shape, dtype=np.uint8)
    out.fill(0)
    out[mask01 != 0] = np.uint8(255)
    return out


# ---------------------------------------------------------------------------
# Spatial filtering (window-local)
# ---------------------------------------------------------------------------


if _HAVE_NUMBA:

    @njit(cache=True, nogil=True)  # type: ignore[misc]
    def _nb_spatial_filter_edge_preserving(
        depth_in: np.ndarray,
        iterations: int,
        radius_px: int,
        delta_m: float,
        alpha: float,
        tmp: np.ndarray,
        out: np.ndarray,
    ) -> None:
        H, W = depth_in.shape
        tmp[:, :] = depth_in

        iters = int(iterations)
        if iters < 1:
            out[:, :] = depth_in
            return
        r = int(radius_px)
        if r < 1:
            r = 1
        if r > 4:
            r = 4  # keep this bounded; window sizes can vary

        a = float(alpha)
        if a < 0.0:
            a = 0.0
        if a > 1.0:
            a = 1.0
        ia = 1.0 - a
        dm = float(delta_m)
        if dm < 0.0:
            dm = 0.0

        for _ in range(iters):
            for y in range(H):
                y0 = y - r
                if y0 < 0:
                    y0 = 0
                y1 = y + r
                if y1 >= H:
                    y1 = H - 1
                for x in range(W):
                    zc = float(tmp[y, x])
                    if zc <= 0.0:
                        out[y, x] = 0.0
                        continue

                    x0 = x - r
                    if x0 < 0:
                        x0 = 0
                    x1 = x + r
                    if x1 >= W:
                        x1 = W - 1

                    acc = 0.0
                    cnt = 0.0
                    for yy in range(y0, y1 + 1):
                        for xx in range(x0, x1 + 1):
                            zn = float(tmp[yy, xx])
                            if zn <= 0.0:
                                continue
                            if dm > 1e-12 and abs(zn - zc) > dm:
                                continue
                            acc += zn
                            cnt += 1.0
                    if cnt > 0.0:
                        avg = acc / cnt
                        out[y, x] = ia * zc + a * avg
                    else:
                        out[y, x] = zc

            tmp[:, :] = out

else:

    def _nb_spatial_filter_edge_preserving(  # type: ignore[no-redef]
        depth_in: np.ndarray,
        iterations: int,
        radius_px: int,
        delta_m: float,
        alpha: float,
        tmp: np.ndarray,
        out: np.ndarray,
    ) -> None:
        # Fallback: a few bilateral passes (C++ implementation), preserving invalid depth as 0.
        iters = int(iterations)
        if int(iters) < 1:
            out[:, :] = depth_in
            return
        out[:, :] = depth_in
        r = int(max(1, min(9, int(radius_px))))
        d = 2 * r + 1
        dm = float(max(0.0, float(delta_m)))
        a = float(np.clip(float(alpha), 0.0, 1.0))
        ia = 1.0 - a
        invalid = out <= 0.0
        for _ in range(iters):
            tmp[:, :] = out
            try:
                filt = cv2.bilateralFilter(tmp, d=int(d), sigmaColor=float(max(1e-6, dm)), sigmaSpace=float(d))
            except Exception:
                filt = cv2.medianBlur(tmp, ksize=int(max(3, d | 1)))
            out[:, :] = ia * tmp + a * filt
            if bool(np.any(invalid)):
                out[invalid] = 0.0


def _apply_spatial_filter(depth_m: np.ndarray, *, settings: SpatialFilterSettings, tmp: np.ndarray, out: np.ndarray) -> np.ndarray:
    try:
        iters = int(getattr(settings, "iterations", 0))
    except Exception:
        iters = 0
    try:
        alpha = float(getattr(settings, "alpha", 0.0))
    except Exception:
        alpha = 0.0
    if int(iters) < 1 or (not np.isfinite(alpha)) or float(alpha) <= 0.0:
        out[:, :] = np.asarray(depth_m, dtype=np.float32)
        return out
    _nb_spatial_filter_edge_preserving(
        np.asarray(depth_m, dtype=np.float32),
        int(settings.iterations),
        int(settings.radius_px),
        float(settings.delta_m),
        float(settings.alpha),
        tmp,
        out,
    )
    return out


# ---------------------------------------------------------------------------
# Sampling kernels
# ---------------------------------------------------------------------------


if _HAVE_NUMBA:

    @njit(cache=True, nogil=True)  # type: ignore[misc]
    def _nb_collect_annulus_uvz(
        depth_m: np.ndarray,
        sx: int,
        sy: int,
        r_in2: int,
        r_out2: int,
        step: int,
        out_u: np.ndarray,
        out_v: np.ndarray,
        out_z: np.ndarray,
    ) -> int:
        """
        Collect (u,v,z) samples from `depth_m` on an annulus around (sx,sy), subsampled by `step`.
        Writes into preallocated arrays and returns the number of samples written.
        """
        H, W = depth_m.shape
        s = int(step)
        if s < 1:
            s = 1
        x0 = int(sx)
        y0 = int(sy)
        if x0 < 0:
            x0 = 0
        if x0 >= W:
            x0 = W - 1
        if y0 < 0:
            y0 = 0
        if y0 >= H:
            y0 = H - 1
        rin2 = int(r_in2)
        rout2 = int(r_out2)
        n = 0
        for v in range(0, H, s):
            dy = int(v) - int(y0)
            dy2 = dy * dy
            for u in range(0, W, s):
                dx = int(u) - int(x0)
                r2 = dx * dx + dy2
                if r2 < rin2 or r2 > rout2:
                    continue
                z = float(depth_m[int(v), int(u)])
                if not (z > 0.0) or not np.isfinite(z):
                    continue
                out_u[n] = int(u)
                out_v[n] = int(v)
                out_z[n] = np.float32(z)
                n += 1
        return int(n)

else:

    def _nb_collect_annulus_uvz(  # type: ignore[misc]
        depth_m: np.ndarray,
        sx: int,
        sy: int,
        r_in2: int,
        r_out2: int,
        step: int,
        out_u: np.ndarray,
        out_v: np.ndarray,
        out_z: np.ndarray,
    ) -> int:
        H, W = depth_m.shape
        s = int(max(1, int(step)))
        x0 = int(_clamp_int(int(sx), 0, int(W - 1)))
        y0 = int(_clamp_int(int(sy), 0, int(H - 1)))
        rin2 = int(r_in2)
        rout2 = int(r_out2)
        n = 0
        for v in range(0, int(H), int(s)):
            dy = int(v) - int(y0)
            dy2 = int(dy * dy)
            for u in range(0, int(W), int(s)):
                dx = int(u) - int(x0)
                r2 = int(dx * dx + dy2)
                if int(r2) < int(rin2) or int(r2) > int(rout2):
                    continue
                z = float(depth_m[int(v), int(u)])
                if not np.isfinite(z) or z <= 0.0:
                    continue
                out_u[n] = int(u)
                out_v[n] = int(v)
                out_z[n] = np.float32(z)
                n += 1
        return int(n)


# ---------------------------------------------------------------------------
# Plane detector kernels
# ---------------------------------------------------------------------------


if _HAVE_NUMBA:

    @njit(cache=True, nogil=True)  # type: ignore[misc]
    def _nb_compute_normals_intrinsics(depth_m: np.ndarray, step: int, fx: float, fy: float, normals_out: np.ndarray) -> None:
        H, W = depth_m.shape
        s = int(step)
        if s < 1:
            s = 1
        if s > (min(H, W) - 1) // 2:
            s = max(1, (min(H, W) - 1) // 2)
        inv = 1.0 / (2.0 * float(s))

        for v in range(s, H - s):
            for u in range(s, W - s):
                z = depth_m[v, u]
                if z <= 0.0:
                    continue

                zl = depth_m[v, u - s]
                zr = depth_m[v, u + s]
                zu = depth_m[v - s, u]
                zd = depth_m[v + s, u]
                if zl <= 0.0 or zr <= 0.0 or zu <= 0.0 or zd <= 0.0:
                    continue

                dzdu = (zr - zl) * inv
                dzdv = (zd - zu) * inv

                nx = -fx * dzdu
                ny = -fy * dzdv
                nz = z

                nrm = (nx * nx + ny * ny + nz * nz) ** 0.5
                if nrm <= 1e-12:
                    continue

                normals_out[v, u, 0] = nx / nrm
                normals_out[v, u, 1] = ny / nrm
                normals_out[v, u, 2] = nz / nrm

    @njit(cache=True, nogil=True)  # type: ignore[misc]
    def _nb_seed_normal_3x3(normals: np.ndarray, depth_m: np.ndarray, sx: int, sy: int, min_norm2: float) -> Tuple[float, float, float, int]:
        H, W, _ = normals.shape
        x0 = int(sx)
        y0 = int(sy)
        if x0 < 0:
            x0 = 0
        if x0 >= W:
            x0 = W - 1
        if y0 < 0:
            y0 = 0
        if y0 >= H:
            y0 = H - 1

        acc0 = 0.0
        acc1 = 0.0
        acc2 = 0.0
        cnt = 0

        for dy in range(-1, 2):
            y = y0 + dy
            if y < 0 or y >= H:
                continue
            for dx in range(-1, 2):
                x = x0 + dx
                if x < 0 or x >= W:
                    continue
                if depth_m[y, x] <= 0.0:
                    continue
                n0 = normals[y, x, 0]
                n1 = normals[y, x, 1]
                n2 = normals[y, x, 2]
                mag2 = n0 * n0 + n1 * n1 + n2 * n2
                if mag2 < min_norm2:
                    continue
                acc0 += n0
                acc1 += n1
                acc2 += n2
                cnt += 1

        if cnt == 0:
            return 0.0, 0.0, 0.0, 0
        nrm = (acc0 * acc0 + acc1 * acc1 + acc2 * acc2) ** 0.5
        if nrm <= 1e-12:
            return 0.0, 0.0, 0.0, 0
        return acc0 / nrm, acc1 / nrm, acc2 / nrm, cnt

    @njit(cache=True, nogil=True)  # type: ignore[misc]
    def _nb_flood_fill_from_seed_plane(
        depth_m: np.ndarray,
        normals: np.ndarray,
        xf: np.ndarray,
        yf: np.ndarray,
        sx: int,
        sy: int,
        plane_delta_thresh: float,
        cos_thresh: float,
        min_norm2: float,
        mask_out: np.ndarray,
        queue: np.ndarray,
    ) -> int:
        H, W = depth_m.shape
        N = H * W
        depth_flat = depth_m.reshape(N)
        normals_flat = normals.reshape(N, 3)
        mask_flat = mask_out.reshape(N)

        if sx < 0 or sx >= W or sy < 0 or sy >= H:
            return 0
        z_seed = depth_m[sy, sx]
        if z_seed <= 0.0:
            return 0

        ns0, ns1, ns2, cnt = _nb_seed_normal_3x3(normals, depth_m, sx, sy, min_norm2)
        if cnt == 0:
            return 0

        seed_idx = sy * W + sx
        head = 0
        tail = 0
        queue[tail] = seed_idx
        tail += 1
        mask_flat[seed_idx] = 1
        filled = 1

        while head < tail:
            idx = queue[head]
            head += 1

            u = idx % W
            v = idx // W

            z_cur = depth_flat[idx]
            if z_cur <= 0.0:
                continue

            px = z_cur * xf[u]
            py = z_cur * yf[v]
            pz = z_cur

            # 8-neighborhood
            for dv in (-1, 0, 1):
                nv = v + dv
                if nv < 0 or nv >= H:
                    continue
                for du in (-1, 0, 1):
                    if du == 0 and dv == 0:
                        continue
                    nu = u + du
                    if nu < 0 or nu >= W:
                        continue
                    nb = nv * W + nu
                    if mask_flat[nb] != 0:
                        continue

                    z_nb = depth_flat[nb]
                    if z_nb <= 0.0:
                        continue

                    qx = z_nb * xf[nu]
                    qy = z_nb * yf[nv]
                    qz = z_nb

                    dpx = qx - px
                    dpy = qy - py
                    dpz = qz - pz

                    dn = abs(ns0 * dpx + ns1 * dpy + ns2 * dpz)
                    if dn > plane_delta_thresh:
                        continue

                    n0 = normals_flat[nb, 0]
                    n1 = normals_flat[nb, 1]
                    n2 = normals_flat[nb, 2]
                    mag2 = n0 * n0 + n1 * n1 + n2 * n2
                    if mag2 < min_norm2:
                        continue
                    dot = ns0 * n0 + ns1 * n1 + ns2 * n2
                    if dot < cos_thresh:
                        continue

                    mask_flat[nb] = 1
                    queue[tail] = nb
                    tail += 1
                    filled += 1

        return filled

    @njit(cache=True, nogil=True)  # type: ignore[misc]
    def _nb_plane_mask_from_plane(  # noqa: PLR0913
        depth_m: np.ndarray,
        dist_to_hole_px: np.ndarray,
        band_out_px: float,
        fx: float,
        fy: float,
        cx_w: float,
        cy_w: float,
        n0: float,
        n1: float,
        n2: float,
        d: float,
        dist_thr_m: float,
        out_mask01: np.ndarray,
    ) -> None:
        """
        Build a plane mask (0/1) for pixels that:
          - have valid depth
          - are within `band_out_px` of the hole boundary (via distanceTransform output)
          - are within `dist_thr_m` of the plane (point-to-plane residual in meters)
        """
        H, W = depth_m.shape
        for v in range(H):
            y = (float(v) - float(cy_w)) / float(max(1e-9, fy))
            for u in range(W):
                if float(dist_to_hole_px[v, u]) > float(band_out_px):
                    out_mask01[v, u] = np.uint8(0)
                    continue
                z = float(depth_m[v, u])
                if not (z > 0.0) or not np.isfinite(z):
                    out_mask01[v, u] = np.uint8(0)
                    continue
                x = (float(u) - float(cx_w)) / float(max(1e-9, fx))
                sd = float(n0) * (x * z) + float(n1) * (y * z) + float(n2) * z + float(d)
                if sd < 0.0:
                    sd = -sd
                out_mask01[v, u] = np.uint8(1 if sd <= float(dist_thr_m) else 0)

    @njit(cache=True, nogil=True)  # type: ignore[misc]
    def _nb_refine_hole_keep_masks_signed_distance(  # noqa: PLR0913
        depth_m: np.ndarray,
        hole_mask01_in: np.ndarray,
        fx: float,
        fy: float,
        cx_w: float,
        cy_w: float,
        n0: float,
        n1: float,
        n2: float,
        d: float,
        margin_m: float,
        keep_valid01_out: np.ndarray,
        keep_unknown01_out: np.ndarray,
    ) -> None:
        """
        Refine a coarse hole mask:
          - Valid depth pixels are kept only if they are behind the plane by `margin_m`
          - Unknown depth pixels are emitted to `keep_unknown01_out` (0/1) for optional later gating
        """
        H, W = depth_m.shape
        for v in range(H):
            y = (float(v) - float(cy_w)) / float(max(1e-9, fy))
            for u in range(W):
                if hole_mask01_in[v, u] == 0:
                    keep_valid01_out[v, u] = np.uint8(0)
                    keep_unknown01_out[v, u] = np.uint8(0)
                    continue
                z = float(depth_m[v, u])
                if not (z > 0.0) or not np.isfinite(z):
                    keep_valid01_out[v, u] = np.uint8(0)
                    keep_unknown01_out[v, u] = np.uint8(1)
                    continue
                x = (float(u) - float(cx_w)) / float(max(1e-9, fx))
                sd = float(n0) * (x * z) + float(n1) * (y * z) + float(n2) * z + float(d)
                keep_valid01_out[v, u] = np.uint8(1 if sd >= float(margin_m) else 0)
                keep_unknown01_out[v, u] = np.uint8(0)

    @njit(cache=True, nogil=True)  # type: ignore[misc]
    def _nb_plane_ransac_best_from_triplets(  # noqa: PLR0913
        P: np.ndarray,
        idx3: np.ndarray,
        thr: float,
        early_cnt: int,
        use_z_ref: int,
        z_ref: float,
        use_normals: int,
        N_cam: np.ndarray,
        normals_dot_min: float,
        use_n_ref: int,
        n_ref: np.ndarray,
        out_n: np.ndarray,
        out_d: np.ndarray,
    ) -> tuple[int, float, float]:
        """
        RANSAC plane fit without per-iteration allocations.

        Returns:
          - best_inlier_count
          - best_rms (meters)
          - best_score (depth closeness score; smaller is better)

        Note: inlier mask is *not* returned; compute it once in NumPy for the best plane.
        """
        N = int(P.shape[0])
        if N < 3:
            out_n[0] = 0.0
            out_n[1] = 0.0
            out_n[2] = 0.0
            out_d[0] = 0.0
            return 0, 1e18, 1e18

        best_cnt = 0
        best_rms = 1e18
        best_score = 1e18
        best_n0 = 0.0
        best_n1 = 0.0
        best_n2 = 0.0
        best_d = 0.0

        thr0 = float(thr)
        ndot_min = float(normals_dot_min)
        useN = int(use_normals) != 0
        useZ = int(use_z_ref) != 0
        useRef = int(use_n_ref) != 0

        for k in range(int(idx3.shape[0])):
            i0 = int(idx3[k, 0])
            i1 = int(idx3[k, 1])
            i2 = int(idx3[k, 2])
            if i0 == i1 or i0 == i2 or i1 == i2:
                continue

            p0x = float(P[i0, 0])
            p0y = float(P[i0, 1])
            p0z = float(P[i0, 2])
            v1x = float(P[i1, 0]) - p0x
            v1y = float(P[i1, 1]) - p0y
            v1z = float(P[i1, 2]) - p0z
            v2x = float(P[i2, 0]) - p0x
            v2y = float(P[i2, 1]) - p0y
            v2z = float(P[i2, 2]) - p0z

            # Cross(v1, v2)
            nx = v1y * v2z - v1z * v2y
            ny = v1z * v2x - v1x * v2z
            nz = v1x * v2y - v1y * v2x
            nn = (nx * nx + ny * ny + nz * nz) ** 0.5
            if not (nn > 1e-12) or not np.isfinite(nn):
                continue
            inv = 1.0 / nn
            nx *= inv
            ny *= inv
            nz *= inv

            # Orient normal consistently
            if useRef:
                dot = nx * float(n_ref[0]) + ny * float(n_ref[1]) + nz * float(n_ref[2])
                if dot < 0.0:
                    nx = -nx
                    ny = -ny
                    nz = -nz
            else:
                if nz < 0.0:
                    nx = -nx
                    ny = -ny
                    nz = -nz

            d = -(nx * p0x + ny * p0y + nz * p0z)

            cnt = 0
            sse = 0.0
            sz = 0.0
            for i in range(int(N)):
                dist = nx * float(P[i, 0]) + ny * float(P[i, 1]) + nz * float(P[i, 2]) + d
                if dist < 0.0:
                    dist = -dist
                if not (dist <= thr0):
                    continue
                if useN:
                    dn = float(N_cam[i, 0]) * nx + float(N_cam[i, 1]) * ny + float(N_cam[i, 2]) * nz
                    if dn < ndot_min:
                        continue
                cnt += 1
                sse += dist * dist
                sz += float(P[i, 2])

            if cnt <= 0:
                continue
            if cnt < best_cnt:
                continue

            rms = (sse / float(cnt)) ** 0.5 if cnt > 0 else 1e18
            mean_z = sz / float(cnt) if cnt > 0 else 1e18
            score = abs(mean_z - float(z_ref)) if useZ else mean_z

            better = False
            if cnt > best_cnt:
                better = True
            else:
                # Tie: prefer depth-closest plane, RMS as secondary criterion.
                if score < best_score - 1e-6:
                    better = True
                elif abs(score - best_score) <= 1e-6 and rms < best_rms:
                    better = True

            if not better:
                continue
            best_cnt = int(cnt)
            best_rms = float(rms)
            best_score = float(score)
            best_n0 = float(nx)
            best_n1 = float(ny)
            best_n2 = float(nz)
            best_d = float(d)
            if int(best_cnt) >= int(early_cnt) and float(best_rms) <= float(thr0):
                break

        out_n[0] = float(best_n0)
        out_n[1] = float(best_n1)
        out_n[2] = float(best_n2)
        out_d[0] = float(best_d)
        return int(best_cnt), float(best_rms), float(best_score)

    @njit(cache=True, nogil=True)  # type: ignore[misc]
    def _nb_plane_ransac_best_cnt_rms(  # noqa: PLR0913
        P: np.ndarray,
        idx3: np.ndarray,
        thr: float,
        early_cnt: int,
        use_normals: int,
        N_cam: np.ndarray,
        normals_dot_min: float,
        use_n_ref: int,
        n_ref: np.ndarray,
        out_n: np.ndarray,
        out_d: np.ndarray,
    ) -> tuple[int, float]:
        """RANSAC plane fit: maximize inliers, tie-break by RMS (no depth-mean scoring)."""
        N = int(P.shape[0])
        if N < 3:
            out_n[0] = 0.0
            out_n[1] = 0.0
            out_n[2] = 0.0
            out_d[0] = 0.0
            return 0, 1e18

        best_cnt = 0
        best_rms = 1e18
        best_n0 = 0.0
        best_n1 = 0.0
        best_n2 = 0.0
        best_d = 0.0

        thr0 = float(thr)
        ndot_min = float(normals_dot_min)
        useN = int(use_normals) != 0
        useRef = int(use_n_ref) != 0

        for k in range(int(idx3.shape[0])):
            i0 = int(idx3[k, 0])
            i1 = int(idx3[k, 1])
            i2 = int(idx3[k, 2])
            if i0 == i1 or i0 == i2 or i1 == i2:
                continue

            p0x = float(P[i0, 0])
            p0y = float(P[i0, 1])
            p0z = float(P[i0, 2])
            v1x = float(P[i1, 0]) - p0x
            v1y = float(P[i1, 1]) - p0y
            v1z = float(P[i1, 2]) - p0z
            v2x = float(P[i2, 0]) - p0x
            v2y = float(P[i2, 1]) - p0y
            v2z = float(P[i2, 2]) - p0z

            nx = v1y * v2z - v1z * v2y
            ny = v1z * v2x - v1x * v2z
            nz = v1x * v2y - v1y * v2x
            nn = (nx * nx + ny * ny + nz * nz) ** 0.5
            if not (nn > 1e-12) or not np.isfinite(nn):
                continue
            inv = 1.0 / nn
            nx *= inv
            ny *= inv
            nz *= inv

            if useRef:
                dot = nx * float(n_ref[0]) + ny * float(n_ref[1]) + nz * float(n_ref[2])
                if dot < 0.0:
                    nx = -nx
                    ny = -ny
                    nz = -nz
            else:
                if nz < 0.0:
                    nx = -nx
                    ny = -ny
                    nz = -nz

            d = -(nx * p0x + ny * p0y + nz * p0z)

            cnt = 0
            sse = 0.0
            for i in range(int(N)):
                dist = nx * float(P[i, 0]) + ny * float(P[i, 1]) + nz * float(P[i, 2]) + d
                if dist < 0.0:
                    dist = -dist
                if not (dist <= thr0):
                    continue
                if useN:
                    dn = float(N_cam[i, 0]) * nx + float(N_cam[i, 1]) * ny + float(N_cam[i, 2]) * nz
                    if dn < ndot_min:
                        continue
                cnt += 1
                sse += dist * dist

            if cnt <= 0:
                continue
            if cnt < best_cnt:
                continue

            rms = (sse / float(cnt)) ** 0.5 if cnt > 0 else 1e18
            better = False
            if cnt > best_cnt:
                better = True
            elif cnt == best_cnt and rms < best_rms:
                better = True
            if not better:
                continue

            best_cnt = int(cnt)
            best_rms = float(rms)
            best_n0 = float(nx)
            best_n1 = float(ny)
            best_n2 = float(nz)
            best_d = float(d)
            if int(best_cnt) >= int(early_cnt) and float(best_rms) <= float(thr0):
                break

        out_n[0] = float(best_n0)
        out_n[1] = float(best_n1)
        out_n[2] = float(best_n2)
        out_d[0] = float(best_d)
        return int(best_cnt), float(best_rms)

    @njit(cache=True, nogil=True)  # type: ignore[misc]
    def _nb_mask01_keep_where_occ_dilated(mask01_u8: np.ndarray, occ_dilated_u8: np.ndarray) -> None:
        H, W = mask01_u8.shape
        for v in range(H):
            for u in range(W):
                if mask01_u8[v, u] != 0 and int(occ_dilated_u8[v, u]) == 0:
                    mask01_u8[v, u] = np.uint8(0)

else:

    def _nb_compute_normals_intrinsics(  # type: ignore[no-redef]
        depth_m: np.ndarray, step: int, fx: float, fy: float, normals_out: np.ndarray
    ) -> None:
        # Fallback: very slow in Python; use only if numba is unavailable.
        H, W = depth_m.shape
        s = int(max(1, int(step)))
        inv = 1.0 / (2.0 * float(s))
        normals_out.fill(0.0)
        for v in range(s, H - s):
            for u in range(s, W - s):
                z = float(depth_m[v, u])
                if z <= 0.0:
                    continue
                zl = float(depth_m[v, u - s])
                zr = float(depth_m[v, u + s])
                zu = float(depth_m[v - s, u])
                zd = float(depth_m[v + s, u])
                if zl <= 0.0 or zr <= 0.0 or zu <= 0.0 or zd <= 0.0:
                    continue
                dzdu = (zr - zl) * inv
                dzdv = (zd - zu) * inv
                nx = -fx * dzdu
                ny = -fy * dzdv
                nz = z
                nrm = float((nx * nx + ny * ny + nz * nz) ** 0.5)
                if nrm <= 1e-12:
                    continue
                normals_out[v, u, 0] = nx / nrm
                normals_out[v, u, 1] = ny / nrm
                normals_out[v, u, 2] = nz / nrm

    def _nb_plane_mask_from_plane(  # noqa: PLR0913
        depth_m: np.ndarray,
        dist_to_hole_px: np.ndarray,
        band_out_px: float,
        fx: float,
        fy: float,
        cx_w: float,
        cy_w: float,
        n0: float,
        n1: float,
        n2: float,
        d: float,
        dist_thr_m: float,
        out_mask01: np.ndarray,
    ) -> None:
        H, W = depth_m.shape
        out_mask01.fill(0)
        for v in range(int(H)):
            y = (float(v) - float(cy_w)) / float(max(1e-9, fy))
            for u in range(int(W)):
                if float(dist_to_hole_px[int(v), int(u)]) > float(band_out_px):
                    continue
                z = float(depth_m[int(v), int(u)])
                if (not np.isfinite(z)) or z <= 0.0:
                    continue
                x = (float(u) - float(cx_w)) / float(max(1e-9, fx))
                sd = float(n0) * (x * z) + float(n1) * (y * z) + float(n2) * z + float(d)
                if sd < 0.0:
                    sd = -sd
                if float(sd) <= float(dist_thr_m):
                    out_mask01[int(v), int(u)] = np.uint8(1)

    def _nb_refine_hole_keep_masks_signed_distance(  # noqa: PLR0913
        depth_m: np.ndarray,
        hole_mask01_in: np.ndarray,
        fx: float,
        fy: float,
        cx_w: float,
        cy_w: float,
        n0: float,
        n1: float,
        n2: float,
        d: float,
        margin_m: float,
        keep_valid01_out: np.ndarray,
        keep_unknown01_out: np.ndarray,
    ) -> None:
        H, W = depth_m.shape
        keep_valid01_out.fill(0)
        keep_unknown01_out.fill(0)
        for v in range(int(H)):
            y = (float(v) - float(cy_w)) / float(max(1e-9, fy))
            for u in range(int(W)):
                if int(hole_mask01_in[int(v), int(u)]) == 0:
                    continue
                z = float(depth_m[int(v), int(u)])
                if (not np.isfinite(z)) or z <= 0.0:
                    keep_unknown01_out[int(v), int(u)] = np.uint8(1)
                    continue
                x = (float(u) - float(cx_w)) / float(max(1e-9, fx))
                sd = float(n0) * (x * z) + float(n1) * (y * z) + float(n2) * z + float(d)
                if float(sd) >= float(margin_m):
                    keep_valid01_out[int(v), int(u)] = np.uint8(1)

    def _nb_mask01_keep_where_occ_dilated(mask01_u8: np.ndarray, occ_dilated_u8: np.ndarray) -> None:
        H, W = mask01_u8.shape
        for v in range(int(H)):
            for u in range(int(W)):
                if int(mask01_u8[int(v), int(u)]) != 0 and int(occ_dilated_u8[int(v), int(u)]) == 0:
                    mask01_u8[int(v), int(u)] = np.uint8(0)

    def _nb_flood_fill_from_seed_plane(  # type: ignore[no-redef]
        depth_m: np.ndarray,
        normals: np.ndarray,
        xf: np.ndarray,
        yf: np.ndarray,
        sx: int,
        sy: int,
        plane_delta_thresh: float,
        cos_thresh: float,
        min_norm2: float,
        mask_out: np.ndarray,
        queue: np.ndarray,
    ) -> int:
        # Fallback BFS in Python (slow); queue is an int32 array.
        H, W = depth_m.shape
        if sx < 0 or sx >= W or sy < 0 or sy >= H:
            return 0
        if float(depth_m[sy, sx]) <= 0.0:
            return 0

        # seed normal (3x3)
        acc = np.zeros(3, dtype=np.float64)
        cnt = 0
        for dy in (-1, 0, 1):
            yy = sy + dy
            if yy < 0 or yy >= H:
                continue
            for dx in (-1, 0, 1):
                xx = sx + dx
                if xx < 0 or xx >= W:
                    continue
                if float(depth_m[yy, xx]) <= 0.0:
                    continue
                n = normals[yy, xx].astype(np.float64, copy=False)
                if float(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]) < float(min_norm2):
                    continue
                acc += n
                cnt += 1
        if cnt <= 0:
            return 0
        nrm = float(np.linalg.norm(acc))
        if nrm <= 1e-12:
            return 0
        ns = acc / nrm

        mask_out.fill(0)
        head = 0
        tail = 0
        seed_idx = int(sy * W + sx)
        queue[tail] = seed_idx
        tail += 1
        mask_out[sy, sx] = 1
        filled = 1

        while head < tail:
            idx = int(queue[head])
            head += 1
            u = int(idx % W)
            v = int(idx // W)
            z_cur = float(depth_m[v, u])
            if z_cur <= 0.0:
                continue
            px = z_cur * float(xf[u])
            py = z_cur * float(yf[v])
            pz = z_cur

            for dv in (-1, 0, 1):
                nv = v + dv
                if nv < 0 or nv >= H:
                    continue
                for du in (-1, 0, 1):
                    if du == 0 and dv == 0:
                        continue
                    nu = u + du
                    if nu < 0 or nu >= W:
                        continue
                    if mask_out[nv, nu] != 0:
                        continue
                    z_nb = float(depth_m[nv, nu])
                    if z_nb <= 0.0:
                        continue
                    qx = z_nb * float(xf[nu])
                    qy = z_nb * float(yf[nv])
                    qz = z_nb
                    dpx = qx - px
                    dpy = qy - py
                    dpz = qz - pz
                    dn = abs(float(ns[0] * dpx + ns[1] * dpy + ns[2] * dpz))
                    if dn > float(plane_delta_thresh):
                        continue
                    n0, n1, n2 = [float(x) for x in normals[nv, nu]]
                    mag2 = n0 * n0 + n1 * n1 + n2 * n2
                    if mag2 < float(min_norm2):
                        continue
                    dot = float(ns[0] * n0 + ns[1] * n1 + ns[2] * n2)
                    if dot < float(cos_thresh):
                        continue
                    mask_out[nv, nu] = 1
                    queue[tail] = int(nv * W + nu)
                    tail += 1
                    filled += 1
        return int(filled)


# ---------------------------------------------------------------------------
# Hole detector kernels
# ---------------------------------------------------------------------------


if _HAVE_NUMBA:

    @njit(cache=True, nogil=True)  # type: ignore[misc]
    def _nb_flood_fill_free_depth_limit(
        depth_m: np.ndarray,
        depth_limit_m: float,
        unknown_is_free: bool,
        sx: int,
        sy: int,
        out_mask: np.ndarray,
        queue: np.ndarray,
    ) -> int:
        """
        Flood fill in image space, treating pixels as "free" if:
          - depth <= 0      (unknown depth -> free only if unknown_is_free)
          - OR depth >= z_ref
        and blocked if 0 < depth < z_ref.
        """
        H, W = depth_m.shape
        if sx < 0 or sx >= W or sy < 0 or sy >= H:
            return 0

        z0 = depth_m[sy, sx]
        if z0 <= 0.0:
            if not unknown_is_free:
                return 0
        else:
            if z0 < depth_limit_m:
                return 0

        head = 0
        tail = 0
        idx0 = sy * W + sx
        queue[tail] = idx0
        tail += 1
        out_mask[sy, sx] = 1
        count = 1

        while head < tail:
            idx = queue[head]
            head += 1
            x = idx % W
            y = idx // W

            # left
            x0 = x - 1
            if x0 >= 0 and out_mask[y, x0] == 0:
                z = depth_m[y, x0]
                if (z <= 0.0 and unknown_is_free) or (z > 0.0 and z >= depth_limit_m):
                    out_mask[y, x0] = 1
                    queue[tail] = y * W + x0
                    tail += 1
                    count += 1

            # right
            x1 = x + 1
            if x1 < W and out_mask[y, x1] == 0:
                z = depth_m[y, x1]
                if (z <= 0.0 and unknown_is_free) or (z > 0.0 and z >= depth_limit_m):
                    out_mask[y, x1] = 1
                    queue[tail] = y * W + x1
                    tail += 1
                    count += 1

            # up
            y0 = y - 1
            if y0 >= 0 and out_mask[y0, x] == 0:
                z = depth_m[y0, x]
                if (z <= 0.0 and unknown_is_free) or (z > 0.0 and z >= depth_limit_m):
                    out_mask[y0, x] = 1
                    queue[tail] = y0 * W + x
                    tail += 1
                    count += 1

            # down
            y1 = y + 1
            if y1 < H and out_mask[y1, x] == 0:
                z = depth_m[y1, x]
                if (z <= 0.0 and unknown_is_free) or (z > 0.0 and z >= depth_limit_m):
                    out_mask[y1, x] = 1
                    queue[tail] = y1 * W + x
                    tail += 1
                    count += 1

        return count

    @njit(cache=True, nogil=True)  # type: ignore[misc]
    def _nb_flood_fill_free_depth_limit_blocked(
        depth_m: np.ndarray,
        depth_limit_m: float,
        unknown_is_free: bool,
        block_mask: np.ndarray,
        sx: int,
        sy: int,
        out_mask: np.ndarray,
        queue: np.ndarray,
    ) -> int:
        """
        Same as `_nb_flood_fill_free_depth_limit`, but blocks traversal through `block_mask != 0`.
        """
        H, W = depth_m.shape
        if sx < 0 or sx >= W or sy < 0 or sy >= H:
            return 0
        if block_mask[sy, sx] != 0:
            return 0

        z0 = depth_m[sy, sx]
        if z0 <= 0.0:
            if not unknown_is_free:
                return 0
        else:
            if z0 < depth_limit_m:
                return 0

        head = 0
        tail = 0
        idx0 = sy * W + sx
        queue[tail] = idx0
        tail += 1
        out_mask[sy, sx] = 1
        count = 1

        while head < tail:
            idx = queue[head]
            head += 1
            x = idx % W
            y = idx // W

            # left
            x0 = x - 1
            if x0 >= 0 and out_mask[y, x0] == 0 and block_mask[y, x0] == 0:
                z = depth_m[y, x0]
                if (z <= 0.0 and unknown_is_free) or (z > 0.0 and z >= depth_limit_m):
                    out_mask[y, x0] = 1
                    queue[tail] = y * W + x0
                    tail += 1
                    count += 1

            # right
            x1 = x + 1
            if x1 < W and out_mask[y, x1] == 0 and block_mask[y, x1] == 0:
                z = depth_m[y, x1]
                if (z <= 0.0 and unknown_is_free) or (z > 0.0 and z >= depth_limit_m):
                    out_mask[y, x1] = 1
                    queue[tail] = y * W + x1
                    tail += 1
                    count += 1

            # up
            y0 = y - 1
            if y0 >= 0 and out_mask[y0, x] == 0 and block_mask[y0, x] == 0:
                z = depth_m[y0, x]
                if (z <= 0.0 and unknown_is_free) or (z > 0.0 and z >= depth_limit_m):
                    out_mask[y0, x] = 1
                    queue[tail] = y0 * W + x
                    tail += 1
                    count += 1

            # down
            y1 = y + 1
            if y1 < H and out_mask[y1, x] == 0 and block_mask[y1, x] == 0:
                z = depth_m[y1, x]
                if (z <= 0.0 and unknown_is_free) or (z > 0.0 and z >= depth_limit_m):
                    out_mask[y1, x] = 1
                    queue[tail] = y1 * W + x
                    tail += 1
                    count += 1

        return count

    @njit(cache=True, nogil=True)  # type: ignore[misc]
    def _nb_flood_fill_free_depth_limit_one_way_unknown(
        depth_m: np.ndarray,
        depth_limit_m: float,
        unknown_is_free: bool,
        sx: int,
        sy: int,
        out_mask: np.ndarray,
        queue: np.ndarray,
    ) -> int:
        """
        Flood fill with a one-way rule for unknown pixels (depth <= 0), when unknown_is_free:
          - valid_free -> unknown allowed
          - unknown    -> unknown allowed
          - unknown    -> valid_free NOT allowed
        """
        H, W = depth_m.shape
        if sx < 0 or sx >= W or sy < 0 or sy >= H:
            return 0

        z0 = depth_m[sy, sx]
        cur_unknown0 = z0 <= 0.0
        if cur_unknown0:
            if not unknown_is_free:
                return 0
        else:
            if z0 < depth_limit_m:
                return 0

        head = 0
        tail = 0
        idx0 = sy * W + sx
        queue[tail] = idx0
        tail += 1
        out_mask[sy, sx] = 1
        count = 1

        while head < tail:
            idx = queue[head]
            head += 1
            x = idx % W
            y = idx // W

            zc = depth_m[y, x]
            cur_unknown = zc <= 0.0

            # left
            x0 = x - 1
            if x0 >= 0 and out_mask[y, x0] == 0:
                z = depth_m[y, x0]
                if cur_unknown:
                    if unknown_is_free and z <= 0.0:
                        out_mask[y, x0] = 1
                        queue[tail] = y * W + x0
                        tail += 1
                        count += 1
                else:
                    if (z > 0.0 and z >= depth_limit_m) or (unknown_is_free and z <= 0.0):
                        out_mask[y, x0] = 1
                        queue[tail] = y * W + x0
                        tail += 1
                        count += 1

            # right
            x1 = x + 1
            if x1 < W and out_mask[y, x1] == 0:
                z = depth_m[y, x1]
                if cur_unknown:
                    if unknown_is_free and z <= 0.0:
                        out_mask[y, x1] = 1
                        queue[tail] = y * W + x1
                        tail += 1
                        count += 1
                else:
                    if (z > 0.0 and z >= depth_limit_m) or (unknown_is_free and z <= 0.0):
                        out_mask[y, x1] = 1
                        queue[tail] = y * W + x1
                        tail += 1
                        count += 1

            # up
            y0 = y - 1
            if y0 >= 0 and out_mask[y0, x] == 0:
                z = depth_m[y0, x]
                if cur_unknown:
                    if unknown_is_free and z <= 0.0:
                        out_mask[y0, x] = 1
                        queue[tail] = y0 * W + x
                        tail += 1
                        count += 1
                else:
                    if (z > 0.0 and z >= depth_limit_m) or (unknown_is_free and z <= 0.0):
                        out_mask[y0, x] = 1
                        queue[tail] = y0 * W + x
                        tail += 1
                        count += 1

            # down
            y1 = y + 1
            if y1 < H and out_mask[y1, x] == 0:
                z = depth_m[y1, x]
                if cur_unknown:
                    if unknown_is_free and z <= 0.0:
                        out_mask[y1, x] = 1
                        queue[tail] = y1 * W + x
                        tail += 1
                        count += 1
                else:
                    if (z > 0.0 and z >= depth_limit_m) or (unknown_is_free and z <= 0.0):
                        out_mask[y1, x] = 1
                        queue[tail] = y1 * W + x
                        tail += 1
                        count += 1

        return count

    @njit(cache=True, nogil=True)  # type: ignore[misc]
    def _nb_flood_fill_free_depth_limit_one_way_unknown_blocked(
        depth_m: np.ndarray,
        depth_limit_m: float,
        unknown_is_free: bool,
        block_mask: np.ndarray,
        sx: int,
        sy: int,
        out_mask: np.ndarray,
        queue: np.ndarray,
    ) -> int:
        H, W = depth_m.shape
        if sx < 0 or sx >= W or sy < 0 or sy >= H:
            return 0
        if block_mask[sy, sx] != 0:
            return 0

        z0 = depth_m[sy, sx]
        cur_unknown0 = z0 <= 0.0
        if cur_unknown0:
            if not unknown_is_free:
                return 0
        else:
            if z0 < depth_limit_m:
                return 0

        head = 0
        tail = 0
        idx0 = sy * W + sx
        queue[tail] = idx0
        tail += 1
        out_mask[sy, sx] = 1
        count = 1

        while head < tail:
            idx = queue[head]
            head += 1
            x = idx % W
            y = idx // W

            zc = depth_m[y, x]
            cur_unknown = zc <= 0.0

            # left
            x0 = x - 1
            if x0 >= 0 and out_mask[y, x0] == 0 and block_mask[y, x0] == 0:
                z = depth_m[y, x0]
                if cur_unknown:
                    if unknown_is_free and z <= 0.0:
                        out_mask[y, x0] = 1
                        queue[tail] = y * W + x0
                        tail += 1
                        count += 1
                else:
                    if (z > 0.0 and z >= depth_limit_m) or (unknown_is_free and z <= 0.0):
                        out_mask[y, x0] = 1
                        queue[tail] = y * W + x0
                        tail += 1
                        count += 1

            # right
            x1 = x + 1
            if x1 < W and out_mask[y, x1] == 0 and block_mask[y, x1] == 0:
                z = depth_m[y, x1]
                if cur_unknown:
                    if unknown_is_free and z <= 0.0:
                        out_mask[y, x1] = 1
                        queue[tail] = y * W + x1
                        tail += 1
                        count += 1
                else:
                    if (z > 0.0 and z >= depth_limit_m) or (unknown_is_free and z <= 0.0):
                        out_mask[y, x1] = 1
                        queue[tail] = y * W + x1
                        tail += 1
                        count += 1

            # up
            y0 = y - 1
            if y0 >= 0 and out_mask[y0, x] == 0 and block_mask[y0, x] == 0:
                z = depth_m[y0, x]
                if cur_unknown:
                    if unknown_is_free and z <= 0.0:
                        out_mask[y0, x] = 1
                        queue[tail] = y0 * W + x
                        tail += 1
                        count += 1
                else:
                    if (z > 0.0 and z >= depth_limit_m) or (unknown_is_free and z <= 0.0):
                        out_mask[y0, x] = 1
                        queue[tail] = y0 * W + x
                        tail += 1
                        count += 1

            # down
            y1 = y + 1
            if y1 < H and out_mask[y1, x] == 0 and block_mask[y1, x] == 0:
                z = depth_m[y1, x]
                if cur_unknown:
                    if unknown_is_free and z <= 0.0:
                        out_mask[y1, x] = 1
                        queue[tail] = y1 * W + x
                        tail += 1
                        count += 1
                else:
                    if (z > 0.0 and z >= depth_limit_m) or (unknown_is_free and z <= 0.0):
                        out_mask[y1, x] = 1
                        queue[tail] = y1 * W + x
                        tail += 1
                        count += 1

        return count

else:

    def _nb_flood_fill_free_depth_limit(  # type: ignore[no-redef]
        depth_m: np.ndarray,
        depth_limit_m: float,
        unknown_is_free: bool,
        sx: int,
        sy: int,
        out_mask: np.ndarray,
        queue: np.ndarray,
    ) -> int:
        H, W = depth_m.shape
        if sx < 0 or sx >= W or sy < 0 or sy >= H:
            return 0
        z0 = float(depth_m[sy, sx])
        if z0 <= 0.0:
            if not bool(unknown_is_free):
                return 0
        else:
            if z0 < float(depth_limit_m):
                return 0
        out_mask.fill(0)
        head = 0
        tail = 0
        queue[tail] = int(sy * W + sx)
        tail += 1
        out_mask[sy, sx] = 1
        count = 1
        while head < tail:
            idx = int(queue[head])
            head += 1
            x = int(idx % W)
            y = int(idx // W)
            for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                if nx < 0 or nx >= W or ny < 0 or ny >= H:
                    continue
                if out_mask[ny, nx] != 0:
                    continue
                z = float(depth_m[ny, nx])
                if (z <= 0.0 and bool(unknown_is_free)) or (z > 0.0 and z >= float(depth_limit_m)):
                    out_mask[ny, nx] = 1
                    queue[tail] = int(ny * W + nx)
                    tail += 1
                    count += 1
        return int(count)

    def _nb_flood_fill_free_depth_limit_blocked(  # type: ignore[no-redef]
        depth_m: np.ndarray,
        depth_limit_m: float,
        unknown_is_free: bool,
        block_mask: np.ndarray,
        sx: int,
        sy: int,
        out_mask: np.ndarray,
        queue: np.ndarray,
    ) -> int:
        H, W = depth_m.shape
        if sx < 0 or sx >= W or sy < 0 or sy >= H:
            return 0
        if int(block_mask[sy, sx]) != 0:
            return 0
        z0 = float(depth_m[sy, sx])
        if z0 <= 0.0:
            if not bool(unknown_is_free):
                return 0
        else:
            if z0 < float(depth_limit_m):
                return 0
        out_mask.fill(0)
        head = 0
        tail = 0
        queue[tail] = int(sy * W + sx)
        tail += 1
        out_mask[sy, sx] = 1
        count = 1
        while head < tail:
            idx = int(queue[head])
            head += 1
            x = int(idx % W)
            y = int(idx // W)
            for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                if nx < 0 or nx >= W or ny < 0 or ny >= H:
                    continue
                if out_mask[ny, nx] != 0 or int(block_mask[ny, nx]) != 0:
                    continue
                z = float(depth_m[ny, nx])
                if (z <= 0.0 and bool(unknown_is_free)) or (z > 0.0 and z >= float(depth_limit_m)):
                    out_mask[ny, nx] = 1
                    queue[tail] = int(ny * W + nx)
                    tail += 1
                    count += 1
        return int(count)

    def _nb_flood_fill_free_depth_limit_one_way_unknown(  # type: ignore[no-redef]
        depth_m: np.ndarray,
        depth_limit_m: float,
        unknown_is_free: bool,
        sx: int,
        sy: int,
        out_mask: np.ndarray,
        queue: np.ndarray,
    ) -> int:
        return _nb_flood_fill_free_depth_limit(depth_m, float(depth_limit_m), bool(unknown_is_free), int(sx), int(sy), out_mask, queue)

    def _nb_flood_fill_free_depth_limit_one_way_unknown_blocked(  # type: ignore[no-redef]
        depth_m: np.ndarray,
        depth_limit_m: float,
        unknown_is_free: bool,
        block_mask: np.ndarray,
        sx: int,
        sy: int,
        out_mask: np.ndarray,
        queue: np.ndarray,
    ) -> int:
        return _nb_flood_fill_free_depth_limit_blocked(
            depth_m, float(depth_limit_m), bool(unknown_is_free), block_mask, int(sx), int(sy), out_mask, queue
        )


# ---------------------------------------------------------------------------
# Detectors
# ---------------------------------------------------------------------------


class PlaneDetector:
    def __init__(self, *, settings: PlaneSettings = PlaneSettings(), spatial: SpatialFilterSettings = SpatialFilterSettings()):
        self.settings = settings
        self.spatial = spatial

        self._shape: Tuple[int, int] = (0, 0)
        self._depth_tmp = np.empty((1, 1), dtype=np.float32)
        self._depth_filt = np.empty((1, 1), dtype=np.float32)
        self._normals = np.empty((1, 1, 3), dtype=np.float32)
        self._mask01 = np.empty((1, 1), dtype=np.uint8)
        self._mask255 = np.empty((1, 1), dtype=np.uint8)
        self._queue = np.empty((1,), dtype=np.int32)

        # Last-run debug
        self.last_error: str = ""
        self.last_filled_px: int = 0

    def _ensure(self, h: int, w: int) -> None:
        if self._shape == (h, w):
            return
        self._shape = (h, w)
        self._depth_tmp = np.empty((h, w), dtype=np.float32)
        self._depth_filt = np.empty((h, w), dtype=np.float32)
        self._normals = np.zeros((h, w, 3), dtype=np.float32)
        self._mask01 = np.zeros((h, w), dtype=np.uint8)
        self._mask255 = np.zeros((h, w), dtype=np.uint8)
        self._queue = np.empty((h * w), dtype=np.int32)

    def detect(
        self,
        rgbd: Union[RgbdWindow, dict],
        *,
        window_wh: Tuple[int, int],
        x: int,
        y: int,
        settings: Optional[PlaneSettings] = None,
        spatial: Optional[SpatialFilterSettings] = None,
    ) -> np.ndarray:
        self.last_error = ""
        self.last_filled_px = 0

        if isinstance(rgbd, dict):
            bgr = rgbd.get("bgr", None)
            depth = rgbd.get("depth", rgbd.get("depth_raw", rgbd.get("depth_m", None)))
            depth_units = float(rgbd.get("depth_units", 0.001))
            intr_any = rgbd.get("intr", rgbd.get("intrinsics", None))
        else:
            bgr = rgbd.bgr
            depth = rgbd.depth
            depth_units = float(rgbd.depth_units)
            intr_any = rgbd.intr

        if bgr is None or not isinstance(bgr, np.ndarray) or bgr.ndim != 3 or bgr.shape[2] != 3:
            self.last_error = "Invalid/missing BGR image."
            return np.zeros((1, 1), dtype=np.uint8)

        depth_m_full = _depth_to_meters(np.asarray(depth), depth_units=depth_units) if isinstance(depth, np.ndarray) else None
        if depth_m_full is None or depth_m_full.ndim != 2:
            self.last_error = "Invalid/missing depth image."
            return np.zeros((1, 1), dtype=np.uint8)

        H, W = int(depth_m_full.shape[0]), int(depth_m_full.shape[1])
        intr = _as_intrinsics(intr_any, fallback_w=W, fallback_h=H)
        if intr is None:
            self.last_error = "Missing/invalid intrinsics (need fx/fy/cx/cy)."
            win_w, win_h = (int(window_wh[0]), int(window_wh[1]))
            if int(win_w) <= 0 or int(win_h) <= 0:
                win_w, win_h = int(W), int(H)
            return np.zeros((int(min(win_h, H)), int(min(win_w, W))), dtype=np.uint8)

        win_w, win_h = int(window_wh[0]), int(window_wh[1])
        x0, y0, ww, hh = _window_bbox(W, H, x=int(x), y=int(y), win_w=win_w, win_h=win_h)

        depth_roi = np.asarray(depth_m_full[y0 : y0 + hh, x0 : x0 + ww], dtype=np.float32)
        sx = int(_clamp_int(int(x) - x0, 0, ww - 1))
        sy = int(_clamp_int(int(y) - y0, 0, hh - 1))

        self._ensure(hh, ww)

        sp = spatial if spatial is not None else self.spatial
        depth_f = _apply_spatial_filter(depth_roi, settings=sp, tmp=self._depth_tmp, out=self._depth_filt)

        ps = settings if settings is not None else self.settings
        angle_deg = float(ps.max_normal_angle_deg)
        cos_thresh = float(np.cos(np.deg2rad(angle_deg)))
        plane_delta = float(ps.neighbor_delta_along_normal_mm) * 0.001
        step = int(max(1, int(ps.normal_grad_step_px)))
        min_norm2 = float(max(0.0, float(ps.min_normal_mag))) ** 2

        if float(depth_f[sy, sx]) <= 0.0:
            self._mask01.fill(0)
            self._mask255.fill(0)
            return self._mask255

        fx = float(intr.fx)
        fy = float(intr.fy)
        cx_w = float(intr.cx) - float(x0)
        cy_w = float(intr.cy) - float(y0)
        xf = (np.arange(ww, dtype=np.float32) - np.float32(cx_w)) / np.float32(max(1e-9, fx))
        yf = (np.arange(hh, dtype=np.float32) - np.float32(cy_w)) / np.float32(max(1e-9, fy))

        self._mask01.fill(0)
        self._normals.fill(0.0)
        _nb_compute_normals_intrinsics(depth_f, step, fx, fy, self._normals)
        filled = _nb_flood_fill_from_seed_plane(
            depth_f,
            self._normals,
            xf,
            yf,
            int(sx),
            int(sy),
            float(plane_delta),
            float(cos_thresh),
            float(min_norm2),
            self._mask01,
            self._queue,
        )
        self.last_filled_px = int(filled)
        _mask01_to_u8(self._mask01, out=self._mask255)
        return self._mask255


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# Demo (optional)
# ---------------------------------------------------------------------------


def _depth_viz(depth_m: np.ndarray, *, min_m: float = 0.25, max_m: float = 15.0) -> np.ndarray:
    d = np.asarray(depth_m, dtype=np.float32)
    d0 = float(min_m)
    d1 = float(max_m)
    scale = 255.0 / max(1e-6, (d1 - d0))
    img = (np.clip(d, d0, d1) - d0) * scale
    u8 = img.astype(np.uint8)
    invalid = d <= 0.0
    if bool(np.any(invalid)):
        u8[invalid] = np.uint8(0)
    bgr = cv2.applyColorMap(u8, cv2.COLORMAP_HOT)
    if bool(np.any(invalid)):
        bgr[invalid] = 0
    return bgr


def _overlay_mask(bgr: np.ndarray, mask255: np.ndarray, color: Tuple[int, int, int], alpha: float = 0.5) -> np.ndarray:
    out = bgr.copy()
    if mask255 is None or mask255.size == 0:
        return out
    m = mask255 != 0
    if not bool(np.any(m)):
        return out
    overlay = np.zeros_like(out)
    overlay[:, :] = np.array(color, dtype=np.uint8).reshape(1, 1, 3)
    out[m] = (alpha * overlay[m] + (1.0 - alpha) * out[m]).astype(np.uint8)
    return out


def main() -> None:
    """Interactive RealSense demo (requires `pyrealsense2`)."""
    try:
        import pyrealsense2 as rs  # type: ignore
    except Exception as exc:
        raise SystemExit(f"pyrealsense2 not available: {exc}")

    W, H, FPS = 640, 480, 30

    pipeline = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, W, H, rs.format.bgr8, FPS)
    cfg.enable_stream(rs.stream.depth, W, H, rs.format.z16, FPS)
    profile = pipeline.start(cfg)
    align = rs.align(rs.stream.color)

    depth_sensor = profile.get_device().first_depth_sensor()
    depth_units = float(depth_sensor.get_depth_scale()) if depth_sensor is not None else 0.001

    vsp = profile.get_stream(rs.stream.color).as_video_stream_profile()
    intr = vsp.get_intrinsics()
    intr_obj = RgbdIntrinsics(
        fx=float(intr.fx),
        fy=float(intr.fy),
        cx=float(intr.ppx),
        cy=float(intr.ppy),
        w=int(intr.width),
        h=int(intr.height),
    )

    plane = PlaneDetector()
    hole = HoleDetector()

    win = "RGBD Detector Demo"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)

    window_wh = (320, 240)
    seed = {"mode": "plane", "x": W // 2, "y": H // 2}

    def on_mouse(event, x, y, flags, param):
        if event not in (cv2.EVENT_MOUSEMOVE, cv2.EVENT_LBUTTONDOWN):
            return
        if 0 <= y < H and 0 <= x < W:
            seed["mode"] = "plane"
            seed["x"] = int(x)
            seed["y"] = int(y)
        elif 0 <= y < H and W <= x < 2 * W:
            seed["mode"] = "hole"
            seed["x"] = int(x - W)
            seed["y"] = int(y)

    cv2.setMouseCallback(win, on_mouse)

    try:
        while True:
            frames = pipeline.wait_for_frames()
            frames = align.process(frames)
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            bgr = np.asanyarray(color_frame.get_data())
            depth_raw = np.asanyarray(depth_frame.get_data())
            depth_m = depth_raw.astype(np.float32) * float(depth_units)

            rgbd = RgbdWindow(bgr=bgr, depth=depth_m, depth_units=float(depth_units), intr=intr_obj)
            depth_vis = _depth_viz(depth_m)

            mode = str(seed["mode"])
            sx = int(seed["x"])
            sy = int(seed["y"])

            if mode == "plane":
                mask = plane.detect(rgbd, window_wh=window_wh, x=sx, y=sy)
                x0, y0, ww, hh = _window_bbox(W, H, x=sx, y=sy, win_w=window_wh[0], win_h=window_wh[1])
                depth_vis_roi = depth_vis[y0 : y0 + hh, x0 : x0 + ww]
                bgr_roi = bgr[y0 : y0 + hh, x0 : x0 + ww]
                depth_vis[y0 : y0 + hh, x0 : x0 + ww] = _overlay_mask(depth_vis_roi, mask, (40, 220, 40), 0.55)
                bgr[y0 : y0 + hh, x0 : x0 + ww] = _overlay_mask(bgr_roi, mask, (40, 220, 40), 0.55)
                cv2.rectangle(depth_vis, (x0, y0), (x0 + ww - 1, y0 + hh - 1), (255, 255, 255), 1)
            else:
                mask = hole.detect(rgbd, window_wh=window_wh, x=sx, y=sy)
                x0, y0, ww, hh = _window_bbox(W, H, x=sx, y=sy, win_w=window_wh[0], win_h=window_wh[1])
                depth_vis_roi = depth_vis[y0 : y0 + hh, x0 : x0 + ww]
                bgr_roi = bgr[y0 : y0 + hh, x0 : x0 + ww]
                depth_vis[y0 : y0 + hh, x0 : x0 + ww] = _overlay_mask(depth_vis_roi, mask, (255, 0, 255), 0.55)
                bgr[y0 : y0 + hh, x0 : x0 + ww] = _overlay_mask(bgr_roi, mask, (255, 0, 255), 0.55)
                cv2.rectangle(bgr, (x0, y0), (x0 + ww - 1, y0 + hh - 1), (255, 255, 255), 1)

            combined = np.hstack([depth_vis, bgr])
            cv2.putText(
                combined,
                f"mode={mode}  window={window_wh[0]}x{window_wh[1]}  filled={plane.last_filled_px if mode=='plane' else hole.last_filled_px}",
                (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                3,
                cv2.LINE_AA,
            )
            cv2.putText(
                combined,
                f"mode={mode}  window={window_wh[0]}x{window_wh[1]}  filled={plane.last_filled_px if mode=='plane' else hole.last_filled_px}",
                (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            if mode == "hole":
                cv2.putText(
                    combined,
                    f"deepest={hole.last_depth_avail_m:.2f}m  r={hole.last_radius_m:.2f}m",
                    (10, 45),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 0),
                    3,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    combined,
                    f"deepest={hole.last_depth_avail_m:.2f}m  r={hole.last_radius_m:.2f}m",
                    (10, 45),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

            cv2.imshow(win, combined)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break

    finally:
        try:
            pipeline.stop()
        except Exception:
            pass
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass


class HoleDetector:
    def __init__(self, *, settings: HoleSettings = HoleSettings(), spatial: SpatialFilterSettings = SpatialFilterSettings()):
        self.settings = settings
        self.spatial = spatial

        self._shape: Tuple[int, int] = (0, 0)
        self._depth_tmp = np.empty((1, 1), dtype=np.float32)
        self._depth_filt = np.empty((1, 1), dtype=np.float32)
        self._mask01 = np.empty((1, 1), dtype=np.uint8)
        self._mask255 = np.empty((1, 1), dtype=np.uint8)
        self._queue = np.empty((1,), dtype=np.int32)

        # Scratch (avoid per-call allocations)
        self._ff_mask = np.empty((3, 3), dtype=np.uint8)  # floodFill mask (h+2,w+2)
        self._dt_pad = np.empty((3, 3), dtype=np.uint8)  # distanceTransform input (h+2,w+2)
        self._inv_u8 = np.empty((1, 1), dtype=np.uint8)  # generic inverse (h,w)
        self._u8_tmp0 = np.empty((1, 1), dtype=np.uint8)
        self._u8_tmp1 = np.empty((1, 1), dtype=np.uint8)

        # Annulus sample scratch (size = h*w, filled by Numba kernels)
        self._s_u = np.empty((1,), dtype=np.int32)
        self._s_v = np.empty((1,), dtype=np.int32)
        self._s_z = np.empty((1,), dtype=np.float32)

        # Morphology kernel cache (ellipse kernels by radius)
        self._ellipse_kernels: dict[int, np.ndarray] = {}

        # Edge-barrier scratch
        self._gray = np.empty((1, 1), dtype=np.uint8)
        self._edge_u8 = np.empty((1, 1), dtype=np.uint8)
        self._edge_mask = np.empty((1, 1), dtype=np.uint8)
        self._edge_block = np.empty((1, 1), dtype=np.uint8)
        self._edge_kernel = np.ones((1, 1), dtype=np.uint8)
        self._edge_dilate_kernel: Optional[np.ndarray] = None

        # Unknown guard scratch
        self._occ_u8 = np.empty((1, 1), dtype=np.uint8)
        self._occ_dilated = np.empty((1, 1), dtype=np.uint8)
        self._col_block_combined = np.empty((1, 1), dtype=np.uint8)
        self._tmp0 = np.empty((1, 1), dtype=np.bool_)
        self._tmp1 = np.empty((1, 1), dtype=np.bool_)
        self._occ_dilate_kernel: Optional[np.ndarray] = None

        # Last-run debug
        self.last_error: str = ""
        self.last_filled_px: int = 0
        self.last_depth_valid_frac: float = 0.0
        self.last_depth_avail_m: float = 0.0
        self.last_radius_m: float = 0.0
        self.last_center_px: Tuple[int, int] = (-1, -1)
        self.last_edge_barrier_used: bool = False
        self.last_edge_barrier_mode: str = ""
        self.last_edge_barrier_method: str = ""
        self.last_hole_wall_depth_m: float = 0.0
        self.last_hole_z_ref_m: float = 0.0
        self.last_hole_plane_band_in_px: int = 0
        self.last_hole_plane_band_out_px: int = 0

        # Plane-guided debug (see `detect_plane_guided()`).
        self.last_plane_ok: bool = False
        self.last_plane_inliers: int = 0
        self.last_plane_rms_m: float = 0.0
        self.last_plane_sector_cov: float = 0.0
        self.last_plane_n_cam: Tuple[float, float, float] = (0.0, 0.0, 0.0)
        self.last_plane_d_cam: float = 0.0
        self.last_plane_normals_used: bool = False
        self.last_plane_normals_valid_frac: float = 0.0
        self.last_circle_center_full_px: Tuple[int, int] = (-1, -1)
        self.last_circle_radius_px: float = 0.0
        self.last_circle_fill_ratio: float = 0.0
        self.last_hole_valid_frac: float = 0.0
        self.last_hole_touches_border: bool = False

        # Plane-guided scratch
        self._plane01 = np.empty((1, 1), dtype=np.uint8)
        self._plane255 = np.empty((1, 1), dtype=np.uint8)
        self._plane_normals = np.zeros((1, 1, 3), dtype=np.float32)

    def _kernel_ellipse(self, r: int) -> np.ndarray:
        r = int(max(0, int(r)))
        if int(r) <= 0:
            return np.ones((1, 1), dtype=np.uint8)
        k = self._ellipse_kernels.get(int(r))
        if k is None or not isinstance(k, np.ndarray) or k.shape != (int(2 * r + 1), int(2 * r + 1)):
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(2 * r + 1), int(2 * r + 1)))
            self._ellipse_kernels[int(r)] = k
        return k

    def _ensure(self, h: int, w: int, *, settings: HoleSettings) -> None:
        resized = self._shape != (h, w)
        if resized:
            self._shape = (h, w)
            self._depth_tmp = np.empty((h, w), dtype=np.float32)
            self._depth_filt = np.empty((h, w), dtype=np.float32)
            self._mask01 = np.zeros((h, w), dtype=np.uint8)
            self._mask255 = np.zeros((h, w), dtype=np.uint8)
            self._plane01 = np.zeros((h, w), dtype=np.uint8)
            self._plane255 = np.zeros((h, w), dtype=np.uint8)
            self._plane_normals = np.zeros((h, w, 3), dtype=np.float32)
            self._queue = np.empty((h * w), dtype=np.int32)

            self._gray = np.empty((h, w), dtype=np.uint8)
            self._edge_u8 = np.empty((h, w), dtype=np.uint8)
            self._edge_mask = np.empty((h, w), dtype=np.uint8)
            self._edge_block = np.empty((h, w), dtype=np.uint8)

            self._occ_u8 = np.empty((h, w), dtype=np.uint8)
            self._occ_dilated = np.empty((h, w), dtype=np.uint8)
            self._col_block_combined = np.empty((h, w), dtype=np.uint8)
            self._tmp0 = np.empty((h, w), dtype=np.bool_)
            self._tmp1 = np.empty((h, w), dtype=np.bool_)

            # Scratch
            self._ff_mask = np.empty((h + 2, w + 2), dtype=np.uint8)
            self._dt_pad = np.empty((h + 2, w + 2), dtype=np.uint8)
            self._inv_u8 = np.empty((h, w), dtype=np.uint8)
            self._u8_tmp0 = np.empty((h, w), dtype=np.uint8)
            self._u8_tmp1 = np.empty((h, w), dtype=np.uint8)

            n = int(max(1, int(h) * int(w)))
            self._s_u = np.empty((n,), dtype=np.int32)
            self._s_v = np.empty((n,), dtype=np.int32)
            self._s_z = np.empty((n,), dtype=np.float32)
            self._ellipse_kernels.clear()

        # Kernels depend on settings; update even when window size is unchanged.
        k = int(max(1, int(settings.edge_morph_ksize)))
        if self._edge_kernel.shape != (k, k):
            self._edge_kernel = np.ones((k, k), dtype=np.uint8)

        d = int(max(0, int(settings.edge_dilate_px)))
        if d > 0:
            kd = 2 * d + 1
            if self._edge_dilate_kernel is None or self._edge_dilate_kernel.shape != (kd, kd):
                self._edge_dilate_kernel = np.ones((kd, kd), dtype=np.uint8)
        else:
            self._edge_dilate_kernel = None

        g = int(max(0, int(settings.unknown_guard_px)))
        if g > 0:
            kg = 2 * g + 1
            if self._occ_dilate_kernel is None or self._occ_dilate_kernel.shape != (kg, kg):
                self._occ_dilate_kernel = np.ones((kg, kg), dtype=np.uint8)
        else:
            self._occ_dilate_kernel = None

    def _edge_mask_from_color(self, color_bgr: np.ndarray, *, settings: HoleSettings) -> np.ndarray:
        # Accept either BGR (H,W,3) or grayscale (H,W). Using grayscale avoids an extra cvtColor when the capture
        # stream is already IR/mono.
        if color_bgr.ndim == 2:
            try:
                np.copyto(self._gray, color_bgr)
            except Exception:
                self._gray[:, :] = np.asarray(color_bgr, dtype=np.uint8)
        else:
            cv2.cvtColor(color_bgr, cv2.COLOR_BGR2GRAY, dst=self._gray)
        method = str(getattr(settings, "edge_method", "laplace") or "laplace").strip().lower()
        if method in ("canny", "cn"):
            try:
                lo = int(getattr(settings, "edge_canny_lo", 60))
                hi = int(getattr(settings, "edge_canny_hi", 140))
            except Exception:
                lo, hi = 60, 140
            if int(hi) < int(lo):
                hi = int(lo)
            try:
                l2 = bool(getattr(settings, "edge_canny_L2gradient", True))
            except Exception:
                l2 = True
            cv2.Canny(self._gray, int(lo), int(hi), L2gradient=bool(l2), dst=self._edge_mask)
        else:
            lap = cv2.Laplacian(self._gray, cv2.CV_16S, ksize=int(settings.edge_laplace_ksize))
            cv2.convertScaleAbs(lap, dst=self._edge_u8)
            cv2.threshold(self._edge_u8, int(settings.edge_threshold), 255, cv2.THRESH_BINARY, dst=self._edge_mask)
        cv2.morphologyEx(self._edge_mask, cv2.MORPH_OPEN, self._edge_kernel, dst=self._edge_mask, iterations=int(settings.edge_morph_iter))
        return self._edge_mask

    def _max_inscribed_circle(self, mask01: np.ndarray, z_ref: float, fx: float, fy: float) -> Tuple[float, Tuple[int, int]]:
        if z_ref <= 1e-6:
            return 0.0, (-1, -1)
        if mask01.size == 0 or int(np.count_nonzero(mask01)) == 0:
            return 0.0, (-1, -1)
        dist = cv2.distanceTransform(mask01, cv2.DIST_L2, 3)
        _, max_px, _, max_loc = cv2.minMaxLoc(dist)
        cx, cy = int(max_loc[0]), int(max_loc[1])
        f = 0.5 * (float(fx) + float(fy))
        if f <= 1e-6:
            return 0.0, (cx, cy)
        return float(max_px) * float(z_ref) / f, (cx, cy)

    def _column_mask_from_depth(
        self,
        depth_m: np.ndarray,
        *,
        seed_x: int,
        seed_y: int,
        z_max: float,
        unknown_is_free: bool,
        block_mask: Optional[np.ndarray],
        out_mask: np.ndarray,
        settings: HoleSettings,
    ) -> int:
        H, W = depth_m.shape
        if seed_x < 0 or seed_x >= W or seed_y < 0 or seed_y >= H:
            out_mask.fill(0)
            return 0

        # Optional "unknown guard": block unknown pixels near occupied pixels (0 < depth < z_ref).
        eff_block = block_mask
        g = int(max(0, int(settings.unknown_guard_px)))
        if bool(unknown_is_free) and g > 0 and self._occ_dilate_kernel is not None:
            occ = self._tmp0
            unk = self._tmp1
            np.less(depth_m, float(z_max), out=occ)
            np.greater(depth_m, 0.0, out=unk)
            np.logical_and(occ, unk, out=occ)
            self._occ_u8.fill(0)
            self._occ_u8[occ] = np.uint8(255)
            cv2.dilate(self._occ_u8, self._occ_dilate_kernel, dst=self._occ_dilated, iterations=1)
            unk2 = (depth_m <= 0.0)
            comb = self._col_block_combined
            if eff_block is not None:
                np.copyto(comb, eff_block)
            else:
                comb.fill(0)
            comb[unk2 & (self._occ_dilated != 0)] = np.uint8(255)
            eff_block = comb

        use_one_way = bool(settings.unknown_one_way) and bool(unknown_is_free)
        out_mask.fill(0)
        if eff_block is not None:
            if use_one_way:
                filled = _nb_flood_fill_free_depth_limit_one_way_unknown_blocked(
                    depth_m, float(z_max), bool(unknown_is_free), eff_block, int(seed_x), int(seed_y), out_mask, self._queue
                )
            else:
                filled = _nb_flood_fill_free_depth_limit_blocked(
                    depth_m, float(z_max), bool(unknown_is_free), eff_block, int(seed_x), int(seed_y), out_mask, self._queue
                )
        else:
            if use_one_way:
                filled = _nb_flood_fill_free_depth_limit_one_way_unknown(
                    depth_m, float(z_max), bool(unknown_is_free), int(seed_x), int(seed_y), out_mask, self._queue
                )
            else:
                filled = _nb_flood_fill_free_depth_limit(depth_m, float(z_max), bool(unknown_is_free), int(seed_x), int(seed_y), out_mask, self._queue)
        return int(filled)

    def _eval_at_depth(
        self,
        depth_m: np.ndarray,
        *,
        seed_x: int,
        seed_y: int,
        z_ref: float,
        gate_m: float,
        fx: float,
        fy: float,
        unknown_is_free: bool,
        block_mask: Optional[np.ndarray],
        settings: HoleSettings,
    ) -> Tuple[bool, int, float, Tuple[int, int]]:
        filled = self._column_mask_from_depth(
            depth_m,
            seed_x=seed_x,
            seed_y=seed_y,
            z_max=float(z_ref),
            unknown_is_free=bool(unknown_is_free),
            block_mask=block_mask,
            out_mask=self._mask01,
            settings=settings,
        )
        if filled <= 0:
            return False, 0, 0.0, (-1, -1)
        radius_m, center_px = self._max_inscribed_circle(self._mask01, float(z_ref), float(fx), float(fy))
        if gate_m > 1e-6 and radius_m < gate_m:
            return False, int(filled), float(radius_m), center_px
        return True, int(filled), float(radius_m), center_px

    def _mask_and_deepest_depth(
        self,
        depth_m: np.ndarray,
        *,
        seed_x: int,
        seed_y: int,
        z_search_max: float,
        gate_m: float,
        fx: float,
        fy: float,
        unknown_is_free: bool,
        block_mask: Optional[np.ndarray],
        settings: HoleSettings,
    ) -> Tuple[int, float, float, Tuple[int, int]]:
        H, W = depth_m.shape
        if seed_x < 0 or seed_x >= W or seed_y < 0 or seed_y >= H:
            self._mask01.fill(0)
            return 0, 0.0, 0.0, (-1, -1)

        z_hi = float(max(0.25, float(z_search_max)))
        z_seed = float(depth_m[seed_y, seed_x])
        if z_seed > 0.0:
            z_hi = min(z_hi, max(0.25, z_seed))

        gate_m = float(max(0.0, float(gate_m)))
        if gate_m <= 1e-6:
            ok, filled, radius_m, center_px = self._eval_at_depth(
                depth_m,
                seed_x=seed_x,
                seed_y=seed_y,
                z_ref=z_hi,
                gate_m=0.0,
                fx=fx,
                fy=fy,
                unknown_is_free=unknown_is_free,
                block_mask=block_mask,
                settings=settings,
            )
            if not ok:
                self._mask01.fill(0)
                return 0, 0.0, 0.0, (-1, -1)
            return int(filled), float(z_hi), float(radius_m), center_px

        ok, filled, radius_m, center_px = self._eval_at_depth(
            depth_m,
            seed_x=seed_x,
            seed_y=seed_y,
            z_ref=z_hi,
            gate_m=gate_m,
            fx=fx,
            fy=fy,
            unknown_is_free=unknown_is_free,
            block_mask=block_mask,
            settings=settings,
        )
        if ok:
            return int(filled), float(z_hi), float(radius_m), center_px

        z_lo = 0.25
        if z_hi <= z_lo + 1e-9:
            self._mask01.fill(0)
            return 0, 0.0, 0.0, (-1, -1)

        ok, _, _, _ = self._eval_at_depth(
            depth_m,
            seed_x=seed_x,
            seed_y=seed_y,
            z_ref=z_lo,
            gate_m=gate_m,
            fx=fx,
            fy=fy,
            unknown_is_free=unknown_is_free,
            block_mask=block_mask,
            settings=settings,
        )
        if not ok:
            self._mask01.fill(0)
            return 0, 0.0, 0.0, (-1, -1)

        low = float(z_lo)
        high = float(z_hi)
        for _ in range(int(max(1, int(settings.avail_search_iters)))):
            if (high - low) <= float(settings.avail_search_min_range_m):
                break
            mid = 0.5 * (low + high)
            ok, _, _, _ = self._eval_at_depth(
                depth_m,
                seed_x=seed_x,
                seed_y=seed_y,
                z_ref=float(mid),
                gate_m=gate_m,
                fx=fx,
                fy=fy,
                unknown_is_free=unknown_is_free,
                block_mask=block_mask,
                settings=settings,
            )
            if ok:
                low = float(mid)
            else:
                high = float(mid)

        ok, filled, radius_m, center_px = self._eval_at_depth(
            depth_m,
            seed_x=seed_x,
            seed_y=seed_y,
            z_ref=float(low),
            gate_m=gate_m,
            fx=fx,
            fy=fy,
            unknown_is_free=unknown_is_free,
            block_mask=block_mask,
            settings=settings,
        )
        if not ok:
            self._mask01.fill(0)
            return 0, 0.0, 0.0, (-1, -1)
        return int(filled), float(low), float(radius_m), center_px

    def detect(
        self,
        rgbd: Union[RgbdWindow, dict],
        *,
        window_wh: Tuple[int, int],
        x: int,
        y: int,
        settings: Optional[HoleSettings] = None,
        spatial: Optional[SpatialFilterSettings] = None,
    ) -> np.ndarray:
        self.last_error = ""
        self.last_filled_px = 0
        self.last_depth_avail_m = 0.0
        self.last_radius_m = 0.0
        self.last_center_px = (-1, -1)
        try:
            self.last_edge_barrier_used = False
            self.last_edge_barrier_mode = ""
            self.last_edge_barrier_method = ""
        except Exception:
            pass

        hs = settings if settings is not None else self.settings
        sp = spatial if spatial is not None else self.spatial

        if isinstance(rgbd, dict):
            bgr = rgbd.get("bgr", None)
            gray = rgbd.get("gray", None)
            depth = rgbd.get("depth", rgbd.get("depth_raw", rgbd.get("depth_m", None)))
            depth_units = float(rgbd.get("depth_units", 0.001))
            intr_any = rgbd.get("intr", rgbd.get("intrinsics", None))
        else:
            bgr = rgbd.bgr
            gray = None
            depth = rgbd.depth
            depth_units = float(rgbd.depth_units)
            intr_any = rgbd.intr

        if gray is not None and (not isinstance(gray, np.ndarray) or gray.ndim != 2):
            gray = None
        if bgr is None or (not isinstance(bgr, np.ndarray)) or (bgr.ndim not in (2, 3)) or (bgr.ndim == 3 and bgr.shape[2] != 3):
            bgr = None
        if bgr is None and gray is None:
            self.last_error = "Invalid/missing image (need bgr or gray)."
            return np.zeros((1, 1), dtype=np.uint8)

        depth_m_full = _depth_to_meters(np.asarray(depth), depth_units=depth_units) if isinstance(depth, np.ndarray) else None
        if depth_m_full is None or depth_m_full.ndim != 2:
            self.last_error = "Invalid/missing depth image."
            return np.zeros((1, 1), dtype=np.uint8)

        H, W = int(depth_m_full.shape[0]), int(depth_m_full.shape[1])
        intr = _as_intrinsics(intr_any, fallback_w=W, fallback_h=H)
        if intr is None:
            self.last_error = "Missing/invalid intrinsics (need fx/fy/cx/cy)."
            win_w, win_h = (int(window_wh[0]), int(window_wh[1]))
            if int(win_w) <= 0 or int(win_h) <= 0:
                win_w, win_h = int(W), int(H)
            return np.zeros((int(min(win_h, H)), int(min(win_w, W))), dtype=np.uint8)

        win_w, win_h = int(window_wh[0]), int(window_wh[1])
        x0, y0, ww, hh = _window_bbox(W, H, x=int(x), y=int(y), win_w=win_w, win_h=win_h)

        depth_roi = np.asarray(depth_m_full[y0 : y0 + hh, x0 : x0 + ww], dtype=np.float32)
        if gray is not None:
            color_roi = np.asarray(gray[y0 : y0 + hh, x0 : x0 + ww], dtype=np.uint8)
        else:
            color_roi = np.asarray(bgr[y0 : y0 + hh, x0 : x0 + ww], dtype=np.uint8)
        sx = int(_clamp_int(int(x) - x0, 0, ww - 1))
        sy = int(_clamp_int(int(y) - y0, 0, hh - 1))

        self._ensure(hh, ww, settings=hs)

        depth_f = _apply_spatial_filter(depth_roi, settings=sp, tmp=self._depth_tmp, out=self._depth_filt)

        z_search_max = float(max(0.25, float(hs.search_max_depth_m)))
        gate_m = float(max(0.0, float(hs.min_inscribed_radius_m)))

        fx = float(intr.fx)
        fy = float(intr.fy)
        unknown_is_free = not bool(hs.unknown_as_full)

        block_mask = None
        try:
            mode = str(getattr(hs, "edge_limit_mode", "unknown") or "unknown").strip().lower()
            method = str(getattr(hs, "edge_method", "laplace") or "laplace").strip().lower()
            try:
                self.last_edge_barrier_mode = str(mode)
                self.last_edge_barrier_method = str(method)
            except Exception:
                pass
            mode_always = mode in ("always", "all", "1", "true", "yes", "on")
            mode_seed = mode in ("seed", "seed_unknown", "seed-unknown", "seedonly", "seed_only", "seedunknown")
            do_barrier = bool(getattr(hs, "edge_limit_enabled", True)) and bool(unknown_is_free)
            if bool(do_barrier) and bool(mode_seed):
                do_barrier = bool(0 <= int(sx) < int(ww) and 0 <= int(sy) < int(hh) and float(depth_f[int(sy), int(sx)]) <= 0.0)
            if bool(do_barrier):
                unk = None
                if (not bool(mode_always)) and (not bool(mode_seed)):
                    # unknown mode: apply only on pixels with unknown depth (<=0 / non-finite).
                    try:
                        unk = (~np.isfinite(depth_f)) | (depth_f <= 0.0)
                    except Exception:
                        unk = None
                    if unk is not None and int(np.count_nonzero(unk)) <= 0:
                        do_barrier = False
                if bool(do_barrier):
                    edge_mask = self._edge_mask_from_color(color_roi, settings=hs)
                    if self._edge_dilate_kernel is not None:
                        cv2.dilate(edge_mask, self._edge_dilate_kernel, dst=self._edge_block, iterations=1)
                    else:
                        np.copyto(self._edge_block, edge_mask)
                    if (unk is not None) and (not bool(mode_always)) and (not bool(mode_seed)):
                        self._edge_block[~unk] = np.uint8(0)
                    if int(np.count_nonzero(self._edge_block)) > 0:
                        block_mask = self._edge_block
            try:
                self.last_edge_barrier_used = bool(block_mask is not None)
            except Exception:
                pass
        except Exception:
            block_mask = None
            try:
                self.last_edge_barrier_used = False
                self.last_edge_barrier_mode = ""
                self.last_edge_barrier_method = ""
            except Exception:
                pass

        filled, z_avail, radius_m, center_px = self._mask_and_deepest_depth(
            depth_f,
            seed_x=sx,
            seed_y=sy,
            z_search_max=z_search_max,
            gate_m=gate_m,
            fx=fx,
            fy=fy,
            unknown_is_free=unknown_is_free,
            block_mask=block_mask,
            settings=hs,
        )

        self.last_filled_px = int(filled)
        self.last_depth_avail_m = float(z_avail)
        self.last_radius_m = float(radius_m)
        self.last_center_px = tuple(int(v) for v in center_px)

        _mask01_to_u8(self._mask01, out=self._mask255)
        return self._mask255

    def detect_plane_guided(
        self,
        rgbd: Union[RgbdWindow, dict],
        *,
        window_wh: Tuple[int, int],
        x: int,
        y: int,
        settings: Optional[HoleSettings] = None,
        spatial: Optional[SpatialFilterSettings] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Plane-guided hole detector:
          - Fit a local plane (RANSAC) on an annulus around the seed pixel.
          - Classify pixels as hole if they are sufficiently behind that plane.
          - Return (hole_mask_u8, plane_mask_u8) in window coordinates.
        """
        self.last_error = ""
        try:
            self.last_seed_snapped = False
        except Exception:
            pass
        try:
            self.last_edge_barrier_used = False
            self.last_edge_barrier_mode = ""
            self.last_edge_barrier_method = ""
        except Exception:
            pass
        try:
            self.last_hole_cc_count = 0
            self.last_hole_cc_pick_label = 0
            self.last_hole_cc_pick_r_px = 0.0
            self.last_hole_cc_pick_min_d2 = 0
        except Exception:
            pass
        self.last_filled_px = 0
        self.last_depth_valid_frac = 0.0
        self.last_depth_avail_m = 0.0
        self.last_radius_m = 0.0
        self.last_center_px = (-1, -1)
        self.last_plane_ok = False
        self.last_plane_inliers = 0
        self.last_plane_rms_m = 0.0
        self.last_plane_sector_cov = 0.0
        self.last_plane_n_cam = (0.0, 0.0, 0.0)
        self.last_plane_d_cam = 0.0
        self.last_plane_normals_used = False
        self.last_plane_normals_valid_frac = 0.0
        self.last_circle_center_full_px = (-1, -1)
        self.last_circle_radius_px = 0.0
        self.last_circle_fill_ratio = 0.0
        self.last_hole_valid_frac = 0.0
        self.last_hole_touches_border = False
        try:
            self.last_hole_wall_depth_m = 0.0
            self.last_hole_z_ref_m = 0.0
            self.last_hole_plane_band_in_px = 0
            self.last_hole_plane_band_out_px = 0
        except Exception:
            pass

        hs = settings if settings is not None else self.settings
        sp = spatial if spatial is not None else self.spatial

        if isinstance(rgbd, dict):
            bgr = rgbd.get("bgr", None)
            gray = rgbd.get("gray", None)
            depth = rgbd.get("depth", rgbd.get("depth_raw", rgbd.get("depth_m", None)))
            depth_units = float(rgbd.get("depth_units", 0.001))
            intr_any = rgbd.get("intr", rgbd.get("intrinsics", None))
        else:
            bgr = rgbd.bgr
            gray = None
            depth = rgbd.depth
            depth_units = float(rgbd.depth_units)
            intr_any = rgbd.intr

        # Image is optional for plane-guided detection (depth + intrinsics are sufficient).
        if gray is not None and (not isinstance(gray, np.ndarray) or gray.ndim != 2):
            gray = None
        if bgr is not None and ((not isinstance(bgr, np.ndarray)) or (bgr.ndim not in (2, 3)) or (bgr.ndim == 3 and bgr.shape[2] != 3)):
            bgr = None

        depth_m_full = _depth_to_meters(np.asarray(depth), depth_units=depth_units) if isinstance(depth, np.ndarray) else None
        if depth_m_full is None or depth_m_full.ndim != 2:
            self.last_error = "Invalid/missing depth image."
            z = np.zeros((1, 1), dtype=np.uint8)
            return z, z

        H, W = int(depth_m_full.shape[0]), int(depth_m_full.shape[1])
        intr = _as_intrinsics(intr_any, fallback_w=W, fallback_h=H)
        if intr is None:
            self.last_error = "Missing/invalid intrinsics (need fx/fy/cx/cy)."
            win_w, win_h = (int(window_wh[0]), int(window_wh[1]))
            if int(win_w) <= 0 or int(win_h) <= 0:
                win_w, win_h = int(W), int(H)
            z = np.zeros((int(min(win_h, H)), int(min(win_w, W))), dtype=np.uint8)
            return z, z

        win_w, win_h = int(window_wh[0]), int(window_wh[1])
        x0, y0, ww, hh = _window_bbox(W, H, x=int(x), y=int(y), win_w=win_w, win_h=win_h)
        sx = int(_clamp_int(int(x) - x0, 0, ww - 1))
        sy = int(_clamp_int(int(y) - y0, 0, hh - 1))

        fx = float(intr.fx)
        fy = float(intr.fy)
        cx0 = float(intr.cx)
        cy0 = float(intr.cy)

        depth_roi = np.asarray(depth_m_full[y0 : y0 + hh, x0 : x0 + ww], dtype=np.float32)
        self._ensure(hh, ww, settings=hs)
        depth_f = _apply_spatial_filter(depth_roi, settings=sp, tmp=self._depth_tmp, out=self._depth_filt)
        try:
            v = (depth_f > 0.0) & np.isfinite(depth_f)
            self.last_depth_valid_frac = float(np.count_nonzero(v)) / float(max(1, int(hh) * int(ww)))
        except Exception:
            pass

        if not bool(getattr(hs, "plane_guided_enabled", True)):
            self.last_error = "plane_guided_disabled"
            z = np.zeros((int(hh), int(ww)), dtype=np.uint8)
            return z, z

        use_normals = False
        n_ref = None
        normals_dot_min = 0.0
        N_cam: Optional[np.ndarray] = None
        try:
            normals_enabled = bool(getattr(hs, "plane_normals_enabled", True))
        except Exception:
            normals_enabled = True
        if bool(normals_enabled):
            try:
                n_step = int(max(1, int(getattr(hs, "plane_normals_step_px", 5))))
            except Exception:
                n_step = 5
            try:
                min_mag = float(getattr(hs, "plane_normals_min_mag", 0.45))
            except Exception:
                min_mag = 0.45
            try:
                max_ang = float(getattr(hs, "plane_normals_max_angle_deg", 30.0))
            except Exception:
                max_ang = 30.0
            try:
                normals_dot_min = float(np.cos(np.deg2rad(float(max_ang))))
            except Exception:
                normals_dot_min = float(np.cos(np.deg2rad(30.0)))
            min_mag2 = float(max(0.0, float(min_mag))) ** 2
            try:
                self._plane_normals.fill(0.0)
                _nb_compute_normals_intrinsics(depth_f, int(n_step), float(fx), float(fy), self._plane_normals)
            except Exception:
                normals_enabled = False

        # Use a deterministic RNG per (seed, ROI) to reduce frame-to-frame jitter caused by random sampling.
        try:
            seed_val = (
                (int(x) & 0xFFFF)
                | ((int(y) & 0xFFFF) << 16)
                ^ ((int(W) & 0xFFFF) << 1)
                ^ ((int(H) & 0xFFFF) << 17)
                ^ ((int(ww) & 0xFFFF) << 5)
                ^ ((int(hh) & 0xFFFF) << 21)
            )
            rng = np.random.default_rng(int(seed_val))
        except Exception:
            rng = np.random.default_rng()

        # Annulus sampling for plane fit.
        r_in = int(max(1, int(getattr(hs, "plane_ring_r_in_px", 14))))
        r_out = int(max(r_in + 1, int(getattr(hs, "plane_ring_r_out_px", 90))))
        step = int(max(1, int(getattr(hs, "plane_ring_step_px", 3))))
        max_samples = int(max(32, int(getattr(hs, "plane_ring_max_samples", 2500))))
        r_in2 = int(r_in * r_in)
        r_out2 = int(r_out * r_out)

        iters = int(max(10, int(getattr(hs, "plane_ransac_iters", 120))))
        thr = float(max(1e-6, float(getattr(hs, "plane_inlier_thresh_m", 0.015))))
        min_inliers = int(max(10, int(getattr(hs, "plane_min_inliers", 120))))
        sector_count = int(max(4, int(getattr(hs, "plane_sector_count", 12))))
        sector_min = float(max(0.0, min(1.0, float(getattr(hs, "plane_sector_min_frac", 0.70)))))
        balance = bool(getattr(hs, "plane_ring_balance", True))

        n_ring = 0
        try:
            n_ring = int(
                _nb_collect_annulus_uvz(
                    depth_f,
                    int(sx),
                    int(sy),
                    int(r_in2),
                    int(r_out2),
                    int(step),
                    self._s_u,
                    self._s_v,
                    self._s_z,
                )
            )
        except Exception:
            n_ring = 0
        if int(n_ring) < 12:
            self.last_error = f"plane_ring_pts_small n={int(n_ring)}"
            z = np.zeros((int(hh), int(ww)), dtype=np.uint8)
            return z, z

        uu_a = np.asarray(self._s_u[: int(n_ring)], dtype=np.int32).reshape(-1)
        vv_a = np.asarray(self._s_v[: int(n_ring)], dtype=np.int32).reshape(-1)
        zz_a = np.asarray(self._s_z[: int(n_ring)], dtype=np.float64).reshape(-1)
        # Estimate a robust "dominant plane depth" for the annulus.
        # Use a sector-balanced median (median of per-sector medians) so a single large region (e.g. floor)
        # doesn't dominate when the desired plane is evenly surrounding the seed (e.g. a wall around a hole).
        try:
            z_med_global = float(np.median(zz_a)) if int(zz_a.size) > 0 else float("nan")
        except Exception:
            z_med_global = float("nan")
        try:
            du0 = uu_a.astype(np.int32, copy=False) - int(sx)
            dv0 = vv_a.astype(np.int32, copy=False) - int(sy)
            ang0 = np.arctan2(dv0.astype(np.float64, copy=False), du0.astype(np.float64, copy=False))
            sec0 = np.floor((ang0 + np.pi) * (float(sector_count) / (2.0 * np.pi))).astype(np.int32)
            sec0 = np.mod(sec0, int(sector_count))
            meds: list[float] = []
            for s in range(int(sector_count)):
                idx = np.nonzero(sec0 == int(s))[0]
                if int(idx.size) < 3:
                    continue
                try:
                    meds.append(float(np.median(zz_a[idx])))
                except Exception:
                    continue
            if int(len(meds)) >= max(4, int(round(0.25 * float(sector_count)))):
                z_med_global = float(np.median(np.asarray(meds, dtype=np.float64)))
        except Exception:
            pass
        # Robust depth-band filtering (reject annulus points far from the dominant depth mode).
        try:
            mad_k = float(max(0.0, float(getattr(hs, "plane_depth_mad_k", 3.0))))
            mad_min = float(max(0.0, float(getattr(hs, "plane_depth_mad_min_m", 0.05))))
        except Exception:
            mad_k, mad_min = 3.0, 0.05
        if np.isfinite(z_med_global) and float(mad_k) > 0.0 and int(zz_a.size) >= 32:
            try:
                mad = float(np.median(np.abs(zz_a - float(z_med_global))))
                thr_z = float(max(float(mad_min), float(mad_k) * float(mad)))
                keep = np.abs(zz_a - float(z_med_global)) <= float(thr_z)
                if int(np.count_nonzero(keep)) >= 12:
                    uu_a = uu_a[keep]
                    vv_a = vv_a[keep]
                    zz_a = zz_a[keep]
                    z_med_global = float(np.median(zz_a)) if int(zz_a.size) > 0 else float("nan")
            except Exception:
                pass

        # Sector-balanced sampling (closest points per sector).
        if bool(balance) and int(uu_a.size) > 0:
            try:
                du = uu_a.astype(np.int32, copy=False) - int(sx)
                dv = vv_a.astype(np.int32, copy=False) - int(sy)
                ang = np.arctan2(dv.astype(np.float64, copy=False), du.astype(np.float64, copy=False))
                sec = np.floor((ang + np.pi) * (float(sector_count) / (2.0 * np.pi))).astype(np.int32)
                sec = np.mod(sec, int(sector_count))

                per_sec = int(max(1, int(np.ceil(float(max_samples) / float(max(1, int(sector_count)))))))
                keep_idx: list[np.ndarray] = []
                for s in range(int(sector_count)):
                    idx_s = np.nonzero(sec == int(s))[0]
                    if int(idx_s.size) <= 0:
                        continue
                    if int(idx_s.size) > int(per_sec):
                        z_s = zz_a[idx_s]
                        if np.isfinite(z_med_global):
                            dz = np.abs(z_s - float(z_med_global))
                            sel = np.argpartition(dz, int(per_sec - 1))[: int(per_sec)]
                        else:
                            sel = np.argpartition(z_s, int(per_sec - 1))[: int(per_sec)]
                        idx_s = idx_s[sel]
                    keep_idx.append(np.asarray(idx_s, dtype=np.int32))
                if keep_idx:
                    idx_k = np.unique(np.concatenate(keep_idx, axis=0).astype(np.int32, copy=False))
                    uu_a = uu_a[idx_k]
                    vv_a = vv_a[idx_k]
                    zz_a = zz_a[idx_k]
            except Exception:
                pass

        # Global cap (after sector balancing).
        if int(uu_a.size) > int(max_samples):
            try:
                idx = rng.choice(int(uu_a.size), size=int(max_samples), replace=False)
                uu_a = uu_a[idx]
                vv_a = vv_a[idx]
                zz_a = zz_a[idx]
            except Exception:
                pass

        # Optionally filter annulus points by normal validity and later gate inliers by normal agreement.
        if bool(normals_enabled):
            try:
                N0 = np.asarray(self._plane_normals[vv_a, uu_a, :], dtype=np.float64).reshape(-1, 3)
                mag2 = np.sum(N0 * N0, axis=1)
                try:
                    min_mag = float(getattr(hs, "plane_normals_min_mag", 0.45))
                except Exception:
                    min_mag = 0.45
                min_mag2 = float(max(0.0, float(min_mag))) ** 2
                vN = (mag2 >= float(min_mag2)) & np.isfinite(mag2)
                vN_cnt = int(np.count_nonzero(vN))
                self.last_plane_normals_valid_frac = float(vN_cnt) / float(max(1, int(vN.size)))
                try:
                    min_frac = float(getattr(hs, "plane_normals_min_valid_frac", 0.15))
                except Exception:
                    min_frac = 0.15
                if int(vN_cnt) >= 12 and float(self.last_plane_normals_valid_frac) >= float(min_frac):
                    use_normals = True
                    self.last_plane_normals_used = True
                    uu_a = uu_a[vN]
                    vv_a = vv_a[vN]
                    zz_a = zz_a[vN]
                    N_cam = np.asarray(N0[vN], dtype=np.float64).reshape(-1, 3)
                    try:
                        n_ref = np.mean(N_cam, axis=0).astype(np.float64, copy=False).reshape(3)
                        nn = float(np.linalg.norm(n_ref))
                        if np.isfinite(nn) and nn > 1e-9:
                            n_ref = n_ref / float(nn)
                        else:
                            n_ref = None
                    except Exception:
                        n_ref = None
            except Exception:
                use_normals = False
                n_ref = None
                N_cam = None

        u_full = uu_a.astype(np.float64, copy=False) + float(x0)
        v_full = vv_a.astype(np.float64, copy=False) + float(y0)
        X = (u_full - float(cx0)) * zz_a / float(max(1e-9, fx))
        Y = (v_full - float(cy0)) * zz_a / float(max(1e-9, fy))
        P = np.stack([X, Y, zz_a], axis=1).astype(np.float64, copy=False)

        best_n: Optional[np.ndarray] = None
        best_d = 0.0
        best_inl: Optional[np.ndarray] = None
        best_cnt = 0
        best_z_med = float("nan")
        best_rms = float("inf")

        N = int(P.shape[0])
        # If the window is clipped near borders, available samples can drop; scale min_inliers down in that case.
        try:
            min_inliers = int(min(int(min_inliers), max(10, int(round(0.25 * float(N))))))
        except Exception:
            min_inliers = int(min_inliers)
        early_cnt = int(max(int(min_inliers), int(round(0.85 * float(N))))) if int(N) > 0 else int(min_inliers)

        # Fast path: Numba RANSAC core (no per-iteration dist/inlier allocations).
        try:
            use_nb = bool(_HAVE_NUMBA and bool(getattr(hs, "plane_ransac_numba", True)))
        except Exception:
            use_nb = False
        if bool(use_nb) and int(N) >= 3:
            try:
                idx3 = rng.integers(0, int(N), size=(int(iters), 3), dtype=np.int64)
                out_n = np.zeros((3,), dtype=np.float64)
                out_d = np.zeros((1,), dtype=np.float64)
                use_z_ref = int(1 if np.isfinite(float(z_med_global)) else 0)
                z_ref = float(z_med_global) if int(use_z_ref) != 0 else 0.0
                use_ref = int(1 if n_ref is not None else 0)
                n_ref_nb = np.asarray(n_ref, dtype=np.float64).reshape(3) if n_ref is not None else np.zeros((3,), dtype=np.float64)
                useN = int(1 if (bool(use_normals) and N_cam is not None) else 0)
                N_cam_nb = np.asarray(N_cam, dtype=np.float64).reshape(-1, 3) if N_cam is not None else np.zeros((1, 3), dtype=np.float64)
                cnt_nb, rms_nb, _score_nb = _nb_plane_ransac_best_from_triplets(
                    np.asarray(P, dtype=np.float64),
                    np.asarray(idx3, dtype=np.int64),
                    float(thr),
                    int(early_cnt),
                    int(use_z_ref),
                    float(z_ref),
                    int(useN),
                    np.asarray(N_cam_nb, dtype=np.float64),
                    float(normals_dot_min),
                    int(use_ref),
                    np.asarray(n_ref_nb, dtype=np.float64),
                    out_n,
                    out_d,
                )
                if int(cnt_nb) > 0 and np.all(np.isfinite(out_n)) and np.isfinite(float(out_d[0])):
                    n_nb = np.asarray(out_n, dtype=np.float64).reshape(3)
                    d_nb = float(out_d[0])
                    dist_nb = np.abs(P @ n_nb + float(d_nb))
                    inl_nb = dist_nb <= float(thr)
                    if bool(use_normals) and N_cam is not None:
                        try:
                            inl_nb = inl_nb & ((np.asarray(N_cam, dtype=np.float64) @ n_nb) >= float(normals_dot_min))
                        except Exception:
                            pass
                    cnt2 = int(np.count_nonzero(inl_nb))
                    if int(cnt2) > 0:
                        try:
                            rms2 = float(np.sqrt(float(np.mean((dist_nb[inl_nb] ** 2).astype(np.float64)))))
                        except Exception:
                            rms2 = float(rms_nb)
                        best_n = np.asarray(n_nb, dtype=np.float64).reshape(3)
                        best_d = float(d_nb)
                        best_inl = np.asarray(inl_nb, dtype=np.bool_).reshape(-1)
                        best_cnt = int(cnt2)
                        best_rms = float(rms2)
                        try:
                            best_z_med = float(np.median(P[inl_nb, 2]))
                        except Exception:
                            best_z_med = float("nan")
            except Exception:
                pass
        for _ in range(0 if (best_n is not None and best_inl is not None) else int(iters)):
            try:
                i0, i1, i2 = rng.integers(0, N, size=3)
            except Exception:
                break
            if int(i0) == int(i1) or int(i0) == int(i2) or int(i1) == int(i2):
                continue
            p0 = P[int(i0)]
            p1 = P[int(i1)]
            p2 = P[int(i2)]
            v1 = p1 - p0
            v2 = p2 - p0
            n = np.cross(v1, v2)
            nn = float(np.linalg.norm(n))
            if not np.isfinite(nn) or nn <= 1e-9:
                continue
            n = n / float(nn)
            # Orient normals consistently (in-plane normals are sign-ambiguous).
            try:
                if n_ref is not None:
                    if float(np.dot(n, n_ref)) < 0.0:
                        n = -n
                elif float(n[2]) < 0.0:
                    n = -n
            except Exception:
                pass
            d = -float(np.dot(n, p0))
            dist = np.abs(P @ n + float(d))
            inl = dist <= float(thr)
            if bool(use_normals) and N_cam is not None:
                try:
                    inl = inl & ((N_cam @ n) >= float(normals_dot_min))
                except Exception:
                    pass
            cnt = int(np.count_nonzero(inl))
            if int(cnt) < int(best_cnt):
                continue
            if int(cnt) <= 0:
                continue

            # Fast path: strictly better inlier count wins without computing the median tie-breaker.
            if int(cnt) > int(best_cnt):
                try:
                    rms = float(np.sqrt(float(np.mean((dist[inl] ** 2).astype(np.float64)))))
                except Exception:
                    rms = float("inf")
                best_n = np.asarray(n, dtype=np.float64).reshape(3)
                best_d = float(d)
                best_inl = np.asarray(inl, dtype=np.bool_).reshape(-1)
                best_cnt = int(cnt)
                best_z_med = float("nan")  # compute lazily only if needed for a tie
                best_rms = float(rms)
                if int(best_cnt) >= int(early_cnt) and np.isfinite(float(best_rms)) and float(best_rms) <= float(thr):
                    break
                continue

            # Tie: prefer the plane closest to the dominant annulus depth (and lower RMS as a secondary criterion).
            if best_inl is None:
                continue
            try:
                z_med = float(np.median(P[inl, 2]))
            except Exception:
                z_med = float("inf")
            if not np.isfinite(float(best_z_med)):
                try:
                    best_z_med = float(np.median(P[np.asarray(best_inl, dtype=np.bool_), 2]))
                except Exception:
                    best_z_med = float("inf")
            try:
                rms = float(np.sqrt(float(np.mean((dist[inl] ** 2).astype(np.float64)))))
            except Exception:
                rms = float("inf")

            better = False
            if np.isfinite(z_med_global):
                dz0 = abs(float(z_med) - float(z_med_global))
                dz1 = abs(float(best_z_med) - float(z_med_global))
                if float(dz0) < float(dz1) - 1e-6:
                    better = True
                elif abs(float(dz0) - float(dz1)) <= 1e-6 and float(rms) < float(best_rms):
                    better = True
            else:
                if float(z_med) < float(best_z_med) - 1e-6:
                    better = True
                elif abs(float(z_med) - float(best_z_med)) <= 1e-6 and float(rms) < float(best_rms):
                    better = True
            if not bool(better):
                continue
            best_n = np.asarray(n, dtype=np.float64).reshape(3)
            best_d = float(d)
            best_inl = np.asarray(inl, dtype=np.bool_).reshape(-1)
            best_cnt = int(cnt)
            best_z_med = float(z_med)
            best_rms = float(rms)
            if int(best_cnt) >= int(early_cnt) and np.isfinite(float(best_rms)) and float(best_rms) <= float(thr):
                break

        if best_n is None or best_inl is None or int(best_cnt) < int(min_inliers):
            self.last_error = f"plane_ransac_fail inliers={int(best_cnt)} min={int(min_inliers)}"
            z = np.zeros((int(hh), int(ww)), dtype=np.uint8)
            return z, z

        # Least-squares refit on the RANSAC inliers (stabilizes the plane and reduces mask jitter).
        try:
            Pinl = np.asarray(P[np.asarray(best_inl, dtype=np.bool_)], dtype=np.float64)
            if int(Pinl.shape[0]) >= 3:
                cen = np.mean(Pinl, axis=0)
                Q = Pinl - cen[None, :]
                # Cheaper than SVD on Nx3: compute the smallest eigenvector of the 3x3 covariance matrix.
                C = (Q.T @ Q).astype(np.float64, copy=False)
                _w, V = np.linalg.eigh(C)
                n_ls = np.asarray(V[:, 0], dtype=np.float64).reshape(3)
                nn = float(np.linalg.norm(n_ls))
                if np.isfinite(nn) and nn > 1e-9:
                    n_ls = n_ls / float(nn)
                    try:
                        if n_ref is not None:
                            if float(np.dot(n_ls, n_ref)) < 0.0:
                                n_ls = -n_ls
                        elif float(n_ls[2]) < 0.0:
                            n_ls = -n_ls
                    except Exception:
                        pass
                    d_ls = -float(np.dot(n_ls, cen))
                    dist_all = np.abs(P @ n_ls + float(d_ls))
                    inl_ls = dist_all <= float(thr)
                    if bool(use_normals) and N_cam is not None:
                        try:
                            inl_ls = inl_ls & ((N_cam @ n_ls) >= float(normals_dot_min))
                        except Exception:
                            pass
                    cnt_ls = int(np.count_nonzero(inl_ls))
                    if int(cnt_ls) >= int(min_inliers):
                        try:
                            rms_ls = float(np.sqrt(float(np.mean((dist_all[inl_ls] ** 2).astype(np.float64)))))
                        except Exception:
                            rms_ls = float(best_rms)
                        best_n = np.asarray(n_ls, dtype=np.float64).reshape(3)
                        best_d = float(d_ls)
                        best_inl = np.asarray(inl_ls, dtype=np.bool_).reshape(-1)
                        best_cnt = int(cnt_ls)
                        best_rms = float(rms_ls)
                        try:
                            best_z_med = float(np.median(P[inl_ls, 2]))
                        except Exception:
                            pass
        except Exception:
            pass

        # Sector coverage.
        try:
            du = uu_a.astype(np.int32, copy=False) - int(sx)
            dv = vv_a.astype(np.int32, copy=False) - int(sy)
            ang = np.arctan2(dv.astype(np.float64), du.astype(np.float64))
            sec = np.floor((ang + np.pi) * (float(sector_count) / (2.0 * np.pi))).astype(np.int32)
            sec = np.mod(sec, int(sector_count))
            sec_inl = sec[np.asarray(best_inl, dtype=np.bool_)]
            hit_all = np.zeros((int(sector_count),), dtype=np.uint8)
            hit_all[sec] = np.uint8(1)
            den = int(np.count_nonzero(hit_all))
            hit = np.zeros((int(sector_count),), dtype=np.uint8)
            hit[sec_inl] = np.uint8(1)
            cov = float(np.count_nonzero(hit)) / float(max(1, int(den)))
        except Exception:
            cov = 0.0
        if float(cov) < float(sector_min):
            allow = False
            try:
                allow_inl = int(max(0, int(getattr(hs, "plane_sector_allow_inliers", 0))))
            except Exception:
                allow_inl = 0
            try:
                allow_rms = float(getattr(hs, "plane_sector_allow_rms_m", 0.0))
            except Exception:
                allow_rms = 0.0
            if int(allow_inl) <= 0:
                allow_inl = int(max(0, int(round(2.0 * float(min_inliers)))))
            if not (np.isfinite(float(allow_rms)) and float(allow_rms) > 0.0):
                allow_rms = float(2.0 * float(thr))
            if int(best_cnt) >= int(allow_inl) and float(best_rms) <= float(allow_rms):
                allow = True
            if not bool(allow):
                self.last_error = f"plane_sector_low cov={float(cov):.2f} min={float(sector_min):.2f}"
                z = np.zeros((int(hh), int(ww)), dtype=np.uint8)
                return z, z

        self.last_plane_ok = True
        self.last_plane_inliers = int(best_cnt)
        self.last_plane_rms_m = float(best_rms)
        self.last_plane_sector_cov = float(cov)
        self.last_plane_n_cam = (float(best_n[0]), float(best_n[1]), float(best_n[2]))
        self.last_plane_d_cam = float(best_d)

        # Per-pixel plane intersection depth z_plane(u,v) in ROI coordinates.
        cx_w = float(cx0) - float(x0)
        cy_w = float(cy0) - float(y0)
        xf = (np.arange(int(ww), dtype=np.float64) - float(cx_w)) / float(max(1e-9, fx))
        yf = (np.arange(int(hh), dtype=np.float64) - float(cy_w)) / float(max(1e-9, fy))
        denom = (float(best_n[0]) * xf[None, :]) + (float(best_n[1]) * yf[:, None]) + float(best_n[2])
        with np.errstate(divide="ignore", invalid="ignore"):
            z_plane = (-float(best_d)) / denom
        z_plane = np.asarray(z_plane, dtype=np.float32)

        # Plane mask for visualization.
        dist_thr = float(max(1e-6, float(getattr(hs, "plane_mask_dist_m", 0.02))))
        dist = np.abs(
            (float(best_n[0]) * (xf[None, :] * depth_f))
            + (float(best_n[1]) * (yf[:, None] * depth_f))
            + (float(best_n[2]) * depth_f)
            + float(best_d)
        )
        self._plane01.fill(0)
        self._plane01[(depth_f > 0.0) & np.isfinite(depth_f) & np.isfinite(z_plane) & (z_plane > 0.0) & (dist <= float(dist_thr))] = np.uint8(1)
        _mask01_to_u8(self._plane01, out=self._plane255)

        # Hole candidate.
        hole_margin_abs = float(max(0.0, float(getattr(hs, "hole_margin_abs_m", 0.06))))
        hole_margin_rel = float(max(0.0, float(getattr(hs, "hole_margin_rel", 0.02))))
        margin = np.maximum(np.float32(hole_margin_abs), np.abs(z_plane) * np.float32(hole_margin_rel))
        try:
            rms_k = float(max(0.0, float(getattr(hs, "hole_margin_rms_k", 3.0))))
            margin = np.maximum(margin, np.float32(rms_k) * np.float32(best_rms))
        except Exception:
            pass
        hole_cand = (depth_f > 0.0) & np.isfinite(depth_f) & np.isfinite(z_plane) & (z_plane > 0.0) & (depth_f > (z_plane + margin))
        unknown = (~np.isfinite(depth_f)) | (depth_f <= 0.0)
        if not bool(hs.unknown_as_full):
            hole_cand = hole_cand | (unknown & np.isfinite(z_plane) & (z_plane > 0.0))
        else:
            # Optionally grow hole regions into unknown pixels near deep pixels (fills holes where depth is missing inside the opening).
            try:
                if bool(getattr(hs, "unknown_one_way", True)) and int(getattr(hs, "unknown_guard_px", 0)) > 0:
                    r = int(max(1, int(getattr(hs, "unknown_guard_px", 5))))
                    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(2 * r + 1), int(2 * r + 1)))
                    tmp = np.zeros((int(hh), int(ww)), dtype=np.uint8)
                    tmp[hole_cand] = np.uint8(255)
                    tmp = cv2.dilate(tmp, k, iterations=1)
                    hole_cand = hole_cand | (unknown & (tmp != 0) & np.isfinite(z_plane) & (z_plane > 0.0))
            except Exception:
                pass

        hole_u8 = np.zeros((int(hh), int(ww)), dtype=np.uint8)
        hole_u8[hole_cand] = np.uint8(255)

        # Optional hole-mask cleanup: close small gaps and fill interior voids (stabilizes inscribed circle / center).
        try:
            close_r = int(max(0, int(getattr(hs, "hole_mask_close_px", 0))))
        except Exception:
            close_r = 0
        if int(close_r) > 0 and int(np.count_nonzero(hole_u8)) > 0:
            try:
                k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(2 * close_r + 1), int(2 * close_r + 1)))
                hole_u8 = cv2.morphologyEx(hole_u8, cv2.MORPH_CLOSE, k, iterations=1)
            except Exception:
                pass
        try:
            fill_holes = bool(getattr(hs, "hole_fill_holes", False))
        except Exception:
            fill_holes = False
        if bool(fill_holes) and int(np.count_nonzero(hole_u8)) > 0:
            try:
                # Pad with a 1px background border to guarantee (0,0) is background even if the hole touches the ROI border.
                pad = self._dt_pad
                pad.fill(0)
                pad[1 : int(hh) + 1, 1 : int(ww) + 1] = hole_u8
                bg = np.asarray(pad, dtype=np.uint8).copy()
                ff = np.zeros((int(hh) + 4, int(ww) + 4), dtype=np.uint8)
                cv2.floodFill(bg, ff, (0, 0), 255)
                filled = cv2.bitwise_or(pad, cv2.bitwise_not(bg))
                hole_u8 = np.asarray(filled[1 : int(hh) + 1, 1 : int(ww) + 1], dtype=np.uint8)
            except Exception:
                pass

        # Optional conservative erosion (helps break thin bridges and reduces spurious mask growth).
        try:
            erode_r = int(max(0, int(getattr(hs, "hole_mask_erode_px", 0))))
        except Exception:
            erode_r = 0
        if int(erode_r) > 0 and int(np.count_nonzero(hole_u8)) > 0:
            try:
                k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(2 * erode_r + 1), int(2 * erode_r + 1)))
                hole_u8 = cv2.erode(hole_u8, k, iterations=1)
            except Exception:
                pass

        # Connected components (choose the seed component, or the "best" component near the seed).
        try:
            num, labels = cv2.connectedComponents(hole_u8, connectivity=4)
        except Exception:
            self.last_error = "hole_cc_exc"
            self._mask01.fill(0)
            _mask01_to_u8(self._mask01, out=self._mask255)
            return self._mask255, self._plane255
        if int(num) <= 1 or not isinstance(labels, np.ndarray):
            self.last_error = "hole_cc_fail"
            self._mask01.fill(0)
            _mask01_to_u8(self._mask01, out=self._mask255)
            return self._mask255, self._plane255

        try:
            self.last_hole_cc_count = int(max(0, int(num) - 1))
        except Exception:
            pass

        lab = 0
        try:
            if 0 <= int(sx) < int(ww) and 0 <= int(sy) < int(hh):
                lab = int(labels[int(sy), int(sx)])
        except Exception:
            lab = 0

        if int(lab) <= 0:
            # If the seed isn't inside a hole CC, pick the best CC within `hole_seed_snap_px`:
            # maximize the CC's inscribed circle radius (distance transform), with a proximity gate to avoid jumping.
            try:
                snap_r = int(max(0, int(getattr(hs, "hole_seed_snap_px", 0))))
            except Exception:
                snap_r = 0
            if int(snap_r) <= 0:
                self.last_error = "hole_seed_not_in_hole"
                self._mask01.fill(0)
                _mask01_to_u8(self._mask01, out=self._mask255)
                return self._mask255, self._plane255
            try:
                pad = self._dt_pad
                pad.fill(0)
                pad[1 : int(hh) + 1, 1 : int(ww) + 1] = hole_u8
                dist_img = cv2.distanceTransform(pad, cv2.DIST_L2, 3)
                dist = np.asarray(dist_img[1 : int(hh) + 1, 1 : int(ww) + 1], dtype=np.float32)

                yy, xx = np.nonzero(labels)
                if int(xx.size) <= 0:
                    raise RuntimeError("no_cc_pixels")
                labs = np.asarray(labels[yy, xx], dtype=np.int32).reshape(-1)
                distv = np.asarray(dist[yy, xx], dtype=np.float32).reshape(-1)

                max_dist = np.zeros((int(num),), dtype=np.float32)
                np.maximum.at(max_dist, labs, distv)

                dx = xx.astype(np.int32, copy=False) - int(sx)
                dy = yy.astype(np.int32, copy=False) - int(sy)
                d2 = (dx.astype(np.int64) * dx.astype(np.int64)) + (dy.astype(np.int64) * dy.astype(np.int64))
                min_d2 = np.full((int(num),), np.int64(1 << 60), dtype=np.int64)
                np.minimum.at(min_d2, labs, d2)

                r2 = int(snap_r) * int(snap_r)
                cand = np.nonzero((np.arange(int(num)) > 0) & (min_d2 <= np.int64(r2)) & (max_dist > np.float32(0.0)))[0]
                if int(cand.size) <= 0:
                    raise RuntimeError("no_cc_near_seed")

                # Prefer CCs fully contained within the ROI (components touching the border are often partial/ambiguous).
                try:
                    border_labs = np.unique(
                        np.concatenate(
                            [
                                np.asarray(labels[0, :], dtype=np.int32).reshape(-1),
                                np.asarray(labels[int(hh - 1), :], dtype=np.int32).reshape(-1),
                                np.asarray(labels[:, 0], dtype=np.int32).reshape(-1),
                                np.asarray(labels[:, int(ww - 1)], dtype=np.int32).reshape(-1),
                            ]
                        )
                    )
                    touch = np.zeros((int(num),), dtype=np.uint8)
                    for li in border_labs.tolist():
                        if int(li) > 0 and int(li) < int(num):
                            touch[int(li)] = np.uint8(1)
                    cand_nb = cand[np.nonzero(touch[cand] == 0)[0]]
                    if int(cand_nb.size) > 0:
                        cand = cand_nb
                except Exception:
                    pass

                # Prefer "nearby and large" components instead of only "largest" (reduces jumping to unrelated regions).
                # First stage uses distance-to-nearest-component-pixel (cheap). Second stage refines using the
                # approximate inscribed-circle center distance for the top few candidates.
                d_min_px = np.sqrt(min_d2[cand].astype(np.float64, copy=False))
                try:
                    w = float(getattr(hs, "hole_seed_snap_score_w", 0.60))
                except Exception:
                    w = 0.60
                if not np.isfinite(w):
                    w = 0.60
                w = float(max(0.0, w))
                try:
                    ratio = float(getattr(hs, "hole_seed_snap_max_dist_ratio", 1.40))
                except Exception:
                    ratio = 1.40
                if not np.isfinite(ratio):
                    ratio = 0.0
                if float(ratio) > 0.0:
                    ok = d_min_px <= (float(ratio) * max_dist[cand].astype(np.float64, copy=False))
                    cand2 = cand[np.nonzero(ok)[0]]
                    if int(cand2.size) > 0:
                        cand = cand2
                        d_min_px = np.sqrt(min_d2[cand].astype(np.float64, copy=False))

                score0 = max_dist[cand].astype(np.float64, copy=False) - (float(w) * d_min_px)
                # Take a small shortlist and refine using inscribed-center distance (much better proxy for "clicked inside the hole").
                try:
                    shortlist_k = int(max(1, min(8, int(cand.size))))
                except Exception:
                    shortlist_k = 4
                if int(cand.size) > int(shortlist_k):
                    try:
                        idx0 = np.argpartition(-score0, int(shortlist_k - 1))[: int(shortlist_k)]
                        cand_s = cand[idx0]
                    except Exception:
                        cand_s = cand
                else:
                    cand_s = cand

                best_lab = int(cand_s[0])
                best_score = float("-inf")
                best_d2_center = float("inf")
                try:
                    max_center_px = int(max(0, int(getattr(hs, "hole_seed_snap_max_center_px", 0))))
                except Exception:
                    max_center_px = 0
                max_center2 = float(max_center_px * max_center_px) if int(max_center_px) > 0 else float("inf")
                for lab_i in np.asarray(cand_s, dtype=np.int32).reshape(-1):
                    li = int(lab_i)
                    if li <= 0:
                        continue
                    try:
                        idx_lab = np.nonzero(labs == int(li))[0]
                    except Exception:
                        idx_lab = np.asarray([], dtype=np.int32)
                    if int(idx_lab.size) <= 0:
                        continue
                    try:
                        j = int(idx_lab[int(np.argmax(distv[idx_lab]))])
                    except Exception:
                        continue
                    cx_i = int(xx[int(j)])
                    cy_i = int(yy[int(j)])
                    dx_c = float(cx_i - int(sx))
                    dy_c = float(cy_i - int(sy))
                    d2_c = float(dx_c * dx_c + dy_c * dy_c)
                    if np.isfinite(float(max_center2)) and float(d2_c) > float(max_center2):
                        continue
                    # Optional guard using center distance.
                    if float(ratio) > 0.0:
                        try:
                            r_i = float(max_dist[int(li)])
                        except Exception:
                            r_i = 0.0
                        if np.isfinite(r_i) and float(r_i) > 0.0:
                            if float(d2_c) > float((float(ratio) * float(r_i)) ** 2):
                                continue
                    sc = float(max_dist[int(li)]) - float(w) * float(np.sqrt(d2_c))
                    if sc > float(best_score) or (abs(sc - float(best_score)) <= 1e-9 and d2_c < float(best_d2_center)):
                        best_score = float(sc)
                        best_lab = int(li)
                        best_d2_center = float(d2_c)

                if not np.isfinite(float(best_score)):
                    # Fallback: nothing passed center-distance gating (rare). Use the cheap score0 winner.
                    try:
                        j_best0 = int(np.argmax(score0))
                        best_lab = int(cand[int(j_best0)])
                        best_d2_center = float("inf")
                    except Exception:
                        pass
                lab = int(best_lab)
                try:
                    self.last_seed_snapped = True
                except Exception:
                    pass
                try:
                    self.last_hole_cc_pick_label = int(lab)
                    self.last_hole_cc_pick_r_px = float(max_dist[int(lab)])
                    # Note: keep the legacy field name, but store the center-distance^2 when available (more interpretable).
                    if np.isfinite(float(best_d2_center)):
                        self.last_hole_cc_pick_min_d2 = int(round(float(best_d2_center)))
                    else:
                        self.last_hole_cc_pick_min_d2 = int(min_d2[int(lab)])
                except Exception:
                    pass

                # Move the effective seed to the best-inscribed point inside the chosen component.
                try:
                    idx_lab = np.nonzero(labs == int(lab))[0]
                    if int(idx_lab.size) > 0:
                        j = int(idx_lab[int(np.argmax(distv[idx_lab]))])
                        sx = int(xx[int(j)])
                        sy = int(yy[int(j)])
                except Exception:
                    pass
            except Exception:
                self.last_error = "hole_seed_not_in_hole"
                self._mask01.fill(0)
                _mask01_to_u8(self._mask01, out=self._mask255)
                return self._mask255, self._plane255
        else:
            try:
                self.last_hole_cc_pick_label = int(lab)
                self.last_hole_cc_pick_r_px = 0.0
                self.last_hole_cc_pick_min_d2 = 0
            except Exception:
                pass

        if int(lab) <= 0:
            self.last_error = "hole_cc_fail"
            self._mask01.fill(0)
            _mask01_to_u8(self._mask01, out=self._mask255)
            return self._mask255, self._plane255
        self._mask01.fill(0)
        self._mask01[labels == int(lab)] = np.uint8(1)
        try:
            touches = False
            if int(np.any(self._mask01[0, :])) or int(np.any(self._mask01[int(hh - 1), :])) or int(np.any(self._mask01[:, 0])) or int(np.any(self._mask01[:, int(ww - 1)])):
                touches = True
            self.last_hole_touches_border = bool(touches)
        except Exception:
            pass
        try:
            m = self._mask01 != 0
            if int(np.count_nonzero(m)) > 0:
                vv = (depth_f > 0.0) & np.isfinite(depth_f) & m
                self.last_hole_valid_frac = float(np.count_nonzero(vv)) / float(max(1, int(np.count_nonzero(m))))
        except Exception:
            pass

        self.last_filled_px = int(np.count_nonzero(self._mask01))
        _mask01_to_u8(self._mask01, out=self._mask255)

        # Inscribed circle in pixels.
        # Use a 1px padded mask so that touching the window border is treated as a boundary.
        try:
            pad = self._dt_pad
            pad.fill(0)
            pad[1 : int(hh) + 1, 1 : int(ww) + 1] = self._mask255
            dist_img = cv2.distanceTransform(pad, cv2.DIST_L2, 3)
            _minv, maxv, _minloc, maxloc = cv2.minMaxLoc(dist_img)
            r_px = float(maxv)
            cx_h, cy_h = int(maxloc[0]) - 1, int(maxloc[1]) - 1
            if np.isfinite(float(r_px)) and float(r_px) > 0.0:
                try:
                    thr_px = max(0.0, float(r_px) - 0.5)
                    yy, xx = np.nonzero(np.asarray(dist_img, dtype=np.float32) >= np.float32(thr_px))
                    if int(xx.size) > 0:
                        cx_h = int(round(float(np.mean(xx.astype(np.float64))) - 1.0))
                        cy_h = int(round(float(np.mean(yy.astype(np.float64))) - 1.0))
                except Exception:
                    pass
            cx_h = int(_clamp_int(int(cx_h), 0, int(ww - 1)))
            cy_h = int(_clamp_int(int(cy_h), 0, int(hh - 1)))
        except Exception:
            r_px = 0.0
            cx_h, cy_h = -1, -1
        self.last_circle_radius_px = float(r_px)
        self.last_center_px = (int(cx_h), int(cy_h))
        u_c = int(x0) + int(cx_h)
        v_c = int(y0) + int(cy_h)
        self.last_circle_center_full_px = (int(u_c), int(v_c))

        # Convert to meters using plane depth at center.
        zc = float("nan")
        if 0 <= int(cy_h) < int(hh) and 0 <= int(cx_h) < int(ww):
            zc = float(z_plane[int(cy_h), int(cx_h)])
        if np.isfinite(zc) and zc > 0.0 and np.isfinite(r_px) and float(r_px) > 0.0:
            f = 0.5 * (float(fx) + float(fy))
            self.last_radius_m = float(r_px) * float(zc) / float(max(1e-9, f))
            self.last_depth_avail_m = float(zc)
        else:
            self.last_radius_m = 0.0
            self.last_depth_avail_m = 0.0

        try:
            fill = (2.0 * float(r_px)) / float(max(1, min(int(intr.w), int(intr.h))))
        except Exception:
            fill = 0.0
        self.last_circle_fill_ratio = float(fill)

        return self._mask255, self._plane255

    def detect_hole_plane(
        self,
        rgbd: Union[RgbdWindow, dict],
        *,
        window_wh: Tuple[int, int],
        x: int,
        y: int,
        settings: Optional[HoleSettings] = None,
        spatial: Optional[SpatialFilterSettings] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Hole-first detector:
          1) Detect a hole mask using a depth-threshold flood fill (column detector inspired by `realsence_2d.py`).
             The threshold is derived from a local depth ring around the cursor (reduces the need for a global "min distance" gate).
          2) Fit a plane around the hole using a thin band around the hole boundary (5% inside, 15% outside of hole size).
          3) Fit the max inscribed circle in pixels, then convert to meters using the plane depth at the circle center.

        Returns: (hole_mask_u8, plane_mask_u8) in window coordinates.
        """
        self.last_error = ""
        try:
            self.last_seed_snapped = False
        except Exception:
            pass
        try:
            self.last_hole_cc_count = 0
            self.last_hole_cc_pick_label = 0
            self.last_hole_cc_pick_r_px = 0.0
            self.last_hole_cc_pick_min_d2 = 0
        except Exception:
            pass
        self.last_filled_px = 0
        self.last_depth_valid_frac = 0.0
        self.last_depth_avail_m = 0.0
        self.last_radius_m = 0.0
        self.last_center_px = (-1, -1)
        self.last_plane_ok = False
        self.last_plane_inliers = 0
        self.last_plane_rms_m = 0.0
        self.last_plane_sector_cov = 0.0
        self.last_plane_n_cam = (0.0, 0.0, 0.0)
        self.last_plane_d_cam = 0.0
        self.last_plane_normals_used = False
        self.last_plane_normals_valid_frac = 0.0
        self.last_circle_center_full_px = (-1, -1)
        self.last_circle_radius_px = 0.0
        self.last_circle_fill_ratio = 0.0
        self.last_hole_valid_frac = 0.0
        self.last_hole_touches_border = False
        try:
            self.last_edge_barrier_used = False
            self.last_edge_barrier_mode = ""
            self.last_edge_barrier_method = ""
            self.last_hole_wall_depth_m = 0.0
            self.last_hole_z_ref_m = 0.0
            self.last_hole_plane_band_in_px = 0
            self.last_hole_plane_band_out_px = 0
            self.last_hole_leak_guard_used = False
            self.last_hole_leak_open_px = 0
        except Exception:
            pass

        hs = settings if settings is not None else self.settings
        sp = spatial if spatial is not None else self.spatial

        if isinstance(rgbd, dict):
            bgr = rgbd.get("bgr", None)
            gray = rgbd.get("gray", None)
            depth = rgbd.get("depth", rgbd.get("depth_raw", rgbd.get("depth_m", None)))
            depth_units = float(rgbd.get("depth_units", 0.001))
            intr_any = rgbd.get("intr", rgbd.get("intrinsics", None))
        else:
            bgr = rgbd.bgr
            gray = None
            depth = rgbd.depth
            depth_units = float(rgbd.depth_units)
            intr_any = rgbd.intr

        if gray is not None and (not isinstance(gray, np.ndarray) or gray.ndim != 2):
            gray = None
        if bgr is None or (not isinstance(bgr, np.ndarray)) or (bgr.ndim not in (2, 3)) or (bgr.ndim == 3 and bgr.shape[2] != 3):
            bgr = None
        if bgr is None and gray is None:
            self.last_error = "Invalid/missing image (need bgr or gray)."
            z = np.zeros((1, 1), dtype=np.uint8)
            return z, z

        depth_m_full = _depth_to_meters(np.asarray(depth), depth_units=depth_units) if isinstance(depth, np.ndarray) else None
        if depth_m_full is None or depth_m_full.ndim != 2:
            self.last_error = "Invalid/missing depth image."
            z = np.zeros((1, 1), dtype=np.uint8)
            return z, z

        H, W = int(depth_m_full.shape[0]), int(depth_m_full.shape[1])
        intr = _as_intrinsics(intr_any, fallback_w=W, fallback_h=H)
        if intr is None:
            self.last_error = "Missing/invalid intrinsics (need fx/fy/cx/cy)."
            win_w, win_h = (int(window_wh[0]), int(window_wh[1]))
            if int(win_w) <= 0 or int(win_h) <= 0:
                win_w, win_h = int(W), int(H)
            z = np.zeros((int(min(win_h, H)), int(min(win_w, W))), dtype=np.uint8)
            return z, z

        fx = float(intr.fx)
        fy = float(intr.fy)
        cx0 = float(intr.cx)
        cy0 = float(intr.cy)
        if float(fx) <= 0.0 or float(fy) <= 0.0:
            self.last_error = "intr_bad"
            z = np.zeros((1, 1), dtype=np.uint8)
            return z, z

        win_w, win_h = int(window_wh[0]), int(window_wh[1])
        x0, y0, ww, hh = _window_bbox(W, H, x=int(x), y=int(y), win_w=win_w, win_h=win_h)
        sx = int(_clamp_int(int(x) - x0, 0, ww - 1))
        sy = int(_clamp_int(int(y) - y0, 0, hh - 1))

        depth_roi = np.asarray(depth_m_full[y0 : y0 + hh, x0 : x0 + ww], dtype=np.float32)
        if gray is not None:
            color_roi = np.asarray(gray[y0 : y0 + hh, x0 : x0 + ww], dtype=np.uint8)
        else:
            color_roi = np.asarray(bgr[y0 : y0 + hh, x0 : x0 + ww], dtype=np.uint8)
        self._ensure(hh, ww, settings=hs)
        depth_f = _apply_spatial_filter(depth_roi, settings=sp, tmp=self._depth_tmp, out=self._depth_filt)
        try:
            v = (depth_f > 0.0) & np.isfinite(depth_f)
            self.last_depth_valid_frac = float(np.count_nonzero(v)) / float(max(1, int(hh) * int(ww)))
        except Exception:
            pass

        # --- 1) Hole mask via depth-threshold flood fill ---
        unknown_is_free = not bool(getattr(hs, "unknown_as_full", False))

        # Estimate a local "wall depth" from an annulus around the cursor.
        # We use this only to set a reasonable depth threshold for the flood fill; the final hole mask is later
        # refined using the fitted wall plane (keeps the detector from "snapping" onto walls when depth is noisy).
        z_seed = float("nan")
        try:
            z_seed = float(depth_f[int(sy), int(sx)])
        except Exception:
            z_seed = float("nan")
        r_in = int(max(1, int(getattr(hs, "plane_ring_r_in_px", 14))))
        r_out = int(max(r_in + 1, int(getattr(hs, "plane_ring_r_out_px", 150))))
        step = int(max(1, int(getattr(hs, "plane_ring_step_px", 3))))
        r_in2 = int(r_in * r_in)
        r_out2 = int(r_out * r_out)
        z_wall = float("nan")
        n_ring = 0
        try:
            n_ring = int(
                _nb_collect_annulus_uvz(
                    depth_f,
                    int(sx),
                    int(sy),
                    int(r_in2),
                    int(r_out2),
                    int(step),
                    self._s_u,
                    self._s_v,
                    self._s_z,
                )
            )
        except Exception:
            n_ring = 0
        if int(n_ring) >= 16:
            try:
                z_arr = np.asarray(self._s_z[: int(n_ring)], dtype=np.float32).reshape(-1)
                z_arr = z_arr[z_arr > np.float32(0.0)]
                if int(z_arr.size) >= 16:
                    z_max0 = float(max(0.25, float(getattr(hs, "search_max_depth_m", 15.0))))
                    z_arr = np.clip(z_arr, 0.25, z_max0)
                    # Pick a "wall-like" depth mode from the annulus: choose the nearer of the top-2 modes.
                    bw = 0.05  # meters
                    nb = int(max(8, min(80, int(np.ceil((z_max0 - 0.25) / float(bw))))))
                    hist, edges = np.histogram(z_arr, bins=int(nb), range=(0.25, float(z_max0)))
                    try:
                        order = np.argsort(hist.astype(np.int64, copy=False))[::-1]
                        j0 = int(order[0]) if int(order.size) > 0 else int(np.argmax(hist))
                        j1 = int(order[1]) if int(order.size) > 1 else int(j0)
                        c0 = 0.5 * (float(edges[int(j0)]) + float(edges[int(j0 + 1)]))
                        c1 = 0.5 * (float(edges[int(j1)]) + float(edges[int(j1 + 1)]))
                        j = int(j0 if float(c0) <= float(c1) else j1)
                    except Exception:
                        j = int(np.argmax(hist))
                    lo = float(edges[int(j)])
                    hi = float(edges[int(j + 1)])
                    in_bin = z_arr[(z_arr >= float(lo)) & (z_arr <= float(hi))]
                    z_wall = float(np.median(in_bin)) if int(in_bin.size) > 0 else float(np.median(z_arr))
                else:
                    z_wall = float("nan")
            except Exception:
                z_wall = float("nan")
        if (not np.isfinite(z_wall)) and np.isfinite(z_seed) and float(z_seed) > 0.0:
            # Ring failed (very sparse depth); fall back to seed depth.
            z_wall = float(z_seed)

        # Depth threshold for "hole"/free space: behind the local wall depth by a small margin.
        hole_margin_abs = float(max(0.0, float(getattr(hs, "hole_margin_abs_m", 0.06))))
        hole_margin_rel = float(max(0.0, float(getattr(hs, "hole_margin_rel", 0.02))))
        if np.isfinite(z_wall) and float(z_wall) > 0.0:
            z_ref = float(z_wall) + float(max(float(hole_margin_abs), float(hole_margin_rel) * float(z_wall)))
            try:
                z_max = float(max(0.25, float(getattr(hs, "search_max_depth_m", 15.0))))
                z_ref = float(min(float(z_ref), float(z_max)))
            except Exception:
                pass
        else:
            # Fallback: treat only very-far pixels as hole (usually only unknown depth will pass).
            z_ref = float(max(0.25, float(getattr(hs, "search_max_depth_m", 15.0))))
        try:
            self.last_hole_wall_depth_m = float(z_wall) if (np.isfinite(float(z_wall)) and float(z_wall) > 0.0) else 0.0
            self.last_hole_z_ref_m = float(z_ref) if (np.isfinite(float(z_ref)) and float(z_ref) > 0.0) else 0.0
        except Exception:
            pass

        # Seed snapping: if the clicked pixel is not "free", snap to a nearby free pixel.
        seed_x = int(sx)
        seed_y = int(sy)
        try:
            snap_px = int(max(0, int(getattr(hs, "hole_seed_snap_px", 0))))
        except Exception:
            snap_px = 0

        def _is_free_px(ix: int, iy: int) -> bool:
            if ix < 0 or ix >= int(ww) or iy < 0 or iy >= int(hh):
                return False
            z = float(depth_f[int(iy), int(ix)])
            if (not np.isfinite(z)) or z <= 0.0:
                return bool(unknown_is_free)
            return bool(z >= float(z_ref))

        if not _is_free_px(int(seed_x), int(seed_y)) and int(snap_px) > 0:
            best_d2 = None
            best_xy = None
            x_lo = int(max(0, int(seed_x) - int(snap_px)))
            x_hi = int(min(int(ww - 1), int(seed_x) + int(snap_px)))
            y_lo = int(max(0, int(seed_y) - int(snap_px)))
            y_hi = int(min(int(hh - 1), int(seed_y) + int(snap_px)))
            for yy in range(int(y_lo), int(y_hi) + 1):
                dy = int(yy) - int(seed_y)
                for xx in range(int(x_lo), int(x_hi) + 1):
                    dx = int(xx) - int(seed_x)
                    d2 = int(dx * dx + dy * dy)
                    if int(d2) > int(snap_px) * int(snap_px):
                        continue
                    if not _is_free_px(int(xx), int(yy)):
                        continue
                    if best_d2 is None or int(d2) < int(best_d2):
                        best_d2 = int(d2)
                        best_xy = (int(xx), int(yy))
                        if int(d2) <= 0:
                            break
                if best_d2 is not None and int(best_d2) <= 0:
                    break
            if best_xy is not None:
                seed_x, seed_y = int(best_xy[0]), int(best_xy[1])
                try:
                    self.last_seed_snapped = True
                except Exception:
                    pass

        if not _is_free_px(int(seed_x), int(seed_y)):
            self.last_error = "hole_seed_not_free"
            self._mask01.fill(0)
            self._mask255.fill(0)
            self._plane01.fill(0)
            self._plane255.fill(0)
            return self._mask255, self._plane255

        # Optional image-edge barrier ("virtual walls" for the hole flood fill).
        block_mask = None
        try:
            mode = str(getattr(hs, "edge_limit_mode", "unknown") or "unknown").strip().lower()
            method = str(getattr(hs, "edge_method", "laplace") or "laplace").strip().lower()
            try:
                self.last_edge_barrier_mode = str(mode)
                self.last_edge_barrier_method = str(method)
            except Exception:
                pass
            mode_always = mode in ("always", "all", "1", "true", "yes", "on")
            mode_seed = mode in ("seed", "seed_unknown", "seed-unknown", "seedonly", "seed_only", "seedunknown")
            do_barrier = bool(getattr(hs, "edge_limit_enabled", True)) and bool(unknown_is_free)
            if bool(do_barrier) and bool(mode_seed):
                do_barrier = bool(
                    0 <= int(seed_x) < int(ww) and 0 <= int(seed_y) < int(hh) and float(depth_f[int(seed_y), int(seed_x)]) <= 0.0
                )
            if bool(do_barrier):
                unk = None
                if (not bool(mode_always)) and (not bool(mode_seed)):
                    # unknown mode: apply only on pixels with unknown depth (<=0 / non-finite).
                    try:
                        unk = (~np.isfinite(depth_f)) | (depth_f <= 0.0)
                    except Exception:
                        unk = None
                    if unk is not None and int(np.count_nonzero(unk)) <= 0:
                        do_barrier = False
                if bool(do_barrier):
                    edge_mask = self._edge_mask_from_color(color_roi, settings=hs)
                    if self._edge_dilate_kernel is not None:
                        cv2.dilate(edge_mask, self._edge_dilate_kernel, dst=self._edge_block, iterations=1)
                    else:
                        np.copyto(self._edge_block, edge_mask)
                    if (unk is not None) and (not bool(mode_always)) and (not bool(mode_seed)):
                        self._edge_block[~unk] = np.uint8(0)
                    if int(np.count_nonzero(self._edge_block)) > 0:
                        block_mask = self._edge_block
            try:
                self.last_edge_barrier_used = bool(block_mask is not None)
            except Exception:
                pass
        except Exception:
            block_mask = None
            try:
                self.last_edge_barrier_used = False
                self.last_edge_barrier_mode = ""
                self.last_edge_barrier_method = ""
            except Exception:
                pass

        # Prefer a wall-relative threshold (`z_ref`) to capture the aperture.
        # The "deepest passing depth" search can lock onto far background / global free space (effectively "the whole frame"),
        # which is not desirable for hole selection.
        try:
            z_search_max = float(max(0.25, float(getattr(hs, "search_max_depth_m", 15.0))))
        except Exception:
            z_search_max = 15.0
        try:
            gate_m = float(max(0.0, float(getattr(hs, "min_inscribed_radius_m", 0.20))))
        except Exception:
            gate_m = 0.20
        if float(gate_m) <= 1e-6:
            # Keep the search well-posed even when the seed depth is unknown (0).
            gate_m = 0.05

        filled0 = 0
        try:
            filled0 = self._column_mask_from_depth(
                depth_f,
                seed_x=int(seed_x),
                seed_y=int(seed_y),
                z_max=float(z_ref),
                unknown_is_free=bool(unknown_is_free),
                block_mask=block_mask,
                out_mask=self._mask01,
                settings=hs,
            )
        except Exception:
            filled0 = 0
        filled = int(filled0)
        z_avail = float(z_ref)
        radius_m = 0.0
        center_px = (-1, -1)
        nz_mask = int(np.count_nonzero(self._mask01))
        if int(filled) > 0 and int(nz_mask) > 0:
            try:
                radius_m, center_px = self._max_inscribed_circle(self._mask01, float(z_avail), float(fx), float(fy))
            except Exception:
                radius_m, center_px = 0.0, (-1, -1)
        ok_gate = bool(int(filled) > 0) and int(nz_mask) > 0

        if not bool(ok_gate):
            # Fallback: deepest passing depth search (useful when z_ref is poor / depth is extremely sparse).
            filled, z_avail, radius_m, center_px = self._mask_and_deepest_depth(
                depth_f,
                seed_x=int(seed_x),
                seed_y=int(seed_y),
                z_search_max=float(z_search_max),
                gate_m=float(gate_m),
                fx=float(fx),
                fy=float(fy),
                unknown_is_free=bool(unknown_is_free),
                block_mask=block_mask,
                settings=hs,
            )
            if int(filled) <= 0 or int(np.count_nonzero(self._mask01)) <= 0:
                self.last_error = "hole_fill_empty"
                self._mask01.fill(0)
                self._mask255.fill(0)
                self._plane01.fill(0)
                self._plane255.fill(0)
                return self._mask255, self._plane255
            try:
                self.last_error = "hole_fill_fallback_deepest"
            except Exception:
                pass
        try:
            self.last_depth_avail_m = float(z_avail)
            self.last_radius_m = float(radius_m)
            self.last_center_px = (int(center_px[0]), int(center_px[1]))
        except Exception:
            pass

        # Optional cleanup (close + fill holes + erode).
        try:
            close_px = int(max(0, int(getattr(hs, "hole_mask_close_px", 0))))
            erode_px = int(max(0, int(getattr(hs, "hole_mask_erode_px", 0))))
            fill_holes = bool(getattr(hs, "hole_fill_holes", True))
        except Exception:
            close_px, erode_px, fill_holes = 0, 0, True
        try:
            mask_u8 = self._mask255
            _mask01_to_u8(self._mask01, out=mask_u8)

            # Leak guard (pre-cleanup): if the free-space flood fill leaks through a tiny gap, it can connect the
            # opening to a large background region and touch the ROI border. Break thin connections by applying a
            # small morphological opening and keeping only the component nearest to the seed.
            try:
                self.last_hole_leak_guard_used = False
                self.last_hole_leak_open_px = 0
            except Exception:
                pass
            try:
                do_leak_guard = bool(getattr(hs, "leak_guard_enabled", True))
            except Exception:
                do_leak_guard = True
            touch_raw = False
            try:
                if bool(do_leak_guard):
                    touch_raw = bool(
                        np.any(mask_u8[0, :] != 0)
                        or np.any(mask_u8[-1, :] != 0)
                        or np.any(mask_u8[:, 0] != 0)
                        or np.any(mask_u8[:, -1] != 0)
                    )
            except Exception:
                touch_raw = False
            if bool(do_leak_guard) and bool(touch_raw) and int(np.count_nonzero(mask_u8)) > 0:
                max_open = int(max(0, int(getattr(hs, "leak_open_max_px", 6))))
                snap_px = int(max(0, int(getattr(hs, "hole_seed_snap_px", 0))))
                if int(max_open) > 0:
                    best_r = 0
                    for r in range(1, int(max_open) + 1):
                        k = self._kernel_ellipse(int(r))
                        opened = self._u8_tmp0
                        cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, k, dst=opened, iterations=1)
                        if int(np.count_nonzero(opened)) <= 0:
                            continue
                        try:
                            num, labels = cv2.connectedComponents(opened, connectivity=4)
                        except Exception:
                            continue
                        if int(num) <= 1 or not isinstance(labels, np.ndarray):
                            continue
                        lab = 0
                        try:
                            if 0 <= int(seed_x) < int(ww) and 0 <= int(seed_y) < int(hh):
                                lab = int(labels[int(seed_y), int(seed_x)])
                        except Exception:
                            lab = 0
                        if int(lab) <= 0 and int(snap_px) > 0:
                            sx_i = int(seed_x)
                            sy_i = int(seed_y)
                            best_d2 = None
                            best_lab = 0
                            x_lo = int(max(0, int(sx_i) - int(snap_px)))
                            x_hi = int(min(int(ww - 1), int(sx_i) + int(snap_px)))
                            y_lo = int(max(0, int(sy_i) - int(snap_px)))
                            y_hi = int(min(int(hh - 1), int(sy_i) + int(snap_px)))
                            for yy in range(int(y_lo), int(y_hi) + 1):
                                dy = int(yy) - int(sy_i)
                                for xx in range(int(x_lo), int(x_hi) + 1):
                                    if opened[int(yy), int(xx)] == 0:
                                        continue
                                    dx = int(xx) - int(sx_i)
                                    d2 = int(dx * dx + dy * dy)
                                    if int(d2) > int(snap_px) * int(snap_px):
                                        continue
                                    if best_d2 is None or int(d2) < int(best_d2):
                                        best_d2 = int(d2)
                                        best_lab = int(labels[int(yy), int(xx)])
                                        if int(d2) <= 0:
                                            break
                                if best_d2 is not None and int(best_d2) <= 0:
                                    break
                            lab = int(best_lab)
                        if int(lab) <= 0:
                            continue
                        try:
                            opened[labels != int(lab)] = np.uint8(0)
                        except Exception:
                            continue
                        try:
                            touch2 = bool(
                                np.any(opened[0, :] != 0)
                                or np.any(opened[-1, :] != 0)
                                or np.any(opened[:, 0] != 0)
                                or np.any(opened[:, -1] != 0)
                            )
                        except Exception:
                            touch2 = True
                        if bool(touch2):
                            continue
                        best_r = int(r)
                        # Copy the repaired mask back in-place.
                        mask_u8[:, :] = np.asarray(opened, dtype=np.uint8)
                        break

                    if int(best_r) > 0:
                        try:
                            self.last_hole_leak_guard_used = True
                            self.last_hole_leak_open_px = int(best_r)
                        except Exception:
                            pass
            if int(close_px) > 0:
                k = self._kernel_ellipse(int(close_px))
                cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, k, dst=self._u8_tmp0, iterations=1)
                mask_u8[:, :] = np.asarray(self._u8_tmp0, dtype=np.uint8)
            if bool(fill_holes):
                cv2.bitwise_not(mask_u8, dst=self._inv_u8)
                self._ff_mask.fill(0)
                cv2.floodFill(self._inv_u8, self._ff_mask, (0, 0), 255)
                cv2.bitwise_not(self._inv_u8, dst=self._u8_tmp0)
                cv2.bitwise_or(mask_u8, self._u8_tmp0, dst=mask_u8)
            if int(erode_px) > 0:
                k = self._kernel_ellipse(int(erode_px))
                cv2.erode(mask_u8, k, dst=self._u8_tmp0, iterations=1)
                mask_u8[:, :] = np.asarray(self._u8_tmp0, dtype=np.uint8)
            # Sync back to mask01/mask255
            self._mask01.fill(0)
            self._mask01[mask_u8 != 0] = np.uint8(1)
            self._mask255[:, :] = np.asarray(mask_u8, dtype=np.uint8)
        except Exception:
            _mask01_to_u8(self._mask01, out=self._mask255)

        self.last_filled_px = int(np.count_nonzero(self._mask01))
        try:
            okz = (self._mask01 != 0) & (depth_f > 0.0) & np.isfinite(depth_f)
            self.last_hole_valid_frac = float(np.count_nonzero(okz)) / float(max(1, int(self.last_filled_px)))
        except Exception:
            pass
        try:
            m0 = np.asarray(self._mask01, dtype=np.uint8)
            touch = bool(np.any(m0[0, :] != 0) or np.any(m0[-1, :] != 0) or np.any(m0[:, 0] != 0) or np.any(m0[:, -1] != 0))
            self.last_hole_touches_border = bool(touch)
        except Exception:
            pass
        # If the mask still touches the ROI border and covers (almost) the whole ROI, treat it as a leak.
        # This avoids selecting "the whole frame" as the opening when the flood fill escapes.
        try:
            if bool(getattr(hs, "leak_guard_enabled", True)) and bool(getattr(self, "last_hole_touches_border", False)):
                area = int(hh) * int(ww)
                if int(area) > 0:
                    fill_frac = float(getattr(self, "last_filled_px", 0)) / float(max(1, int(area)))
                    try:
                        leak_border_fill_frac = float(getattr(hs, "leak_border_fill_frac", 0.25))
                    except Exception:
                        leak_border_fill_frac = 0.25
                    try:
                        leak_full_fill_frac = float(getattr(hs, "leak_full_fill_frac", 0.95))
                    except Exception:
                        leak_full_fill_frac = 0.95
                    if not np.isfinite(float(leak_border_fill_frac)):
                        leak_border_fill_frac = 0.25
                    if not np.isfinite(float(leak_full_fill_frac)):
                        leak_full_fill_frac = 0.95
                    leak_border_fill_frac = float(max(0.0, min(1.0, float(leak_border_fill_frac))))
                    leak_full_fill_frac = float(max(0.0, min(1.0, float(leak_full_fill_frac))))

                    # Heuristic: on full-frame detection, any large border-touching mask is almost always a leak
                    # (jumping between the real hole and "the whole frame"). When using a smaller ROI window, a
                    # valid hole can legitimately touch the window border, so we keep the strong threshold there.
                    full_frame = False
                    try:
                        full_frame = (
                            int(x0) == 0
                            and int(y0) == 0
                            and intr is not None
                            and int(ww) == int(getattr(intr, "w", -1))
                            and int(hh) == int(getattr(intr, "h", -1))
                        )
                    except Exception:
                        full_frame = False

                    leak_thr = float(leak_border_fill_frac) if bool(full_frame) else float(leak_full_fill_frac)
                    if float(fill_frac) >= float(leak_thr):
                        err_kind = "hole_leak_border" if bool(full_frame) and float(leak_thr) < float(leak_full_fill_frac) else "hole_leak_full"
                        self.last_error = f"{err_kind} fill={float(fill_frac):.2f}"
                        self._mask01.fill(0)
                        self._mask255.fill(0)
                        self._plane01.fill(0)
                        self._plane255.fill(0)
                        return self._mask255, self._plane255
        except Exception:
            pass

        # Inscribed circle in pixels.
        try:
            pad = self._dt_pad
            pad.fill(0)
            pad[1 : int(hh) + 1, 1 : int(ww) + 1] = self._mask255
            dist_img = cv2.distanceTransform(pad, cv2.DIST_L2, 3)
            _minv, maxv, _minloc, maxloc = cv2.minMaxLoc(dist_img)
            r_px = float(maxv)
            cx_h, cy_h = int(maxloc[0]) - 1, int(maxloc[1]) - 1
            cx_h = int(_clamp_int(int(cx_h), 0, int(ww - 1)))
            cy_h = int(_clamp_int(int(cy_h), 0, int(hh - 1)))
        except Exception:
            r_px = 0.0
            cx_h, cy_h = -1, -1
        self.last_circle_radius_px = float(r_px)
        self.last_center_px = (int(cx_h), int(cy_h))
        u_c = int(x0) + int(cx_h)
        v_c = int(y0) + int(cy_h)
        self.last_circle_center_full_px = (int(u_c), int(v_c))
        try:
            fill = (2.0 * float(r_px)) / float(max(1, min(int(intr.w), int(intr.h))))
        except Exception:
            fill = 0.0
        self.last_circle_fill_ratio = float(fill)

        if not (np.isfinite(r_px) and float(r_px) >= 2.0 and 0 <= int(cx_h) < int(ww) and 0 <= int(cy_h) < int(hh)):
            self.last_error = "hole_circle_small"
            self._plane255.fill(0)
            return self._mask255, self._plane255

        # --- 2) Plane fit around the hole boundary ---
        # Band = 5% inside, 15% outside of hole size (pixel radius).
        band_in_px = int(max(1, int(round(0.05 * float(r_px)))))
        band_out_px = int(max(1, int(round(0.15 * float(r_px)))))

        # Distance transforms for band selection.
        try:
            dist_in = cv2.distanceTransform(np.asarray(self._mask255, dtype=np.uint8), cv2.DIST_L2, 3).astype(np.float32, copy=False)
        except Exception:
            dist_in = np.zeros((int(hh), int(ww)), dtype=np.float32)
        try:
            cv2.bitwise_not(self._mask255, dst=self._inv_u8)
            dist_out = cv2.distanceTransform(self._inv_u8, cv2.DIST_L2, 3).astype(np.float32, copy=False)
        except Exception:
            dist_out = np.zeros((int(hh), int(ww)), dtype=np.float32)

        # Collect plane samples: prefer the OUTSIDE band only (plane should be outside the opening).
        # If the hole boundary is surrounded by unknown depth, expand the outside band until we hit valid depth.
        # Fall back to including a thin inside band only if outside still has too few valid samples.
        try:
            m_h = self._mask255 != 0
            ring_in = m_h & (dist_in <= np.float32(float(band_in_px)))
            valid = (depth_f > 0.0) & np.isfinite(depth_f)
            band_out_try = int(band_out_px)
            ring_out = (~m_h) & (dist_out <= np.float32(float(band_out_try)))
            yy, xx = np.nonzero(ring_out & valid)

            try:
                min_pts = int(max(12, min(256, int(getattr(hs, "plane_min_inliers", 120)))))
            except Exception:
                min_pts = 120
            max_band_out = int(min(200, max(int(band_out_try), int(round(0.75 * float(r_px))), 40)))
            while int(xx.size) < int(min_pts) and int(band_out_try) < int(max_band_out):
                band_out_try = int(min(int(max_band_out), max(int(band_out_try) + 1, int(round(1.5 * float(band_out_try))))))
                ring_out = (~m_h) & (dist_out <= np.float32(float(band_out_try)))
                yy, xx = np.nonzero(ring_out & valid)

            band_out_px = int(band_out_try)
            try:
                self.last_hole_plane_band_in_px = int(band_in_px)
                self.last_hole_plane_band_out_px = int(band_out_px)
            except Exception:
                pass

            if int(xx.size) < 12:
                yy, xx = np.nonzero((ring_out | ring_in) & valid)
        except Exception:
            yy, xx = np.zeros((0,), dtype=np.int32), np.zeros((0,), dtype=np.int32)
        if int(xx.size) < 12:
            self.last_error = f"plane_ring_pts_small n={int(xx.size)}"
            self._plane255.fill(0)
            return self._mask255, self._plane255

        # Downsample uniformly to a cap.
        max_samples = int(max(64, int(getattr(hs, "plane_ring_max_samples", 2500))))
        if int(xx.size) > int(max_samples):
            try:
                rng = np.random.default_rng(int((int(x) & 0xFFFF) | ((int(y) & 0xFFFF) << 16)))
                idx = rng.choice(int(xx.size), size=int(max_samples), replace=False)
                xx = xx[idx]
                yy = yy[idx]
            except Exception:
                xx = xx[: int(max_samples)]
                yy = yy[: int(max_samples)]

        # Compute 3D points in camera frame.
        u_full = xx.astype(np.float64, copy=False) + float(x0)
        v_full = yy.astype(np.float64, copy=False) + float(y0)
        zz_a = np.asarray(depth_f[yy, xx], dtype=np.float64).reshape(-1)
        X = (u_full - float(cx0)) * zz_a / float(max(1e-9, fx))
        Y = (v_full - float(cy0)) * zz_a / float(max(1e-9, fy))
        P = np.stack([X, Y, zz_a], axis=1).astype(np.float64, copy=False)

        # Optional normals gating.
        use_normals = False
        n_ref = None
        normals_dot_min = 0.0
        N_cam = None
        try:
            normals_enabled = bool(getattr(hs, "plane_normals_enabled", True))
        except Exception:
            normals_enabled = True
        if bool(normals_enabled):
            try:
                n_step = int(max(1, int(getattr(hs, "plane_normals_step_px", 5))))
            except Exception:
                n_step = 5
            try:
                min_mag = float(getattr(hs, "plane_normals_min_mag", 0.45))
            except Exception:
                min_mag = 0.45
            try:
                max_ang = float(getattr(hs, "plane_normals_max_angle_deg", 30.0))
            except Exception:
                max_ang = 30.0
            try:
                normals_dot_min = float(np.cos(np.deg2rad(float(max_ang))))
            except Exception:
                normals_dot_min = float(np.cos(np.deg2rad(30.0)))
            min_mag2 = float(max(0.0, float(min_mag))) ** 2
            try:
                self._plane_normals.fill(0.0)
                _nb_compute_normals_intrinsics(depth_f, int(n_step), float(fx), float(fy), self._plane_normals)
                N0 = np.asarray(self._plane_normals[yy, xx, :], dtype=np.float64).reshape(-1, 3)
                mag2 = np.sum(N0 * N0, axis=1)
                vN = (mag2 >= float(min_mag2)) & np.isfinite(mag2)
                vN_cnt = int(np.count_nonzero(vN))
                self.last_plane_normals_valid_frac = float(vN_cnt) / float(max(1, int(vN.size)))
                try:
                    min_frac = float(getattr(hs, "plane_normals_min_valid_frac", 0.15))
                except Exception:
                    min_frac = 0.15
                if int(vN_cnt) >= 12 and float(self.last_plane_normals_valid_frac) >= float(min_frac):
                    use_normals = True
                    self.last_plane_normals_used = True
                    P = P[vN]
                    xx = xx[vN]
                    yy = yy[vN]
                    N_cam = np.asarray(N0[vN], dtype=np.float64).reshape(-1, 3)
                    try:
                        n_ref = np.mean(N_cam, axis=0).astype(np.float64, copy=False).reshape(3)
                        nn = float(np.linalg.norm(n_ref))
                        if np.isfinite(nn) and nn > 1e-9:
                            n_ref = n_ref / float(nn)
                        else:
                            n_ref = None
                    except Exception:
                        n_ref = None
            except Exception:
                use_normals = False
                n_ref = None
                N_cam = None

        # RANSAC plane fit.
        best_n: Optional[np.ndarray] = None
        best_d = 0.0
        best_inl: Optional[np.ndarray] = None
        best_cnt = 0
        best_rms = float("inf")

        iters = int(max(10, int(getattr(hs, "plane_ransac_iters", 120))))
        thr = float(max(1e-6, float(getattr(hs, "plane_inlier_thresh_m", 0.025))))
        min_inliers = int(max(10, int(getattr(hs, "plane_min_inliers", 120))))
        sector_count = int(max(4, int(getattr(hs, "plane_sector_count", 12))))
        sector_min = float(max(0.0, min(1.0, float(getattr(hs, "plane_sector_min_frac", 0.70)))))
        Np = int(P.shape[0])
        try:
            min_inliers = int(min(int(min_inliers), max(10, int(round(0.25 * float(Np))))))
        except Exception:
            min_inliers = int(min_inliers)
        early_cnt = int(max(int(min_inliers), int(round(0.85 * float(Np))))) if int(Np) > 0 else int(min_inliers)

        try:
            rng = np.random.default_rng(int((int(x) & 0xFFFF) | ((int(y) & 0xFFFF) << 16) ^ ((int(Np) & 0xFFFF) << 3)))
        except Exception:
            rng = np.random.default_rng()

        # Fast path: Numba RANSAC core (no per-iteration dist/inlier allocations).
        try:
            use_nb = bool(_HAVE_NUMBA and bool(getattr(hs, "plane_ransac_numba", True)))
        except Exception:
            use_nb = False
        if bool(use_nb) and int(Np) >= 3:
            try:
                idx3 = rng.integers(0, int(Np), size=(int(iters), 3), dtype=np.int64)
                out_n = np.zeros((3,), dtype=np.float64)
                out_d = np.zeros((1,), dtype=np.float64)
                use_ref = int(1 if n_ref is not None else 0)
                n_ref_nb = np.asarray(n_ref, dtype=np.float64).reshape(3) if n_ref is not None else np.zeros((3,), dtype=np.float64)
                useN = int(1 if (bool(use_normals) and N_cam is not None) else 0)
                N_cam_nb = np.asarray(N_cam, dtype=np.float64).reshape(-1, 3) if N_cam is not None else np.zeros((1, 3), dtype=np.float64)
                cnt_nb, rms_nb = _nb_plane_ransac_best_cnt_rms(
                    np.asarray(P, dtype=np.float64),
                    np.asarray(idx3, dtype=np.int64),
                    float(thr),
                    int(early_cnt),
                    int(useN),
                    np.asarray(N_cam_nb, dtype=np.float64),
                    float(normals_dot_min),
                    int(use_ref),
                    np.asarray(n_ref_nb, dtype=np.float64),
                    out_n,
                    out_d,
                )
                if int(cnt_nb) > 0 and np.all(np.isfinite(out_n)) and np.isfinite(float(out_d[0])):
                    n_nb = np.asarray(out_n, dtype=np.float64).reshape(3)
                    d_nb = float(out_d[0])
                    dist_nb = np.abs(P @ n_nb + float(d_nb))
                    inl_nb = dist_nb <= float(thr)
                    if bool(use_normals) and N_cam is not None:
                        try:
                            inl_nb = inl_nb & ((np.asarray(N_cam, dtype=np.float64) @ n_nb) >= float(normals_dot_min))
                        except Exception:
                            pass
                    cnt2 = int(np.count_nonzero(inl_nb))
                    if int(cnt2) > 0:
                        try:
                            rms2 = float(np.sqrt(float(np.mean((dist_nb[inl_nb] ** 2).astype(np.float64)))))
                        except Exception:
                            rms2 = float(rms_nb)
                        best_n = np.asarray(n_nb, dtype=np.float64).reshape(3)
                        best_d = float(d_nb)
                        best_inl = np.asarray(inl_nb, dtype=np.bool_).reshape(-1)
                        best_cnt = int(cnt2)
                        best_rms = float(rms2)
            except Exception:
                pass

        for _ in range(0 if (best_n is not None and best_inl is not None) else int(iters)):
            try:
                i0, i1, i2 = rng.integers(0, int(Np), size=3)
            except Exception:
                break
            if int(i0) == int(i1) or int(i0) == int(i2) or int(i1) == int(i2):
                continue
            p0 = P[int(i0)]
            p1 = P[int(i1)]
            p2 = P[int(i2)]
            v1 = p1 - p0
            v2 = p2 - p0
            n = np.cross(v1, v2)
            nn = float(np.linalg.norm(n))
            if not np.isfinite(nn) or nn <= 1e-9:
                continue
            n = n / float(nn)
            try:
                if n_ref is not None:
                    if float(np.dot(n, n_ref)) < 0.0:
                        n = -n
                elif float(n[2]) < 0.0:
                    n = -n
            except Exception:
                pass
            d = -float(np.dot(n, p0))
            dist = np.abs(P @ n + float(d))
            inl = dist <= float(thr)
            if bool(use_normals) and N_cam is not None:
                try:
                    inl = inl & ((N_cam @ n) >= float(normals_dot_min))
                except Exception:
                    pass
            cnt = int(np.count_nonzero(inl))
            if int(cnt) < int(best_cnt):
                continue
            if int(cnt) <= 0:
                continue
            try:
                rms = float(np.sqrt(float(np.mean((dist[inl] ** 2).astype(np.float64)))))
            except Exception:
                rms = float("inf")
            better = False
            if int(cnt) > int(best_cnt):
                better = True
            elif int(cnt) == int(best_cnt) and float(rms) < float(best_rms):
                better = True
            if not bool(better):
                continue
            best_n = np.asarray(n, dtype=np.float64).reshape(3)
            best_d = float(d)
            best_inl = np.asarray(inl, dtype=np.bool_).reshape(-1)
            best_cnt = int(cnt)
            best_rms = float(rms)
            if int(best_cnt) >= int(early_cnt) and np.isfinite(float(best_rms)) and float(best_rms) <= float(thr):
                break

        if best_n is None or best_inl is None or int(best_cnt) < int(min_inliers):
            self.last_error = f"plane_ransac_fail inliers={int(best_cnt)} min={int(min_inliers)}"
            self._plane255.fill(0)
            return self._mask255, self._plane255

        # Least-squares refit on inliers.
        try:
            Pinl = np.asarray(P[np.asarray(best_inl, dtype=np.bool_)], dtype=np.float64)
            if int(Pinl.shape[0]) >= 3:
                cen = np.mean(Pinl, axis=0)
                Q = Pinl - cen[None, :]
                C = (Q.T @ Q).astype(np.float64, copy=False)
                _w, V = np.linalg.eigh(C)
                n_ls = np.asarray(V[:, 0], dtype=np.float64).reshape(3)
                nn = float(np.linalg.norm(n_ls))
                if np.isfinite(nn) and nn > 1e-9:
                    n_ls = n_ls / float(nn)
                    try:
                        if n_ref is not None:
                            if float(np.dot(n_ls, n_ref)) < 0.0:
                                n_ls = -n_ls
                        elif float(n_ls[2]) < 0.0:
                            n_ls = -n_ls
                    except Exception:
                        pass
                    d_ls = -float(np.dot(n_ls, cen))
                    dist_all = np.abs(P @ n_ls + float(d_ls))
                    inl_ls = dist_all <= float(thr)
                    if bool(use_normals) and N_cam is not None:
                        try:
                            inl_ls = inl_ls & ((N_cam @ n_ls) >= float(normals_dot_min))
                        except Exception:
                            pass
                    cnt_ls = int(np.count_nonzero(inl_ls))
                    if int(cnt_ls) >= int(min_inliers):
                        try:
                            rms_ls = float(np.sqrt(float(np.mean((dist_all[inl_ls] ** 2).astype(np.float64)))))
                        except Exception:
                            rms_ls = float(best_rms)
                        best_n = np.asarray(n_ls, dtype=np.float64).reshape(3)
                        best_d = float(d_ls)
                        best_inl = np.asarray(inl_ls, dtype=np.bool_).reshape(-1)
                        best_cnt = int(cnt_ls)
                        best_rms = float(rms_ls)
        except Exception:
            pass

        # Sector coverage around hole center.
        try:
            du = xx.astype(np.int32, copy=False) - int(cx_h)
            dv = yy.astype(np.int32, copy=False) - int(cy_h)
            ang = np.arctan2(dv.astype(np.float64), du.astype(np.float64))
            sec = np.floor((ang + np.pi) * (float(sector_count) / (2.0 * np.pi))).astype(np.int32)
            sec = np.mod(sec, int(sector_count))
            sec_inl = sec[np.asarray(best_inl, dtype=np.bool_)]
            hit_all = np.zeros((int(sector_count),), dtype=np.uint8)
            hit_all[sec] = np.uint8(1)
            den = int(np.count_nonzero(hit_all))
            hit = np.zeros((int(sector_count),), dtype=np.uint8)
            hit[sec_inl] = np.uint8(1)
            cov = float(np.count_nonzero(hit)) / float(max(1, int(den)))
        except Exception:
            cov = 0.0
        if float(cov) < float(sector_min):
            self.last_error = f"plane_sector_low cov={float(cov):.2f} min={float(sector_min):.2f}"
            self._plane255.fill(0)
            return self._mask255, self._plane255

        self.last_plane_ok = True
        self.last_plane_inliers = int(best_cnt)
        self.last_plane_rms_m = float(best_rms)
        self.last_plane_sector_cov = float(cov)
        self.last_plane_n_cam = (float(best_n[0]), float(best_n[1]), float(best_n[2]))
        self.last_plane_d_cam = float(best_d)

        # Refine the coarse hole mask by requiring hole pixels (with valid depth) to lie *behind* the fitted plane.
        # This greatly reduces false positives where the flood fill leaks onto the wall plane.
        try:
            cx_w = float(cx0) - float(x0)
            cy_w = float(cy0) - float(y0)

            # Plane depth at circle center (used to convert pixel radius -> meters).
            denom = (float(best_n[0]) * ((float(u_c) - float(cx0)) / float(max(1e-9, fx)))) + (
                float(best_n[1]) * ((float(v_c) - float(cy0)) / float(max(1e-9, fy)))
            ) + float(best_n[2])
            zc = float("nan")
            if np.isfinite(float(denom)) and abs(float(denom)) > 1e-9:
                zc = (-float(best_d)) / float(denom)

            # Behind-plane margin: abs/rel + k*rms (robust to plane fit noise).
            rms_k = float(max(0.0, float(getattr(hs, "hole_margin_rms_k", 3.0))))
            margin = float(max(float(hole_margin_abs), float(hole_margin_rel) * float(zc if np.isfinite(zc) and zc > 0.0 else z_wall), float(rms_k) * float(best_rms)))
            margin = float(max(0.0, margin))

            # Keep original mask/circle in case refinement becomes unstable (bad plane fit / bad depth).
            orig_mask01 = self._u8_tmp0
            np.copyto(orig_mask01, self._mask01)
            orig_count = int(np.count_nonzero(orig_mask01))
            r_px0 = float(r_px)
            cx_h0 = int(cx_h)
            cy_h0 = int(cy_h)
            u_c0 = int(u_c)
            v_c0 = int(v_c)

            keep_valid01 = self._u8_tmp1
            keep_unknown01 = self._plane01
            _nb_refine_hole_keep_masks_signed_distance(
                depth_f,
                orig_mask01,
                float(fx),
                float(fy),
                float(cx_w),
                float(cy_w),
                float(best_n[0]),
                float(best_n[1]),
                float(best_n[2]),
                float(best_d),
                float(margin),
                keep_valid01,
                keep_unknown01,
            )

            np.copyto(self._mask01, keep_valid01)
            if bool(unknown_is_free):
                # Keep unknown pixels only when they're close to valid behind-plane pixels; this suppresses rare
                # "unknown-only" leak paths that can connect the opening to large missing-depth regions.
                try:
                    g = int(max(0, int(getattr(hs, "unknown_guard_px", 0))))
                except Exception:
                    g = 0
                if (
                    int(g) > 0
                    and self._occ_dilate_kernel is not None
                    and int(np.count_nonzero(keep_valid01)) > 0
                    and int(np.count_nonzero(keep_unknown01)) > 0
                ):
                    np.multiply(keep_valid01, np.uint8(255), out=self._occ_u8, casting="unsafe")
                    cv2.dilate(self._occ_u8, self._occ_dilate_kernel, dst=self._occ_dilated, iterations=1)
                    _nb_mask01_keep_where_occ_dilated(keep_unknown01, self._occ_dilated)
                self._mask01 |= keep_unknown01
            _mask01_to_u8(self._mask01, out=self._mask255)

            # Recompute circle after refinement (important for stable GUI + PID gating).
            try:
                pad = self._dt_pad
                pad.fill(0)
                pad[1 : int(hh) + 1, 1 : int(ww) + 1] = self._mask255
                dist_img = cv2.distanceTransform(pad, cv2.DIST_L2, 3)
                _minv, maxv, _minloc, maxloc = cv2.minMaxLoc(dist_img)
                r_px = float(maxv)
                cx_h, cy_h = int(maxloc[0]) - 1, int(maxloc[1]) - 1
                cx_h = int(_clamp_int(int(cx_h), 0, int(ww - 1)))
                cy_h = int(_clamp_int(int(cy_h), 0, int(hh - 1)))
            except Exception:
                r_px = 0.0
                cx_h, cy_h = -1, -1
            self.last_circle_radius_px = float(r_px)
            self.last_center_px = (int(cx_h), int(cy_h))
            u_c = int(x0) + int(cx_h)
            v_c = int(y0) + int(cy_h)
            self.last_circle_center_full_px = (int(u_c), int(v_c))
            try:
                fill = (2.0 * float(r_px)) / float(max(1, min(int(intr.w), int(intr.h))))
            except Exception:
                fill = 0.0
            self.last_circle_fill_ratio = float(fill)

            # If refinement collapses the mask/circle, revert to the pre-refine mask (keeps the detector usable).
            refined_ok = True
            try:
                new_count = int(np.count_nonzero(self._mask01))
                if int(orig_count) > 0:
                    if int(new_count) < int(max(16, int(round(0.25 * float(orig_count))))):
                        refined_ok = False
                if bool(refined_ok) and np.isfinite(float(r_px0)) and float(r_px0) > 1e-6:
                    if not (np.isfinite(float(r_px)) and float(r_px) >= 0.5 * float(r_px0)):
                        refined_ok = False
            except Exception:
                refined_ok = True

            if not bool(refined_ok):
                try:
                    self.last_error = "hole_refine_revert"
                except Exception:
                    pass
                # Restore original mask.
                np.copyto(self._mask01, orig_mask01)
                _mask01_to_u8(self._mask01, out=self._mask255)
                # Restore circle.
                r_px = float(r_px0)
                cx_h = int(cx_h0)
                cy_h = int(cy_h0)
                u_c = int(u_c0)
                v_c = int(v_c0)
                self.last_circle_radius_px = float(r_px)
                self.last_center_px = (int(cx_h), int(cy_h))
                self.last_circle_center_full_px = (int(u_c), int(v_c))
                try:
                    fill0 = (2.0 * float(r_px)) / float(max(1, min(int(intr.w), int(intr.h))))
                except Exception:
                    fill0 = 0.0
                self.last_circle_fill_ratio = float(fill0)
                # Use plane depth at the original circle center.
                if np.isfinite(zc) and float(zc) > 0.0:
                    self.last_depth_avail_m = float(zc)
                    f = 0.5 * (float(fx) + float(fy))
                    self.last_radius_m = float(r_px) * float(zc) / float(max(1e-9, f))
                else:
                    self.last_depth_avail_m = 0.0
                    self.last_radius_m = 0.0
            else:
                # Recompute plane depth at the refined center.
                denom2 = (float(best_n[0]) * ((float(u_c) - float(cx0)) / float(max(1e-9, fx)))) + (
                    float(best_n[1]) * ((float(v_c) - float(cy0)) / float(max(1e-9, fy)))
                ) + float(best_n[2])
                zc2 = float("nan")
                if np.isfinite(float(denom2)) and abs(float(denom2)) > 1e-9:
                    zc2 = (-float(best_d)) / float(denom2)
                if np.isfinite(zc2) and zc2 > 0.0:
                    self.last_depth_avail_m = float(zc2)
                    f = 0.5 * (float(fx) + float(fy))
                    self.last_radius_m = float(r_px) * float(zc2) / float(max(1e-9, f))
                else:
                    self.last_depth_avail_m = 0.0
                    self.last_radius_m = 0.0
        except Exception:
            # Fallback: keep the coarse mask + circle and still compute plane depth at the coarse center.
            denom = (float(best_n[0]) * ((float(u_c) - float(cx0)) / float(max(1e-9, fx)))) + (
                float(best_n[1]) * ((float(v_c) - float(cy0)) / float(max(1e-9, fy)))
            ) + float(best_n[2])
            zc = float("nan")
            if np.isfinite(float(denom)) and abs(float(denom)) > 1e-9:
                zc = (-float(best_d)) / float(denom)
            if np.isfinite(zc) and zc > 0.0:
                self.last_depth_avail_m = float(zc)
                f = 0.5 * (float(fx) + float(fy))
                self.last_radius_m = float(r_px) * float(zc) / float(max(1e-9, f))
            else:
                self.last_depth_avail_m = 0.0
                self.last_radius_m = 0.0

        # Plane mask for visualization: pixels near the plane AND near the hole boundary (outer band).
        try:
            dist_thr = float(max(1e-6, float(getattr(hs, "plane_mask_dist_m", 0.02))))
            # Recompute dist_out against the refined mask (keeps the visual plane band tight).
            cx_w = float(cx0) - float(x0)
            cy_w = float(cy0) - float(y0)
            cv2.bitwise_not(self._mask255, dst=self._inv_u8)
            dist_out = cv2.distanceTransform(self._inv_u8, cv2.DIST_L2, 3).astype(np.float32, copy=False)
            self._plane01.fill(0)
            _nb_plane_mask_from_plane(
                depth_f,
                dist_out,
                float(band_out_px),
                float(fx),
                float(fy),
                float(cx_w),
                float(cy_w),
                float(best_n[0]),
                float(best_n[1]),
                float(best_n[2]),
                float(best_d),
                float(dist_thr),
                self._plane01,
            )
            _mask01_to_u8(self._plane01, out=self._plane255)
        except Exception:
            self._plane01.fill(0)
            _mask01_to_u8(self._plane01, out=self._plane255)

        return self._mask255, self._plane255
