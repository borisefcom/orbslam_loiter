"""Polygon tracker (VO-projection + LK edge-point lock) in a separate process.

This tracker:
- Detects a polygon from a user bbox (via Canny + contour + convex hull).
- Lifts polygon vertices (and many sampled edge points) to world coordinates using aligned depth.
- Each frame: projects those world points using current VO pose (Twc) to get a predicted polygon.
- Runs MedianFlow-style LK (FW/BW) + ZNCC gates on sampled edge points to estimate a rigid 2D similarity correction.
- Applies the correction to the VO-projected polygon (shape cannot deform).
"""

from __future__ import annotations

# NOTE: VO-specific logic stays in this process; VO-agnostic vision helpers live under `rpi_position_hold/vision/`
# so we can swap VO backends later without rewriting the tracker.

import time
import queue
import threading
from dataclasses import replace
from typing import Optional

import cv2
import numpy as np

try:
    from numba import njit  # type: ignore

    _HAVE_NUMBA = True
except Exception:
    njit = None  # type: ignore
    _HAVE_NUMBA = False

from ipc.shm_ring import ShmRing, ShmRingSpec
from ipc.shm_state import ShmOdomState, ShmStateSpec

from ..config import _set_cv_threads
from ..odom_core import _sample_depth_stats_u16
from ..rgbd_detector import HoleDetector, HoleSettings, SpatialFilterSettings, _window_bbox
from ..utils import _intrinsics_from_dict, _setup_stdio_logging
from ..vision.edge_snap import EdgeSnapConfig, edge_normal_snap_poly_uv
from ..vision.feature_acquire import BandMaskConfig, FeatureAcquireConfig, LineAcquireConfig, OrbAcquireConfig, PolyFeatureAcquirer
from .common import _proc_setup_signals


def _wrap_pi(a: float) -> float:
    try:
        x = float(a)
    except Exception:
        return 0.0
    x = (x + np.pi) % (2.0 * np.pi) - np.pi
    return float(x)


def _sim_affine_from_params(*, tx: float, ty: float, theta_rad: float, log_s: float) -> np.ndarray:
    th = float(theta_rad)
    s = float(np.exp(float(log_s)))
    ct = float(np.cos(th))
    st = float(np.sin(th))
    return np.asarray([[s * ct, -s * st, float(tx)], [s * st, s * ct, float(ty)]], dtype=np.float64).reshape(2, 3)


def _sim_params_from_affine(*, A: np.ndarray) -> Optional[tuple[float, float, float, float]]:
    try:
        M = np.asarray(A, dtype=np.float64).reshape(2, 3)
    except Exception:
        return None
    a, b, tx = float(M[0, 0]), float(M[0, 1]), float(M[0, 2])
    c, d, ty = float(M[1, 0]), float(M[1, 1]), float(M[1, 2])
    if not (np.isfinite(a) and np.isfinite(b) and np.isfinite(c) and np.isfinite(d) and np.isfinite(tx) and np.isfinite(ty)):
        return None
    s = float(np.sqrt(max(1e-12, 0.5 * ((a * a + c * c) + (b * b + d * d)))))
    if not np.isfinite(s) or s <= 1e-9:
        return None
    th = float(np.arctan2(c, a))
    log_s = float(np.log(s))
    return float(tx), float(ty), float(_wrap_pi(th)), float(log_s)


def _affine_to_similarity(*, A: np.ndarray) -> Optional[np.ndarray]:
    try:
        M = np.asarray(A, dtype=np.float64).reshape(2, 3)
    except Exception:
        return None
    R = np.asarray(M[:, :2], dtype=np.float64).reshape(2, 2)
    t = np.asarray(M[:, 2], dtype=np.float64).reshape(2)
    if not (np.isfinite(R).all() and np.isfinite(t).all()):
        return None
    try:
        U, S, Vt = np.linalg.svd(R)
        R0 = (U @ Vt).reshape(2, 2)
        if float(np.linalg.det(R0)) < 0.0:
            Vt2 = np.asarray(Vt, dtype=np.float64).copy()
            Vt2[-1, :] *= -1.0
            R0 = (U @ Vt2).reshape(2, 2)
        s = float(np.mean(np.asarray(S, dtype=np.float64)))
    except Exception:
        return None
    if not np.isfinite(s) or s <= 1e-9:
        return None
    return np.asarray([[s * R0[0, 0], s * R0[0, 1], t[0]], [s * R0[1, 0], s * R0[1, 1], t[1]]], dtype=np.float64).reshape(2, 3)


def _bbox_from_poly_uv(
    *, poly_uv: np.ndarray, w: int, h: int, margin_px: int = 0, min_size_px: int = 2
) -> Optional[tuple[int, int, int, int]]:
    try:
        pts = np.asarray(poly_uv, dtype=np.float64).reshape(-1, 2)
    except Exception:
        return None
    if pts.size <= 0 or int(pts.shape[0]) < 3:
        return None
    if not np.isfinite(pts).all():
        return None
    try:
        x0 = float(np.min(pts[:, 0]))
        x1 = float(np.max(pts[:, 0]))
        y0 = float(np.min(pts[:, 1]))
        y1 = float(np.max(pts[:, 1]))
    except Exception:
        return None
    m = int(max(0, int(margin_px)))
    x0i = int(max(0, min(int(w - 1), int(np.floor(x0)) - m)))
    y0i = int(max(0, min(int(h - 1), int(np.floor(y0)) - m)))
    x1i = int(max(x0i + 1, min(int(w), int(np.ceil(x1)) + m + 1)))
    y1i = int(max(y0i + 1, min(int(h), int(np.ceil(y1)) + m + 1)))
    if int(x1i - x0i) < int(min_size_px) or int(y1i - y0i) < int(min_size_px):
        return None
    return int(x0i), int(y0i), int(x1i), int(y1i)


def _poly_select_from_bbox(
    *,
    gray: np.ndarray,
    bbox: tuple[int, int, int, int],
    canny_lo: int,
    canny_hi: int,
    edge_kernel,
    concavity_max: float,
    approx_eps_frac: float,
    min_vertex_angle_deg: float,
    max_vertices: int,
    min_edge_len_px: float,
) -> Optional[np.ndarray]:
    x0, y0, x1, y1 = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
    if not (x0 < x1 and y0 < y1):
        return None
    try:
        roi = np.asarray(gray[int(y0) : int(y1 + 1), int(x0) : int(x1 + 1)], dtype=np.uint8)
    except Exception:
        return None
    if int(roi.size) <= 0 or int(roi.shape[0]) < 12 or int(roi.shape[1]) < 12:
        return None
    try:
        e_roi = cv2.Canny(roi, int(canny_lo), int(canny_hi), L2gradient=True)
        if edge_kernel is not None:
            e_roi = cv2.dilate(e_roi, edge_kernel, iterations=1)
        fc_res = cv2.findContours(e_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if isinstance(fc_res, tuple) and int(len(fc_res)) == 3:
            _img, conts, _hier = fc_res
        else:
            conts, _hier = fc_res
    except Exception:
        conts = []
    if not conts:
        return None
    c = max(conts, key=lambda cc: float(cv2.contourArea(cc)))
    try:
        area = float(cv2.contourArea(c))
    except Exception:
        area = 0.0
    if float(area) < 25.0:
        return None
    try:
        c_full = np.asarray(c, dtype=np.int32).copy()
        c_full[:, 0, 0] += int(x0)
        c_full[:, 0, 1] += int(y0)
        hull = cv2.convexHull(c_full)
        concavity = 0.0
        try:
            a_c = float(cv2.contourArea(c_full))
            a_h = float(cv2.contourArea(hull))
            if float(a_h) > 1e-9:
                concavity = float(max(0.0, 1.0 - (float(a_c) / float(a_h))))
        except Exception:
            concavity = 0.0
        conc_max = float(max(0.0, min(0.99, float(concavity_max))))
        c_use = hull if float(concavity) > float(conc_max) else c_full
        peri = float(cv2.arcLength(c_use, True))
        eps_f = float(max(1e-6, float(approx_eps_frac)))
        eps = float(max(1.0, float(eps_f) * float(peri)))
        pts = None
        for _ in range(12):
            approx = cv2.approxPolyDP(c_use, float(eps), True)
            pts0 = np.asarray(approx, dtype=np.float32).reshape(-1, 2)
            pts1 = _poly_remove_sharp_vertices(poly_uv=pts0, min_angle_deg=float(min_vertex_angle_deg))
            pts2 = _poly_remove_short_edges(poly_uv=pts1, min_edge_len_px=float(min_edge_len_px))
            pts = pts2 if int(pts2.shape[0]) >= 3 else (pts1 if int(pts1.shape[0]) >= 3 else pts0)
            mv = int(max_vertices)
            if mv > 0 and int(pts.shape[0]) > int(mv):
                eps *= 1.35
                continue
            break
    except Exception:
        return None
    if pts is None or int(pts.shape[0]) < 3:
        return None
    return np.asarray(pts, dtype=np.float32).reshape(-1, 2)


def _poly_fit_from_contour(
    *,
    contour_win: np.ndarray,
    x0: int,
    y0: int,
    concavity_max: float,
    approx_eps_frac: float,
    min_vertex_angle_deg: float,
    max_vertices: int,
    min_edge_len_px: float,
) -> tuple[Optional[np.ndarray], dict]:
    """
    Convert a binary-mask contour (in window coordinates) to a cleaned polygon in full-image coordinates.
    Returns (poly_uv or None, debug_info dict).
    """
    dbg: dict = {"concavity": 0.0, "used_hull": False, "peri": 0.0, "eps": 0.0, "n_raw": 0}
    try:
        c = np.asarray(contour_win, dtype=np.int32).reshape(-1, 1, 2).copy()
    except Exception:
        return None, dbg
    if int(c.shape[0]) < 3:
        return None, dbg
    dbg["n_raw"] = int(c.shape[0])
    try:
        c[:, 0, 0] += int(x0)
        c[:, 0, 1] += int(y0)
    except Exception:
        return None, dbg
    try:
        hull = cv2.convexHull(c)
        concavity = 0.0
        try:
            a_c = float(cv2.contourArea(c))
            a_h = float(cv2.contourArea(hull))
            if float(a_h) > 1e-9:
                concavity = float(max(0.0, 1.0 - (float(a_c) / float(a_h))))
        except Exception:
            concavity = 0.0
        dbg["concavity"] = float(concavity)
        conc_max = float(max(0.0, min(0.99, float(concavity_max))))
        c_use = hull if float(concavity) > float(conc_max) else c
        dbg["used_hull"] = bool(c_use is hull)
        peri = float(cv2.arcLength(c_use, True))
        dbg["peri"] = float(peri)
        eps_f = float(max(1e-6, float(approx_eps_frac)))
        eps = float(max(1.0, float(eps_f) * float(peri)))
        pts = None
        for _ in range(12):
            dbg["eps"] = float(eps)
            approx = cv2.approxPolyDP(c_use, float(eps), True)
            pts0 = np.asarray(approx, dtype=np.float32).reshape(-1, 2)
            pts1 = _poly_remove_sharp_vertices(poly_uv=pts0, min_angle_deg=float(min_vertex_angle_deg))
            pts2 = _poly_remove_short_edges(poly_uv=pts1, min_edge_len_px=float(min_edge_len_px))
            pts = pts2 if int(pts2.shape[0]) >= 3 else (pts1 if int(pts1.shape[0]) >= 3 else pts0)
            mv = int(max_vertices)
            if mv > 0 and int(pts.shape[0]) > int(mv):
                eps *= 1.35
                continue
            break
    except Exception:
        return None, dbg
    if pts is None or int(pts.shape[0]) < 3:
        return None, dbg
    return np.asarray(pts, dtype=np.float32).reshape(-1, 2), dbg


def _poly_perimeter_px(*, poly_uv: np.ndarray) -> float:
    try:
        pts = np.asarray(poly_uv, dtype=np.float64).reshape(-1, 2)
    except Exception:
        return 0.0
    if int(pts.shape[0]) < 2:
        return 0.0
    try:
        nxt = np.roll(pts, -1, axis=0)
        d = nxt - pts
        seg = np.hypot(d[:, 0], d[:, 1]).astype(np.float64, copy=False)
        return float(np.sum(seg))
    except Exception:
        return 0.0


def _poly_sample_edge_points_by_s(*, poly_uv: np.ndarray, s: np.ndarray) -> Optional[np.ndarray]:
    """Sample points along the polygon perimeter at normalized arc-length positions `s` in [0,1)."""
    try:
        pts = np.asarray(poly_uv, dtype=np.float64).reshape(-1, 2)
        s0 = np.asarray(s, dtype=np.float64).reshape(-1)
    except Exception:
        return None
    if pts.size <= 0 or int(pts.shape[0]) < 2 or s0.size <= 0:
        return None
    if not (np.isfinite(pts).all() and np.isfinite(s0).all()):
        return None
    n = int(pts.shape[0])
    try:
        nxt = np.roll(pts, -1, axis=0)
        d = nxt - pts
        seg = np.hypot(d[:, 0], d[:, 1]).astype(np.float64, copy=False)
        cum = np.cumsum(seg).astype(np.float64, copy=False)
        peri = float(cum[-1]) if int(cum.size) > 0 else 0.0
    except Exception:
        return None
    if not np.isfinite(peri) or float(peri) <= 1e-9:
        return None

    s_wrapped = np.mod(s0, 1.0).astype(np.float64, copy=False)
    dist = (s_wrapped * float(peri)).astype(np.float64, copy=False)
    idx = np.searchsorted(cum, dist, side="right").astype(np.int32, copy=False)
    idx = np.clip(idx, 0, int(n - 1)).astype(np.int32, copy=False)
    try:
        cum_prev = np.concatenate(([0.0], cum[:-1])).astype(np.float64, copy=False)
    except Exception:
        cum_prev = np.zeros((int(n),), dtype=np.float64)
        cum_prev[1:] = cum[:-1]
    seg_i = np.maximum(seg[idx].astype(np.float64, copy=False), 1e-9)
    t = ((dist - cum_prev[idx]) / seg_i).astype(np.float64, copy=False)
    p0 = pts[idx].astype(np.float64, copy=False)
    p1 = nxt[idx].astype(np.float64, copy=False)
    uv = (p0 + t.reshape(-1, 1) * (p1 - p0)).astype(np.float32, copy=False)
    return np.asarray(uv, dtype=np.float32).reshape(-1, 2)


def _poly_sample_edge_points(*, poly_uv: np.ndarray, step_px: float, max_pts: int) -> Optional[np.ndarray]:
    try:
        peri = float(_poly_perimeter_px(poly_uv=np.asarray(poly_uv, dtype=np.float32)))
    except Exception:
        peri = 0.0
    if not np.isfinite(peri) or float(peri) <= 1e-6:
        return None
    step = float(max(1.0, float(step_px)))
    n = int(max(4, int(np.ceil(float(peri) / float(step)))))
    if int(max_pts) > 0:
        n = int(min(int(max_pts), int(n)))
    if int(n) < 2:
        return None
    s = (np.arange(int(n), dtype=np.float64) + 0.5) / float(max(1, int(n)))
    return _poly_sample_edge_points_by_s(poly_uv=np.asarray(poly_uv, dtype=np.float32), s=np.asarray(s, dtype=np.float64))


def _project_world_to_uv_z(*, Pw: np.ndarray, Twc: np.ndarray, intr) -> tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    if intr is None:
        return None, None, None
    try:
        fx = float(intr.fx)
        fy = float(intr.fy)
        cx = float(intr.cx)
        cy = float(intr.cy)
    except Exception:
        return None, None, None
    if fx <= 0.0 or fy <= 0.0:
        return None, None, None
    try:
        P = np.asarray(Pw, dtype=np.float64).reshape(-1, 3)
        T = np.asarray(Twc, dtype=np.float64).reshape(4, 4)
    except Exception:
        return None, None, None
    try:
        Rwc = np.asarray(T[:3, :3], dtype=np.float64).reshape(3, 3)
        twc = np.asarray(T[:3, 3], dtype=np.float64).reshape(1, 3)
    except Exception:
        return None, None, None
    try:
        Pc = (P - twc) @ Rwc
    except Exception:
        return None, None, None
    z = Pc[:, 2].astype(np.float64, copy=False)
    ok = (np.isfinite(z)) & (z > 1e-6)
    if int(np.count_nonzero(ok)) <= 0:
        return None, None, None
    uv = np.zeros((int(Pc.shape[0]), 2), dtype=np.float64)
    try:
        uv[:, 0] = (float(fx) * (Pc[:, 0] / z)) + float(cx)
        uv[:, 1] = (float(fy) * (Pc[:, 1] / z)) + float(cy)
    except Exception:
        return None, None, None
    return uv.astype(np.float64, copy=False), z.astype(np.float64, copy=False), ok.astype(bool, copy=False)


def _zncc_patch(*, img0: np.ndarray, img1: np.ndarray, u0: float, v0: float, u1: float, v1: float, r: int) -> float:
    rr = int(max(1, int(r)))
    try:
        x0 = int(round(float(u0)))
        y0 = int(round(float(v0)))
        x1 = int(round(float(u1)))
        y1 = int(round(float(v1)))
    except Exception:
        return float("nan")
    h0, w0 = int(img0.shape[0]), int(img0.shape[1])
    h1, w1 = int(img1.shape[0]), int(img1.shape[1])
    if x0 - rr < 0 or x0 + rr >= int(w0) or y0 - rr < 0 or y0 + rr >= int(h0):
        return float("nan")
    if x1 - rr < 0 or x1 + rr >= int(w1) or y1 - rr < 0 or y1 + rr >= int(h1):
        return float("nan")
    p0 = np.asarray(img0[int(y0 - rr) : int(y0 + rr + 1), int(x0 - rr) : int(x0 + rr + 1)], dtype=np.float32)
    p1 = np.asarray(img1[int(y1 - rr) : int(y1 + rr + 1), int(x1 - rr) : int(x1 + rr + 1)], dtype=np.float32)
    if int(p0.size) <= 0 or int(p1.size) != int(p0.size):
        return float("nan")
    m0 = float(np.mean(p0))
    m1 = float(np.mean(p1))
    q0 = p0 - float(m0)
    q1 = p1 - float(m1)
    d0 = float(np.sum(q0 * q0))
    d1 = float(np.sum(q1 * q1))
    den = float(np.sqrt(max(1e-12, float(d0) * float(d1))))
    return float(np.sum(q0 * q1) / float(den))


if _HAVE_NUMBA:

    @njit(cache=True)  # type: ignore[misc]
    def _nb_zncc_gate_u8(
        img0: np.ndarray,
        img1: np.ndarray,
        p0_xy: np.ndarray,
        p1_xy: np.ndarray,
        keep_in_u8: np.ndarray,
        r: int,
        ncc_min: float,
        out_u8: np.ndarray,
    ) -> None:
        """Batch ZNCC gate for many points (no per-point slicing/allocations)."""
        rr = int(r)
        if rr < 1:
            rr = 1
        h0 = int(img0.shape[0])
        w0 = int(img0.shape[1])
        h1 = int(img1.shape[0])
        w1 = int(img1.shape[1])
        n = int(p0_xy.shape[0])
        if int(out_u8.shape[0]) != int(n):
            return
        for i in range(int(n)):
            out_u8[int(i)] = 0
        if n <= 0:
            return

        for i in range(int(n)):
            if int(keep_in_u8[int(i)]) == 0:
                continue
            x0 = int(np.rint(p0_xy[int(i), 0]))
            y0 = int(np.rint(p0_xy[int(i), 1]))
            x1 = int(np.rint(p1_xy[int(i), 0]))
            y1 = int(np.rint(p1_xy[int(i), 1]))
            if x0 - rr < 0 or x0 + rr >= w0 or y0 - rr < 0 or y0 + rr >= h0:
                continue
            if x1 - rr < 0 or x1 + rr >= w1 or y1 - rr < 0 or y1 + rr >= h1:
                continue

            # Means
            s0 = 0.0
            s1 = 0.0
            cnt = 0
            for dy in range(-rr, rr + 1):
                yy0 = int(y0 + dy)
                yy1 = int(y1 + dy)
                for dx in range(-rr, rr + 1):
                    s0 += float(img0[int(yy0), int(x0 + dx)])
                    s1 += float(img1[int(yy1), int(x1 + dx)])
                    cnt += 1
            if cnt <= 0:
                continue
            m0 = s0 / float(cnt)
            m1 = s1 / float(cnt)

            num = 0.0
            d0 = 0.0
            d1 = 0.0
            for dy in range(-rr, rr + 1):
                yy0 = int(y0 + dy)
                yy1 = int(y1 + dy)
                for dx in range(-rr, rr + 1):
                    a = float(img0[int(yy0), int(x0 + dx)]) - float(m0)
                    b = float(img1[int(yy1), int(x1 + dx)]) - float(m1)
                    num += float(a * b)
                    d0 += float(a * a)
                    d1 += float(b * b)
            den = float(d0 * d1)
            if not (den > 1e-12):
                continue
            cc = float(num) / float(np.sqrt(den))
            if float(cc) >= float(ncc_min):
                out_u8[int(i)] = 1


def _poly_remove_sharp_vertices(*, poly_uv: np.ndarray, min_angle_deg: float, max_remove: int = 128) -> np.ndarray:
    """
    Removes vertices with very small interior angles (sharp spikes).

    This keeps some jaggedness but prevents extreme, unstable polygon shapes.
    Runs only at selection time, so we bias for robustness over speed.
    """
    try:
        pts = np.asarray(poly_uv, dtype=np.float64).reshape(-1, 2)
    except Exception:
        return np.asarray(poly_uv, dtype=np.float32).reshape(-1, 2)
    if int(pts.shape[0]) < 4:
        return pts.astype(np.float32, copy=False)
    thr = float(max(0.0, float(min_angle_deg)))
    if thr <= 0.0:
        return pts.astype(np.float32, copy=False)
    thr_rad = float(np.radians(float(thr)))
    removed = 0
    while int(pts.shape[0]) > 3 and int(removed) < int(max_remove):
        n = int(pts.shape[0])
        ang = np.full((n,), np.pi, dtype=np.float64)
        for i in range(int(n)):
            p = pts[int(i)]
            p0 = pts[int((i - 1) % int(n))]
            p1 = pts[int((i + 1) % int(n))]
            v0 = p0 - p
            v1 = p1 - p
            d0 = float(np.hypot(float(v0[0]), float(v0[1])))
            d1 = float(np.hypot(float(v1[0]), float(v1[1])))
            if float(d0) <= 1e-6 or float(d1) <= 1e-6:
                continue
            c = float((float(v0[0]) * float(v1[0]) + float(v0[1]) * float(v1[1])) / float(d0 * d1))
            c = float(np.clip(c, -1.0, 1.0))
            ang[int(i)] = float(np.arccos(c))
        i_min = int(np.argmin(ang))
        if not (float(ang[int(i_min)]) < float(thr_rad)):
            break
        pts = np.delete(pts, int(i_min), axis=0)
        removed += 1
    return pts.astype(np.float32, copy=False).reshape(-1, 2)


def _poly_remove_short_edges(*, poly_uv: np.ndarray, min_edge_len_px: float, max_remove: int = 256) -> np.ndarray:
    """
    Removes vertices that form very short edges (pixel-scale jaggedness).
    """
    try:
        pts = np.asarray(poly_uv, dtype=np.float64).reshape(-1, 2)
    except Exception:
        return np.asarray(poly_uv, dtype=np.float32).reshape(-1, 2)
    if int(pts.shape[0]) < 4:
        return pts.astype(np.float32, copy=False)
    thr = float(max(0.0, float(min_edge_len_px)))
    if thr <= 0.0:
        return pts.astype(np.float32, copy=False)

    removed = 0
    while int(pts.shape[0]) > 3 and int(removed) < int(max_remove):
        n = int(pts.shape[0])
        lens = np.empty((n,), dtype=np.float64)
        for i in range(int(n)):
            j = int((i + 1) % int(n))
            dx = float(pts[j, 0] - pts[i, 0])
            dy = float(pts[j, 1] - pts[i, 1])
            lens[int(i)] = float(np.hypot(dx, dy))
        i_min = int(np.argmin(lens))
        if not (float(lens[int(i_min)]) < float(thr)):
            break
        # Drop the start vertex of the shortest edge.
        pts = np.delete(pts, int(i_min), axis=0)
        removed += 1
    return pts.astype(np.float32, copy=False).reshape(-1, 2)


def _sample_depth_stats_u16_outside_mask(
    depth_raw: np.ndarray,
    *,
    x: int,
    y: int,
    r: int,
    mask_win: Optional[np.ndarray],
    mask_x0: int,
    mask_y0: int,
) -> tuple[int, float, int]:
    """
    Median/MAD depth sample (uint16) in a square window, ignoring:
      - invalid depth (0)
      - pixels inside `mask_win != 0` (mask is in window coords at (mask_x0,mask_y0))

    Used for hole-seeded selection: the hole mask often covers pixels with invalid/"infinite" depth,
    but the polygon boundary should be lifted using the *surrounding* surface depth.
    """
    if depth_raw is None or not isinstance(depth_raw, np.ndarray) or depth_raw.ndim != 2:
        return 0, 0.0, 0
    h, w = depth_raw.shape[:2]
    x0c = max(0, min(int(w - 1), int(x)))
    y0c = max(0, min(int(h - 1), int(y)))
    rr = int(max(0, int(r)))
    x0 = max(0, x0c - rr)
    y0 = max(0, y0c - rr)
    x1 = min(int(w - 1), x0c + rr)
    y1 = min(int(h - 1), y0c + rr)
    vals: list[int] = []
    have_mask = mask_win is not None and isinstance(mask_win, np.ndarray) and mask_win.ndim == 2 and int(mask_win.size) > 0
    mh = int(mask_win.shape[0]) if have_mask else 0
    mw = int(mask_win.shape[1]) if have_mask else 0
    for yy in range(int(y0), int(y1) + 1):
        ly = int(yy) - int(mask_y0)
        in_y = have_mask and (0 <= ly < mh)
        for xx in range(int(x0), int(x1) + 1):
            d = int(depth_raw[int(yy), int(xx)])
            if int(d) <= 0:
                continue
            if in_y:
                lx = int(xx) - int(mask_x0)
                if 0 <= lx < mw:
                    try:
                        if int(mask_win[int(ly), int(lx)]) != 0:
                            continue
                    except Exception:
                        pass
            vals.append(int(d))
    if not vals:
        return 0, 0.0, 0
    try:
        arr = np.asarray(vals, dtype=np.float64).reshape(-1)
        med = float(np.median(arr))
        mad = float(np.median(np.abs(arr - float(med))))
        return int(round(med)), float(mad), int(arr.size)
    except Exception:
        return 0, 0.0, int(len(vals))


def _poly_vo_lk_track_process_main(
    *,
    stop_event,
    ring_spec_dict: dict,
    state_spec_dict: Optional[dict] = None,
    cfg: dict,
    intr: Optional[dict] = None,
    depth_units: float = 0.0,
    cmd_q,
    out_q,
) -> None:
    _proc_setup_signals(stop_event=stop_event)

    cfg = dict(cfg or {})
    _setup_stdio_logging(log_spec=dict(cfg.get("_log_spec") or {}), role="poly_vo_lk")
    _set_cv_threads(cfg=cfg, role="poly")
    log_enabled = bool((cfg.get("_log_spec") or {}).get("enabled", False))

    app_cfg = dict(cfg.get("app") or {})
    cap_cfg = dict(cfg.get("capture") or {})
    proc_cfg = dict(app_cfg.get("poly_process") or {})
    poly_cfg = dict(app_cfg.get("poly_vo_lk_track") or {})

    drop_frames = bool(proc_cfg.get("drop_frames", True))
    idle_sleep_s = float(max(0.0, float(proc_cfg.get("idle_sleep_s", 0.002))))
    max_gap_frames = int(max(1, int(proc_cfg.get("max_gap_frames", app_cfg.get("max_gap_frames", 8)))))
    every_n_frames = int(max(1, int(proc_cfg.get("every_n_frames", 1))))

    canny_lo = int(proc_cfg.get("canny_lo", 60))
    canny_hi = int(proc_cfg.get("canny_hi", 140))
    edge_kernel = None
    try:
        if bool(proc_cfg.get("dilate_edges", True)):
            edge_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    except Exception:
        edge_kernel = None

    # Stage-1 micro-refinement: edge-normal snap (VO-agnostic).
    try:
        stage1_snap_cfg = dict(proc_cfg.get("stage1_edge_snap") or {})
    except Exception:
        stage1_snap_cfg = {}
    try:
        stage1_edge_snap = EdgeSnapConfig(
            enabled=bool(stage1_snap_cfg.get("enabled", False)),
            search_px=int(stage1_snap_cfg.get("search_px", 12)),
            max_step_px=float(stage1_snap_cfg.get("max_step_px", 2.0)),
            canny_lo=int(stage1_snap_cfg.get("canny_lo", canny_lo)),
            canny_hi=int(stage1_snap_cfg.get("canny_hi", canny_hi)),
            canny_L2gradient=bool(stage1_snap_cfg.get("canny_L2gradient", True)),
            blur_ksize=int(stage1_snap_cfg.get("blur_ksize", 5)),
            sobel_ksize=int(stage1_snap_cfg.get("sobel_ksize", 3)),
            edge_bonus=float(stage1_snap_cfg.get("edge_bonus", 2000.0)),
            grad_weight=float(stage1_snap_cfg.get("grad_weight", 1.0)),
            min_score_improve=float(stage1_snap_cfg.get("min_score_improve", 0.0)),
        )
    except Exception:
        stage1_edge_snap = EdgeSnapConfig(enabled=False)

    # Stage-1 -> Stage-2 transition (PID confirm) automation (VO-agnostic).
    try:
        stage2_cfg = dict(proc_cfg.get("stage2_transition") or {})
    except Exception:
        stage2_cfg = {}
    try:
        stage2_transition_mode = str(stage2_cfg.get("mode", "manual") or "manual").strip().lower()
    except Exception:
        stage2_transition_mode = "manual"
    try:
        stage2_roi_fill_frac = float(stage2_cfg.get("roi_fill_frac", 0.65))
    except Exception:
        stage2_roi_fill_frac = 0.65
    stage2_roi_fill_frac = float(min(1.0, max(0.0, stage2_roi_fill_frac)))
    try:
        stage2_cooldown_s = float(stage2_cfg.get("cooldown_s", 0.5))
    except Exception:
        stage2_cooldown_s = 0.5
    stage2_cooldown_s = float(max(0.0, stage2_cooldown_s))
    stage2_require_ok = bool(stage2_cfg.get("require_ok", True))
    stage2_auto_last_t = 0.0

    # Stage-0 auto-lock: when preview finds a polygon, immediately activate selection and start tracking it.
    # This makes acquisition robust when the detector flickers or the user clicks at a moment with no contour.
    try:
        auto_lock_cfg = dict(proc_cfg.get("auto_lock") or {})
    except Exception:
        auto_lock_cfg = {}
    auto_lock_enabled = bool(auto_lock_cfg.get("enabled", False))
    try:
        auto_lock_kind = str(auto_lock_cfg.get("kind", "hole") or "hole").strip().lower()
    except Exception:
        auto_lock_kind = "hole"
    try:
        auto_lock_cooldown_s = float(auto_lock_cfg.get("cooldown_s", 0.5))
    except Exception:
        auto_lock_cooldown_s = 0.5
    auto_lock_cooldown_s = float(max(0.0, auto_lock_cooldown_s))
    auto_lock_last_t = 0.0

    # Optional: ORB/Line acquisition (VO-agnostic) for future reseed/recovery strategies.
    try:
        feat_cfg = dict(proc_cfg.get("feature_acquire") or {})
    except Exception:
        feat_cfg = {}
    feat_enabled = bool(feat_cfg.get("enabled", False))
    try:
        feat_band_cfg = dict(feat_cfg.get("band") or {})
    except Exception:
        feat_band_cfg = {}
    feat_band = BandMaskConfig(
        mode=str(feat_band_cfg.get("mode", "both") or "both"),
        outer_px=int(feat_band_cfg.get("outer_px", 16)),
        inner_px=int(feat_band_cfg.get("inner_px", 8)),
    )
    try:
        feat_orb_cfg = dict(feat_cfg.get("orb") or {})
    except Exception:
        feat_orb_cfg = {}
    feat_orb = OrbAcquireConfig(
        enabled=bool(feat_orb_cfg.get("enabled", True)),
        nfeatures=int(feat_orb_cfg.get("nfeatures", 600)),
        scaleFactor=float(feat_orb_cfg.get("scaleFactor", 1.2)),
        nlevels=int(feat_orb_cfg.get("nlevels", 8)),
        edgeThreshold=int(feat_orb_cfg.get("edgeThreshold", 31)),
        firstLevel=int(feat_orb_cfg.get("firstLevel", 0)),
        WTA_K=int(feat_orb_cfg.get("WTA_K", 2)),
        scoreType=int(feat_orb_cfg.get("scoreType", 0)),
        patchSize=int(feat_orb_cfg.get("patchSize", 31)),
        fastThreshold=int(feat_orb_cfg.get("fastThreshold", 20)),
        max_matches=int(feat_orb_cfg.get("max_matches", 200)),
        ratio_test=float(feat_orb_cfg.get("ratio_test", 0.75)),
        ransac_reproj_thresh_px=float(feat_orb_cfg.get("ransac_reproj_thresh_px", 3.0)),
        ransac_confidence=float(feat_orb_cfg.get("ransac_confidence", 0.99)),
        min_inliers=int(feat_orb_cfg.get("min_inliers", 12)),
    )
    try:
        feat_lines_cfg = dict(feat_cfg.get("lines") or {})
    except Exception:
        feat_lines_cfg = {}
    feat_lines = LineAcquireConfig(
        enabled=bool(feat_lines_cfg.get("enabled", False)),
        min_length_px=float(feat_lines_cfg.get("min_length_px", 25.0)),
        refine=int(feat_lines_cfg.get("refine", 1)),
    )
    feature_acq_cfg = FeatureAcquireConfig(enabled=bool(feat_enabled), band=feat_band, orb=feat_orb, lines=feat_lines)
    feature_acq = PolyFeatureAcquirer(feature_acq_cfg) if bool(feature_acq_cfg.enabled) else None
    try:
        feature_acq_rate_hz = float(feat_cfg.get("rate_hz", 0.0))
    except Exception:
        feature_acq_rate_hz = 0.0
    feature_acq_rate_hz = float(max(0.0, feature_acq_rate_hz))
    feature_acq_stage1 = bool(feat_cfg.get("stage1", True))
    feature_acq_stage2 = bool(feat_cfg.get("stage2", True))
    feature_acq_on_select = bool(feat_cfg.get("on_select", True))
    feature_acq_on_confirm = bool(feat_cfg.get("on_confirm", True))
    feature_acq_publish_orb = bool(feat_cfg.get("publish_orb", False))
    feature_acq_publish_lines = bool(feat_cfg.get("publish_lines", False))
    feature_acq_publish_max_orb = int(max(0, int(feat_cfg.get("publish_max_orb", 120))))
    feature_acq_publish_max_lines = int(max(0, int(feat_cfg.get("publish_max_lines", 40))))

    # Hole detector window (pixels). Use 0/0 for full-frame detection.
    hole_win = proc_cfg.get("hole_window_wh", (0, 0))
    try:
        hole_win_w = int(hole_win[0])
        hole_win_h = int(hole_win[1])
    except Exception:
        hole_win_w, hole_win_h = 0, 0
    if int(hole_win_w) <= 0 or int(hole_win_h) <= 0:
        hole_win_w, hole_win_h = 0, 0
    else:
        hole_win_w = int(max(32, hole_win_w))
        hole_win_h = int(max(32, hole_win_h))

    hole_min_area_px = float(max(0.0, float(proc_cfg.get("hole_min_area_px", 25.0))))
    hole_unknown_as_full = bool(proc_cfg.get("hole_unknown_as_full", True))
    hole_select_snap_px = proc_cfg.get("hole_select_snap_px", None)
    hole_lock_snap_px = int(max(0, int(proc_cfg.get("hole_lock_snap_px", 12))))
    hole_preview_enabled = bool(proc_cfg.get("hole_preview_enabled", True))
    # Runtime toggle (controlled by headless JSON cmd "hole_enable").
    hole_runtime_enabled = True
    hole_preview_rate_hz = float(max(0.0, float(proc_cfg.get("hole_preview_rate_hz", 10.0))))
    hole_preview_snap_px = int(max(0, int(proc_cfg.get("hole_preview_snap_px", 0))))
    hole_pid_min_radius_m = float(max(0.0, float(proc_cfg.get("hole_pid_min_radius_m", 0.55))))

    # Plane selection (3D plane tracker) — uses the hole detector's RANSAC plane fit to propose a solid plane patch.
    plane_preview_enabled = bool(proc_cfg.get("plane_preview_enabled", True))
    # Runtime toggle (controlled by headless JSON cmd "plane_enable"). Plane mode is off by default.
    plane_runtime_enabled = False
    try:
        plane_preview_rate_hz = float(proc_cfg.get("plane_preview_rate_hz", hole_preview_rate_hz))
    except Exception:
        plane_preview_rate_hz = float(hole_preview_rate_hz)
    plane_preview_rate_hz = float(max(0.0, float(plane_preview_rate_hz)))
    plane_pid_min_radius_m = float(max(0.0, float(proc_cfg.get("plane_pid_min_radius_m", 1.0))))
    plane_poly_n_verts = int(max(8, int(proc_cfg.get("plane_poly_n_verts", 24))))
    plane_mask_close_px = int(max(0, int(proc_cfg.get("plane_mask_close_px", 3))))
    hole_done_enabled = bool(proc_cfg.get("hole_done_enabled", True))
    hole_done_fill_ratio = float(proc_cfg.get("hole_done_fill_ratio", 0.95))
    if (not bool(hole_done_enabled)) or (not np.isfinite(float(hole_done_fill_ratio))) or float(hole_done_fill_ratio) <= 0.0:
        hole_done_enabled = False
        hole_done_fill_ratio = 0.0
    else:
        hole_done_fill_ratio = float(max(0.0, min(1.0, float(hole_done_fill_ratio))))
    hole_refine_rate_hz = float(max(0.0, float(proc_cfg.get("hole_refine_rate_hz", 10.0))))
    # If true: run the expensive hole-refinement detector in a background thread.
    hole_refine_thread = bool(proc_cfg.get("hole_refine_thread", True))
    # If true: copy gray/depth into a persistent buffer before handing to the thread (safe with shm ring buffers).
    hole_refine_thread_copy = bool(proc_cfg.get("hole_refine_thread_copy", True))

    # BEGIN_STAGE1_HYSTERESIS
    # Stage 1 (after first click, before second click): refine the hole outline while holding it with the tracker.
    # We bias the refinement toward the *largest* stable opening using the detector's max-inscribed-circle radius
    # (pixels) and add hysteresis (grow fast, shrink only after repeated evidence) to cut through depth jitter.
    #
    # Marker for easy rollback / A/B tests:
    # - Remove the code between BEGIN_STAGE1_HYSTERESIS / END_STAGE1_HYSTERESIS to revert to "always update".
    hole_refine_hyst_enabled = bool(proc_cfg.get("hole_refine_hyst_enabled", True))
    hole_refine_gain_abs_px = float(max(0.0, float(proc_cfg.get("hole_refine_gain_abs_px", 3.0))))
    hole_refine_gain_rel = float(max(0.0, float(proc_cfg.get("hole_refine_gain_rel", 0.03))))
    hole_refine_shrink_rel = float(max(0.0, float(proc_cfg.get("hole_refine_shrink_rel", 0.06))))
    hole_refine_shrink_votes_req = int(max(1, int(proc_cfg.get("hole_refine_shrink_votes_req", 4))))
    # END_STAGE1_HYSTERESIS
    # Selection-time snapping: should be large enough so a click near the hole center can still snap to the hole edges,
    # but not so large that it jumps to unrelated regions (walls / background) inside the ROI.
    if hole_select_snap_px is None:
        img_w0 = int(max(1, int(cap_cfg.get("out_width", 640))))
        img_h0 = int(max(1, int(cap_cfg.get("out_height", 480))))
        base_w = int(hole_win_w) if int(hole_win_w) > 0 else int(img_w0)
        base_h = int(hole_win_h) if int(hole_win_h) > 0 else int(img_h0)
        hole_select_snap_px = int(round(0.35 * float(min(int(base_w), int(base_h)))))
    else:
        try:
            hole_select_snap_px = int(hole_select_snap_px)
        except Exception:
            img_w0 = int(max(1, int(cap_cfg.get("out_width", 640))))
            img_h0 = int(max(1, int(cap_cfg.get("out_height", 480))))
            base_w = int(hole_win_w) if int(hole_win_w) > 0 else int(img_w0)
            base_h = int(hole_win_h) if int(hole_win_h) > 0 else int(img_h0)
            hole_select_snap_px = int(round(0.35 * float(min(int(base_w), int(base_h)))))
    hole_select_snap_px = int(max(0, int(hole_select_snap_px)))

    # Hole-detector depth spatial filter (reduce "3D smoothing" by default).
    hole_spatial_cfg = proc_cfg.get("hole_spatial", None)
    hole_spatial_iters = 2
    hole_spatial_radius_px = 1
    hole_spatial_delta_m = 0.02
    hole_spatial_alpha = 0.50
    if isinstance(hole_spatial_cfg, dict):
        try:
            hole_spatial_iters = int(hole_spatial_cfg.get("iterations", hole_spatial_iters))
        except Exception:
            pass
        try:
            hole_spatial_radius_px = int(hole_spatial_cfg.get("radius_px", hole_spatial_radius_px))
        except Exception:
            pass
        try:
            hole_spatial_delta_m = float(hole_spatial_cfg.get("delta_m", hole_spatial_delta_m))
        except Exception:
            pass
        try:
            hole_spatial_alpha = float(hole_spatial_cfg.get("alpha", hole_spatial_alpha))
        except Exception:
            pass
    hole_spatial = SpatialFilterSettings(
        iterations=int(hole_spatial_iters),
        radius_px=int(hole_spatial_radius_px),
        delta_m=float(hole_spatial_delta_m),
        alpha=float(hole_spatial_alpha),
    )

    # Hole detector leak guard (prevents "whole-frame" mask blowups when flood fill escapes).
    hole_leak_cfg = proc_cfg.get("hole_leak_guard", None)
    hole_leak_guard_enabled = True
    hole_leak_open_max_px = 6
    hole_leak_border_fill_frac = 0.25
    hole_leak_full_fill_frac = 0.95
    if isinstance(hole_leak_cfg, dict):
        try:
            hole_leak_guard_enabled = bool(hole_leak_cfg.get("enabled", hole_leak_guard_enabled))
        except Exception:
            pass
        try:
            hole_leak_open_max_px = int(hole_leak_cfg.get("open_max_px", hole_leak_open_max_px))
        except Exception:
            pass
        try:
            hole_leak_border_fill_frac = float(hole_leak_cfg.get("border_fill_frac", hole_leak_border_fill_frac))
        except Exception:
            pass
        try:
            hole_leak_full_fill_frac = float(hole_leak_cfg.get("full_fill_frac", hole_leak_full_fill_frac))
        except Exception:
            pass
    hole_leak_open_max_px = int(max(0, int(hole_leak_open_max_px)))
    try:
        hole_leak_border_fill_frac = float(max(0.0, min(1.0, float(hole_leak_border_fill_frac))))
    except Exception:
        hole_leak_border_fill_frac = 0.25
    try:
        hole_leak_full_fill_frac = float(max(0.0, min(1.0, float(hole_leak_full_fill_frac))))
    except Exception:
        hole_leak_full_fill_frac = 0.95

    # Hole-detector plane sector coverage gate (used by the plane-guided hole mode).
    # `plane_sector_low` triggers when the fitted plane inliers do not cover enough angular sectors around the seed.
    hole_plane_sector_cfg = proc_cfg.get("hole_plane_sector", None)
    hole_plane_sector_count = 12
    hole_plane_sector_min_frac = 0.70
    if isinstance(hole_plane_sector_cfg, dict):
        try:
            hole_plane_sector_count = int(hole_plane_sector_cfg.get("count", hole_plane_sector_count))
        except Exception:
            pass
        try:
            hole_plane_sector_min_frac = float(hole_plane_sector_cfg.get("min_frac", hole_plane_sector_min_frac))
        except Exception:
            pass
    hole_plane_sector_count = int(max(4, int(hole_plane_sector_count)))
    try:
        hole_plane_sector_min_frac = float(max(0.0, min(1.0, float(hole_plane_sector_min_frac))))
    except Exception:
        hole_plane_sector_min_frac = 0.70

    # Hole-detector annulus normal estimation (plane-guided mode).
    hole_plane_norm_cfg = proc_cfg.get("hole_plane_normals", None)
    hole_plane_norm_enabled = True
    hole_plane_norm_step_px = 5
    hole_plane_norm_min_mag = 0.45
    hole_plane_norm_max_angle_deg = 30.0
    hole_plane_norm_min_valid_frac = 0.15
    hole_plane_ransac_numba = bool(proc_cfg.get("hole_plane_ransac_numba", True))
    if isinstance(hole_plane_norm_cfg, dict):
        try:
            hole_plane_norm_enabled = bool(hole_plane_norm_cfg.get("enabled", hole_plane_norm_enabled))
        except Exception:
            pass
        try:
            hole_plane_norm_step_px = int(hole_plane_norm_cfg.get("step_px", hole_plane_norm_step_px))
        except Exception:
            pass
        try:
            hole_plane_norm_min_mag = float(hole_plane_norm_cfg.get("min_mag", hole_plane_norm_min_mag))
        except Exception:
            pass
        try:
            hole_plane_norm_max_angle_deg = float(hole_plane_norm_cfg.get("max_angle_deg", hole_plane_norm_max_angle_deg))
        except Exception:
            pass
        try:
            hole_plane_norm_min_valid_frac = float(hole_plane_norm_cfg.get("min_valid_frac", hole_plane_norm_min_valid_frac))
        except Exception:
            pass

    hole_edge_cfg = proc_cfg.get("hole_edge_barrier", None)
    hole_edge_enabled = True
    hole_edge_mode = "unknown"
    hole_edge_method = "laplace"
    hole_edge_canny_lo = int(canny_lo)
    hole_edge_canny_hi = int(canny_hi)
    hole_edge_canny_l2 = True
    hole_edge_laplace_ksize = 3
    hole_edge_threshold = 25
    hole_edge_morph_ksize = 2
    hole_edge_morph_iter = 1
    hole_edge_dilate_px = 2
    if isinstance(hole_edge_cfg, dict):
        try:
            hole_edge_enabled = bool(hole_edge_cfg.get("enabled", hole_edge_enabled))
        except Exception:
            pass
        try:
            hole_edge_mode = str(hole_edge_cfg.get("mode", hole_edge_mode) or hole_edge_mode)
        except Exception:
            pass
        try:
            hole_edge_method = str(hole_edge_cfg.get("method", hole_edge_method) or hole_edge_method)
        except Exception:
            pass
        try:
            hole_edge_canny_lo = int(hole_edge_cfg.get("canny_lo", hole_edge_canny_lo))
        except Exception:
            pass
        try:
            hole_edge_canny_hi = int(hole_edge_cfg.get("canny_hi", hole_edge_canny_hi))
        except Exception:
            pass
        try:
            hole_edge_canny_l2 = bool(hole_edge_cfg.get("canny_L2gradient", hole_edge_canny_l2))
        except Exception:
            pass
        try:
            hole_edge_laplace_ksize = int(hole_edge_cfg.get("laplace_ksize", hole_edge_laplace_ksize))
        except Exception:
            pass
        try:
            hole_edge_threshold = int(hole_edge_cfg.get("threshold", hole_edge_threshold))
        except Exception:
            pass
        try:
            hole_edge_morph_ksize = int(hole_edge_cfg.get("morph_ksize", hole_edge_morph_ksize))
        except Exception:
            pass
        try:
            hole_edge_morph_iter = int(hole_edge_cfg.get("morph_iter", hole_edge_morph_iter))
        except Exception:
            pass
        try:
            hole_edge_dilate_px = int(hole_edge_cfg.get("dilate_px", hole_edge_dilate_px))
        except Exception:
            pass

    # Polygon fitting shape controls (used for both bbox selection and hole-mask selection).
    # - concavity_max: allow some concavity (jaggedness), but clamp extreme concave shapes to convex hull.
    # - min_vertex_angle_deg: drop very sharp spikes from the approximated polygon.
    # - approx_eps_frac: contour approximation strength (larger -> smoother polygon).
    poly_concavity_max = float(max(0.0, min(0.99, float(proc_cfg.get("poly_concavity_max", 0.15)))))
    poly_min_vertex_angle_deg = float(max(0.0, float(proc_cfg.get("poly_min_vertex_angle_deg", 45.0))))
    poly_approx_eps_frac = float(max(1e-6, float(proc_cfg.get("poly_approx_eps_frac", 0.03))))
    poly_max_vertices = int(max(0, int(proc_cfg.get("poly_max_vertices", 18))))
    poly_min_edge_len_px = float(max(0.0, float(proc_cfg.get("poly_min_edge_len_px", 10.0))))
    # Hole-mask polygon fitting: default to a less-aggressive approximation (keeps a jagged outline).
    hole_poly_concavity_max = float(max(0.0, min(0.99, float(proc_cfg.get("hole_poly_concavity_max", poly_concavity_max)))))
    hole_poly_min_vertex_angle_deg = float(
        max(0.0, float(proc_cfg.get("hole_poly_min_vertex_angle_deg", min(poly_min_vertex_angle_deg, 30.0))))
    )
    hole_poly_min_edge_len_px = float(max(0.0, float(proc_cfg.get("hole_poly_min_edge_len_px", min(poly_min_edge_len_px, 6.0)))))
    hole_poly_max_vertices = int(max(0, int(proc_cfg.get("hole_poly_max_vertices", max(poly_max_vertices, 32)))))
    hole_poly_approx_eps_frac = float(max(1e-6, float(proc_cfg.get("hole_poly_approx_eps_frac", min(poly_approx_eps_frac, 0.012)))))

    enabled = bool(poly_cfg.get("enabled", False))
    depth_gate_enabled = bool(poly_cfg.get("depth_gate_enabled", True))
    depth_r_px = int(max(0, int(poly_cfg.get("depth_r_px", 2))))
    depth_prior_rel = float(max(0.0, float(poly_cfg.get("depth_prior_rel", 0.15))))
    depth_prior_abs = float(max(0.0, float(poly_cfg.get("depth_prior_abs", 0.10))))

    pts_min = int(max(4, int(poly_cfg.get("points_min", 50))))
    pts_step = int(max(4, int(poly_cfg.get("points_step", 50))))
    pts_max = int(max(4, int(poly_cfg.get("points_max", 500))))
    min_fit_inliers = int(max(4, int(poly_cfg.get("min_fit_inliers", 10))))

    lk_win = int(max(7, int(poly_cfg.get("lk_win", 31))))
    lk_levels = int(max(0, int(poly_cfg.get("lk_levels", 4))))
    lk_min_eig = float(poly_cfg.get("lk_min_eig", 1e-4))
    lk_use_initial_flow = bool(poly_cfg.get("lk_use_initial_flow", True))
    lk_params = dict(
        winSize=(int(lk_win), int(lk_win)),
        maxLevel=int(lk_levels),
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        minEigThreshold=float(lk_min_eig),
    )
    fb_px = float(max(0.5, float(poly_cfg.get("fb_thresh_px", 2.0))))
    pred_gate_px = float(max(0.0, float(poly_cfg.get("pred_gate_px", 25.0))))
    if not bool(lk_use_initial_flow):
        # No VO-projected initial guess -> pred gate becomes misleading; disable.
        pred_gate_px = 0.0
    ncc_r_px = int(max(1, int(poly_cfg.get("ncc_r_px", 4))))
    ncc_min = float(poly_cfg.get("ncc_min", 0.60))

    ransac_px = float(max(0.5, float(poly_cfg.get("ransac_px", 3.0))))
    min_inlier_ratio = float(max(0.0, min(1.0, float(poly_cfg.get("min_inlier_ratio", 0.15)))))

    # Dynamic point-count adaptation (CPU vs robustness):
    # - Measure "point success" as: (# points passing all LK gates) / (requested point count)
    # - If < low threshold: increase tracked points by points_step
    # - If >= high threshold: decrease tracked points by points_step (only when locked)
    success_ratio_low = float(max(0.0, min(1.0, float(poly_cfg.get("success_ratio_low", 0.30)))))
    success_ratio_high = float(max(0.0, min(1.0, float(poly_cfg.get("success_ratio_high", 0.60)))))
    if float(success_ratio_high) < float(success_ratio_low):
        success_ratio_high = float(success_ratio_low)

    # Additional acceptance gate: require at least this fraction of the requested points to survive all LK gates.
    # This prevents "locking" on a tiny cluster of points (common source of slow drift).
    accept_keep_ratio_min = float(max(0.0, min(1.0, float(poly_cfg.get("accept_keep_ratio_min", 0.20)))))

    # Plane-guided point pruning/reseed (movement stage after PID lock):
    # Points used for LK lock can drift onto background surfaces visible through the opening (not on the wall plane).
    # When enabled, we reject those points by checking their observed 3D position against the fitted plane and
    # optionally reseed a fresh perimeter point set on that same plane.
    plane_cancel_enabled = bool(poly_cfg.get("plane_cancel_enabled", True))
    plane_reseed_enabled = bool(poly_cfg.get("plane_reseed_enabled", True))
    plane_dist_thr_m = float(max(0.0, float(poly_cfg.get("plane_dist_thr_m", 0.05))))
    plane_bad_count = int(max(1, int(poly_cfg.get("plane_bad_count", 2))))
    plane_reseed_min_alive = int(max(0, int(poly_cfg.get("plane_reseed_min_alive", max(int(pts_min), int(3 * min_fit_inliers))))))
    plane_reseed_cooldown_s = float(max(0.0, float(poly_cfg.get("plane_reseed_cooldown_s", 0.75))))
    plane_reseed_edge_offset_px = float(max(0.0, float(poly_cfg.get("plane_reseed_edge_offset_px", 3.0))))

    corr_ema_alpha = float(max(0.0, min(1.0, float(poly_cfg.get("corr_ema_alpha", 0.50)))))
    corr_decay_half_life_s = float(max(1e-3, float(poly_cfg.get("corr_decay_half_life_s", 0.50))))

    show_pts = bool(poly_cfg.get("show_lock_points", True))
    debug_log = bool(poly_cfg.get("debug_log", False))
    dbg_log = bool(log_enabled and debug_log)
    perf_log = bool(dbg_log and bool(poly_cfg.get("perf_log", False)))
    perf_rate_hz = float(max(0.1, float(poly_cfg.get("perf_rate_hz", 1.0))))
    jit_warmup = bool(poly_cfg.get("jit_warmup", True))

    ring = ShmRing.attach(ShmRingSpec.from_dict(dict(ring_spec_dict)))
    intr_obj = _intrinsics_from_dict(dict(intr or {}), fallback_w=int(ring.spec.w), fallback_h=int(ring.spec.h))
    depth_units = float(depth_units) if np.isfinite(float(depth_units)) else 0.0

    state = None
    if isinstance(state_spec_dict, dict):
        try:
            state = ShmOdomState.attach(ShmStateSpec.from_dict(dict(state_spec_dict)))
        except Exception:
            state = None

    # Numba warmup: compile critical kernels up-front so first interaction doesn't JIT mid-flight.
    if bool(_HAVE_NUMBA) and bool(jit_warmup):
        try:
            img0 = np.zeros((9, 9), dtype=np.uint8)
            img1 = np.zeros((9, 9), dtype=np.uint8)
            p0 = np.zeros((1, 2), dtype=np.float32)
            p1 = np.zeros((1, 2), dtype=np.float32)
            keep_u8 = np.ones((1,), dtype=np.uint8)
            out_u8 = np.zeros((1,), dtype=np.uint8)
            _nb_zncc_gate_u8(img0, img1, p0, p1, keep_u8, 1, 0.0, out_u8)  # type: ignore[name-defined]
        except Exception:
            pass

    poly_active = False
    poly_uv: Optional[np.ndarray] = None
    poly_bbox: Optional[tuple[int, int, int, int]] = None
    poly_world_pts_w: Optional[np.ndarray] = None
    edge_world_pts_w: Optional[np.ndarray] = None
    edge_s: Optional[np.ndarray] = None  # normalized arc-length positions for `edge_world_pts_w` (stable reseed)
    poly_z_prior_m: Optional[float] = None

    # Refinement-stage (pre-lock) polygon published to the GUI.
    # This polygon is allowed to change shape based on repeated hole-mask detections,
    # and is propagated frame-to-frame using the tracked polygon motion.
    ref_poly_uv: Optional[np.ndarray] = None  # full-image pixels
    ref_mask_u8: Optional[np.ndarray] = None  # window coords (for depth sampling on confirm)
    ref_mask_x0 = 0
    ref_mask_y0 = 0

    # BEGIN_STAGE1_HYSTERESIS
    # "Best so far" score during Stage 1 refinement (max-inscribed-circle radius in pixels).
    stage1_best_r_px = float("nan")
    stage1_shrink_votes = int(0)
    # END_STAGE1_HYSTERESIS

    # Previous tracked polygon for propagating `ref_poly_uv`.
    trk_poly_prev: Optional[np.ndarray] = None  # full-image pixels

    # Selection mode metadata.
    sel_kind: str = "none"  # none|bbox|hole|plane
    pid_confirmed = False
    hole_center_uv: Optional[tuple[float, float]] = None  # full-image pixels
    hole_r_px = 0.0
    hole_r_m = 0.0
    hole_fill = 0.0
    hole_ok_pid = False
    hole_plane_inliers = 0
    hole_plane_rms_m = 0.0
    hole_plane_cov = 0.0
    # Range (meters) from the camera to the cached hole annulus plane, evaluated along the ray through the polygon center.
    # Used for operator debug readout (stage1+stage2), even after the detector is disabled (post-confirm).
    hole_plane_center_range_m = float("nan")
    hole_err = ""
    hole_last_refine_ts = 0.0
    # Plane in world coordinates (updated by the hole detector during refine; used for movement-stage gating/reseed).
    hole_plane_w_n: Optional[np.ndarray] = None  # (3,)
    hole_plane_w_d = 0.0
    hole_plane_w_ok = False

    # Plane-selection metadata (set only for plane-based selection).
    plane_center_uv: Optional[tuple[float, float]] = None  # full-image pixels
    plane_r_px = 0.0
    plane_r_m = 0.0
    plane_ok_pid = False
    plane_plane_inliers = 0
    plane_plane_rms_m = 0.0
    plane_plane_cov = 0.0
    # Range (meters) from the camera to the cached plane patch, evaluated along the ray through the polygon center.
    # Used for PID_MOVE auto-exit on the server side (stop when close enough to the plane).
    plane_plane_center_range_m = float("nan")
    plane_err = ""
    # Plane in world coordinates (updated from the plane detector at selection time).
    plane_plane_w_n: Optional[np.ndarray] = None  # (3,)
    plane_plane_w_d = 0.0
    plane_plane_w_ok = False

    # Scratch/caches for Stage-1 hole-mask gating (allocated lazily to avoid per-call allocations).
    hole_gate_kernel_cache: dict[int, np.ndarray] = {}
    hole_trk_mask_u8: Optional[np.ndarray] = None
    hole_trk_dil_u8: Optional[np.ndarray] = None
    hole_fused_u8: Optional[np.ndarray] = None
    hole_fc_u8: Optional[np.ndarray] = None

    corr_params = np.zeros((4,), dtype=np.float64)  # tx,ty,theta,log_s
    corr_last_ts: Optional[float] = None

    prev_gray: Optional[np.ndarray] = None
    prev_Twc: Optional[np.ndarray] = None
    prev_corr_params: Optional[np.ndarray] = None

    poly_locked = False
    lock_used_uv: Optional[np.ndarray] = None
    last_inliers = 0
    last_n = 0
    last_reason = ""
    pts_target_n = int(pts_min)
    last_try_n = 0
    last_keep_ratio = float("nan")
    # Per-edge-point plane gating state (only used when `hole_plane_w_ok` and PID confirmed).
    edge_alive: Optional[np.ndarray] = None  # bool, shape (M,)
    edge_plane_bad: Optional[np.ndarray] = None  # uint8, shape (M,)
    _plane_reseed_last_t = 0.0

    # Scratch buffers for Numba ZNCC gate (allocated lazily).
    zncc_out_u8: Optional[np.ndarray] = None

    last_fi = -1
    fps = 0.0
    _fps_n = 0
    _fps_t0 = float(time.time())
    _last_dbg_t = 0.0

    # Perf counters (reported at low rate when enabled).
    _perf_t0 = float(time.time())
    _perf_n = 0
    _perf_sum_total_ms = 0.0
    _perf_sum_zncc_ms = 0.0
    _perf_zncc_n = 0
    _perf_sum_refine_apply_ms = 0.0
    _perf_ref_n = 0
    _perf_last_refine_det_ms = float("nan")

    # Headless Stage-0 hover preview (candidate polygon) state.
    hover_uv: Optional[tuple[int, int]] = None
    preview_last_t = 0.0
    preview_pending: Optional[dict] = None
    preview_cache = {
        "hole": None,  # type: Optional[dict]
        "plane": None,  # type: Optional[dict]
    }

    # VO-agnostic feature acquisition (optional) state.
    feat_anchor = None
    feat_last = None
    feat_last_t = 0.0
    feat_orb_A = None
    feat_orb_ok = False
    feat_orb_n = 0
    feat_orb_inl_n = 0
    feat_lines_n = 0
    feat_orb_uv_pub = None
    feat_lines_pub = None

    def _publish(*, fi: int, ts: float, reason: str) -> None:
        nonlocal preview_pending
        poly_uv_pub = poly_uv
        try:
            if bool(poly_active) and str(sel_kind) == "hole" and (not bool(pid_confirmed)) and ref_poly_uv is not None:
                poly_uv_pub = np.asarray(ref_poly_uv, dtype=np.float32).reshape(-1, 2)
        except Exception:
            poly_uv_pub = poly_uv

        msg = {
            "fi": int(fi),
            "ts": float(ts),
            "active": bool(poly_active),
            "sel_kind": str(sel_kind),
            "pid_confirmed": bool(pid_confirmed),
            "hole_center_uv": (tuple(float(v) for v in hole_center_uv) if hole_center_uv is not None else None),
            "hole_r_px": float(hole_r_px),
            "hole_r_m": float(hole_r_m),
            "hole_fill": float(hole_fill),
            "hole_ok_pid": bool(hole_ok_pid),
            "hole_plane_inliers": int(hole_plane_inliers),
            "hole_plane_rms_m": float(hole_plane_rms_m),
            "hole_plane_cov": float(hole_plane_cov),
            "hole_plane_center_range_m": float(hole_plane_center_range_m),
            "hole_err": str(hole_err),
            "plane_center_uv": (tuple(float(v) for v in plane_center_uv) if plane_center_uv is not None else None),
            "plane_r_px": float(plane_r_px),
            "plane_r_m": float(plane_r_m),
            "plane_ok_pid": bool(plane_ok_pid),
            "plane_plane_inliers": int(plane_plane_inliers),
            "plane_plane_rms_m": float(plane_plane_rms_m),
            "plane_plane_cov": float(plane_plane_cov),
            "plane_plane_center_range_m": float(plane_plane_center_range_m),
            "plane_err": str(plane_err),
            "poly_uv": (np.asarray(poly_uv_pub, dtype=np.float32).reshape(-1, 2) if poly_uv_pub is not None else None),
            "bbox": (tuple(poly_bbox) if poly_bbox is not None else None),
            "method": str("VO_LK") if bool(poly_active) else str("none"),
            "inliers": int(last_inliers),
            "n": int(last_n),
            "lines": int(0),
            "chi2": float("nan"),
            "kf_x": None,
            "miss_s": float(0.0),
            "miss_n": int(0),
            "fail": int(0),
            "margin_xy": (int(0), int(0)),
            "dir": str(""),
            "ld_lines": None,
            # Reuse legacy field names for GUI compatibility (these are LK points, not ORB).
            "orb_locked": bool(poly_locked),
            "orb_used_uv": (np.asarray(lock_used_uv, dtype=np.float32).reshape(-1, 2) if bool(show_pts) and lock_used_uv is not None else None),
            "orb_inliers": int(last_inliers),
            "orb_matches": int(last_n),
            "orb_kp": int(0),
            "orb_desc": int(0),
            "orb_good": int(last_inliers),
            "orb_anchor_n": int(0),
            "orb_roi_wh": (int(0), int(0)),
            "orb_mask_nz": int(0),
            "orb_gate": (int(0), int(0), int(0), int(0)),
            "orb_reason": str(last_reason),
            "feat_orb_ok": bool(feat_orb_ok),
            "feat_orb_n": int(feat_orb_n),
            "feat_orb_inliers": int(feat_orb_inl_n),
            "feat_lines_n": int(feat_lines_n),
            "feat_orb_uv": (
                np.asarray(feat_orb_uv_pub, dtype=np.float32).reshape(-1, 2)
                if feat_orb_uv_pub is not None and bool(feature_acq_publish_orb)
                else None
            ),
            "feat_lines_xyxy": (
                np.asarray(feat_lines_pub, dtype=np.float32).reshape(-1, 4)
                if feat_lines_pub is not None and bool(feature_acq_publish_lines)
                else None
            ),
            "try_n": int(last_try_n),
            "keep_ratio": float(last_keep_ratio),
            "corr_chi2": float("nan"),
            "corr_x4": [float(x) for x in np.asarray(corr_params, dtype=np.float64).reshape(4).tolist()],
            "fps": float(fps),
            "reason": str(reason),
        }
        if preview_pending is not None:
            try:
                msg["preview"] = dict(preview_pending)
            except Exception:
                msg["preview"] = preview_pending
            preview_pending = None
        try:
            out_q.put_nowait(msg)
            return
        except Exception:
            pass
        try:
            _ = out_q.get_nowait()
        except Exception:
            pass
        try:
            out_q.put_nowait(msg)
        except Exception:
            pass

    def _reset_state(*, keep_selection: bool = False) -> None:
        nonlocal poly_active, poly_uv, poly_bbox, poly_world_pts_w, edge_world_pts_w, edge_s, poly_z_prior_m
        nonlocal ref_poly_uv, ref_mask_u8, ref_mask_x0, ref_mask_y0, trk_poly_prev
        nonlocal sel_kind, pid_confirmed, hole_center_uv, hole_r_px, hole_r_m, hole_fill, hole_ok_pid
        nonlocal hole_plane_inliers, hole_plane_rms_m, hole_plane_cov, hole_err, hole_last_refine_ts
        nonlocal hole_plane_center_range_m
        nonlocal hole_plane_w_n, hole_plane_w_d, hole_plane_w_ok
        nonlocal plane_center_uv, plane_r_px, plane_r_m, plane_ok_pid
        nonlocal plane_plane_inliers, plane_plane_rms_m, plane_plane_cov, plane_plane_center_range_m, plane_err
        nonlocal plane_plane_w_n, plane_plane_w_d, plane_plane_w_ok
        nonlocal corr_params, corr_last_ts, prev_gray, prev_Twc, prev_corr_params
        nonlocal poly_locked, lock_used_uv, last_inliers, last_n, last_reason, pts_target_n, last_try_n, last_keep_ratio
        nonlocal edge_alive, edge_plane_bad, _plane_reseed_last_t
        # BEGIN_STAGE1_HYSTERESIS
        nonlocal stage1_best_r_px, stage1_shrink_votes
        # END_STAGE1_HYSTERESIS
        nonlocal stage2_auto_last_t, auto_lock_last_t
        nonlocal feat_anchor, feat_last, feat_last_t, feat_orb_A, feat_orb_ok, feat_orb_n, feat_orb_inl_n, feat_lines_n, feat_orb_uv_pub, feat_lines_pub
        poly_locked = False
        lock_used_uv = None
        last_inliers = 0
        last_n = 0
        last_reason = ""
        pts_target_n = int(pts_min)
        last_try_n = 0
        last_keep_ratio = float("nan")
        edge_alive = None
        edge_plane_bad = None
        _plane_reseed_last_t = 0.0
        corr_params = np.zeros((4,), dtype=np.float64)
        corr_last_ts = None
        prev_gray = None
        prev_Twc = None
        prev_corr_params = None
        ref_poly_uv = None
        ref_mask_u8 = None
        ref_mask_x0 = 0
        ref_mask_y0 = 0
        trk_poly_prev = None
        hole_plane_center_range_m = float("nan")
        plane_plane_center_range_m = float("nan")
        # BEGIN_STAGE1_HYSTERESIS
        stage1_best_r_px = float("nan")
        stage1_shrink_votes = int(0)
        # END_STAGE1_HYSTERESIS

        # Feature acquisition dynamic state (anchor may persist when preserving selection).
        feat_last = None
        feat_last_t = 0.0
        feat_orb_A = None
        feat_orb_ok = False
        feat_orb_n = 0
        feat_orb_inl_n = 0
        feat_lines_n = 0
        feat_orb_uv_pub = None
        feat_lines_pub = None

        if not bool(keep_selection):
            poly_active = False
            poly_uv = None
            poly_bbox = None
            poly_world_pts_w = None
            edge_world_pts_w = None
            edge_s = None
            poly_z_prior_m = None
            sel_kind = "none"
            pid_confirmed = False
            hole_center_uv = None
            hole_r_px = 0.0
            hole_r_m = 0.0
            hole_fill = 0.0
            hole_ok_pid = False
            hole_plane_inliers = 0
            hole_plane_rms_m = 0.0
            hole_plane_cov = 0.0
            hole_plane_center_range_m = float("nan")
            hole_err = ""
            hole_last_refine_ts = 0.0
            hole_plane_w_n = None
            hole_plane_w_d = 0.0
            hole_plane_w_ok = False

            plane_center_uv = None
            plane_r_px = 0.0
            plane_r_m = 0.0
            plane_ok_pid = False
            plane_plane_inliers = 0
            plane_plane_rms_m = 0.0
            plane_plane_cov = 0.0
            plane_plane_center_range_m = float("nan")
            plane_err = ""
            plane_plane_w_n = None
            plane_plane_w_d = 0.0
            plane_plane_w_ok = False
            feat_anchor = None
            stage2_auto_last_t = 0.0
            auto_lock_last_t = 0.0

    def _activate_selection_from_poly_uv(
        *,
        poly_uv0: np.ndarray,
        fi_sel: int,
        ts_sel: float,
        gray: np.ndarray,
        depth_raw,
        depth_mask_win: Optional[np.ndarray] = None,
        depth_mask_x0: int = 0,
        depth_mask_y0: int = 0,
        depth_sample_r_px: Optional[int] = None,
        depth_prefer_outside_mask: bool = False,
        lift_plane_w_n: Optional[np.ndarray] = None,
        lift_plane_w_d: Optional[float] = None,
        lift_plane_edge_offset_px: float = 0.0,
    ) -> None:
        nonlocal poly_active, poly_uv, poly_bbox, poly_world_pts_w, edge_world_pts_w, edge_s, poly_z_prior_m
        nonlocal corr_params, corr_last_ts, prev_gray, prev_Twc, prev_corr_params
        nonlocal poly_locked, lock_used_uv, last_inliers, last_n, last_reason, pts_target_n, last_try_n, last_keep_ratio
        nonlocal edge_alive, edge_plane_bad, _plane_reseed_last_t
        nonlocal sel_kind

        def _sel_fail(code: str, extra: str = "") -> None:
            nonlocal last_reason
            try:
                sk_prev = str(sel_kind)
            except Exception:
                sk_prev = "none"
            try:
                last_reason = str(code)
            except Exception:
                last_reason = ""
            if bool(dbg_log):
                try:
                    extra_s = str(extra).strip()
                    extra_s = f" {extra_s}" if extra_s else ""
                    print(f"[poly_vo_lk] select_fail fi={int(fi_sel)} code={str(code)}{extra_s}", flush=True)
                except Exception:
                    pass
            _reset_state(keep_selection=False)
            # Preserve the intended selection kind so external integrations can attribute failures.
            try:
                if str(sk_prev) in ("hole", "plane", "bbox"):
                    sel_kind = str(sk_prev)
            except Exception:
                pass
            try:
                last_reason = str(code)
            except Exception:
                last_reason = ""
            _publish(fi=int(fi_sel), ts=float(ts_sel), reason=str(code))

        try:
            poly_uv0 = np.asarray(poly_uv0, dtype=np.float32).reshape(-1, 2)
        except Exception:
            _sel_fail("poly_uv_invalid")
            return
        if int(poly_uv0.shape[0]) < 3:
            _sel_fail("poly_uv_small", extra=f"n={int(poly_uv0.shape[0])}")
            return

        h_img = int(gray.shape[0])
        w_img = int(gray.shape[1])
        poly_bbox0 = _bbox_from_poly_uv(poly_uv=poly_uv0, w=int(w_img), h=int(h_img), margin_px=0, min_size_px=2)
        if poly_bbox0 is None:
            _sel_fail("bbox_fail")
            return

        Twc_sel = None
        if state is not None:
            st = state.read(copy=False)
            if st is not None and int(len(st)) >= 12:
                try:
                    Twc_sel = np.asarray(st[11], dtype=np.float64).reshape(4, 4)
                except Exception:
                    Twc_sel = None
        if Twc_sel is None:
            _sel_fail("no_Twc")
            return
        if intr_obj is None:
            _sel_fail("no_intr")
            return

        # Optional: lift the selection polygon + edge points by intersecting with a fitted plane.
        # This is used by the 3D plane tracker to:
        # - keep the "lock" points on the selected plane (independent of depth dropouts)
        # - drive PID to the plane patch under the cursor (polygon is centered at the cursor)
        try:
            if lift_plane_w_n is not None and lift_plane_w_d is not None:
                plane_w_n0 = np.asarray(lift_plane_w_n, dtype=np.float64).reshape(3)
                plane_w_d0 = float(lift_plane_w_d)
            else:
                plane_w_n0 = None
                plane_w_d0 = 0.0
        except Exception:
            plane_w_n0 = None
            plane_w_d0 = 0.0

        if plane_w_n0 is not None and np.all(np.isfinite(plane_w_n0)) and np.isfinite(float(plane_w_d0)):
            Rwc0 = np.asarray(Twc_sel[:3, :3], dtype=np.float64).reshape(3, 3)
            twc0 = np.asarray(Twc_sel[:3, 3], dtype=np.float64).reshape(1, 3)
            fx = float(intr_obj.fx)
            fy = float(intr_obj.fy)
            cx0 = float(intr_obj.cx)
            cy0 = float(intr_obj.cy)
            if float(fx) <= 0.0 or float(fy) <= 0.0:
                _sel_fail("intr_bad", extra=f"fx={float(fx):.3f} fy={float(fy):.3f}")
                return

            try:
                plane_n_c, plane_d_c = _plane_cam_from_world(
                    n_w=np.asarray(plane_w_n0, dtype=np.float64).reshape(3),
                    d_w=float(plane_w_d0),
                    Twc_now=np.asarray(Twc_sel, dtype=np.float64).reshape(4, 4),
                )
            except Exception:
                _sel_fail("plane_cam_fail")
                return

            uvv = np.asarray(poly_uv0, dtype=np.float64).reshape(-1, 2)
            uu = uvv[:, 0]
            vv = uvv[:, 1]
            x = (uu - float(cx0)) / float(max(1e-9, fx))
            y = (vv - float(cy0)) / float(max(1e-9, fy))
            denom = (float(plane_n_c[0]) * x + float(plane_n_c[1]) * y + float(plane_n_c[2])).astype(np.float64, copy=False)
            valid = np.isfinite(denom) & (np.abs(denom) > 1e-9)
            z = np.full((int(denom.size),), float("nan"), dtype=np.float64)
            z[valid] = (-float(plane_d_c) / denom[valid]).astype(np.float64, copy=False)
            valid = valid & np.isfinite(z) & (z > 0.0)
            if int(np.count_nonzero(valid)) != int(uvv.shape[0]):
                _sel_fail("vert_plane_lift_fail")
                return
            poly_z_prior_m = float(np.median(np.asarray(z, dtype=np.float64).reshape(-1))) if int(z.size) > 0 else None

            Pc = np.stack([x * z, y * z, z], axis=1).astype(np.float64, copy=False)
            poly_world_pts_w = (Pc @ Rwc0.T) + twc0
            poly_world_pts_w = np.asarray(poly_world_pts_w, dtype=np.float64).reshape(-1, 3)

            # Sample many edge points (2D, evenly spaced by arc-length), then lift them to world on the plane.
            peri = float(_poly_perimeter_px(poly_uv=np.asarray(poly_uv0, dtype=np.float32)))
            step_px = float(max(1.0, float(peri) / float(max(1, int(pts_max)))))
            n_edge = int(max(4, int(np.ceil(float(peri) / float(step_px)))))
            n_edge = int(min(int(pts_max), int(n_edge)))
            s_edge = (np.arange(int(n_edge), dtype=np.float64) + 0.5) / float(max(1, int(n_edge)))
            uv_edge = _poly_sample_edge_points_by_s(poly_uv=np.asarray(poly_uv0, dtype=np.float32), s=np.asarray(s_edge, dtype=np.float64))
            if uv_edge is None or int(uv_edge.shape[0]) < int(min_fit_inliers):
                _sel_fail("edge_sample_fail", extra=f"peri={float(peri):.1f} step={float(step_px):.2f} n={int(0 if uv_edge is None else uv_edge.shape[0])}")
                return

            uv_edge_f = np.asarray(uv_edge, dtype=np.float64).reshape(-1, 2)
            off_px = float(max(0.0, float(lift_plane_edge_offset_px)))
            if float(off_px) > 1e-6:
                try:
                    cen = np.mean(np.asarray(poly_uv0, dtype=np.float64).reshape(-1, 2), axis=0).reshape(2)
                except Exception:
                    cen = np.asarray([0.0, 0.0], dtype=np.float64)
                dv = uv_edge_f - cen[None, :]
                dn = np.hypot(dv[:, 0], dv[:, 1]).reshape(-1, 1)
                good = dn[:, 0] > 1e-6
                dv2 = np.zeros_like(dv, dtype=np.float64)
                dv2[good] = dv[good] / dn[good]
                uv_edge_f = uv_edge_f + float(off_px) * dv2

            uu_e = uv_edge_f[:, 0]
            vv_e = uv_edge_f[:, 1]
            x_e = (uu_e - float(cx0)) / float(max(1e-9, fx))
            y_e = (vv_e - float(cy0)) / float(max(1e-9, fy))
            denom_e = (float(plane_n_c[0]) * x_e + float(plane_n_c[1]) * y_e + float(plane_n_c[2])).astype(np.float64, copy=False)
            valid_e = np.isfinite(denom_e) & (np.abs(denom_e) > 1e-9)
            z_e = np.full((int(denom_e.size),), float("nan"), dtype=np.float64)
            z_e[valid_e] = (-float(plane_d_c) / denom_e[valid_e]).astype(np.float64, copy=False)
            valid_e = valid_e & np.isfinite(z_e) & (z_e > 0.0)
            if int(np.count_nonzero(valid_e)) < int(min_fit_inliers):
                _sel_fail("edge_plane_lift_fail", extra=f"n={int(np.count_nonzero(valid_e))} min={int(min_fit_inliers)}")
                return
            Pc_e = np.stack([x_e * z_e, y_e * z_e, z_e], axis=1).astype(np.float64, copy=False)
            Pw_e = (Pc_e @ Rwc0.T) + twc0
            edge_world_pts_w = np.asarray(Pw_e, dtype=np.float64).reshape(-1, 3)[valid_e]
            try:
                edge_s = np.asarray(s_edge, dtype=np.float64).reshape(-1)[valid_e]
            except Exception:
                edge_s = None
            edge_alive = np.ones((int(edge_world_pts_w.shape[0]),), dtype=np.bool_)
            edge_plane_bad = np.zeros((int(edge_world_pts_w.shape[0]),), dtype=np.uint8)
            _plane_reseed_last_t = 0.0

            poly_active = True
            poly_uv = np.asarray(poly_uv0, dtype=np.float32).reshape(-1, 2)
            poly_bbox = tuple(poly_bbox0)
            poly_locked = False
            lock_used_uv = None
            last_inliers = 0
            last_n = 0
            last_reason = "select_ok"
            pts_target_n = int(pts_min)
            last_try_n = int(pts_target_n)
            last_keep_ratio = float("nan")
            corr_params = np.zeros((4,), dtype=np.float64)
            corr_last_ts = float(ts_sel)
            prev_gray = np.asarray(gray, dtype=np.uint8).copy()
            prev_Twc = np.asarray(Twc_sel, dtype=np.float64).reshape(4, 4).copy()
            prev_corr_params = np.asarray(corr_params, dtype=np.float64).copy()

            if bool(dbg_log):
                try:
                    print(
                        f"[poly_vo_lk] select_ok (plane_lift) verts={int(poly_uv0.shape[0])} edge_pts={int(edge_world_pts_w.shape[0])}",
                        flush=True,
                    )
                except Exception:
                    pass
            _publish(fi=int(fi_sel), ts=float(ts_sel), reason="select_ok")
            return

        if float(depth_units) <= 0.0 or depth_raw is None:
            _sel_fail("no_depth")
            return

        try:
            depth_raw = np.asarray(depth_raw, dtype=np.uint16)
        except Exception:
            _sel_fail("depth_cast")
            return
        if depth_raw.ndim != 2 or int(depth_raw.shape[0]) != int(h_img) or int(depth_raw.shape[1]) != int(w_img):
            _sel_fail("depth_shape", extra=f"got={tuple(int(x) for x in depth_raw.shape)} img=({int(h_img)},{int(w_img)})")
            return

        Rwc0 = np.asarray(Twc_sel[:3, :3], dtype=np.float64).reshape(3, 3)
        twc0 = np.asarray(Twc_sel[:3, 3], dtype=np.float64).reshape(1, 3)
        fx = float(intr_obj.fx)
        fy = float(intr_obj.fy)
        cx0 = float(intr_obj.cx)
        cy0 = float(intr_obj.cy)
        if float(fx) <= 0.0 or float(fy) <= 0.0:
            _sel_fail("intr_bad", extra=f"fx={float(fx):.3f} fy={float(fy):.3f}")
            return

        r_depth_use = int(depth_r_px if depth_sample_r_px is None else max(0, int(depth_sample_r_px)))

        def _depth_stats(u: float, v: float) -> tuple[int, float, int]:
            uu = int(round(float(u)))
            vv = int(round(float(v)))
            if bool(depth_prefer_outside_mask) and depth_mask_win is not None:
                # Try outside-mask depth first (often needed for holes where depth inside is invalid).
                for rr in (int(r_depth_use), int(max(r_depth_use + 2, r_depth_use)), int(max(r_depth_use + 4, r_depth_use))):
                    med_u16, mad_u16, nn = _sample_depth_stats_u16_outside_mask(
                        np.asarray(depth_raw, dtype=np.uint16),
                        x=int(uu),
                        y=int(vv),
                        r=int(rr),
                        mask_win=depth_mask_win,
                        mask_x0=int(depth_mask_x0),
                        mask_y0=int(depth_mask_y0),
                    )
                    if int(nn) > 0 and int(med_u16) > 0:
                        return int(med_u16), float(mad_u16), int(nn)
            return _sample_depth_stats_u16(np.asarray(depth_raw, dtype=np.uint16), int(uu), int(vv), int(r_depth_use))

        # Depth prior from vertices.
        z_list: list[float] = []
        for u0, v0 in np.asarray(poly_uv0, dtype=np.float64).reshape(-1, 2).tolist():
            med_u16, _mad, nn = _depth_stats(float(u0), float(v0))
            if int(med_u16) <= 0 or int(nn) <= 0:
                continue
            z = float(med_u16) * float(depth_units)
            if not np.isfinite(z) or z <= 0.0:
                continue
            z_list.append(float(z))
        poly_z_prior_m = float(np.median(np.asarray(z_list, dtype=np.float64))) if z_list else None

        # Lift polygon vertices.
        Pw_list: list[list[float]] = []
        ok_all = True
        for u0, v0 in np.asarray(poly_uv0, dtype=np.float64).reshape(-1, 2).tolist():
            med_u16, _mad, nn = _depth_stats(float(u0), float(v0))
            z = float(med_u16) * float(depth_units) if int(med_u16) > 0 else float("nan")
            if (not np.isfinite(z)) or z <= 0.0:
                if poly_z_prior_m is not None and np.isfinite(float(poly_z_prior_m)) and float(poly_z_prior_m) > 0.0:
                    z = float(poly_z_prior_m)
            if (not np.isfinite(z)) or z <= 0.0:
                ok_all = False
                break
            x = (float(u0) - float(cx0)) * float(z) / float(fx)
            y = (float(v0) - float(cy0)) * float(z) / float(fy)
            Pc = np.asarray([[float(x), float(y), float(z)]], dtype=np.float64)
            Pw = (Pc @ Rwc0.T) + twc0
            Pw_list.append([float(Pw[0, 0]), float(Pw[0, 1]), float(Pw[0, 2])])
        if not bool(ok_all) or int(len(Pw_list)) != int(np.asarray(poly_uv0).shape[0]):
            _sel_fail("vert_lift_fail", extra=f"n={int(poly_uv0.shape[0])} z_ok={int(len(z_list))} z_prior={float(poly_z_prior_m) if poly_z_prior_m is not None else float('nan'):.3f}")
            return
        poly_world_pts_w = np.asarray(Pw_list, dtype=np.float64).reshape(-1, 3)

        # Sample many edge points (2D, evenly spaced by arc-length), then lift them to world.
        peri = float(_poly_perimeter_px(poly_uv=np.asarray(poly_uv0, dtype=np.float32)))
        step_px = float(max(1.0, float(peri) / float(max(1, int(pts_max)))))
        n_edge = int(max(4, int(np.ceil(float(peri) / float(step_px)))))
        n_edge = int(min(int(pts_max), int(n_edge)))
        s_edge = (np.arange(int(n_edge), dtype=np.float64) + 0.5) / float(max(1, int(n_edge)))
        uv_edge = _poly_sample_edge_points_by_s(poly_uv=np.asarray(poly_uv0, dtype=np.float32), s=np.asarray(s_edge, dtype=np.float64))
        if uv_edge is None or int(uv_edge.shape[0]) < int(min_fit_inliers):
            _sel_fail("edge_sample_fail", extra=f"peri={float(peri):.1f} step={float(step_px):.2f} n={int(0 if uv_edge is None else uv_edge.shape[0])}")
            return

        Pw_edge: list[list[float]] = []
        s_keep: list[float] = []
        uv_edge_f = np.asarray(uv_edge, dtype=np.float64).reshape(-1, 2)
        for i_e, (u0, v0) in enumerate(uv_edge_f.tolist()):
            med_u16, _mad, nn = _depth_stats(float(u0), float(v0))
            z = float(med_u16) * float(depth_units) if int(med_u16) > 0 else float("nan")
            if (not np.isfinite(z)) or z <= 0.0:
                if poly_z_prior_m is not None and np.isfinite(float(poly_z_prior_m)) and float(poly_z_prior_m) > 0.0:
                    z = float(poly_z_prior_m)
            if (not np.isfinite(z)) or z <= 0.0:
                continue
            x = (float(u0) - float(cx0)) * float(z) / float(fx)
            y = (float(v0) - float(cy0)) * float(z) / float(fy)
            Pc = np.asarray([[float(x), float(y), float(z)]], dtype=np.float64)
            Pw = (Pc @ Rwc0.T) + twc0
            Pw_edge.append([float(Pw[0, 0]), float(Pw[0, 1]), float(Pw[0, 2])])
            try:
                s_keep.append(float(s_edge[int(i_e)]))
            except Exception:
                pass
        if int(len(Pw_edge)) < int(min_fit_inliers):
            _sel_fail("edge_lift_fail", extra=f"n={int(len(Pw_edge))} min={int(min_fit_inliers)}")
            return
        edge_world_pts_w = np.asarray(Pw_edge, dtype=np.float64).reshape(-1, 3)
        edge_s = np.asarray(s_keep, dtype=np.float64).reshape(-1) if s_keep else None
        edge_alive = np.ones((int(edge_world_pts_w.shape[0]),), dtype=np.bool_)
        edge_plane_bad = np.zeros((int(edge_world_pts_w.shape[0]),), dtype=np.uint8)
        _plane_reseed_last_t = 0.0

        poly_active = True
        poly_uv = np.asarray(poly_uv0, dtype=np.float32).reshape(-1, 2)
        poly_bbox = tuple(poly_bbox0)
        poly_locked = False
        lock_used_uv = None
        last_inliers = 0
        last_n = 0
        last_reason = "select_ok"
        pts_target_n = int(pts_min)
        last_try_n = int(pts_target_n)
        last_keep_ratio = float("nan")
        corr_params = np.zeros((4,), dtype=np.float64)
        corr_last_ts = float(ts_sel)
        prev_gray = np.asarray(gray, dtype=np.uint8).copy()
        prev_Twc = np.asarray(Twc_sel, dtype=np.float64).reshape(4, 4).copy()
        prev_corr_params = np.asarray(corr_params, dtype=np.float64).copy()

        if bool(dbg_log):
            try:
                print(f"[poly_vo_lk] select_ok verts={int(poly_uv0.shape[0])} edge_pts={int(edge_world_pts_w.shape[0])}", flush=True)
            except Exception:
                pass

        _publish(fi=int(fi_sel), ts=float(ts_sel), reason="select_ok")

    def _handle_select_bbox(msg: dict) -> None:
        nonlocal poly_active, poly_uv, poly_bbox, poly_world_pts_w, edge_world_pts_w, poly_z_prior_m
        nonlocal ref_poly_uv, ref_mask_u8, ref_mask_x0, ref_mask_y0, trk_poly_prev
        nonlocal sel_kind, pid_confirmed, hole_center_uv, hole_r_px, hole_r_m, hole_fill, hole_ok_pid
        nonlocal hole_plane_inliers, hole_plane_rms_m, hole_plane_cov, hole_err, hole_last_refine_ts
        nonlocal hole_plane_w_n, hole_plane_w_d, hole_plane_w_ok
        nonlocal corr_params, corr_last_ts, prev_gray, prev_Twc, prev_corr_params
        nonlocal poly_locked, lock_used_uv, last_inliers, last_n, last_reason, pts_target_n, last_try_n, last_keep_ratio

        bbox = msg.get("bbox", None)
        if not (isinstance(bbox, (list, tuple)) and int(len(bbox)) == 4):
            return
        fi_sel = int(msg.get("fi", -1))
        ts_sel = float(msg.get("ts", 0.0))
        slot = ring.find_slot_by_frame_idx(int(fi_sel))
        if slot is None:
            latest = ring.read_latest()
            if latest is None:
                return
            slot, fi_sel, ts_sel = latest

        if bool(dbg_log):
            try:
                print(f"[poly_vo_lk] select_start fi={int(fi_sel)} bbox={tuple(int(x) for x in bbox)}", flush=True)
            except Exception:
                pass

        gray = np.asarray(ring.gray[int(slot)], dtype=np.uint8)
        depth_raw = None
        try:
            if bool(cap_cfg.get("depth", True)) and float(depth_units) > 0.0:
                depth_raw = np.asarray(ring.depth_raw[int(slot)], dtype=np.uint16)
        except Exception:
            depth_raw = None

        _t_frame0 = float(time.perf_counter()) if bool(perf_log) else 0.0

        poly = _poly_select_from_bbox(
            gray=gray,
            bbox=tuple(bbox),
            canny_lo=int(canny_lo),
            canny_hi=int(canny_hi),
            edge_kernel=edge_kernel,
            concavity_max=float(poly_concavity_max),
            approx_eps_frac=float(poly_approx_eps_frac),
            min_vertex_angle_deg=float(poly_min_vertex_angle_deg),
            max_vertices=int(poly_max_vertices),
            min_edge_len_px=float(poly_min_edge_len_px),
        )
        if poly is None or int(np.asarray(poly).shape[0]) < 3:
            last_reason = "bbox_no_poly"
            if bool(dbg_log):
                try:
                    print(f"[poly_vo_lk] select_fail fi={int(fi_sel)} code=bbox_no_poly", flush=True)
                except Exception:
                    pass
            _reset_state(keep_selection=False)
            last_reason = "bbox_no_poly"
            _publish(fi=int(fi_sel), ts=float(ts_sel), reason="bbox_no_poly")
            return

        sel_kind = "bbox"
        ref_poly_uv = None
        ref_mask_u8 = None
        ref_mask_x0 = 0
        ref_mask_y0 = 0
        trk_poly_prev = None
        pid_confirmed = False
        hole_center_uv = None
        hole_r_px = 0.0
        hole_r_m = 0.0
        hole_fill = 0.0
        hole_ok_pid = False
        hole_plane_inliers = 0
        hole_plane_rms_m = 0.0
        hole_plane_cov = 0.0
        hole_err = ""
        hole_last_refine_ts = 0.0
        hole_plane_w_ok = False
        hole_plane_w_n = None
        hole_plane_w_d = 0.0
        _activate_selection_from_poly_uv(poly_uv0=np.asarray(poly, dtype=np.float32).reshape(-1, 2), fi_sel=int(fi_sel), ts_sel=float(ts_sel), gray=gray, depth_raw=depth_raw)

    hole_settings = HoleSettings(
        unknown_as_full=bool(hole_unknown_as_full),
        hole_seed_snap_px=int(hole_select_snap_px),
        hole_seed_snap_max_center_px=int(hole_select_snap_px),
        leak_guard_enabled=bool(hole_leak_guard_enabled),
        leak_open_max_px=int(hole_leak_open_max_px),
        leak_border_fill_frac=float(hole_leak_border_fill_frac),
        leak_full_fill_frac=float(hole_leak_full_fill_frac),
        plane_ransac_numba=bool(hole_plane_ransac_numba),
        plane_sector_count=int(hole_plane_sector_count),
        plane_sector_min_frac=float(hole_plane_sector_min_frac),
        plane_normals_enabled=bool(hole_plane_norm_enabled),
        plane_normals_step_px=int(max(1, int(hole_plane_norm_step_px))),
        plane_normals_min_mag=float(max(0.0, float(hole_plane_norm_min_mag))),
        plane_normals_max_angle_deg=float(max(0.0, float(hole_plane_norm_max_angle_deg))),
        plane_normals_min_valid_frac=float(max(0.0, min(1.0, float(hole_plane_norm_min_valid_frac)))),
        edge_limit_enabled=bool(hole_edge_enabled),
        edge_limit_mode=str(hole_edge_mode),
        edge_method=str(hole_edge_method),
        edge_canny_lo=int(hole_edge_canny_lo),
        edge_canny_hi=int(hole_edge_canny_hi),
        edge_canny_L2gradient=bool(hole_edge_canny_l2),
        edge_laplace_ksize=int(max(1, int(hole_edge_laplace_ksize))),
        edge_threshold=int(max(0, int(hole_edge_threshold))),
        edge_morph_ksize=int(max(1, int(hole_edge_morph_ksize))),
        edge_morph_iter=int(max(0, int(hole_edge_morph_iter))),
        edge_dilate_px=int(max(0, int(hole_edge_dilate_px))),
    )
    # While locked/refining keep snapping much tighter to avoid jumping to a different opening.
    hole_settings_lock = replace(
        hole_settings,
        hole_seed_snap_px=int(hole_lock_snap_px),
        hole_seed_snap_max_center_px=int(hole_lock_snap_px),
    )
    hole_settings_preview = replace(
        hole_settings,
        hole_seed_snap_px=int(hole_preview_snap_px),
        hole_seed_snap_max_center_px=int(hole_preview_snap_px),
    )
    # If preview snap is disabled (0), allow a small fallback snap (use lock snap) so we can still propose a polygon
    # when hovering near an opening edge.
    hole_settings_preview_fallback = replace(
        hole_settings_preview,
        hole_seed_snap_px=int(hole_lock_snap_px),
        hole_seed_snap_max_center_px=int(hole_lock_snap_px),
    )
    hole = HoleDetector(settings=hole_settings, spatial=hole_spatial)

    # Hole-refinement worker thread (runs expensive depth floodfill + plane fit asynchronously).
    # This keeps the per-frame LK loop responsive and allows later tuning of the refine rate independently.
    hole_refine_req_q: Optional[queue.Queue] = None
    hole_refine_res_q: Optional[queue.Queue] = None
    hole_refine_busy = threading.Event()
    hole_refine_stop = threading.Event()
    hole_refine_worker: Optional[threading.Thread] = None
    hole_refine_gray_buf: Optional[np.ndarray] = None
    hole_refine_depth_buf: Optional[np.ndarray] = None
    hole_refine_last_applied_fi = -1

    def _hole_refine_worker_main() -> None:
        # Use an isolated detector instance so scratch buffers stay thread-local.
        hdet = HoleDetector(settings=hole_settings, spatial=hole_spatial)
        while (not stop_event.is_set()) and (not hole_refine_stop.is_set()):
            try:
                req = hole_refine_req_q.get(timeout=0.1) if hole_refine_req_q is not None else None
            except Exception:
                req = None
            if req is None:
                continue
            try:
                fi0 = int(req.get("fi", -1))
                ts0 = float(req.get("ts", 0.0))
                x0_det = int(req.get("x0", 0))
                y0_det = int(req.get("y0", 0))
                ww_det = int(req.get("ww", 0))
                hh_det = int(req.get("hh", 0))
                x_seed = int(req.get("x", 0))
                y_seed = int(req.get("y", 0))
                gray0 = req.get("gray", None)
                depth0 = req.get("depth_raw", None)

                t0 = float(time.perf_counter())
                m_u8, p_u8 = hdet.detect_hole_plane(
                    {"bgr": None, "gray": gray0, "depth_raw": depth0, "depth_units": float(depth_units), "intr": intr_obj},
                    window_wh=(int(hole_win_w), int(hole_win_h)),
                    x=int(x_seed),
                    y=int(y_seed),
                    settings=hole_settings_lock,
                )
                dt_ms = 1000.0 * float(time.perf_counter() - float(t0))

                err_s = str(getattr(hdet, "last_error", "") or "")
                try:
                    plane_ok = bool(getattr(hdet, "last_plane_ok", False))
                except Exception:
                    plane_ok = False
                try:
                    n_cam = tuple(float(x) for x in np.asarray(getattr(hdet, "last_plane_n_cam", (0.0, 0.0, 0.0)), dtype=np.float64).reshape(3))
                    d_cam = float(getattr(hdet, "last_plane_d_cam", 0.0))
                except Exception:
                    n_cam = (0.0, 0.0, 0.0)
                    d_cam = 0.0
                try:
                    cx_full, cy_full = getattr(hdet, "last_circle_center_full_px", (-1, -1))
                    cx_full = int(cx_full)
                    cy_full = int(cy_full)
                except Exception:
                    cx_full, cy_full = -1, -1

                res = {
                    "fi": int(fi0),
                    "ts": float(ts0),
                    "x0": int(x0_det),
                    "y0": int(y0_det),
                    "ww": int(ww_det),
                    "hh": int(hh_det),
                    "mask_u8": m_u8,
                    "plane_u8": p_u8,
                    "err": str(err_s),
                    "dt_ms": float(dt_ms),
                    "plane_ok": bool(plane_ok),
                    "plane_n_cam": tuple(n_cam),
                    "plane_d_cam": float(d_cam),
                    "plane_inliers": int(getattr(hdet, "last_plane_inliers", 0)),
                    "plane_rms_m": float(getattr(hdet, "last_plane_rms_m", 0.0)),
                    "plane_cov": float(getattr(hdet, "last_plane_sector_cov", 0.0)),
                    "circle_r_px": float(getattr(hdet, "last_circle_radius_px", 0.0)),
                    "radius_m": float(getattr(hdet, "last_radius_m", 0.0)),
                    "fill": float(getattr(hdet, "last_circle_fill_ratio", 0.0)),
                    "center_full": (float(cx_full), float(cy_full)) if (cx_full >= 0 and cy_full >= 0) else None,
                    "touches_border": bool(getattr(hdet, "last_hole_touches_border", False)),
                }
            except Exception:
                res = {
                    "fi": int(req.get("fi", -1)),
                    "ts": float(req.get("ts", 0.0)),
                    "x0": int(req.get("x0", 0)),
                    "y0": int(req.get("y0", 0)),
                    "ww": int(req.get("ww", 0)),
                    "hh": int(req.get("hh", 0)),
                    "mask_u8": None,
                    "plane_u8": None,
                    "err": "hole_refine_exc",
                    "dt_ms": float("nan"),
                    "plane_ok": False,
                    "plane_n_cam": (0.0, 0.0, 0.0),
                    "plane_d_cam": 0.0,
                    "plane_inliers": 0,
                    "plane_rms_m": 0.0,
                    "plane_cov": 0.0,
                    "circle_r_px": 0.0,
                    "radius_m": 0.0,
                    "fill": 0.0,
                    "center_full": None,
                    "touches_border": False,
                }

            # Keep only the newest result (drop older ones).
            if hole_refine_res_q is not None:
                try:
                    while True:
                        _ = hole_refine_res_q.get_nowait()
                except Exception:
                    pass
                try:
                    hole_refine_res_q.put_nowait(res)
                except Exception:
                    pass
            hole_refine_busy.clear()

    if bool(hole_refine_thread):
        try:
            hole_refine_req_q = queue.Queue(maxsize=1)
            hole_refine_res_q = queue.Queue(maxsize=1)
            hole_refine_worker = threading.Thread(target=_hole_refine_worker_main, name="hole_refine", daemon=True)
            hole_refine_worker.start()
        except Exception:
            hole_refine_req_q = None
            hole_refine_res_q = None
            hole_refine_worker = None
            hole_refine_thread = False

    def _plane_world_from_cam(*, n_cam: np.ndarray, d_cam: float, Twc_now: np.ndarray) -> tuple[np.ndarray, float]:
        Rwc = np.asarray(Twc_now[:3, :3], dtype=np.float64).reshape(3, 3)
        twc = np.asarray(Twc_now[:3, 3], dtype=np.float64).reshape(3)
        n_cam = np.asarray(n_cam, dtype=np.float64).reshape(3)
        n_w = (Rwc @ n_cam.reshape(3, 1)).reshape(3)
        d_w = float(d_cam) - float(np.dot(n_w, twc))
        nn = float(np.linalg.norm(n_w))
        if np.isfinite(nn) and nn > 1e-9:
            n_w = n_w / float(nn)
            d_w = float(d_w) / float(nn)
        return np.asarray(n_w, dtype=np.float64).reshape(3), float(d_w)

    def _plane_cam_from_world(*, n_w: np.ndarray, d_w: float, Twc_now: np.ndarray) -> tuple[np.ndarray, float]:
        Rwc = np.asarray(Twc_now[:3, :3], dtype=np.float64).reshape(3, 3)
        twc = np.asarray(Twc_now[:3, 3], dtype=np.float64).reshape(3)
        n_w = np.asarray(n_w, dtype=np.float64).reshape(3)
        n_c = (Rwc.T @ n_w.reshape(3, 1)).reshape(3)
        d_c = float(np.dot(n_w, twc) + float(d_w))
        return np.asarray(n_c, dtype=np.float64).reshape(3), float(d_c)

    def _update_hole_plane_world_from_detector(
        *,
        Twc_now: Optional[np.ndarray],
        plane_ok: Optional[bool] = None,
        n_cam: Optional[np.ndarray] = None,
        d_cam: Optional[float] = None,
    ) -> None:
        nonlocal hole_plane_w_n, hole_plane_w_d, hole_plane_w_ok
        if Twc_now is None:
            hole_plane_w_ok = False
            hole_plane_w_n = None
            hole_plane_w_d = 0.0
            return
        if plane_ok is None:
            try:
                ok = bool(getattr(hole, "last_plane_ok", False))
            except Exception:
                ok = False
        else:
            ok = bool(plane_ok)
        if not bool(ok):
            hole_plane_w_ok = False
            hole_plane_w_n = None
            hole_plane_w_d = 0.0
            return
        if n_cam is None or d_cam is None:
            try:
                n_cam = np.asarray(getattr(hole, "last_plane_n_cam", (0.0, 0.0, 0.0)), dtype=np.float64).reshape(3)
                d_cam = float(getattr(hole, "last_plane_d_cam", 0.0))
            except Exception:
                hole_plane_w_ok = False
                hole_plane_w_n = None
                hole_plane_w_d = 0.0
                return
        if not (np.all(np.isfinite(np.asarray(n_cam, dtype=np.float64))) and np.isfinite(float(d_cam))):
            hole_plane_w_ok = False
            hole_plane_w_n = None
            hole_plane_w_d = 0.0
            return
        try:
            n_w, d_w = _plane_world_from_cam(
                n_cam=np.asarray(n_cam, dtype=np.float64).reshape(3),
                d_cam=float(d_cam),
                Twc_now=np.asarray(Twc_now, dtype=np.float64).reshape(4, 4),
            )
            hole_plane_w_n = np.asarray(n_w, dtype=np.float64).reshape(3)
            hole_plane_w_d = float(d_w)
            hole_plane_w_ok = True
        except Exception:
            hole_plane_w_ok = False
            hole_plane_w_n = None
            hole_plane_w_d = 0.0

    def _update_plane_plane_world_from_detector(
        *,
        Twc_now: Optional[np.ndarray],
        plane_ok: Optional[bool] = None,
        n_cam: Optional[np.ndarray] = None,
        d_cam: Optional[float] = None,
    ) -> None:
        nonlocal plane_plane_w_n, plane_plane_w_d, plane_plane_w_ok
        if Twc_now is None:
            plane_plane_w_ok = False
            plane_plane_w_n = None
            plane_plane_w_d = 0.0
            return
        if plane_ok is None:
            try:
                ok = bool(getattr(hole, "last_plane_ok", False))
            except Exception:
                ok = False
        else:
            ok = bool(plane_ok)
        if not bool(ok):
            plane_plane_w_ok = False
            plane_plane_w_n = None
            plane_plane_w_d = 0.0
            return
        if n_cam is None or d_cam is None:
            try:
                n_cam = np.asarray(getattr(hole, "last_plane_n_cam", (0.0, 0.0, 0.0)), dtype=np.float64).reshape(3)
                d_cam = float(getattr(hole, "last_plane_d_cam", 0.0))
            except Exception:
                plane_plane_w_ok = False
                plane_plane_w_n = None
                plane_plane_w_d = 0.0
                return
        if not (np.all(np.isfinite(np.asarray(n_cam, dtype=np.float64))) and np.isfinite(float(d_cam))):
            plane_plane_w_ok = False
            plane_plane_w_n = None
            plane_plane_w_d = 0.0
            return
        try:
            n_w, d_w = _plane_world_from_cam(
                n_cam=np.asarray(n_cam, dtype=np.float64).reshape(3),
                d_cam=float(d_cam),
                Twc_now=np.asarray(Twc_now, dtype=np.float64).reshape(4, 4),
            )
            plane_plane_w_n = np.asarray(n_w, dtype=np.float64).reshape(3)
            plane_plane_w_d = float(d_w)
            plane_plane_w_ok = True
        except Exception:
            plane_plane_w_ok = False
            plane_plane_w_n = None
            plane_plane_w_d = 0.0

    def _reseed_edge_points_on_plane(
        *,
        poly_uv_img: np.ndarray,
        Twc_now: np.ndarray,
        plane_w_n: np.ndarray,
        plane_w_d: float,
        edge_offset_px: float,
    ) -> Optional[np.ndarray]:
        nonlocal edge_s
        if intr_obj is None:
            return None
        try:
            poly_uv_img = np.asarray(poly_uv_img, dtype=np.float32).reshape(-1, 2)
        except Exception:
            return None
        if int(poly_uv_img.shape[0]) < 3:
            return None
        try:
            plane_w_n = np.asarray(plane_w_n, dtype=np.float64).reshape(3)
            plane_w_d = float(plane_w_d)
        except Exception:
            return None
        if not (np.all(np.isfinite(np.asarray(plane_w_n, dtype=np.float64))) and np.isfinite(float(plane_w_d))):
            return None

        try:
            peri = float(_poly_perimeter_px(poly_uv=np.asarray(poly_uv_img, dtype=np.float32)))
        except Exception:
            peri = 0.0
        if not np.isfinite(peri) or float(peri) <= 1e-6:
            return None
        step_px = float(max(1.0, float(peri) / float(max(4, int(pts_max)))))
        # Keep the edge-point pool stable: prefer reusing the previous arc-length sampling `edge_s` when available.
        try:
            prev_n = int(np.asarray(edge_s, dtype=np.float64).size) if edge_s is not None else 0
        except Exception:
            prev_n = 0
        n_edge = int(max(int(min_fit_inliers), int(np.ceil(float(peri) / float(step_px)))))
        n_edge = int(min(int(pts_max), int(n_edge)))
        if int(prev_n) >= int(min_fit_inliers):
            n_edge = int(min(int(n_edge), int(prev_n)))
        if int(n_edge) < int(min_fit_inliers):
            return None
        if edge_s is not None and int(np.asarray(edge_s).size) == int(n_edge):
            s_use = np.asarray(edge_s, dtype=np.float64).reshape(-1)
        else:
            s_use = (np.arange(int(n_edge), dtype=np.float64) + 0.5) / float(max(1, int(n_edge)))
        uv_edge = _poly_sample_edge_points_by_s(poly_uv=np.asarray(poly_uv_img, dtype=np.float32), s=np.asarray(s_use, dtype=np.float64))
        if uv_edge is None or int(np.asarray(uv_edge).shape[0]) < int(min_fit_inliers):
            return None
        uv_edge = np.asarray(uv_edge, dtype=np.float64).reshape(-1, 2)

        # Offset edge points away from the hole interior (toward the surrounding plane) to avoid tracking points "through" the opening.
        try:
            cen = np.mean(np.asarray(poly_uv_img, dtype=np.float64).reshape(-1, 2), axis=0).reshape(2)
        except Exception:
            cen = np.asarray([0.0, 0.0], dtype=np.float64)
        off_px = float(max(0.0, float(edge_offset_px)))
        if float(off_px) > 1e-6:
            dv = uv_edge - cen[None, :]
            dn = np.hypot(dv[:, 0], dv[:, 1]).reshape(-1, 1)
            good = dn[:, 0] > 1e-6
            dv2 = np.zeros_like(dv, dtype=np.float64)
            dv2[good] = dv[good] / dn[good]
            uv_edge = uv_edge + float(off_px) * dv2

        # Plane in current camera frame.
        try:
            n_c, d_c = _plane_cam_from_world(
                n_w=np.asarray(plane_w_n, dtype=np.float64).reshape(3),
                d_w=float(plane_w_d),
                Twc_now=Twc_now,
            )
        except Exception:
            return None

        fx = float(intr_obj.fx)
        fy = float(intr_obj.fy)
        cx0 = float(intr_obj.cx)
        cy0 = float(intr_obj.cy)
        if float(fx) <= 0.0 or float(fy) <= 0.0:
            return None

        Rwc = np.asarray(Twc_now[:3, :3], dtype=np.float64).reshape(3, 3)
        twc = np.asarray(Twc_now[:3, 3], dtype=np.float64).reshape(1, 3)
        h_img = int(ring.spec.h)
        w_img = int(ring.spec.w)

        uv = np.asarray(uv_edge, dtype=np.float64).reshape(-1, 2)
        uu = np.clip(uv[:, 0], 0.0, float(max(0, int(w_img - 1)))).astype(np.float64, copy=False)
        vv = np.clip(uv[:, 1], 0.0, float(max(0, int(h_img - 1)))).astype(np.float64, copy=False)
        x = (uu - float(cx0)) / float(max(1e-9, fx))
        y = (vv - float(cy0)) / float(max(1e-9, fy))
        denom = (float(n_c[0]) * x + float(n_c[1]) * y + float(n_c[2])).astype(np.float64, copy=False)
        valid = np.isfinite(denom) & (np.abs(denom) > 1e-9)
        if int(np.count_nonzero(valid)) < int(min_fit_inliers):
            return None
        z = np.full((int(denom.size),), float("nan"), dtype=np.float64)
        z[valid] = (-float(d_c) / denom[valid]).astype(np.float64, copy=False)
        valid = valid & np.isfinite(z) & (z > 0.0)
        if int(np.count_nonzero(valid)) < int(min_fit_inliers):
            return None
        Pc = np.stack([x * z, y * z, z], axis=1).astype(np.float64, copy=False)
        Pw = (Pc @ Rwc.T) + twc
        Pw = np.asarray(Pw, dtype=np.float64).reshape(-1, 3)[valid]
        if int(Pw.shape[0]) < int(min_fit_inliers):
            return None
        # Keep `edge_s` aligned with the current point pool.
        try:
            edge_s = np.asarray(s_use, dtype=np.float64).reshape(-1)[valid]
        except Exception:
            edge_s = None
        return np.asarray(Pw, dtype=np.float64).reshape(-1, 3)

    def _apply_hole_refine_mask(
        *,
        fi: int,
        mask_u8: np.ndarray,
        x0_det: int,
        y0_det: int,
        ww_det: int,
        hh_det: int,
        Twc_det: Optional[np.ndarray],
    ) -> None:
        """Stage-1 refinement: fuse detector mask with tracked poly, fit polygon, apply hysteresis, reseed hold points."""
        nonlocal ref_mask_u8, ref_mask_x0, ref_mask_y0, ref_poly_uv, trk_poly_prev
        nonlocal pts_target_n, edge_world_pts_w, edge_alive, edge_plane_bad, _plane_reseed_last_t
        # BEGIN_STAGE1_HYSTERESIS
        nonlocal stage1_best_r_px, stage1_shrink_votes
        # END_STAGE1_HYSTERESIS
        nonlocal hole_trk_mask_u8, hole_trk_dil_u8, hole_fused_u8, hole_fc_u8, hole_gate_kernel_cache

        try:
            mask_u8 = np.asarray(mask_u8, dtype=np.uint8)
        except Exception:
            return
        if int(mask_u8.ndim) != 2 or int(mask_u8.shape[0]) != int(hh_det) or int(mask_u8.shape[1]) != int(ww_det):
            return

        fused = mask_u8
        if poly_uv is not None and int(np.asarray(poly_uv).shape[0]) >= 3:
            try:
                trk = np.asarray(poly_uv, dtype=np.float32).reshape(-1, 2)
                trk_win = (trk - np.asarray([[float(x0_det), float(y0_det)]], dtype=np.float32)).astype(np.int32, copy=False)
                if (
                    hole_trk_mask_u8 is None
                    or hole_trk_dil_u8 is None
                    or hole_fused_u8 is None
                    or hole_fc_u8 is None
                    or hole_trk_mask_u8.shape != (int(hh_det), int(ww_det))
                ):
                    hole_trk_mask_u8 = np.zeros((int(hh_det), int(ww_det)), dtype=np.uint8)
                    hole_trk_dil_u8 = np.zeros((int(hh_det), int(ww_det)), dtype=np.uint8)
                    hole_fused_u8 = np.zeros((int(hh_det), int(ww_det)), dtype=np.uint8)
                    hole_fc_u8 = np.zeros((int(hh_det), int(ww_det)), dtype=np.uint8)
                hole_trk_mask_u8.fill(0)
                cv2.fillPoly(hole_trk_mask_u8, [trk_win], 255)

                rpx = float(hole_r_px)
                if not np.isfinite(rpx):
                    rpx = 0.0
                gate_px = int(round(0.15 * max(10.0, rpx)))
                gate_px = int(max(2, min(30, int(gate_px))))
                k = hole_gate_kernel_cache.get(int(gate_px), None)
                if k is None:
                    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(2 * gate_px + 1), int(2 * gate_px + 1)))
                    hole_gate_kernel_cache[int(gate_px)] = np.asarray(k, dtype=np.uint8)
                cv2.dilate(hole_trk_mask_u8, k, hole_trk_dil_u8, iterations=1)
                cv2.bitwise_and(mask_u8, hole_trk_dil_u8, hole_fused_u8)
                fused = hole_fused_u8
                if int(np.count_nonzero(hole_fused_u8)) < int(0.15 * float(max(1, int(np.count_nonzero(mask_u8))))):
                    fused = mask_u8
            except Exception:
                fused = mask_u8

        cand_mask_u8 = np.asarray(fused, dtype=np.uint8)
        cand_mask_x0 = int(x0_det)
        cand_mask_y0 = int(y0_det)

        # Fit polygon from mask contour.
        if hole_fc_u8 is None or hole_fc_u8.shape != (int(hh_det), int(ww_det)):
            hole_fc_u8 = np.zeros((int(hh_det), int(ww_det)), dtype=np.uint8)
        try:
            np.copyto(hole_fc_u8, np.asarray(cand_mask_u8, dtype=np.uint8), casting="unsafe")
        except Exception:
            hole_fc_u8[:, :] = np.asarray(cand_mask_u8, dtype=np.uint8)
        fc_res = cv2.findContours(hole_fc_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if isinstance(fc_res, tuple) and int(len(fc_res)) == 3:
            _img2, conts2, _hier2 = fc_res
        else:
            conts2, _hier2 = fc_res
        if not conts2:
            return
        c2 = max(conts2, key=lambda cc: float(cv2.contourArea(cc)))
        poly_uv1, _dbg2 = _poly_fit_from_contour(
            contour_win=np.asarray(c2, dtype=np.int32),
            x0=int(x0_det),
            y0=int(y0_det),
            concavity_max=float(hole_poly_concavity_max),
            approx_eps_frac=float(hole_poly_approx_eps_frac),
            min_vertex_angle_deg=float(hole_poly_min_vertex_angle_deg),
            max_vertices=int(hole_poly_max_vertices),
            min_edge_len_px=float(hole_poly_min_edge_len_px),
        )
        if poly_uv1 is None or int(np.asarray(poly_uv1).shape[0]) < 3:
            return
        cand_poly_uv = np.asarray(poly_uv1, dtype=np.float32).reshape(-1, 2)

        # BEGIN_STAGE1_HYSTERESIS
        # Primary score for Stage-1 refine updates: max-inscribed-circle radius (pixels) from the detector.
        try:
            cand_r_px = float(hole_r_px)
        except Exception:
            cand_r_px = float("nan")
        best_r = float(stage1_best_r_px) if np.isfinite(float(stage1_best_r_px)) else float("nan")

        accept_update = True
        accept_reason = "update"
        if bool(hole_refine_hyst_enabled):
            accept_update = False
            accept_reason = "hold"
            if not (np.isfinite(float(cand_r_px)) and float(cand_r_px) > 0.0):
                accept_update = False
                accept_reason = "cand_r_bad"
            elif not (np.isfinite(float(best_r)) and float(best_r) > 0.0):
                accept_update = True
                accept_reason = "init"
                stage1_shrink_votes = int(0)
            else:
                improve_min = float(max(float(hole_refine_gain_abs_px), float(hole_refine_gain_rel) * float(best_r)))
                shrink_min = float(max(1.0, float(hole_refine_shrink_rel) * float(best_r)))
                if float(cand_r_px) >= float(best_r) + float(improve_min):
                    accept_update = True
                    accept_reason = "grow"
                    stage1_shrink_votes = int(0)
                elif float(cand_r_px) <= float(best_r) - float(shrink_min):
                    stage1_shrink_votes = int(stage1_shrink_votes) + 1
                    if int(stage1_shrink_votes) >= int(hole_refine_shrink_votes_req):
                        accept_update = True
                        accept_reason = "shrink_consensus"
                        stage1_shrink_votes = int(0)
                    else:
                        accept_update = False
                        accept_reason = f"shrink_vote_{int(stage1_shrink_votes)}/{int(hole_refine_shrink_votes_req)}"
                else:
                    stage1_shrink_votes = int(0)
                    accept_update = False
                    accept_reason = "hold"

        if bool(accept_update):
            # Commit mask/poly with no per-call allocations (reuse buffers where possible).
            if ref_mask_u8 is None or int(getattr(ref_mask_u8, "ndim", 0)) != 2 or ref_mask_u8.shape != cand_mask_u8.shape:
                ref_mask_u8 = np.zeros_like(cand_mask_u8, dtype=np.uint8)
            try:
                np.copyto(ref_mask_u8, np.asarray(cand_mask_u8, dtype=np.uint8), casting="unsafe")
            except Exception:
                ref_mask_u8[:, :] = np.asarray(cand_mask_u8, dtype=np.uint8)
            ref_mask_x0 = int(cand_mask_x0)
            ref_mask_y0 = int(cand_mask_y0)
            poly_uv_commit = np.asarray(cand_poly_uv, dtype=np.float32).reshape(-1, 2)
            # Optional micro-refinement: edge-normal snap (small displacement, stage-1 only).
            try:
                if bool(stage1_edge_snap.enabled):
                    poly_uv_commit = edge_normal_snap_poly_uv(gray=np.asarray(gray, dtype=np.uint8), poly_uv=poly_uv_commit, cfg=stage1_edge_snap)
            except Exception:
                pass
            ref_poly_uv = np.asarray(poly_uv_commit, dtype=np.float32).reshape(-1, 2).copy()
            # Reset propagation anchor to current tracked polygon (updated later in the tick).
            try:
                trk_poly_prev = np.asarray(poly_uv, dtype=np.float32).reshape(-1, 2).copy() if poly_uv is not None else None
            except Exception:
                trk_poly_prev = None
            stage1_best_r_px = float(cand_r_px) if np.isfinite(float(cand_r_px)) else float("nan")

            # BEGIN_STAGE1_RESEED_POINTS
            # Keep the LK "hold" features aligned with the refined polygon (so the circles follow the updated perimeter).
            try:
                new_edge = (
                    _reseed_edge_points_on_plane(
                        poly_uv_img=np.asarray(poly_uv_commit, dtype=np.float32).reshape(-1, 2),
                        Twc_now=Twc_det,
                        plane_w_n=np.asarray(hole_plane_w_n, dtype=np.float64).reshape(3),
                        plane_w_d=float(hole_plane_w_d),
                        edge_offset_px=float(plane_reseed_edge_offset_px),
                    )
                    if (Twc_det is not None and bool(hole_plane_w_ok) and hole_plane_w_n is not None)
                    else None
                )
                if new_edge is not None and int(np.asarray(new_edge).shape[0]) >= int(min_fit_inliers):
                    edge_world_pts_w = np.asarray(new_edge, dtype=np.float64).reshape(-1, 3)
                    edge_alive = None
                    edge_plane_bad = None
                    _plane_reseed_last_t = 0.0
                    pts_target_n = int(pts_min)
                    if bool(dbg_log):
                        try:
                            print(f"[poly_vo_lk] stage1_reseed edge_pts={int(edge_world_pts_w.shape[0])}", flush=True)
                        except Exception:
                            pass
            except Exception:
                pass
            # END_STAGE1_RESEED_POINTS
            if bool(dbg_log):
                try:
                    prev_best = float(best_r) if np.isfinite(float(best_r)) else float("nan")
                    print(
                        f"[poly_vo_lk] hole_refine_accept fi={int(fi)} r_px={float(cand_r_px):.1f} best_prev={float(prev_best):.1f} reason={str(accept_reason)}",
                        flush=True,
                    )
                except Exception:
                    pass
        else:
            if bool(dbg_log) and str(accept_reason).startswith("shrink_vote"):
                try:
                    prev_best = float(best_r) if np.isfinite(float(best_r)) else float("nan")
                    print(
                        f"[poly_vo_lk] hole_refine_hold fi={int(fi)} r_px={float(cand_r_px):.1f} best={float(prev_best):.1f} reason={str(accept_reason)}",
                        flush=True,
                    )
                except Exception:
                    pass
        # END_STAGE1_HYSTERESIS

    def _plane_poly_from_plane_mask(
        *,
        plane_u8: np.ndarray,
        x0: int,
        y0: int,
        depth_raw: np.ndarray,
        close_px: int,
        n_verts: int,
        depth_sample_r_px: int,
        seed_uv_roi: Optional[tuple[int, int]] = None,
    ) -> Optional[dict]:
        try:
            plane_u8 = np.asarray(plane_u8, dtype=np.uint8)
        except Exception:
            return None
        if int(getattr(plane_u8, "ndim", 0)) != 2 or int(plane_u8.size) <= 0:
            return None
        hh, ww = int(plane_u8.shape[0]), int(plane_u8.shape[1])
        if int(hh) <= 1 or int(ww) <= 1:
            return None
        if int(np.count_nonzero(plane_u8)) <= 0:
            return None

        # Optional close: fill tiny gaps so distanceTransform is stable under missing depth jitter.
        try:
            if int(close_px) > 0:
                r = int(max(1, int(close_px)))
                k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(2 * r + 1), int(2 * r + 1)))
                plane_u8 = cv2.morphologyEx(np.asarray(plane_u8, dtype=np.uint8), cv2.MORPH_CLOSE, k, iterations=1)
        except Exception:
            pass

        try:
            mask01 = (np.asarray(plane_u8, dtype=np.uint8) != 0).astype(np.uint8, copy=False)
        except Exception:
            mask01 = None
        if mask01 is None or int(np.count_nonzero(mask01)) <= 0:
            return None

        try:
            dist = cv2.distanceTransform(mask01, cv2.DIST_L2, 3)
        except Exception:
            return None

        # Prefer a solid patch *under the cursor* when provided:
        # - center on the seed pixel (or the nearest plane pixel)
        # - radius from distanceTransform at that location (largest inscribed circle around the cursor)
        cx_roi = cy_roi = -1
        r_px = float("nan")
        if seed_uv_roi is not None:
            try:
                sx, sy = int(seed_uv_roi[0]), int(seed_uv_roi[1])
            except Exception:
                sx = sy = -1
            if 0 <= int(sx) < int(ww) and 0 <= int(sy) < int(hh):
                try:
                    if int(mask01[int(sy), int(sx)]) != 0:
                        cx_roi = int(sx)
                        cy_roi = int(sy)
                    else:
                        # Snap to nearest plane pixel (handles clicks on missing-depth pixels).
                        yy, xx = np.nonzero(mask01)
                        if int(xx.size) > 0:
                            dy = yy.astype(np.int32, copy=False) - int(sy)
                            dx = xx.astype(np.int32, copy=False) - int(sx)
                            j = int(np.argmin(dx * dx + dy * dy))
                            cx_roi = int(xx[j])
                            cy_roi = int(yy[j])
                except Exception:
                    cx_roi = cy_roi = -1
        # Fallback: maximum inscribed circle anywhere in the plane mask.
        if cx_roi < 0 or cy_roi < 0:
            try:
                _min, max_px, _min_loc, max_loc = cv2.minMaxLoc(dist)
                cx_roi = int(max_loc[0])
                cy_roi = int(max_loc[1])
            except Exception:
                cx_roi = cy_roi = -1
        try:
            if cx_roi >= 0 and cy_roi >= 0:
                r_px = float(dist[int(cy_roi), int(cx_roi)])
        except Exception:
            r_px = float("nan")
        if cx_roi < 0 or cy_roi < 0 or (not np.isfinite(float(r_px))) or float(r_px) <= 1e-3:
            return None

        # Depth at the circle center (median in a small neighborhood). Fall back to median of plane-mask depths.
        z_ref_m = float("nan")
        try:
            med_u16, _mad_u16, nn = _sample_depth_stats_u16(
                np.asarray(depth_raw, dtype=np.uint16),
                int(x0 + cx_roi),
                int(y0 + cy_roi),
                int(depth_sample_r_px),
            )
            if int(nn) > 0 and int(med_u16) > 0:
                z_ref_m = float(med_u16) * float(depth_units)
        except Exception:
            z_ref_m = float("nan")
        if (not np.isfinite(float(z_ref_m))) or float(z_ref_m) <= 0.0:
            try:
                d_win = np.asarray(depth_raw, dtype=np.uint16)[int(y0) : int(y0) + int(hh), int(x0) : int(x0) + int(ww)]
                vals = d_win[(mask01 != 0) & (d_win > 0)]
                if int(vals.size) > 0:
                    z_ref_m = float(np.median(np.asarray(vals, dtype=np.float64).reshape(-1))) * float(depth_units)
            except Exception:
                z_ref_m = float("nan")
        if (not np.isfinite(float(z_ref_m))) or float(z_ref_m) <= 0.0:
            return None

        try:
            fx = float(getattr(intr_obj, "fx", 0.0))
            fy = float(getattr(intr_obj, "fy", 0.0))
        except Exception:
            fx = fy = 0.0
        f = 0.5 * (float(fx) + float(fy))
        if (not np.isfinite(float(f))) or float(f) <= 1e-6:
            return None
        r_m = float(r_px) * float(z_ref_m) / float(f)

        # Build a solid N-gon approximating the max-inscribed circle (full-image coords).
        try:
            n = int(max(8, int(n_verts)))
        except Exception:
            n = 24
        shrink = 0.95
        rr = max(0.0, float(r_px) * float(shrink))
        ang = np.linspace(0.0, 2.0 * np.pi, int(n), endpoint=False, dtype=np.float64)
        cx_full = float(int(x0) + int(cx_roi))
        cy_full = float(int(y0) + int(cy_roi))
        try:
            w_img = int(depth_raw.shape[1])
            h_img = int(depth_raw.shape[0])
        except Exception:
            w_img = h_img = 0
        u = cx_full + rr * np.cos(ang)
        v = cy_full + rr * np.sin(ang)
        if int(w_img) > 0:
            u = np.clip(u, 0.0, float(w_img - 1))
        if int(h_img) > 0:
            v = np.clip(v, 0.0, float(h_img - 1))
        poly_uv0 = np.stack([u, v], axis=1).astype(np.float32, copy=False).reshape(-1, 2)
        if int(poly_uv0.shape[0]) < 3:
            return None
        return {
            "poly_uv": np.asarray(poly_uv0, dtype=np.float32).reshape(-1, 2),
            "center_uv": (float(cx_full), float(cy_full)),
            "r_px": float(r_px),
            "r_m": float(r_m),
            "z_ref_m": float(z_ref_m),
        }

    def _handle_select_plane(msg: dict) -> None:
        nonlocal poly_active, poly_uv, poly_bbox, poly_world_pts_w, edge_world_pts_w, poly_z_prior_m
        nonlocal ref_poly_uv, ref_mask_u8, ref_mask_x0, ref_mask_y0, trk_poly_prev
        nonlocal sel_kind, pid_confirmed
        nonlocal hole_center_uv, hole_r_px, hole_r_m, hole_fill, hole_ok_pid
        nonlocal hole_plane_inliers, hole_plane_rms_m, hole_plane_cov, hole_err, hole_last_refine_ts
        nonlocal hole_plane_w_n, hole_plane_w_d, hole_plane_w_ok
        nonlocal plane_center_uv, plane_r_px, plane_r_m, plane_ok_pid
        nonlocal plane_plane_inliers, plane_plane_rms_m, plane_plane_cov, plane_plane_center_range_m, plane_err
        nonlocal plane_plane_w_n, plane_plane_w_d, plane_plane_w_ok
        nonlocal corr_params, corr_last_ts, prev_gray, prev_Twc, prev_corr_params
        nonlocal poly_locked, lock_used_uv, last_inliers, last_n, last_reason, pts_target_n, last_try_n, last_keep_ratio
        # BEGIN_STAGE1_HYSTERESIS
        nonlocal stage1_best_r_px, stage1_shrink_votes
        # END_STAGE1_HYSTERESIS
        nonlocal feat_anchor, feat_last, feat_last_t, feat_orb_A, feat_orb_ok, feat_orb_n, feat_orb_inl_n, feat_lines_n, feat_orb_uv_pub, feat_lines_pub

        try:
            x = int(msg.get("x", -1))
            y = int(msg.get("y", -1))
        except Exception:
            return
        if x < 0 or y < 0:
            return

        fi_sel = int(msg.get("fi", -1))
        ts_sel = float(msg.get("ts", 0.0))
        slot = ring.find_slot_by_frame_idx(int(fi_sel))
        if slot is None:
            latest = ring.read_latest()
            if latest is None:
                return
            slot, fi_sel, ts_sel = latest

        if bool(dbg_log):
            try:
                print(f"[poly_vo_lk] plane_select_start fi={int(fi_sel)} xy=({int(x)},{int(y)})", flush=True)
            except Exception:
                pass

        gray = np.asarray(ring.gray[int(slot)], dtype=np.uint8)
        depth_raw = None
        try:
            if bool(cap_cfg.get("depth", True)) and float(depth_units) > 0.0:
                depth_raw = np.asarray(ring.depth_raw[int(slot)], dtype=np.uint16)
        except Exception:
            depth_raw = None

        def _plane_fail(code: str, extra: str = "") -> None:
            nonlocal last_reason, plane_err, sel_kind
            det_err = ""
            if str(code) != "plane_no_depth_or_intr":
                try:
                    det_err = str(getattr(hole, "last_error", "") or "").strip()
                except Exception:
                    det_err = ""
            if not det_err:
                try:
                    det_err = str(extra).strip()
                except Exception:
                    det_err = ""
            try:
                last_reason = str(code)
            except Exception:
                last_reason = ""
            if bool(dbg_log):
                try:
                    extra_s = str(extra).strip()
                    extra_s = f" {extra_s}" if extra_s else ""
                    print(f"[poly_vo_lk] plane_select_fail fi={int(fi_sel)} code={str(code)}{extra_s}", flush=True)
                except Exception:
                    pass
            _reset_state(keep_selection=False)
            # Preserve the intended selection kind so external integrations can attribute failures.
            sel_kind = "plane"
            try:
                plane_err = str(det_err)
            except Exception:
                plane_err = ""
            try:
                last_reason = str(code)
            except Exception:
                last_reason = ""
            _publish(fi=int(fi_sel), ts=float(ts_sel), reason=str(code))

        if intr_obj is None or depth_raw is None or float(depth_units) <= 0.0:
            _plane_fail("plane_no_depth_or_intr")
            return

        h_img = int(gray.shape[0])
        w_img = int(gray.shape[1])
        x0, y0, ww, hh = _window_bbox(int(w_img), int(h_img), x=int(x), y=int(y), win_w=int(hole_win_w), win_h=int(hole_win_h))

        # Plane-guided detection: get a local plane mask around the seed.
        try:
            _hole_u8, plane_u8 = hole.detect_plane_guided(
                {"bgr": None, "gray": gray, "depth_raw": depth_raw, "depth_units": float(depth_units), "intr": intr_obj},
                window_wh=(int(hole_win_w), int(hole_win_h)),
                x=int(x),
                y=int(y),
                settings=hole_settings,
            )
        except Exception:
            plane_u8 = np.zeros((1, 1), dtype=np.uint8)
        try:
            plane_u8 = np.asarray(plane_u8, dtype=np.uint8)
        except Exception:
            plane_u8 = np.zeros((1, 1), dtype=np.uint8)
        nz0 = int(np.count_nonzero(plane_u8)) if isinstance(plane_u8, np.ndarray) else 0
        if int(plane_u8.size) <= 0 or int(plane_u8.shape[0]) <= 1 or int(plane_u8.shape[1]) <= 1:
            _plane_fail("plane_mask_invalid", extra=f"shape={tuple(int(x) for x in getattr(plane_u8,'shape',()))}")
            return
        if int(nz0) <= 0:
            _plane_fail("plane_mask_empty", extra=f"nz={int(nz0)}")
            return

        # Compute a solid plane patch polygon from the max-inscribed circle of the plane mask.
        hole_depth_r_px = int(max(0, int(proc_cfg.get("hole_depth_r_px", 8))))
        try:
            depth_sample_r_px = int(max(int(depth_r_px), int(hole_depth_r_px)))
        except Exception:
            depth_sample_r_px = int(max(0, int(hole_depth_r_px)))
        out = _plane_poly_from_plane_mask(
            plane_u8=np.asarray(plane_u8, dtype=np.uint8),
            x0=int(x0),
            y0=int(y0),
            depth_raw=np.asarray(depth_raw, dtype=np.uint16),
            close_px=int(plane_mask_close_px),
            n_verts=int(plane_poly_n_verts),
            depth_sample_r_px=int(depth_sample_r_px),
            seed_uv_roi=(int(x) - int(x0), int(y) - int(y0)),
        )
        if out is None:
            _plane_fail("plane_poly_fail")
            return

        # Cache plane stats for GUI + PID gating.
        sel_kind = "plane"
        pid_confirmed = False
        trk_poly_prev = None
        plane_plane_inliers = int(getattr(hole, "last_plane_inliers", 0))
        plane_plane_rms_m = float(getattr(hole, "last_plane_rms_m", 0.0))
        plane_plane_cov = float(getattr(hole, "last_plane_sector_cov", 0.0))
        plane_err = str(getattr(hole, "last_error", "") or "")
        plane_r_px = float(out.get("r_px", 0.0) or 0.0)
        plane_r_m = float(out.get("r_m", 0.0) or 0.0)
        try:
            c_u, c_v = out.get("center_uv", (-1.0, -1.0))
            plane_center_uv = (float(c_u), float(c_v))
        except Exception:
            plane_center_uv = None
        try:
            _plane_ok = bool(getattr(hole, "last_plane_ok", False))
        except Exception:
            _plane_ok = False
        plane_ok_pid = bool(_plane_ok) and bool(np.isfinite(float(plane_r_m))) and float(plane_r_m) >= float(plane_pid_min_radius_m)
        plane_plane_center_range_m = float("nan")
        # Cache the fitted plane in world coordinates (used for center-range readout + server auto-exit).
        Twc_now = None
        if state is not None:
            st = state.read(copy=False)
            if st is not None and int(len(st)) >= 12:
                try:
                    Twc_now = np.asarray(st[11], dtype=np.float64).reshape(4, 4)
                except Exception:
                    Twc_now = None
        _update_plane_plane_world_from_detector(Twc_now=Twc_now)

        # Clear any hole-specific metadata (not relevant in plane mode).
        hole_center_uv = None
        hole_r_px = 0.0
        hole_r_m = 0.0
        hole_fill = 0.0
        hole_ok_pid = False
        hole_plane_inliers = 0
        hole_plane_rms_m = 0.0
        hole_plane_cov = 0.0
        hole_err = ""
        hole_last_refine_ts = 0.0
        hole_plane_w_ok = False
        hole_plane_w_n = None
        hole_plane_w_d = 0.0
        ref_poly_uv = None
        ref_mask_u8 = None
        ref_mask_x0 = 0
        ref_mask_y0 = 0
        # BEGIN_STAGE1_HYSTERESIS
        stage1_best_r_px = float("nan")
        stage1_shrink_votes = int(0)
        # END_STAGE1_HYSTERESIS

        _activate_selection_from_poly_uv(
            poly_uv0=np.asarray(out.get("poly_uv", None), dtype=np.float32).reshape(-1, 2),
            fi_sel=int(fi_sel),
            ts_sel=float(ts_sel),
            gray=gray,
            depth_raw=depth_raw,
            depth_sample_r_px=int(depth_sample_r_px),
            depth_prefer_outside_mask=False,
            lift_plane_w_n=(np.asarray(plane_plane_w_n, dtype=np.float64).reshape(3) if bool(plane_plane_w_ok) and plane_plane_w_n is not None else None),
            lift_plane_w_d=(float(plane_plane_w_d) if bool(plane_plane_w_ok) and plane_plane_w_n is not None else None),
            lift_plane_edge_offset_px=0.0,
        )

        # Optional: capture an anchor snapshot (ORB/lines) for this selection.
        try:
            if (
                feature_acq is not None
                and bool(feature_acq_stage1)
                and bool(feature_acq_on_select)
                and bool(poly_active)
                and poly_uv is not None
                and int(np.asarray(poly_uv).shape[0]) >= 3
            ):
                feat_anchor = feature_acq.acquire(
                    gray=np.asarray(gray, dtype=np.uint8),
                    poly_uv=np.asarray(poly_uv, dtype=np.float32).reshape(-1, 2),
                )
                feat_last = None
                feat_last_t = 0.0
                feat_orb_A = None
                feat_orb_ok = False
                try:
                    feat_orb_n = int(0 if feat_anchor.orb_uv is None else np.asarray(feat_anchor.orb_uv).shape[0])
                except Exception:
                    feat_orb_n = 0
                feat_orb_inl_n = 0
                try:
                    feat_lines_n = int(0 if feat_anchor.lines_xyxy is None else np.asarray(feat_anchor.lines_xyxy).shape[0])
                except Exception:
                    feat_lines_n = 0
                feat_orb_uv_pub = None
                feat_lines_pub = None
                try:
                    if bool(feature_acq_publish_orb) and feat_anchor.orb_uv is not None:
                        uv0 = np.asarray(feat_anchor.orb_uv, dtype=np.float32).reshape(-1, 2)
                        nmax = int(max(0, int(feature_acq_publish_max_orb)))
                        if int(nmax) > 0 and int(uv0.shape[0]) > int(nmax):
                            uv0 = np.asarray(uv0[: int(nmax)], dtype=np.float32).reshape(-1, 2)
                        feat_orb_uv_pub = uv0
                except Exception:
                    feat_orb_uv_pub = None
                try:
                    if bool(feature_acq_publish_lines) and feat_anchor.lines_xyxy is not None:
                        L0 = np.asarray(feat_anchor.lines_xyxy, dtype=np.float32).reshape(-1, 4)
                        nmax = int(max(0, int(feature_acq_publish_max_lines)))
                        if int(nmax) > 0 and int(L0.shape[0]) > int(nmax):
                            L0 = np.asarray(L0[: int(nmax)], dtype=np.float32).reshape(-1, 4)
                        feat_lines_pub = L0
                except Exception:
                    feat_lines_pub = None
                if bool(dbg_log):
                    try:
                        n0 = int(0 if feat_anchor.orb_uv is None else np.asarray(feat_anchor.orb_uv).shape[0])
                        print(f"[poly_vo_lk] feat_anchor (plane) orb_n={int(n0)}", flush=True)
                    except Exception:
                        pass
        except Exception:
            pass

    def _handle_select_hole(msg: dict) -> None:
        nonlocal poly_active, poly_uv, poly_bbox, poly_world_pts_w, edge_world_pts_w, poly_z_prior_m
        nonlocal ref_poly_uv, ref_mask_u8, ref_mask_x0, ref_mask_y0, trk_poly_prev
        nonlocal sel_kind, pid_confirmed, hole_center_uv, hole_r_px, hole_r_m, hole_fill, hole_ok_pid
        nonlocal hole_plane_inliers, hole_plane_rms_m, hole_plane_cov, hole_err, hole_last_refine_ts
        nonlocal hole_plane_w_n, hole_plane_w_d, hole_plane_w_ok
        nonlocal corr_params, corr_last_ts, prev_gray, prev_Twc, prev_corr_params
        nonlocal poly_locked, lock_used_uv, last_inliers, last_n, last_reason, pts_target_n, last_try_n, last_keep_ratio
        # BEGIN_STAGE1_HYSTERESIS
        nonlocal stage1_best_r_px, stage1_shrink_votes
        # END_STAGE1_HYSTERESIS
        nonlocal feat_anchor, feat_last, feat_last_t, feat_orb_A, feat_orb_ok, feat_orb_n, feat_orb_inl_n, feat_lines_n, feat_orb_uv_pub, feat_lines_pub

        try:
            x = int(msg.get("x", -1))
            y = int(msg.get("y", -1))
        except Exception:
            return
        if x < 0 or y < 0:
            return

        fi_sel = int(msg.get("fi", -1))
        ts_sel = float(msg.get("ts", 0.0))
        slot = ring.find_slot_by_frame_idx(int(fi_sel))
        if slot is None:
            latest = ring.read_latest()
            if latest is None:
                return
            slot, fi_sel, ts_sel = latest

        if bool(dbg_log):
            try:
                print(f"[poly_vo_lk] hole_select_start fi={int(fi_sel)} xy=({int(x)},{int(y)})", flush=True)
            except Exception:
                pass

        # Hole detector can run on gray-only (IR stream); avoid unnecessary BGR materialization.
        bgr = None
        try:
            bgr = np.asarray(ring.bgr[int(slot)], dtype=np.uint8)
        except Exception:
            bgr = None
        gray = np.asarray(ring.gray[int(slot)], dtype=np.uint8)
        depth_raw = None
        try:
            if bool(cap_cfg.get("depth", True)) and float(depth_units) > 0.0:
                depth_raw = np.asarray(ring.depth_raw[int(slot)], dtype=np.uint16)
        except Exception:
            depth_raw = None

        def _hole_fail(code: str, extra: str = "") -> None:
            nonlocal last_reason, hole_err, sel_kind

            # Preserve the detector's last_error in the published state even when selection fails.
            # This allows external integrations (server.py) to debug failure modes without scraping the raw log.
            det_err = ""
            if str(code) != "hole_no_depth_or_intr":
                try:
                    det_err = str(getattr(hole, "last_error", "") or "").strip()
                except Exception:
                    det_err = ""
            if not det_err:
                try:
                    det_err = str(extra).strip()
                except Exception:
                    det_err = ""
            try:
                last_reason = str(code)
            except Exception:
                last_reason = ""
            if bool(dbg_log):
                try:
                    extra_s = str(extra).strip()
                    extra_s = f" {extra_s}" if extra_s else ""
                    print(f"[poly_vo_lk] hole_select_fail fi={int(fi_sel)} code={str(code)}{extra_s}", flush=True)
                except Exception:
                    pass
            _reset_state(keep_selection=False)
            # Preserve the intended selection kind so external integrations can attribute failures.
            sel_kind = "hole"
            try:
                hole_err = str(det_err)
            except Exception:
                hole_err = ""
            try:
                last_reason = str(code)
            except Exception:
                last_reason = ""
            _publish(fi=int(fi_sel), ts=float(ts_sel), reason=str(code))

        if intr_obj is None or depth_raw is None or float(depth_units) <= 0.0:
            _hole_fail("hole_no_depth_or_intr")
            return

        h_img = int(gray.shape[0])
        w_img = int(gray.shape[1])
        x0, y0, ww, hh = _window_bbox(int(w_img), int(h_img), x=int(x), y=int(y), win_w=int(hole_win_w), win_h=int(hole_win_h))

        # If the GUI provides a recent hover-preview mask/polygon, prefer it for selection.
        # This makes selection robust when the detector flickers (e.g., leak guard triggers intermittently).
        try:
            poly_cached = msg.get("poly_uv", None)
            mask_cached = msg.get("mask_u8", None)
            if poly_cached is not None and mask_cached is not None:
                poly_uv0 = np.asarray(poly_cached, dtype=np.float32).reshape(-1, 2)
                mask_u8_cached = np.asarray(mask_cached, dtype=np.uint8)
                x0_cached = int(msg.get("mask_x0", x0))
                y0_cached = int(msg.get("mask_y0", y0))
                nz_cached = int(np.count_nonzero(mask_u8_cached)) if isinstance(mask_u8_cached, np.ndarray) else 0
                if (
                    int(nz_cached) > 0
                    and int(mask_u8_cached.ndim) == 2
                    and int(mask_u8_cached.shape[0]) > 1
                    and int(mask_u8_cached.shape[1]) > 1
                    and int(poly_uv0.shape[0]) >= 3
                ):
                    x0 = int(x0_cached)
                    y0 = int(y0_cached)
                    ww = int(mask_u8_cached.shape[1])
                    hh = int(mask_u8_cached.shape[0])

                    # Seed hole/plane stats from the preview (will be refined by the detector loop).
                    sel_kind = "hole"
                    pid_confirmed = False
                    trk_poly_prev = None
                    hole_plane_inliers = 0
                    hole_plane_rms_m = 0.0
                    hole_plane_cov = 0.0
                    hole_err = ""
                    hole_r_px = 0.0
                    hole_r_m = 0.0
                    hole_fill = 0.0
                    hole_center_uv = None
                    _plane_ok = False
                    _bdr = False
                    try:
                        hs = msg.get("hole_stats", None)
                        if isinstance(hs, dict):
                            hole_plane_inliers = int(hs.get("plane_inliers", hole_plane_inliers))
                            hole_plane_rms_m = float(hs.get("plane_rms_m", hole_plane_rms_m))
                            hole_plane_cov = float(hs.get("plane_cov", hole_plane_cov))
                            hole_err = str(hs.get("err", hole_err) or "")
                            hole_r_px = float(hs.get("r_px", hole_r_px))
                            hole_r_m = float(hs.get("r_m", hole_r_m))
                            hole_fill = float(hs.get("fill", hole_fill))
                            _plane_ok = bool(int(hs.get("plane_ok", 0))) if "plane_ok" in hs else bool(_plane_ok)
                            _bdr = bool(int(hs.get("touches_border", 0))) if "touches_border" in hs else bool(_bdr)
                            cuv = hs.get("center_uv", None)
                            if cuv is not None and isinstance(cuv, (tuple, list)) and int(len(cuv)) == 2:
                                try:
                                    hole_center_uv = (float(cuv[0]), float(cuv[1]))
                                except Exception:
                                    hole_center_uv = None
                    except Exception:
                        pass

                    hole_ok_pid = (
                        bool(_plane_ok)
                        and (not bool(_bdr))
                        and bool(np.isfinite(float(hole_r_m)))
                        and float(hole_r_m) >= float(hole_pid_min_radius_m)
                    )
                    hole_last_refine_ts = float(time.time())
                    # Cached preview does not provide a reliable world-plane; reset it.
                    hole_plane_w_ok = False
                    hole_plane_w_n = None
                    hole_plane_w_d = 0.0

                    # Initialize refinement mask from the cached preview (used later for depth sampling on confirm).
                    try:
                        ref_mask_u8 = np.asarray(mask_u8_cached, dtype=np.uint8).copy()
                        ref_mask_x0 = int(x0)
                        ref_mask_y0 = int(y0)
                    except Exception:
                        ref_mask_u8 = None
                        ref_mask_x0 = 0
                        ref_mask_y0 = 0

                    try:
                        ref_poly_uv = np.asarray(poly_uv0, dtype=np.float32).reshape(-1, 2).copy()
                        # Optional micro-refinement: snap vertices along edge normals (small per-call displacement).
                        try:
                            if bool(stage1_edge_snap.enabled):
                                ref_poly_uv = edge_normal_snap_poly_uv(
                                    gray=np.asarray(gray, dtype=np.uint8),
                                    poly_uv=ref_poly_uv,
                                    cfg=stage1_edge_snap,
                                )
                        except Exception:
                            pass
                        trk_poly_prev = np.asarray(ref_poly_uv, dtype=np.float32).reshape(-1, 2).copy()
                        # BEGIN_STAGE1_HYSTERESIS
                        stage1_best_r_px = float(hole_r_px) if np.isfinite(float(hole_r_px)) else float("nan")
                        stage1_shrink_votes = int(0)
                        # END_STAGE1_HYSTERESIS
                    except Exception:
                        ref_poly_uv = None
                        trk_poly_prev = None
                        # BEGIN_STAGE1_HYSTERESIS
                        stage1_best_r_px = float("nan")
                        stage1_shrink_votes = int(0)
                        # END_STAGE1_HYSTERESIS

                    if bool(dbg_log):
                        try:
                            print(
                                f"[poly_vo_lk] hole_select_use_preview fi={int(fi_sel)} nz={int(nz_cached)} verts={int(poly_uv0.shape[0])} "
                                f"r_px={float(hole_r_px):.1f} r_m={float(hole_r_m):.3f} fill={100.0*float(hole_fill):.0f}%",
                                flush=True,
                            )
                        except Exception:
                            pass

                    hole_depth_r_px = int(max(0, int(proc_cfg.get("hole_depth_r_px", 8))))
                    _activate_selection_from_poly_uv(
                        poly_uv0=np.asarray(poly_uv0, dtype=np.float32).reshape(-1, 2),
                        fi_sel=int(fi_sel),
                        ts_sel=float(ts_sel),
                        gray=gray,
                        depth_raw=depth_raw,
                        depth_mask_win=np.asarray(mask_u8_cached, dtype=np.uint8),
                        depth_mask_x0=int(x0),
                        depth_mask_y0=int(y0),
                        depth_sample_r_px=int(max(int(depth_r_px), int(hole_depth_r_px))),
                        depth_prefer_outside_mask=True,
                    )
                    # Optional: capture an anchor snapshot (ORB/lines) for this selection.
                    try:
                        if (
                            feature_acq is not None
                            and bool(feature_acq_stage1)
                            and bool(feature_acq_on_select)
                            and ref_poly_uv is not None
                            and int(np.asarray(ref_poly_uv).shape[0]) >= 3
                        ):
                            feat_anchor = feature_acq.acquire(
                                gray=np.asarray(gray, dtype=np.uint8),
                                poly_uv=np.asarray(ref_poly_uv, dtype=np.float32).reshape(-1, 2),
                            )
                            feat_last = None
                            feat_last_t = 0.0
                            feat_orb_A = None
                            feat_orb_ok = False
                            try:
                                feat_orb_n = int(0 if feat_anchor.orb_uv is None else np.asarray(feat_anchor.orb_uv).shape[0])
                            except Exception:
                                feat_orb_n = 0
                            feat_orb_inl_n = 0
                            try:
                                feat_lines_n = int(0 if feat_anchor.lines_xyxy is None else np.asarray(feat_anchor.lines_xyxy).shape[0])
                            except Exception:
                                feat_lines_n = 0
                            feat_orb_uv_pub = None
                            feat_lines_pub = None
                            try:
                                if bool(feature_acq_publish_orb) and feat_anchor.orb_uv is not None:
                                    uv0 = np.asarray(feat_anchor.orb_uv, dtype=np.float32).reshape(-1, 2)
                                    nmax = int(max(0, int(feature_acq_publish_max_orb)))
                                    if int(nmax) > 0 and int(uv0.shape[0]) > int(nmax):
                                        uv0 = np.asarray(uv0[: int(nmax)], dtype=np.float32).reshape(-1, 2)
                                    feat_orb_uv_pub = uv0
                            except Exception:
                                feat_orb_uv_pub = None
                            try:
                                if bool(feature_acq_publish_lines) and feat_anchor.lines_xyxy is not None:
                                    L0 = np.asarray(feat_anchor.lines_xyxy, dtype=np.float32).reshape(-1, 4)
                                    nmax = int(max(0, int(feature_acq_publish_max_lines)))
                                    if int(nmax) > 0 and int(L0.shape[0]) > int(nmax):
                                        L0 = np.asarray(L0[: int(nmax)], dtype=np.float32).reshape(-1, 4)
                                    feat_lines_pub = L0
                            except Exception:
                                feat_lines_pub = None
                            if bool(dbg_log):
                                try:
                                    n0 = int(0 if feat_anchor.orb_uv is None else np.asarray(feat_anchor.orb_uv).shape[0])
                                    print(f"[poly_vo_lk] feat_anchor (hole_preview) orb_n={int(n0)}", flush=True)
                                except Exception:
                                    pass
                    except Exception:
                        pass
                    return
        except Exception:
            pass

        mask_u8, _plane_u8 = hole.detect_hole_plane(
            {"bgr": bgr, "gray": gray, "depth_raw": depth_raw, "depth_units": float(depth_units), "intr": intr_obj},
            window_wh=(int(hole_win_w), int(hole_win_h)),
            x=int(x),
            y=int(y),
        )
        try:
            mask_u8 = np.asarray(mask_u8, dtype=np.uint8)
        except Exception:
            mask_u8 = np.zeros((1, 1), dtype=np.uint8)
        nz0 = int(np.count_nonzero(mask_u8)) if isinstance(mask_u8, np.ndarray) else 0
        if bool(dbg_log):
            try:
                err_s = str(getattr(hole, "last_error", "") or "").strip()
                err_s = err_s if err_s else "-"
                print(
                    f"[poly_vo_lk] hole_mask fi={int(fi_sel)} win=({int(x0)},{int(y0)},{int(ww)},{int(hh)}) "
                    f"filled={int(getattr(hole,'last_filled_px',0))} nz={int(nz0)} "
                    f"d_valid={100.0*float(getattr(hole,'last_depth_valid_frac',0.0)):.0f}% "
                    f"h_valid={100.0*float(getattr(hole,'last_hole_valid_frac',0.0)):.0f}% "
                    f"bdr={int(bool(getattr(hole,'last_hole_touches_border',False)))} "
                    f"leak={int(bool(getattr(hole,'last_hole_leak_guard_used',False)))} "
                    f"leak_open={int(getattr(hole,'last_hole_leak_open_px',0))} "
                    f"snap_px={int(hole_select_snap_px)} "
                    f"snapped={int(bool(getattr(hole,'last_seed_snapped',False)))} "
                    f"edge={int(bool(getattr(hole,'last_edge_barrier_used',False)))} "
                    f"emode={str(getattr(hole,'last_edge_barrier_mode','') or '-')} "
                    f"emeth={str(getattr(hole,'last_edge_barrier_method','') or '-')} "
                    f"cc={int(getattr(hole,'last_hole_cc_count',0))} lab={int(getattr(hole,'last_hole_cc_pick_label',0))} "
                    f"cc_r_px={float(getattr(hole,'last_hole_cc_pick_r_px',0.0)):.1f} cc_d={int(getattr(hole,'last_hole_cc_pick_min_d2',0))} "
                    f"plane_ok={int(bool(getattr(hole,'last_plane_ok',False)))} inl={int(getattr(hole,'last_plane_inliers',0))} "
                    f"rms={1000.0*float(getattr(hole,'last_plane_rms_m',0.0)):.1f}mm cov={100.0*float(getattr(hole,'last_plane_sector_cov',0.0)):.0f}% "
                    f"n_ok={int(bool(getattr(hole,'last_plane_normals_used',False)))} "
                    f"n_valid={100.0*float(getattr(hole,'last_plane_normals_valid_frac',0.0)):.0f}% "
                    f"n=({float(getattr(hole,'last_plane_n_cam',(0.0,0.0,0.0))[0]):+.3f},{float(getattr(hole,'last_plane_n_cam',(0.0,0.0,0.0))[1]):+.3f},{float(getattr(hole,'last_plane_n_cam',(0.0,0.0,0.0))[2]):+.3f}) "
                    f"z_wall={float(getattr(hole,'last_hole_wall_depth_m',0.0)):.3f} z_ref={float(getattr(hole,'last_hole_z_ref_m',0.0)):.3f} "
                    f"band={int(getattr(hole,'last_hole_plane_band_in_px',0))}/{int(getattr(hole,'last_hole_plane_band_out_px',0))} "
                    f"z_avail={float(getattr(hole,'last_depth_avail_m',0.0)):.3f} r={float(getattr(hole,'last_radius_m',0.0)):.3f} "
                    f"r_px={float(getattr(hole,'last_circle_radius_px',0.0)):.1f} fill={100.0*float(getattr(hole,'last_circle_fill_ratio',0.0)):.0f}% "
                    f"err={err_s}",
                    flush=True,
                )
            except Exception:
                pass
        if int(mask_u8.size) <= 0 or int(mask_u8.shape[0]) <= 1 or int(mask_u8.shape[1]) <= 1:
            _hole_fail("hole_mask_invalid", extra=f"shape={tuple(int(x) for x in getattr(mask_u8,'shape',()))}")
            return
        if int(nz0) <= 0:
            _hole_fail("hole_mask_empty", extra=f"nz={int(nz0)}")
            return

        # Cache hole/plane stats for GUI + PID gating.
        sel_kind = "hole"
        pid_confirmed = False
        trk_poly_prev = None
        hole_plane_inliers = int(getattr(hole, "last_plane_inliers", 0))
        hole_plane_rms_m = float(getattr(hole, "last_plane_rms_m", 0.0))
        hole_plane_cov = float(getattr(hole, "last_plane_sector_cov", 0.0))
        hole_err = str(getattr(hole, "last_error", "") or "")
        hole_r_px = float(getattr(hole, "last_circle_radius_px", 0.0))
        hole_r_m = float(getattr(hole, "last_radius_m", 0.0))
        hole_fill = float(getattr(hole, "last_circle_fill_ratio", 0.0))
        try:
            cx_full, cy_full = getattr(hole, "last_circle_center_full_px", (-1, -1))
            cx_full = int(cx_full)
            cy_full = int(cy_full)
        except Exception:
            cx_full, cy_full = -1, -1
        hole_center_uv = (float(cx_full), float(cy_full)) if (cx_full >= 0 and cy_full >= 0) else None
        # PID "OK" requires a reliable plane fit (otherwise `r` in meters is not meaningful) and a non-leaky hole mask.
        try:
            _plane_ok = bool(getattr(hole, "last_plane_ok", False))
        except Exception:
            _plane_ok = False
        try:
            _bdr = bool(getattr(hole, "last_hole_touches_border", False))
        except Exception:
            _bdr = False
        hole_ok_pid = (
            bool(_plane_ok)
            and (not bool(_bdr))
            and bool(np.isfinite(float(hole_r_m)))
            and float(hole_r_m) >= float(hole_pid_min_radius_m)
        )
        hole_last_refine_ts = float(time.time())
        # Cache the fitted wall plane in world coordinates (used later during PID movement to prune/reseed lock points).
        Twc_now = None
        if state is not None:
            st = state.read(copy=False)
            if st is not None and int(len(st)) >= 12:
                try:
                    Twc_now = np.asarray(st[11], dtype=np.float64).reshape(4, 4)
                except Exception:
                    Twc_now = None
        _update_hole_plane_world_from_detector(Twc_now=Twc_now)
        # Initialize refinement mask from the current detection (used later for depth sampling on confirm).
        try:
            ref_mask_u8 = np.asarray(mask_u8, dtype=np.uint8).copy()
            ref_mask_x0 = int(x0)
            ref_mask_y0 = int(y0)
        except Exception:
            ref_mask_u8 = None
            ref_mask_x0 = 0
            ref_mask_y0 = 0
        ref_poly_uv = None

        try:
            fc_res = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if isinstance(fc_res, tuple) and int(len(fc_res)) == 3:
                _img, conts, _hier = fc_res
            else:
                conts, _hier = fc_res
        except Exception:
            conts = []
        if not conts:
            _hole_fail("hole_no_contours", extra=f"nz={int(nz0)}")
            return
        c = max(conts, key=lambda cc: float(cv2.contourArea(cc)))
        try:
            area = float(cv2.contourArea(c))
        except Exception:
            area = 0.0
        if float(area) < float(hole_min_area_px):
            _hole_fail("hole_area_small", extra=f"area={float(area):.1f} min={float(hole_min_area_px):.1f}")
            return

        poly_uv0, poly_dbg = _poly_fit_from_contour(
            contour_win=np.asarray(c, dtype=np.int32),
            x0=int(x0),
            y0=int(y0),
            concavity_max=float(hole_poly_concavity_max),
            approx_eps_frac=float(hole_poly_approx_eps_frac),
            min_vertex_angle_deg=float(hole_poly_min_vertex_angle_deg),
            max_vertices=int(hole_poly_max_vertices),
            min_edge_len_px=float(hole_poly_min_edge_len_px),
        )
        if poly_uv0 is None or int(np.asarray(poly_uv0).shape[0]) < 3:
            _hole_fail("hole_poly_fail", extra=f"n={int(0 if poly_uv0 is None else np.asarray(poly_uv0).shape[0])}")
            return
        if bool(dbg_log):
            try:
                print(
                    f"[poly_vo_lk] hole_poly fi={int(fi_sel)} area={float(area):.1f} verts={int(np.asarray(poly_uv0).shape[0])} "
                    f"conc={float(poly_dbg.get('concavity',0.0)):.2f} hull={int(bool(poly_dbg.get('used_hull',False)))} "
                    f"eps={float(poly_dbg.get('eps',0.0)):.1f} peri={float(poly_dbg.get('peri',0.0)):.1f} n_raw={int(poly_dbg.get('n_raw',0))}",
                    flush=True,
                )
            except Exception:
                pass

        # Initialize the refinement polygon from the detected contour. This is what the GUI will show while we are
        # still in the "refine" stage (before PID confirmation). It will then be propagated using the tracked polygon
        # motion and periodically updated with new depth-based detections.
        try:
            ref_poly_uv = np.asarray(poly_uv0, dtype=np.float32).reshape(-1, 2).copy()
            # Optional micro-refinement: snap vertices along edge normals (small per-call displacement).
            try:
                if bool(stage1_edge_snap.enabled):
                    ref_poly_uv = edge_normal_snap_poly_uv(
                        gray=np.asarray(gray, dtype=np.uint8),
                        poly_uv=ref_poly_uv,
                        cfg=stage1_edge_snap,
                    )
            except Exception:
                pass
            trk_poly_prev = np.asarray(ref_poly_uv, dtype=np.float32).reshape(-1, 2).copy()
            # BEGIN_STAGE1_HYSTERESIS
            stage1_best_r_px = float(hole_r_px) if np.isfinite(float(hole_r_px)) else float("nan")
            stage1_shrink_votes = int(0)
            # END_STAGE1_HYSTERESIS
        except Exception:
            ref_poly_uv = None
            trk_poly_prev = None
            # BEGIN_STAGE1_HYSTERESIS
            stage1_best_r_px = float("nan")
            stage1_shrink_votes = int(0)
            # END_STAGE1_HYSTERESIS

        hole_depth_r_px = int(max(0, int(proc_cfg.get("hole_depth_r_px", 8))))
        _activate_selection_from_poly_uv(
            poly_uv0=np.asarray(poly_uv0, dtype=np.float32).reshape(-1, 2),
            fi_sel=int(fi_sel),
            ts_sel=float(ts_sel),
            gray=gray,
            depth_raw=depth_raw,
            depth_mask_win=np.asarray(mask_u8, dtype=np.uint8),
            depth_mask_x0=int(x0),
            depth_mask_y0=int(y0),
            depth_sample_r_px=int(max(int(depth_r_px), int(hole_depth_r_px))),
            depth_prefer_outside_mask=True,
        )
        # Optional: capture an anchor snapshot (ORB/lines) for this selection.
        try:
            if (
                feature_acq is not None
                and bool(feature_acq_stage1)
                and bool(feature_acq_on_select)
                and ref_poly_uv is not None
                and int(np.asarray(ref_poly_uv).shape[0]) >= 3
            ):
                feat_anchor = feature_acq.acquire(
                    gray=np.asarray(gray, dtype=np.uint8),
                    poly_uv=np.asarray(ref_poly_uv, dtype=np.float32).reshape(-1, 2),
                )
                feat_last = None
                feat_last_t = 0.0
                feat_orb_A = None
                feat_orb_ok = False
                try:
                    feat_orb_n = int(0 if feat_anchor.orb_uv is None else np.asarray(feat_anchor.orb_uv).shape[0])
                except Exception:
                    feat_orb_n = 0
                feat_orb_inl_n = 0
                try:
                    feat_lines_n = int(0 if feat_anchor.lines_xyxy is None else np.asarray(feat_anchor.lines_xyxy).shape[0])
                except Exception:
                    feat_lines_n = 0
                feat_orb_uv_pub = None
                feat_lines_pub = None
                try:
                    if bool(feature_acq_publish_orb) and feat_anchor.orb_uv is not None:
                        uv0 = np.asarray(feat_anchor.orb_uv, dtype=np.float32).reshape(-1, 2)
                        nmax = int(max(0, int(feature_acq_publish_max_orb)))
                        if int(nmax) > 0 and int(uv0.shape[0]) > int(nmax):
                            uv0 = np.asarray(uv0[: int(nmax)], dtype=np.float32).reshape(-1, 2)
                        feat_orb_uv_pub = uv0
                except Exception:
                    feat_orb_uv_pub = None
                try:
                    if bool(feature_acq_publish_lines) and feat_anchor.lines_xyxy is not None:
                        L0 = np.asarray(feat_anchor.lines_xyxy, dtype=np.float32).reshape(-1, 4)
                        nmax = int(max(0, int(feature_acq_publish_max_lines)))
                        if int(nmax) > 0 and int(L0.shape[0]) > int(nmax):
                            L0 = np.asarray(L0[: int(nmax)], dtype=np.float32).reshape(-1, 4)
                        feat_lines_pub = L0
                except Exception:
                    feat_lines_pub = None
                if bool(dbg_log):
                    try:
                        n0 = int(0 if feat_anchor.orb_uv is None else np.asarray(feat_anchor.orb_uv).shape[0])
                        print(f"[poly_vo_lk] feat_anchor (hole) orb_n={int(n0)}", flush=True)
                    except Exception:
                        pass
        except Exception:
            pass

    def _handle_confirm_hole(msg: dict) -> None:
        nonlocal pid_confirmed, sel_kind, hole_ok_pid, hole_r_m, last_reason
        nonlocal ref_poly_uv, ref_mask_u8, ref_mask_x0, ref_mask_y0, trk_poly_prev
        nonlocal poly_active, poly_uv
        nonlocal feat_anchor, feat_last, feat_last_t, feat_orb_A, feat_orb_ok, feat_orb_n, feat_orb_inl_n, feat_lines_n, feat_orb_uv_pub, feat_lines_pub
        fi = int(msg.get("fi", last_fi if last_fi >= 0 else 0))
        ts = float(msg.get("ts", 0.0))
        if (not bool(poly_active)) or str(sel_kind) != "hole":
            pid_confirmed = False
            last_reason = "pid_no_hole"
            _publish(fi=int(fi), ts=float(ts), reason="pid_no_hole")
            return
        # Lock can be enabled even when the opening is smaller than the PID safety gate; the GUI still exposes
        # `hole_ok_pid` so the operator can see the size warning.
        if bool(dbg_log):
            try:
                print(
                    f"[poly_vo_lk] hole_confirm_start fi={int(fi)} ok_pid={int(bool(hole_ok_pid))} "
                    f"r={float(hole_r_m):.3f} fill={100.0*float(hole_fill):.0f}% "
                    f"inl={int(hole_plane_inliers)} rms={1000.0*float(hole_plane_rms_m):.1f}mm cov={100.0*float(hole_plane_cov):.0f}%",
                    flush=True,
                )
            except Exception:
                pass

        # Reinitialize the 3D polygon using the refined depth-based contour (if available) so that the lock starts
        # from the best/cleanest outline the detector+refiner converged to.
        try:
            slot = ring.find_slot_by_frame_idx(int(fi))
        except Exception:
            slot = None
        if slot is None:
            latest2 = ring.read_latest()
            if latest2 is not None:
                slot, fi, ts = latest2
        if slot is None:
            pid_confirmed = False
            last_reason = "pid_no_frame"
            _publish(fi=int(fi), ts=float(ts), reason="pid_no_frame")
            return

        try:
            gray = np.asarray(ring.gray[int(slot)], dtype=np.uint8)
        except Exception:
            pid_confirmed = False
            last_reason = "pid_no_gray"
            _publish(fi=int(fi), ts=float(ts), reason="pid_no_gray")
            return
        depth_raw = None
        try:
            if bool(cap_cfg.get("depth", True)) and float(depth_units) > 0.0:
                depth_raw = np.asarray(ring.depth_raw[int(slot)], dtype=np.uint16)
        except Exception:
            depth_raw = None
        if depth_raw is None:
            pid_confirmed = False
            last_reason = "pid_no_depth"
            _publish(fi=int(fi), ts=float(ts), reason="pid_no_depth")
            return

        try:
            poly_uv_use = (
                np.asarray(ref_poly_uv, dtype=np.float32).reshape(-1, 2)
                if ref_poly_uv is not None and int(np.asarray(ref_poly_uv).shape[0]) >= 3
                else np.asarray(poly_uv, dtype=np.float32).reshape(-1, 2)
            )
        except Exception:
            pid_confirmed = False
            last_reason = "pid_poly_bad"
            _publish(fi=int(fi), ts=float(ts), reason="pid_poly_bad")
            return

        hole_depth_r_px = int(max(0, int(proc_cfg.get("hole_depth_r_px", 8))))
        _activate_selection_from_poly_uv(
            poly_uv0=np.asarray(poly_uv_use, dtype=np.float32).reshape(-1, 2),
            fi_sel=int(fi),
            ts_sel=float(ts),
            gray=gray,
            depth_raw=depth_raw,
            depth_mask_win=(np.asarray(ref_mask_u8, dtype=np.uint8) if ref_mask_u8 is not None else None),
            depth_mask_x0=int(ref_mask_x0),
            depth_mask_y0=int(ref_mask_y0),
            depth_sample_r_px=int(max(int(depth_r_px), int(hole_depth_r_px))),
            depth_prefer_outside_mask=True,
        )

        # Optional: refresh the feature anchor on confirm (start of movement stage).
        try:
            if feature_acq is not None and bool(feature_acq_stage2) and bool(feature_acq_on_confirm):
                feat_anchor = feature_acq.acquire(
                    gray=np.asarray(gray, dtype=np.uint8),
                    poly_uv=np.asarray(poly_uv_use, dtype=np.float32).reshape(-1, 2),
                )
                feat_last = None
                feat_last_t = 0.0
                feat_orb_A = None
                feat_orb_ok = False
                try:
                    feat_orb_n = int(0 if feat_anchor.orb_uv is None else np.asarray(feat_anchor.orb_uv).shape[0])
                except Exception:
                    feat_orb_n = 0
                feat_orb_inl_n = 0
                try:
                    feat_lines_n = int(0 if feat_anchor.lines_xyxy is None else np.asarray(feat_anchor.lines_xyxy).shape[0])
                except Exception:
                    feat_lines_n = 0
                feat_orb_uv_pub = None
                feat_lines_pub = None
                try:
                    if bool(feature_acq_publish_orb) and feat_anchor.orb_uv is not None:
                        uv0 = np.asarray(feat_anchor.orb_uv, dtype=np.float32).reshape(-1, 2)
                        nmax = int(max(0, int(feature_acq_publish_max_orb)))
                        if int(nmax) > 0 and int(uv0.shape[0]) > int(nmax):
                            uv0 = np.asarray(uv0[: int(nmax)], dtype=np.float32).reshape(-1, 2)
                        feat_orb_uv_pub = uv0
                except Exception:
                    feat_orb_uv_pub = None
                try:
                    if bool(feature_acq_publish_lines) and feat_anchor.lines_xyxy is not None:
                        L0 = np.asarray(feat_anchor.lines_xyxy, dtype=np.float32).reshape(-1, 4)
                        nmax = int(max(0, int(feature_acq_publish_max_lines)))
                        if int(nmax) > 0 and int(L0.shape[0]) > int(nmax):
                            L0 = np.asarray(L0[: int(nmax)], dtype=np.float32).reshape(-1, 4)
                        feat_lines_pub = L0
                except Exception:
                    feat_lines_pub = None
                if bool(dbg_log):
                    try:
                        n0 = int(0 if feat_anchor.orb_uv is None else np.asarray(feat_anchor.orb_uv).shape[0])
                        print(f"[poly_vo_lk] feat_anchor_confirm (hole) orb_n={int(n0)}", flush=True)
                    except Exception:
                        pass
        except Exception:
            pass

        # Confirmed: stop running the hole detector; continue tracking with the locked outline only.
        pid_confirmed = True
        if bool(hole_ok_pid):
            last_reason = "pid_on"
            pub_reason = "pid_on"
        else:
            try:
                last_reason = f"pid_on_small_r={float(hole_r_m):.2f}"
            except Exception:
                last_reason = "pid_on_small"
            pub_reason = "pid_on_small"
        # Disable refine overlay for publish once confirmed.
        ref_poly_uv = None
        trk_poly_prev = None
        _publish(fi=int(fi), ts=float(ts), reason=str(pub_reason))
        if bool(dbg_log):
            try:
                print(f"[poly_vo_lk] hole_confirm_done fi={int(fi)} reason={str(pub_reason)}", flush=True)
            except Exception:
                pass

    def _handle_confirm_plane(msg: dict) -> None:
        nonlocal pid_confirmed, sel_kind, plane_ok_pid, plane_r_m, last_reason
        nonlocal poly_active, poly_uv
        nonlocal feat_anchor, feat_last, feat_last_t, feat_orb_A, feat_orb_ok, feat_orb_n, feat_orb_inl_n, feat_lines_n, feat_orb_uv_pub, feat_lines_pub
        fi = int(msg.get("fi", last_fi if last_fi >= 0 else 0))
        ts = float(msg.get("ts", 0.0))
        if (not bool(poly_active)) or str(sel_kind) != "plane":
            pid_confirmed = False
            last_reason = "pid_no_plane"
            _publish(fi=int(fi), ts=float(ts), reason="pid_no_plane")
            return
        if bool(dbg_log):
            try:
                print(
                    f"[poly_vo_lk] plane_confirm_start fi={int(fi)} ok_pid={int(bool(plane_ok_pid))} r={float(plane_r_m):.3f} "
                    f"inl={int(plane_plane_inliers)} rms={1000.0*float(plane_plane_rms_m):.1f}mm cov={100.0*float(plane_plane_cov):.0f}%",
                    flush=True,
                )
            except Exception:
                pass

        # Reinitialize the 3D polygon from the current outline so the lock starts from the latest tracked shape.
        try:
            slot = ring.find_slot_by_frame_idx(int(fi))
        except Exception:
            slot = None
        if slot is None:
            latest2 = ring.read_latest()
            if latest2 is not None:
                slot, fi, ts = latest2
        if slot is None:
            pid_confirmed = False
            last_reason = "pid_no_frame"
            _publish(fi=int(fi), ts=float(ts), reason="pid_no_frame")
            return

        try:
            gray = np.asarray(ring.gray[int(slot)], dtype=np.uint8)
        except Exception:
            pid_confirmed = False
            last_reason = "pid_no_gray"
            _publish(fi=int(fi), ts=float(ts), reason="pid_no_gray")
            return
        depth_raw = None
        try:
            if bool(cap_cfg.get("depth", True)) and float(depth_units) > 0.0:
                depth_raw = np.asarray(ring.depth_raw[int(slot)], dtype=np.uint16)
        except Exception:
            depth_raw = None
        if depth_raw is None:
            pid_confirmed = False
            last_reason = "pid_no_depth"
            _publish(fi=int(fi), ts=float(ts), reason="pid_no_depth")
            return

        try:
            poly_uv_use = np.asarray(poly_uv, dtype=np.float32).reshape(-1, 2)
        except Exception:
            pid_confirmed = False
            last_reason = "pid_poly_bad"
            _publish(fi=int(fi), ts=float(ts), reason="pid_poly_bad")
            return

        hole_depth_r_px = int(max(0, int(proc_cfg.get("hole_depth_r_px", 8))))
        _activate_selection_from_poly_uv(
            poly_uv0=np.asarray(poly_uv_use, dtype=np.float32).reshape(-1, 2),
            fi_sel=int(fi),
            ts_sel=float(ts),
            gray=gray,
            depth_raw=depth_raw,
            depth_sample_r_px=int(max(int(depth_r_px), int(hole_depth_r_px))),
            depth_prefer_outside_mask=False,
        )

        # Optional: refresh the feature anchor on confirm (start of movement stage).
        try:
            if feature_acq is not None and bool(feature_acq_stage2) and bool(feature_acq_on_confirm):
                feat_anchor = feature_acq.acquire(
                    gray=np.asarray(gray, dtype=np.uint8),
                    poly_uv=np.asarray(poly_uv_use, dtype=np.float32).reshape(-1, 2),
                )
                feat_last = None
                feat_last_t = 0.0
                feat_orb_A = None
                feat_orb_ok = False
                try:
                    feat_orb_n = int(0 if feat_anchor.orb_uv is None else np.asarray(feat_anchor.orb_uv).shape[0])
                except Exception:
                    feat_orb_n = 0
                feat_orb_inl_n = 0
                try:
                    feat_lines_n = int(0 if feat_anchor.lines_xyxy is None else np.asarray(feat_anchor.lines_xyxy).shape[0])
                except Exception:
                    feat_lines_n = 0
                feat_orb_uv_pub = None
                feat_lines_pub = None
                try:
                    if bool(feature_acq_publish_orb) and feat_anchor.orb_uv is not None:
                        uv0 = np.asarray(feat_anchor.orb_uv, dtype=np.float32).reshape(-1, 2)
                        nmax = int(max(0, int(feature_acq_publish_max_orb)))
                        if int(nmax) > 0 and int(uv0.shape[0]) > int(nmax):
                            uv0 = np.asarray(uv0[: int(nmax)], dtype=np.float32).reshape(-1, 2)
                        feat_orb_uv_pub = uv0
                except Exception:
                    feat_orb_uv_pub = None
                try:
                    if bool(feature_acq_publish_lines) and feat_anchor.lines_xyxy is not None:
                        L0 = np.asarray(feat_anchor.lines_xyxy, dtype=np.float32).reshape(-1, 4)
                        nmax = int(max(0, int(feature_acq_publish_max_lines)))
                        if int(nmax) > 0 and int(L0.shape[0]) > int(nmax):
                            L0 = np.asarray(L0[: int(nmax)], dtype=np.float32).reshape(-1, 4)
                        feat_lines_pub = L0
                except Exception:
                    feat_lines_pub = None
                if bool(dbg_log):
                    try:
                        n0 = int(0 if feat_anchor.orb_uv is None else np.asarray(feat_anchor.orb_uv).shape[0])
                        print(f"[poly_vo_lk] feat_anchor_confirm (plane) orb_n={int(n0)}", flush=True)
                    except Exception:
                        pass
        except Exception:
            pass

        pid_confirmed = True
        if bool(plane_ok_pid):
            last_reason = "pid_on"
            pub_reason = "pid_on"
        else:
            try:
                last_reason = f"pid_on_small_r={float(plane_r_m):.2f}"
            except Exception:
                last_reason = "pid_on_small"
            pub_reason = "pid_on_small"
        _publish(fi=int(fi), ts=float(ts), reason=str(pub_reason))
        if bool(dbg_log):
            try:
                print(f"[poly_vo_lk] plane_confirm_done fi={int(fi)} reason={str(pub_reason)}", flush=True)
            except Exception:
                pass

    while not stop_event.is_set():
        # Drain commands (selection/clear).
        try:
            while True:
                msg = cmd_q.get_nowait()
                if not isinstance(msg, dict):
                    continue
                cmd = str(msg.get("cmd", "")).strip().lower()
                if cmd == "clear":
                    _reset_state(keep_selection=False)
                    _publish(fi=int(last_fi if last_fi >= 0 else 0), ts=float(0.0), reason="clear")
                elif cmd in ("hole_enable", "hole_detector_enable"):
                    # Runtime enable for hole acquisition (mouse-driven). This is exclusive with plane mode.
                    en = msg.get("enable", True)
                    try:
                        en_i = int(en)
                        hole_runtime_enabled = bool(en_i != 0)
                    except Exception:
                        hole_runtime_enabled = bool(en)
                    if bool(hole_runtime_enabled):
                        # Switching into hole mode: disable plane mode and clear any plane selection.
                        if bool(plane_runtime_enabled) or str(sel_kind) == "plane":
                            hover_uv = None
                            preview_pending = None
                            _reset_state(keep_selection=False)
                            _publish(fi=int(last_fi if last_fi >= 0 else 0), ts=float(0.0), reason="clear")
                        plane_runtime_enabled = False
                    if not bool(hole_runtime_enabled):
                        # Stop expensive preview detection immediately and clear any active selection/PID.
                        hover_uv = None
                        preview_pending = None
                        _reset_state(keep_selection=False)
                        _publish(fi=int(last_fi if last_fi >= 0 else 0), ts=float(0.0), reason="clear")
                elif cmd in ("plane_enable", "plane_detector_enable"):
                    # Runtime enable for plane acquisition (mouse-driven). This is exclusive with hole mode.
                    en = msg.get("enable", True)
                    try:
                        en_i = int(en)
                        plane_runtime_enabled = bool(en_i != 0)
                    except Exception:
                        plane_runtime_enabled = bool(en)
                    if bool(plane_runtime_enabled):
                        # Switching into plane mode: disable hole mode and clear any hole selection.
                        if bool(hole_runtime_enabled) or str(sel_kind) == "hole":
                            hover_uv = None
                            preview_pending = None
                            _reset_state(keep_selection=False)
                            _publish(fi=int(last_fi if last_fi >= 0 else 0), ts=float(0.0), reason="clear")
                        hole_runtime_enabled = False
                    if not bool(plane_runtime_enabled):
                        hover_uv = None
                        preview_pending = None
                        _reset_state(keep_selection=False)
                        _publish(fi=int(last_fi if last_fi >= 0 else 0), ts=float(0.0), reason="clear")
                elif cmd == "select_bbox":
                    _handle_select_bbox(dict(msg))
                elif cmd == "select_hole":
                    if bool(hole_runtime_enabled):
                        _handle_select_hole(dict(msg))
                elif cmd in ("confirm_hole", "confirm"):
                    if bool(hole_runtime_enabled):
                        _handle_confirm_hole(dict(msg))
                elif cmd == "select_plane":
                    if bool(plane_runtime_enabled):
                        _handle_select_plane(dict(msg))
                elif cmd == "confirm_plane":
                    if bool(plane_runtime_enabled):
                        _handle_confirm_plane(dict(msg))
                elif cmd in ("hover", "mouse_move", "pointer"):
                    if bool(hole_runtime_enabled) or bool(plane_runtime_enabled):
                        try:
                            hx = int(msg.get("x", -1))
                            hy = int(msg.get("y", -1))
                            hover_uv = (int(hx), int(hy)) if (int(hx) >= 0 and int(hy) >= 0) else None
                        except Exception:
                            hover_uv = None
                    else:
                        hover_uv = None
        except Exception:
            pass

        latest = ring.read_latest() if bool(drop_frames) else ring.read_next(after_frame_idx=int(last_fi), timeout=0.05)
        if latest is None:
            time.sleep(float(idle_sleep_s))
            continue
        slot, fi, ts = latest
        if int(fi) <= int(last_fi):
            time.sleep(float(idle_sleep_s))
            continue

        prev_fi = int(last_fi)
        gap = int(fi) - int(last_fi) if int(last_fi) >= 0 else 1
        last_fi = int(fi)
        hole_plane_center_range_m = float("nan")

        # Allow frame gating (run every N frames).
        if int(every_n_frames) > 1 and (int(fi) % int(every_n_frames)) != 0:
            _publish(fi=int(fi), ts=float(ts), reason="skip")
            continue

        if (not bool(enabled)) or (not bool(poly_active)) or poly_world_pts_w is None or edge_world_pts_w is None:
            _fps_n += 1
            now = float(time.time())
            dt_fps = float(now - float(_fps_t0))
            if float(dt_fps) >= 0.5:
                fps = float(_fps_n) / float(max(1e-6, dt_fps))
                _fps_n = 0
                _fps_t0 = float(now)
 
             # Stage-0 hover preview (headless integration):
             # When no selection is active, run the hole detector at a bounded rate based on the latest hover pixel.
             # This is kept inside the poly process so it cannot stall the main app loop (which drives headless JSON I/O).
            try:
                  if (
                      bool(plane_runtime_enabled)
                      and bool(plane_preview_enabled)
                      and (not bool(poly_active))
                      and hover_uv is not None
                      and intr_obj is not None
                      and bool(cap_cfg.get("depth", True))
                      and float(depth_units) > 0.0
                      and float(plane_preview_rate_hz) > 0.0
                  ):
                     min_dt = float(1.0 / float(max(1e-6, float(plane_preview_rate_hz))))
                     if float(now - float(preview_last_t)) >= float(min_dt):
                         preview_last_t = float(now)
                         hx, hy = int(hover_uv[0]), int(hover_uv[1])
                         try:
                             gray0 = np.asarray(ring.gray[int(slot)], dtype=np.uint8)
                             depth0 = np.asarray(ring.depth_raw[int(slot)], dtype=np.uint16)
                         except Exception:
                             gray0 = None
                             depth0 = None
 
                         if (
                             gray0 is not None
                             and depth0 is not None
                             and isinstance(gray0, np.ndarray)
                             and isinstance(depth0, np.ndarray)
                             and int(gray0.ndim) == 2
                             and int(depth0.ndim) == 2
                             and int(depth0.max()) > 0
                         ):
                             h_img = int(gray0.shape[0])
                             w_img = int(gray0.shape[1])
                             if 0 <= int(hx) < int(w_img) and 0 <= int(hy) < int(h_img):
                                 x0, y0, ww, hh = _window_bbox(
                                     int(w_img),
                                     int(h_img),
                                     x=int(hx),
                                     y=int(hy),
                                     win_w=int(hole_win_w),
                                     win_h=int(hole_win_h),
                                 )
 
                                 try:
                                     _m_u8, p_u8 = hole.detect_plane_guided(
                                         {"bgr": None, "gray": gray0, "depth_raw": depth0, "depth_units": float(depth_units), "intr": intr_obj},
                                         window_wh=(int(hole_win_w), int(hole_win_h)),
                                         x=int(hx),
                                         y=int(hy),
                                         settings=hole_settings_preview,
                                     )
                                 except Exception:
                                     p_u8 = None
                                 try:
                                     plane_u8 = np.asarray(p_u8, dtype=np.uint8) if p_u8 is not None else np.zeros((1, 1), dtype=np.uint8)
                                 except Exception:
                                     plane_u8 = np.zeros((1, 1), dtype=np.uint8)
 
                                 if int(np.count_nonzero(plane_u8)) > 0:
                                     hole_depth_r_px = int(max(0, int(proc_cfg.get("hole_depth_r_px", 8))))
                                     try:
                                         depth_sample_r_px = int(max(int(depth_r_px), int(hole_depth_r_px)))
                                     except Exception:
                                         depth_sample_r_px = int(max(0, int(hole_depth_r_px)))
                                     out = _plane_poly_from_plane_mask(
                                         plane_u8=np.asarray(plane_u8, dtype=np.uint8),
                                         x0=int(x0),
                                         y0=int(y0),
                                         depth_raw=np.asarray(depth0, dtype=np.uint16),
                                         close_px=int(plane_mask_close_px),
                                         n_verts=int(plane_poly_n_verts),
                                         depth_sample_r_px=int(depth_sample_r_px),
                                         seed_uv_roi=(int(hx) - int(x0), int(hy) - int(y0)),
                                     )
                                     if out is not None:
                                         try:
                                             _plane_ok = bool(getattr(hole, "last_plane_ok", False))
                                         except Exception:
                                             _plane_ok = False
                                         try:
                                             r_m = float(out.get("r_m", 0.0) or 0.0)
                                         except Exception:
                                             r_m = 0.0
                                         prev_ok_pid = bool(_plane_ok) and bool(np.isfinite(float(r_m))) and float(r_m) >= float(plane_pid_min_radius_m)
                                         try:
                                             c_u, c_v = out.get("center_uv", (float(getattr(intr_obj, "cx", 0.0)), float(getattr(intr_obj, "cy", 0.0))))
                                             c_u = float(c_u)
                                             c_v = float(c_v)
                                         except Exception:
                                             c_u = float(getattr(intr_obj, "cx", 0.0))
                                             c_v = float(getattr(intr_obj, "cy", 0.0))
 
                                         preview_pending = {
                                             "fi": int(fi),
                                             "sel_kind": str("plane"),
                                             "poly_uv": np.asarray(out.get("poly_uv", None), dtype=np.float32).reshape(-1, 2),
                                             "center_uv": (float(c_u), float(c_v)),
                                             "ok_pid": bool(prev_ok_pid),
                                             "plane_stats": {
                                                 "plane_ok": int(bool(_plane_ok)),
                                                 "plane_inliers": int(getattr(hole, "last_plane_inliers", 0)),
                                                 "plane_rms_m": float(getattr(hole, "last_plane_rms_m", 0.0)),
                                                 "plane_cov": float(getattr(hole, "last_plane_sector_cov", 0.0)),
                                                 "r_m": float(r_m),
                                                 "z_ref_m": float(out.get("z_ref_m", float("nan"))),
                                                 "err": str(getattr(hole, "last_error", "") or ""),
                                             },
                                         }
                  elif (
                      bool(hole_runtime_enabled)
                      and bool(hole_preview_enabled)
                      and (not bool(poly_active))
                      and hover_uv is not None
                      and intr_obj is not None
                      and bool(cap_cfg.get("depth", True))
                      and float(depth_units) > 0.0
                      and float(hole_preview_rate_hz) > 0.0
                  ):
                     min_dt = float(1.0 / float(max(1e-6, float(hole_preview_rate_hz))))
                     if float(now - float(preview_last_t)) >= float(min_dt):
                         preview_last_t = float(now)
                         hx, hy = int(hover_uv[0]), int(hover_uv[1])
                         try:
                             gray0 = np.asarray(ring.gray[int(slot)], dtype=np.uint8)
                             depth0 = np.asarray(ring.depth_raw[int(slot)], dtype=np.uint16)
                         except Exception:
                             gray0 = None
                             depth0 = None
 
                         if (
                             gray0 is not None
                             and depth0 is not None
                             and isinstance(gray0, np.ndarray)
                             and isinstance(depth0, np.ndarray)
                             and int(gray0.ndim) == 2
                             and int(depth0.ndim) == 2
                             and int(depth0.max()) > 0
                         ):
                             h_img = int(gray0.shape[0])
                             w_img = int(gray0.shape[1])
                             if 0 <= int(hx) < int(w_img) and 0 <= int(hy) < int(h_img):
                                 x0, y0, ww, hh = _window_bbox(
                                     int(w_img),
                                     int(h_img),
                                     x=int(hx),
                                     y=int(hy),
                                     win_w=int(hole_win_w),
                                     win_h=int(hole_win_h),
                                 )
 
                                 # Strict preview first (no snap); if empty and snap is 0, retry with a small fallback snap.
                                 mask_u8, _plane_u8 = hole.detect_hole_plane(
                                     {"bgr": None, "gray": gray0, "depth_raw": depth0, "depth_units": float(depth_units), "intr": intr_obj},
                                     window_wh=(int(hole_win_w), int(hole_win_h)),
                                     x=int(hx),
                                     y=int(hy),
                                     settings=hole_settings_preview,
                                 )
                                 try:
                                     mask_u8 = np.asarray(mask_u8, dtype=np.uint8)
                                 except Exception:
                                     mask_u8 = np.zeros((1, 1), dtype=np.uint8)
                                 try:
                                     nz0 = int(np.count_nonzero(mask_u8))
                                 except Exception:
                                     nz0 = 0
                                 if int(nz0) <= 0 and int(hole_preview_snap_px) <= 0 and int(hole_lock_snap_px) > 0:
                                     try:
                                         m2, _p2 = hole.detect_hole_plane(
                                             {"bgr": None, "gray": gray0, "depth_raw": depth0, "depth_units": float(depth_units), "intr": intr_obj},
                                             window_wh=(int(hole_win_w), int(hole_win_h)),
                                             x=int(hx),
                                             y=int(hy),
                                             settings=hole_settings_preview_fallback,
                                         )
                                         m2 = np.asarray(m2, dtype=np.uint8)
                                         if int(np.count_nonzero(m2)) > 0:
                                             mask_u8 = m2
                                     except Exception:
                                         pass
 
                                 if int(np.count_nonzero(mask_u8)) > 0:
                                     try:
                                         fc_res = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                                         if isinstance(fc_res, tuple) and int(len(fc_res)) == 3:
                                             _img, conts, _hier = fc_res
                                         else:
                                             conts, _hier = fc_res
                                     except Exception:
                                         conts = []
                                     if conts:
                                         c = max(conts, key=lambda cc: float(cv2.contourArea(cc)))
                                         try:
                                             area = float(cv2.contourArea(c))
                                         except Exception:
                                             area = 0.0
                                         if float(area) >= float(hole_min_area_px):
                                             poly_uv0, _dbg = _poly_fit_from_contour(
                                                 contour_win=np.asarray(c, dtype=np.int32),
                                                 x0=int(x0),
                                                 y0=int(y0),
                                                 concavity_max=float(hole_poly_concavity_max),
                                                 approx_eps_frac=float(hole_poly_approx_eps_frac),
                                                 min_vertex_angle_deg=float(hole_poly_min_vertex_angle_deg),
                                                 max_vertices=int(hole_poly_max_vertices),
                                                 min_edge_len_px=float(hole_poly_min_edge_len_px),
                                             )
                                             if poly_uv0 is not None and int(np.asarray(poly_uv0).shape[0]) >= 3:
                                                 # Center from detector when available; fallback to polygon centroid.
                                                 c_u = c_v = None
                                                 try:
                                                     cx_full, cy_full = getattr(hole, "last_circle_center_full_px", (-1, -1))
                                                     cx_full = int(cx_full)
                                                     cy_full = int(cy_full)
                                                     if int(cx_full) >= 0 and int(cy_full) >= 0:
                                                         c_u = float(cx_full)
                                                         c_v = float(cy_full)
                                                 except Exception:
                                                     c_u = c_v = None
                                                 if c_u is None or c_v is None:
                                                     try:
                                                         pu = np.asarray(poly_uv0, dtype=np.float64).reshape(-1, 2)
                                                         c_u = float(np.mean(pu[:, 0]))
                                                         c_v = float(np.mean(pu[:, 1]))
                                                     except Exception:
                                                         c_u = float(getattr(intr_obj, "cx", 0.0))
                                                         c_v = float(getattr(intr_obj, "cy", 0.0))
 
                                                 try:
                                                     _plane_ok = bool(getattr(hole, "last_plane_ok", False))
                                                 except Exception:
                                                     _plane_ok = False
                                                 try:
                                                     _bdr = bool(getattr(hole, "last_hole_touches_border", False))
                                                 except Exception:
                                                     _bdr = False
                                                 try:
                                                     r_m = float(getattr(hole, "last_radius_m", 0.0))
                                                 except Exception:
                                                     r_m = 0.0
                                                 prev_ok_pid = (
                                                     bool(_plane_ok)
                                                     and (not bool(_bdr))
                                                     and bool(np.isfinite(float(r_m)))
                                                     and float(r_m) >= float(hole_pid_min_radius_m)
                                                 )

                                                 # Cache preview for possible auto-lock (keep heavy mask out of published preview_pending).
                                                 try:
                                                     r_px0 = float(getattr(hole, "last_circle_radius_px", 0.0))
                                                 except Exception:
                                                     r_px0 = 0.0
                                                 try:
                                                     cov0 = float(getattr(hole, "last_plane_sector_cov", 0.0))
                                                 except Exception:
                                                     cov0 = 0.0
                                                 try:
                                                     inl0 = int(getattr(hole, "last_plane_inliers", 0))
                                                 except Exception:
                                                     inl0 = 0
                                                 try:
                                                     rms0 = float(getattr(hole, "last_plane_rms_m", 0.0))
                                                 except Exception:
                                                     rms0 = 0.0
                                                 try:
                                                     err0 = str(getattr(hole, "last_error", "") or "")
                                                 except Exception:
                                                     err0 = ""
                                                 preview_cache["hole"] = {
                                                     "x": int(hx),
                                                     "y": int(hy),
                                                     "fi": int(fi),
                                                     "ts": float(ts),
                                                     "poly_uv": np.asarray(poly_uv0, dtype=np.float32).reshape(-1, 2),
                                                     "mask_u8": np.asarray(mask_u8, dtype=np.uint8),
                                                     "mask_x0": int(x0),
                                                     "mask_y0": int(y0),
                                                     "hole_stats": {
                                                         "plane_ok": int(bool(_plane_ok)),
                                                         "touches_border": int(bool(_bdr)),
                                                         "plane_inliers": int(inl0),
                                                         "plane_rms_m": float(rms0),
                                                         "plane_cov": float(cov0),
                                                         "r_px": float(r_px0),
                                                         "r_m": float(r_m),
                                                         "fill": float(getattr(hole, "last_circle_fill_ratio", 0.0)),
                                                         "center_uv": (float(c_u), float(c_v)),
                                                         "err": str(err0),
                                                     },
                                                 }

                                                 preview_pending = {
                                                     "fi": int(fi),
                                                     "sel_kind": str("hole"),
                                                     "poly_uv": np.asarray(poly_uv0, dtype=np.float32).reshape(-1, 2),
                                                     "center_uv": (float(c_u), float(c_v)),
                                                     "ok_pid": bool(prev_ok_pid),
                                                     "hole_stats": {
                                                         "plane_ok": int(bool(_plane_ok)),
                                                         "touches_border": int(bool(_bdr)),
                                                         "plane_inliers": int(getattr(hole, "last_plane_inliers", 0)),
                                                         "plane_rms_m": float(getattr(hole, "last_plane_rms_m", 0.0)),
                                                         "plane_cov": float(getattr(hole, "last_plane_sector_cov", 0.0)),
                                                         "r_px": float(r_px0),
                                                         "r_m": float(r_m),
                                                         "fill": float(getattr(hole, "last_circle_fill_ratio", 0.0)),
                                                         "center_uv": (float(c_u), float(c_v)),
                                                         "err": str(getattr(hole, "last_error", "") or ""),
                                                     },
                                                 }

                                                 # Auto-lock: immediately select this polygon to start tracking (stage 1).
                                                 try:
                                                     if bool(auto_lock_enabled) and str(auto_lock_kind) in ("hole", "both", "all"):
                                                         now_wall = float(time.time())
                                                         if float(now_wall - float(auto_lock_last_t)) >= float(auto_lock_cooldown_s):
                                                             auto_lock_last_t = float(now_wall)
                                                             try:
                                                                 _handle_select_hole(dict(preview_cache.get("hole") or {}))
                                                             except Exception:
                                                                 pass
                                                 except Exception:
                                                     pass
            except Exception:
                pass
            # If auto-lock activated a selection, avoid publishing an extra "idle" tick.
            if bool(poly_active):
                continue
            _publish(fi=int(fi), ts=float(ts), reason="idle")
            continue

        if int(gap) > int(max_gap_frames):
            # Large gap: keep selection but reset dynamic state.
            _reset_state(keep_selection=True)
            last_reason = f"gap_reset_{int(gap)}"

        gray = np.asarray(ring.gray[int(slot)], dtype=np.uint8)
        depth_raw = None
        try:
            if bool(cap_cfg.get("depth", True)) and float(depth_units) > 0.0:
                depth_raw = np.asarray(ring.depth_raw[int(slot)], dtype=np.uint16)
        except Exception:
            depth_raw = None

        # Hole refinement loop: keep updating the hole circle/plane estimate at low rate while tracking,
        # but only before PID confirmation (saves CPU during "movement").
        try:
            if (
                bool(poly_active)
                and str(sel_kind) == "hole"
                and (not bool(pid_confirmed))
                and float(hole_refine_rate_hz) > 0.0
                and depth_raw is not None
                and intr_obj is not None
            ):
                now_wall = float(time.time())
                min_dt = float(1.0 / float(max(1e-6, hole_refine_rate_hz)))
                if float(now_wall - float(hole_last_refine_ts)) >= float(min_dt):
                    # Use current hole center as the seed (more stable than polygon centroid when LK drifts).
                    try:
                        if hole_center_uv is not None:
                            u_det = float(hole_center_uv[0])
                            v_det = float(hole_center_uv[1])
                        elif poly_uv is not None and int(np.asarray(poly_uv).shape[0]) >= 3:
                            u_det = float(np.mean(np.asarray(poly_uv, dtype=np.float64)[:, 0]))
                            v_det = float(np.mean(np.asarray(poly_uv, dtype=np.float64)[:, 1]))
                        else:
                            u_det = 0.5 * float(np.asarray(gray).shape[1])
                            v_det = 0.5 * float(np.asarray(gray).shape[0])
                    except Exception:
                        u_det = 0.5 * float(np.asarray(gray).shape[1])
                        v_det = 0.5 * float(np.asarray(gray).shape[0])

                    try:
                        h_img = int(gray.shape[0])
                        w_img = int(gray.shape[1])
                        x0_det, y0_det, ww_det, hh_det = _window_bbox(
                            int(w_img), int(h_img), x=int(round(float(u_det))), y=int(round(float(v_det))), win_w=int(hole_win_w), win_h=int(hole_win_h)
                        )
                        if bool(hole_refine_thread) and hole_refine_req_q is not None and hole_refine_res_q is not None:
                            # Threaded refine: schedule a request (if idle) and apply the newest available result.
                            try:
                                if (not bool(hole_refine_busy.is_set())):
                                    if bool(hole_refine_thread_copy):
                                        if hole_refine_gray_buf is None or hole_refine_gray_buf.shape != np.asarray(gray).shape:
                                            hole_refine_gray_buf = np.empty_like(np.asarray(gray, dtype=np.uint8))
                                        np.copyto(hole_refine_gray_buf, np.asarray(gray, dtype=np.uint8), casting="unsafe")
                                        if hole_refine_depth_buf is None or hole_refine_depth_buf.shape != np.asarray(depth_raw).shape:
                                            hole_refine_depth_buf = np.empty_like(np.asarray(depth_raw, dtype=np.uint16))
                                        np.copyto(hole_refine_depth_buf, np.asarray(depth_raw, dtype=np.uint16), casting="unsafe")
                                        g_in = hole_refine_gray_buf
                                        d_in = hole_refine_depth_buf
                                    else:
                                        g_in = gray
                                        d_in = depth_raw
                                    hole_refine_busy.set()
                                    try:
                                        hole_refine_req_q.put_nowait(
                                            {
                                                "fi": int(fi),
                                                "ts": float(ts),
                                                "x": int(round(float(u_det))),
                                                "y": int(round(float(v_det))),
                                                "x0": int(x0_det),
                                                "y0": int(y0_det),
                                                "ww": int(ww_det),
                                                "hh": int(hh_det),
                                                "gray": g_in,
                                                "depth_raw": d_in,
                                            }
                                        )
                                    except Exception:
                                        hole_refine_busy.clear()
                            except Exception:
                                pass

                            res = None
                            try:
                                while True:
                                    res = hole_refine_res_q.get_nowait()
                            except Exception:
                                pass
                            if isinstance(res, dict):
                                hole_err = str(res.get("err", hole_err) or "")
                                _m_u8 = res.get("mask_u8", None)
                                det_ok = False
                                try:
                                    det_ok = (
                                        isinstance(_m_u8, np.ndarray)
                                        and int(np.count_nonzero(np.asarray(_m_u8, dtype=np.uint8))) > 0
                                        and bool(res.get("plane_ok", False))
                                        and not str(hole_err).strip()
                                    )
                                except Exception:
                                    det_ok = False
                                if bool(det_ok):
                                    hole_plane_inliers = int(res.get("plane_inliers", hole_plane_inliers))
                                    hole_plane_rms_m = float(res.get("plane_rms_m", hole_plane_rms_m))
                                    hole_plane_cov = float(res.get("plane_cov", hole_plane_cov))
                                    hole_r_px = float(res.get("circle_r_px", hole_r_px))
                                    hole_r_m = float(res.get("radius_m", hole_r_m))
                                    hole_fill = float(res.get("fill", hole_fill))
                                    try:
                                        c_full = res.get("center_full", None)
                                        if c_full is not None and int(len(c_full)) == 2:
                                            hole_center_uv = (float(c_full[0]), float(c_full[1]))
                                    except Exception:
                                        pass
                                    hole_ok_pid = (
                                        bool(res.get("plane_ok", False))
                                        and (not bool(res.get("touches_border", False)))
                                        and bool(np.isfinite(float(hole_r_m)))
                                        and float(hole_r_m) >= float(hole_pid_min_radius_m)
                                    )
                                    Twc_det = None
                                    if state is not None:
                                        st2 = state.read(copy=False)
                                        if st2 is not None and int(len(st2)) >= 12:
                                            try:
                                                Twc_det = np.asarray(st2[11], dtype=np.float64).reshape(4, 4)
                                            except Exception:
                                                Twc_det = None
                                    try:
                                        n_cam = np.asarray(res.get("plane_n_cam", (0.0, 0.0, 0.0)), dtype=np.float64).reshape(3)
                                        d_cam = float(res.get("plane_d_cam", 0.0))
                                    except Exception:
                                        n_cam = np.asarray((0.0, 0.0, 0.0), dtype=np.float64).reshape(3)
                                        d_cam = 0.0
                                    _update_hole_plane_world_from_detector(
                                        Twc_now=Twc_det, plane_ok=bool(res.get("plane_ok", False)), n_cam=n_cam, d_cam=float(d_cam)
                                    )
                                    try:
                                        if bool(perf_log):
                                            try:
                                                _perf_last_refine_det_ms = float(res.get("dt_ms", _perf_last_refine_det_ms))
                                            except Exception:
                                                pass
                                            _t_ref = float(time.perf_counter())
                                        _apply_hole_refine_mask(
                                            fi=int(fi),
                                            mask_u8=np.asarray(_m_u8, dtype=np.uint8),
                                            x0_det=int(res.get("x0", int(x0_det))),
                                            y0_det=int(res.get("y0", int(y0_det))),
                                            ww_det=int(res.get("ww", int(ww_det))),
                                            hh_det=int(res.get("hh", int(hh_det))),
                                            Twc_det=Twc_det,
                                        )
                                        if bool(perf_log):
                                            _perf_sum_refine_apply_ms += 1000.0 * float(time.perf_counter() - float(_t_ref))
                                            _perf_ref_n += 1
                                    except Exception:
                                        pass
                        else:
                            # Synchronous refine (legacy path).
                            _m_u8, _p_u8 = hole.detect_hole_plane(
                                {"bgr": None, "gray": gray, "depth_raw": depth_raw, "depth_units": float(depth_units), "intr": intr_obj},
                                window_wh=(int(hole_win_w), int(hole_win_h)),
                                x=int(round(float(u_det))),
                                y=int(round(float(v_det))),
                                settings=hole_settings_lock,
                            )
                            # Update cached hole stats only on successful detection; keep last-good values otherwise.
                            hole_err = str(getattr(hole, "last_error", hole_err) or "")
                            det_ok = False
                            try:
                                det_ok = (
                                    isinstance(_m_u8, np.ndarray)
                                    and int(np.count_nonzero(np.asarray(_m_u8, dtype=np.uint8))) > 0
                                    and bool(getattr(hole, "last_plane_ok", False))
                                    and not str(hole_err).strip()
                                )
                            except Exception:
                                det_ok = False
                            if bool(det_ok):
                                hole_plane_inliers = int(getattr(hole, "last_plane_inliers", hole_plane_inliers))
                                hole_plane_rms_m = float(getattr(hole, "last_plane_rms_m", hole_plane_rms_m))
                                hole_plane_cov = float(getattr(hole, "last_plane_sector_cov", hole_plane_cov))
                                hole_r_px = float(getattr(hole, "last_circle_radius_px", hole_r_px))
                                hole_r_m = float(getattr(hole, "last_radius_m", hole_r_m))
                                hole_fill = float(getattr(hole, "last_circle_fill_ratio", hole_fill))
                                try:
                                    cx_full, cy_full = getattr(hole, "last_circle_center_full_px", (-1, -1))
                                    cx_full = int(cx_full)
                                    cy_full = int(cy_full)
                                except Exception:
                                    cx_full, cy_full = -1, -1
                                hole_center_uv = (float(cx_full), float(cy_full)) if (cx_full >= 0 and cy_full >= 0) else hole_center_uv
                                try:
                                    _plane_ok = bool(getattr(hole, "last_plane_ok", False))
                                except Exception:
                                    _plane_ok = False
                                try:
                                    _bdr = bool(getattr(hole, "last_hole_touches_border", False))
                                except Exception:
                                    _bdr = False
                                hole_ok_pid = (
                                    bool(_plane_ok)
                                    and (not bool(_bdr))
                                    and bool(np.isfinite(float(hole_r_m)))
                                    and float(hole_r_m) >= float(hole_pid_min_radius_m)
                                )

                                Twc_det = None
                                if state is not None:
                                    st2 = state.read(copy=False)
                                    if st2 is not None and int(len(st2)) >= 12:
                                        try:
                                            Twc_det = np.asarray(st2[11], dtype=np.float64).reshape(4, 4)
                                        except Exception:
                                            Twc_det = None
                                _update_hole_plane_world_from_detector(Twc_now=Twc_det)
                                # Commit Stage-1 polygon/mask updates via a shared helper (stable + low-alloc).
                                try:
                                    if bool(perf_log):
                                        _t_ref = float(time.perf_counter())
                                    _apply_hole_refine_mask(
                                        fi=int(fi),
                                        mask_u8=np.asarray(_m_u8, dtype=np.uint8),
                                        x0_det=int(x0_det),
                                        y0_det=int(y0_det),
                                        ww_det=int(ww_det),
                                        hh_det=int(hh_det),
                                        Twc_det=Twc_det,
                                    )
                                    if bool(perf_log):
                                        _perf_sum_refine_apply_ms += 1000.0 * float(time.perf_counter() - float(_t_ref))
                                        _perf_ref_n += 1
                                except Exception:
                                    pass
                    except Exception:
                        hole_err = "hole_refine_exc"
                    hole_last_refine_ts = float(now_wall)
        except Exception:
            pass

        # Auto-stop when the locked polygon bbox nearly fills the frame (movement completed).
        # NOTE: in drone integration this is typically disabled and stop/reset is managed by the server.
        try:
            if (
                bool(hole_done_enabled)
                and bool(poly_active)
                and str(sel_kind) == "hole"
                and bool(pid_confirmed)
                and poly_bbox is not None
                and float(hole_done_fill_ratio) > 0.0
            ):
                h_img = int(gray.shape[0])
                w_img = int(gray.shape[1])
                bx0, by0, bx1, by1 = (int(poly_bbox[0]), int(poly_bbox[1]), int(poly_bbox[2]), int(poly_bbox[3]))
                bw = float(max(1, int(bx1) - int(bx0))) / float(max(1, int(w_img)))
                bh = float(max(1, int(by1) - int(by0))) / float(max(1, int(h_img)))
                fill_bb = float(min(float(bw), float(bh)))
                if np.isfinite(fill_bb) and float(fill_bb) >= float(hole_done_fill_ratio):
                    if bool(dbg_log):
                        try:
                            print(f"[poly_vo_lk] hole_done fi={int(fi)} bbox_fill={100.0*float(fill_bb):.0f}%", flush=True)
                        except Exception:
                            pass
                    _reset_state(keep_selection=False)
                    _publish(fi=int(fi), ts=float(ts), reason="hole_done")
                    continue
        except Exception:
            pass

        # Read latest VO pose.
        Twc_now = None
        if state is not None:
            st = state.read(copy=False)
            if st is not None and int(len(st)) >= 12:
                try:
                    Twc_now = np.asarray(st[11], dtype=np.float64).reshape(4, 4)
                except Exception:
                    Twc_now = None
        if Twc_now is None or intr_obj is None:
            poly_locked = False
            lock_used_uv = None
            last_inliers = 0
            last_n = 0
            last_reason = "no_pose"
            _publish(fi=int(fi), ts=float(ts), reason="no_pose")
            prev_gray = np.asarray(gray, dtype=np.uint8).copy()
            prev_Twc = None
            prev_corr_params = None
            continue

        # Predict polygon from VO pose.
        uv_poly_vo, _z_poly, ok_poly = _project_world_to_uv_z(
            Pw=np.asarray(poly_world_pts_w, dtype=np.float64),
            Twc=np.asarray(Twc_now, dtype=np.float64),
            intr=intr_obj,
        )
        if uv_poly_vo is None or ok_poly is None or (not bool(np.all(np.asarray(ok_poly, dtype=bool).reshape(-1)))):
            poly_locked = False
            lock_used_uv = None
            last_inliers = 0
            last_n = 0
            last_reason = "proj_fail"
            poly_uv = None
            poly_bbox = None
            _publish(fi=int(fi), ts=float(ts), reason="proj_fail")
            prev_gray = np.asarray(gray, dtype=np.uint8).copy()
            prev_Twc = np.asarray(Twc_now, dtype=np.float64).reshape(4, 4).copy()
            prev_corr_params = np.asarray(corr_params, dtype=np.float64).copy()
            continue

        # Initialize prev state on first tick after selection/reset.
        if prev_gray is None or prev_Twc is None or prev_corr_params is None:
            prev_gray = np.asarray(gray, dtype=np.uint8).copy()
            prev_Twc = np.asarray(Twc_now, dtype=np.float64).reshape(4, 4).copy()
            prev_corr_params = np.asarray(corr_params, dtype=np.float64).copy()
            poly_uv = np.asarray(uv_poly_vo, dtype=np.float32).reshape(-1, 2)
            poly_locked = False
            lock_used_uv = None
            last_inliers = 0
            last_n = 0
            last_reason = "warmup"
            _publish(fi=int(fi), ts=float(ts), reason="warmup")
            continue

        # Project edge world points for prev/cur.
        uv_edge_prev_vo, _z_prev, ok_prev = _project_world_to_uv_z(
            Pw=np.asarray(edge_world_pts_w, dtype=np.float64),
            Twc=np.asarray(prev_Twc, dtype=np.float64),
            intr=intr_obj,
        )
        uv_edge_cur_vo, z_cur, ok_cur = _project_world_to_uv_z(
            Pw=np.asarray(edge_world_pts_w, dtype=np.float64),
            Twc=np.asarray(Twc_now, dtype=np.float64),
            intr=intr_obj,
        )
        if uv_edge_prev_vo is None or uv_edge_cur_vo is None or z_cur is None or ok_prev is None or ok_cur is None:
            poly_locked = False
            lock_used_uv = None
            last_inliers = 0
            last_n = 0
            last_reason = "edge_proj_fail"
            poly_uv = np.asarray(uv_poly_vo, dtype=np.float32).reshape(-1, 2)
            _publish(fi=int(fi), ts=float(ts), reason="edge_proj_fail")
            prev_gray = np.asarray(gray, dtype=np.uint8).copy()
            prev_Twc = np.asarray(Twc_now, dtype=np.float64).reshape(4, 4).copy()
            prev_corr_params = np.asarray(corr_params, dtype=np.float64).copy()
            continue

        # Correction dt for decay.
        now_ts = float(ts)
        dt_corr = None
        try:
            if corr_last_ts is not None and np.isfinite(float(corr_last_ts)):
                dt_corr = float(now_ts) - float(corr_last_ts)
        except Exception:
            dt_corr = None
        if dt_corr is None or (not np.isfinite(float(dt_corr))) or float(dt_corr) <= 0.0:
            try:
                dt_corr = 1.0 / float(max(1.0, float(cap_cfg.get("fps", 30))))
            except Exception:
                dt_corr = 1.0 / 30.0
        corr_last_ts = float(now_ts)

        # Predict points in image using current correction (for LK initial flow).
        A_prev = _sim_affine_from_params(
            tx=float(prev_corr_params[0]), ty=float(prev_corr_params[1]), theta_rad=float(prev_corr_params[2]), log_s=float(prev_corr_params[3])
        )
        A_cur = _sim_affine_from_params(
            tx=float(corr_params[0]), ty=float(corr_params[1]), theta_rad=float(corr_params[2]), log_s=float(corr_params[3])
        )
        uv_edge_prev_pred = cv2.transform(np.asarray(uv_edge_prev_vo, dtype=np.float32).reshape(-1, 1, 2), np.asarray(A_prev, dtype=np.float64)).reshape(-1, 2)
        uv_edge_cur_pred = cv2.transform(np.asarray(uv_edge_cur_vo, dtype=np.float32).reshape(-1, 1, 2), np.asarray(A_cur, dtype=np.float64)).reshape(-1, 2)

        # Dynamic point count: maintain a working point budget and adapt based on per-frame gating success ratio.
        M = int(uv_edge_cur_vo.shape[0])
        # Per-edge-point plane gating state (kept aligned with the current edge pool).
        if edge_alive is None or edge_plane_bad is None or int(getattr(edge_alive, "size", 0)) != int(M):
            edge_alive = np.ones((int(M),), dtype=np.bool_)
            edge_plane_bad = np.zeros((int(M),), dtype=np.uint8)

        try:
            alive_idx = np.flatnonzero(np.asarray(edge_alive, dtype=bool).reshape(-1))
        except Exception:
            alive_idx = np.arange(int(M), dtype=np.int32)
        M_alive = int(alive_idx.size)
        pts_max_use = int(min(int(pts_max), int(M_alive)))
        if int(pts_max_use) > 0:
            pts_target_n = int(max(int(pts_min), min(int(pts_target_n), int(pts_max_use))))
        else:
            pts_target_n = int(pts_min)
        got = False
        used_uv = None
        used_n = 0
        used_inl = 0
        used_reason = ""
        used_keep_ratio = float("nan")
        attempted_n = 0

        h_img = int(gray.shape[0])
        w_img = int(gray.shape[1])

        # Plane in current camera frame (movement-stage gate: reject points that are not on the wall plane).
        plane_n_c = None
        plane_d_c = 0.0
        if bool(plane_cancel_enabled) and bool(pid_confirmed):
            plane_w_n = None
            plane_w_d = 0.0
            try:
                sk0 = str(sel_kind)
            except Exception:
                sk0 = "none"
            if sk0 == "hole" and bool(hole_plane_w_ok) and hole_plane_w_n is not None:
                plane_w_n = np.asarray(hole_plane_w_n, dtype=np.float64).reshape(3)
                plane_w_d = float(hole_plane_w_d)
            elif sk0 == "plane" and bool(plane_plane_w_ok) and plane_plane_w_n is not None:
                plane_w_n = np.asarray(plane_plane_w_n, dtype=np.float64).reshape(3)
                plane_w_d = float(plane_plane_w_d)
            if plane_w_n is not None:
                try:
                    plane_n_c, plane_d_c = _plane_cam_from_world(
                        n_w=np.asarray(plane_w_n, dtype=np.float64).reshape(3),
                        d_w=float(plane_w_d),
                        Twc_now=np.asarray(Twc_now, dtype=np.float64).reshape(4, 4),
                    )
                except Exception:
                    plane_n_c = None
                    plane_d_c = 0.0

        # One optional in-frame "escalation" retry when point success is very low.
        N_req = int(pts_target_n)
        max_attempts = 2
        for _attempt in range(int(max_attempts)):
            N = int(min(int(N_req), int(pts_max_use)))
            attempted_n = int(N)
            if int(N) < 4:
                break
            try:
                idx0 = np.linspace(0, int(max(0, int(M_alive - 1))), int(N)).astype(np.int32)
            except Exception:
                idx0 = np.arange(0, int(min(int(N), int(M_alive))), dtype=np.int32)
            idx0 = np.unique(idx0).astype(np.int32, copy=False)
            try:
                idx = np.asarray(alive_idx, dtype=np.int32).reshape(-1)[idx0]
            except Exception:
                idx = idx0
            idx = np.unique(np.asarray(idx, dtype=np.int32).reshape(-1)).astype(np.int32, copy=False)
            if int(idx.size) < 4:
                break
            N_eff = int(idx.size)

            ok_m = np.asarray(ok_prev, dtype=bool).reshape(-1)[idx] & np.asarray(ok_cur, dtype=bool).reshape(-1)[idx]
            if int(np.count_nonzero(ok_m)) < 4:
                used_n = 0
                used_inl = 0
                used_keep_ratio = 0.0
                used_reason = f"no_vis_n={int(N)}"
                if float(used_keep_ratio) < float(success_ratio_low) and int(N_req) < int(pts_max_use):
                    N_req = int(min(int(pts_max_use), int(N_req + pts_step)))
                    pts_target_n = int(N_req)
                    continue
                break
            ii = idx[ok_m]
            p0 = np.asarray(uv_edge_prev_pred, dtype=np.float32).reshape(-1, 2)[ii]
            p1_init = np.asarray(uv_edge_cur_pred, dtype=np.float32).reshape(-1, 2)[ii]
            src_vo = np.asarray(uv_edge_cur_vo, dtype=np.float32).reshape(-1, 2)[ii]
            z_pred = np.asarray(z_cur, dtype=np.float64).reshape(-1)[ii]

            if int(p0.shape[0]) < 4:
                used_n = 0
                used_inl = 0
                used_keep_ratio = 0.0
                used_reason = f"no_vis_n={int(N)}"
                if float(used_keep_ratio) < float(success_ratio_low) and int(N_req) < int(pts_max_use):
                    N_req = int(min(int(pts_max_use), int(N_req + pts_step)))
                    pts_target_n = int(N_req)
                    continue
                break
            pts0 = p0.reshape(-1, 1, 2).astype(np.float32, copy=False)
            try:
                if bool(lk_use_initial_flow):
                    pts1_guess = p1_init.reshape(-1, 1, 2).astype(np.float32, copy=False)
                    pts1, st1, _err1 = cv2.calcOpticalFlowPyrLK(
                        prev_gray, gray, pts0, pts1_guess, flags=cv2.OPTFLOW_USE_INITIAL_FLOW, **lk_params
                    )
                else:
                    pts1, st1, _err1 = cv2.calcOpticalFlowPyrLK(prev_gray, gray, pts0, None, flags=0, **lk_params)
            except Exception:
                pts1, st1 = None, None
            if pts1 is None or st1 is None:
                used_n = 0
                used_inl = 0
                used_keep_ratio = 0.0
                used_reason = f"lk_fail_n={int(N)}"
                if float(used_keep_ratio) < float(success_ratio_low) and int(N_req) < int(pts_max_use):
                    N_req = int(min(int(pts_max_use), int(N_req + pts_step)))
                    pts_target_n = int(N_req)
                    continue
                break
            try:
                if bool(lk_use_initial_flow):
                    pts0_back, st0, _err0 = cv2.calcOpticalFlowPyrLK(
                        gray, prev_gray, pts1, pts0, flags=cv2.OPTFLOW_USE_INITIAL_FLOW, **lk_params
                    )
                else:
                    pts0_back, st0, _err0 = cv2.calcOpticalFlowPyrLK(gray, prev_gray, pts1, None, flags=0, **lk_params)
            except Exception:
                pts0_back, st0 = None, None
            if pts0_back is None or st0 is None:
                used_n = 0
                used_inl = 0
                used_keep_ratio = 0.0
                used_reason = f"lk_bw_fail_n={int(N)}"
                if float(used_keep_ratio) < float(success_ratio_low) and int(N_req) < int(pts_max_use):
                    N_req = int(min(int(pts_max_use), int(N_req + pts_step)))
                    pts_target_n = int(N_req)
                    continue
                break

            st1 = np.asarray(st1, dtype=np.uint8).reshape(-1)
            st0 = np.asarray(st0, dtype=np.uint8).reshape(-1)
            p0_xy = np.asarray(pts0, dtype=np.float32).reshape(-1, 2)
            p1_xy = np.asarray(pts1, dtype=np.float32).reshape(-1, 2)
            p0b_xy = np.asarray(pts0_back, dtype=np.float32).reshape(-1, 2)

            inb = (p1_xy[:, 0] >= 0.0) & (p1_xy[:, 0] < float(w_img)) & (p1_xy[:, 1] >= 0.0) & (p1_xy[:, 1] < float(h_img))
            fb = np.linalg.norm((p0_xy - p0b_xy).astype(np.float32, copy=False), axis=1)
            keep = (st1 == 1) & (st0 == 1) & inb & (fb <= float(fb_px))
            if float(pred_gate_px) > 0.0:
                pred_err = np.linalg.norm((p1_xy - np.asarray(p1_init, dtype=np.float32).reshape(-1, 2)).astype(np.float32, copy=False), axis=1)
                keep = keep & (pred_err <= float(pred_gate_px))
            if int(np.count_nonzero(keep)) < 4:
                used_n = int(np.count_nonzero(keep))
                used_inl = 0
                used_keep_ratio = float(used_n) / float(max(1, int(N_eff)))
                used_reason = f"fb_n={int(N)}"
                if float(used_keep_ratio) < float(success_ratio_low) and int(N_req) < int(pts_max_use):
                    N_req = int(min(int(pts_max_use), int(N_req + pts_step)))
                    pts_target_n = int(N_req)
                    continue
                break

            # ZNCC gate (batch: Numba fast-path avoids per-point slicing/allocations)
            if bool(perf_log):
                _t_zncc = float(time.perf_counter())
            if _HAVE_NUMBA:
                try:
                    if zncc_out_u8 is None or int(getattr(zncc_out_u8, "size", 0)) != int(N_eff):
                        zncc_out_u8 = np.zeros((int(N_eff),), dtype=np.uint8)
                    _nb_zncc_gate_u8(
                        prev_gray,
                        gray,
                        np.asarray(p0_xy, dtype=np.float32).reshape(-1, 2),
                        np.asarray(p1_xy, dtype=np.float32).reshape(-1, 2),
                        np.asarray(keep, dtype=bool).reshape(-1).view(np.uint8),
                        int(ncc_r_px),
                        float(ncc_min),
                        zncc_out_u8,
                    )
                    keep &= (np.asarray(zncc_out_u8, dtype=np.uint8).reshape(-1) != 0)
                except Exception:
                    keep2 = np.zeros_like(keep, dtype=bool)
                    for k in np.flatnonzero(keep).tolist():
                        cc = _zncc_patch(
                            img0=prev_gray,
                            img1=gray,
                            u0=float(p0_xy[k, 0]),
                            v0=float(p0_xy[k, 1]),
                            u1=float(p1_xy[k, 0]),
                            v1=float(p1_xy[k, 1]),
                            r=int(ncc_r_px),
                        )
                        if np.isfinite(cc) and float(cc) >= float(ncc_min):
                            keep2[int(k)] = True
                    keep = keep2
            else:
                keep2 = np.zeros_like(keep, dtype=bool)
                for k in np.flatnonzero(keep).tolist():
                    cc = _zncc_patch(
                        img0=prev_gray,
                        img1=gray,
                        u0=float(p0_xy[k, 0]),
                        v0=float(p0_xy[k, 1]),
                        u1=float(p1_xy[k, 0]),
                        v1=float(p1_xy[k, 1]),
                        r=int(ncc_r_px),
                    )
                    if np.isfinite(cc) and float(cc) >= float(ncc_min):
                        keep2[int(k)] = True
                keep = keep2
            if bool(perf_log):
                _perf_sum_zncc_ms += 1000.0 * float(time.perf_counter() - float(_t_zncc))
                _perf_zncc_n += 1
            if int(np.count_nonzero(keep)) < 4:
                used_n = int(np.count_nonzero(keep))
                used_inl = 0
                used_keep_ratio = float(used_n) / float(max(1, int(N_eff)))
                used_reason = f"zncc_n={int(N)}"
                if float(used_keep_ratio) < float(success_ratio_low) and int(N_req) < int(pts_max_use):
                    N_req = int(min(int(pts_max_use), int(N_req + pts_step)))
                    pts_target_n = int(N_req)
                    continue
                break

            # Depth gate (optional)
            if bool(depth_gate_enabled) and depth_raw is not None and float(depth_units) > 0.0:
                keep3 = np.zeros_like(keep, dtype=bool)
                for k in np.flatnonzero(keep).tolist():
                    u = float(p1_xy[k, 0])
                    v = float(p1_xy[k, 1])
                    med_u16, _mad, nn = _sample_depth_stats_u16(
                        np.asarray(depth_raw, dtype=np.uint16), int(round(u)), int(round(v)), int(depth_r_px)
                    )
                    if int(med_u16) <= 0 or int(nn) <= 0:
                        continue
                    z_obs = float(med_u16) * float(depth_units)
                    z_p = float(z_pred[k])
                    if (not np.isfinite(z_obs)) or (not np.isfinite(z_p)) or float(z_p) <= 1e-6:
                        continue
                    thr = float(max(float(depth_prior_abs), float(depth_prior_rel) * float(z_p)))
                    if float(abs(float(z_obs) - float(z_p))) > float(thr):
                        continue
                    # Movement-stage plane gate: reject points whose observed depth does not lie on the wall plane.
                    if plane_n_c is not None and bool(plane_cancel_enabled):
                        try:
                            g_idx = int(ii[int(k)])
                        except Exception:
                            g_idx = -1
                        try:
                            fx_i = float(intr_obj.fx) if intr_obj is not None else 0.0
                            fy_i = float(intr_obj.fy) if intr_obj is not None else 0.0
                            cx_i = float(intr_obj.cx) if intr_obj is not None else 0.0
                            cy_i = float(intr_obj.cy) if intr_obj is not None else 0.0
                        except Exception:
                            fx_i, fy_i, cx_i, cy_i = 0.0, 0.0, 0.0, 0.0
                        if float(fx_i) > 1e-9 and float(fy_i) > 1e-9 and int(g_idx) >= 0:
                            xr = (float(u) - float(cx_i)) * float(z_obs) / float(fx_i)
                            yr = (float(v) - float(cy_i)) * float(z_obs) / float(fy_i)
                            dist = abs(
                                float(plane_n_c[0]) * float(xr)
                                + float(plane_n_c[1]) * float(yr)
                                + float(plane_n_c[2]) * float(z_obs)
                                + float(plane_d_c)
                            )
                            if (not np.isfinite(dist)) or float(dist) > float(plane_dist_thr_m):
                                # Mark as persistently bad when it consistently violates the plane (prevents slow drift).
                                try:
                                    if edge_plane_bad is not None and edge_alive is not None and int(g_idx) < int(edge_plane_bad.size):
                                        edge_plane_bad[int(g_idx)] = np.uint8(min(255, int(edge_plane_bad[int(g_idx)]) + 1))
                                        if int(edge_plane_bad[int(g_idx)]) >= int(plane_bad_count):
                                            edge_alive[int(g_idx)] = False
                                except Exception:
                                    pass
                                continue
                            else:
                                try:
                                    if edge_plane_bad is not None and int(g_idx) < int(edge_plane_bad.size) and int(edge_plane_bad[int(g_idx)]) > 0:
                                        edge_plane_bad[int(g_idx)] = np.uint8(max(0, int(edge_plane_bad[int(g_idx)]) - 1))
                                except Exception:
                                    pass
                    keep3[int(k)] = True
                keep = keep3

            used_n = int(np.count_nonzero(keep))
            used_inl = 0
            used_keep_ratio = float(used_n) / float(max(1, int(N_eff)))
            if int(used_n) < int(min_fit_inliers):
                used_reason = f"few_pass_n={int(N)}"
                if float(used_keep_ratio) < float(success_ratio_low) and int(N_req) < int(pts_max_use):
                    N_req = int(min(int(pts_max_use), int(N_req + pts_step)))
                    pts_target_n = int(N_req)
                    continue
                break

            src = np.asarray(src_vo, dtype=np.float32).reshape(-1, 2)[keep]
            dst = np.asarray(p1_xy, dtype=np.float32).reshape(-1, 2)[keep]
            used_n = int(src.shape[0])
            # For GUI debugging: publish the LK points that passed all gates even if RANSAC later fails.
            # (GUI shows them red when not locked, yellow when locked.)
            try:
                used_uv = np.asarray(dst, dtype=np.float32).reshape(-1, 2) if bool(show_pts) else None
            except Exception:
                used_uv = None
            used_keep_ratio = float(used_n) / float(max(1, int(N_eff)))
            if int(used_n) < int(min_fit_inliers):
                used_inl = 0
                used_reason = f"few_pass_n={int(N)}"
                if float(used_keep_ratio) < float(success_ratio_low) and int(N_req) < int(pts_max_use):
                    N_req = int(min(int(pts_max_use), int(N_req + pts_step)))
                    pts_target_n = int(N_req)
                    continue
                break

            try:
                A_tmp, inl = cv2.estimateAffinePartial2D(
                    src,
                    dst,
                    method=cv2.RANSAC,
                    ransacReprojThreshold=float(ransac_px),
                    maxIters=2000,
                    confidence=0.99,
                    refineIters=0,
                )
            except Exception:
                A_tmp, inl = None, None
            if A_tmp is None or inl is None:
                used_inl = 0
                used_reason = f"ransac_fail_n={int(N)}"
                if float(used_keep_ratio) < float(success_ratio_low) and int(N_req) < int(pts_max_use):
                    N_req = int(min(int(pts_max_use), int(N_req + pts_step)))
                    pts_target_n = int(N_req)
                    continue
                break
            A_sim = _affine_to_similarity(A=np.asarray(A_tmp, dtype=np.float64))
            A_tmp = np.asarray(A_sim if A_sim is not None else A_tmp, dtype=np.float64).reshape(2, 3)
            inl_m = np.asarray(inl, dtype=bool).reshape(-1)
            n_inl = int(np.count_nonzero(inl_m))
            r_inl = float(n_inl) / float(max(1, int(inl_m.size)))
            used_inl = int(n_inl)
            keep_ok = (float(accept_keep_ratio_min) <= 0.0) or (np.isfinite(float(used_keep_ratio)) and float(used_keep_ratio) >= float(accept_keep_ratio_min))
            if int(n_inl) >= int(min_fit_inliers) and float(r_inl) >= float(min_inlier_ratio) and bool(keep_ok):
                got = True
                used_uv = np.asarray(dst, dtype=np.float32).reshape(-1, 2)[inl_m] if bool(show_pts) else None
                used_reason = f"ok_n={int(N)}"

                params = _sim_params_from_affine(A=np.asarray(A_tmp, dtype=np.float64))
                if params is not None:
                    tx_m, ty_m, th_m, ls_m = params
                    a = float(corr_ema_alpha)
                    corr_params = np.asarray(corr_params, dtype=np.float64).reshape(4)
                    corr_params[0] = (1.0 - float(a)) * float(corr_params[0]) + float(a) * float(tx_m)
                    corr_params[1] = (1.0 - float(a)) * float(corr_params[1]) + float(a) * float(ty_m)
                    dth = float(_wrap_pi(float(th_m) - float(corr_params[2])))
                    corr_params[2] = float(_wrap_pi(float(corr_params[2]) + float(a) * float(dth)))
                    corr_params[3] = (1.0 - float(a)) * float(corr_params[3]) + float(a) * float(ls_m)

                # Adapt point budget:
                # - increase when too few points pass gates (< success_ratio_low)
                # - decrease when very healthy (>= success_ratio_high and locked), to save CPU
                if float(used_keep_ratio) < float(success_ratio_low) and int(pts_target_n) < int(pts_max_use):
                    pts_target_n = int(min(int(pts_max_use), int(pts_target_n + pts_step)))
                elif float(used_keep_ratio) >= float(success_ratio_high) and int(pts_target_n) > int(pts_min):
                    pts_target_n = int(max(int(pts_min), int(pts_target_n - pts_step)))
                break

            if int(n_inl) >= int(min_fit_inliers) and float(r_inl) >= float(min_inlier_ratio) and (not bool(keep_ok)):
                used_reason = f"keep_low_n={int(N)}"
                if float(used_keep_ratio) < float(success_ratio_low) and int(N_req) < int(pts_max_use):
                    N_req = int(min(int(pts_max_use), int(N_req + pts_step)))
                    pts_target_n = int(N_req)
                    continue
                break

            # Rejected: increase budget if point success is poor; optionally retry once in-frame.
            used_reason = f"rej_n={int(N)}"
            if float(used_keep_ratio) < float(success_ratio_low) and int(N_req) < int(pts_max_use):
                N_req = int(min(int(pts_max_use), int(N_req + pts_step)))
                pts_target_n = int(N_req)
                continue
            break

        last_try_n = int(attempted_n)
        last_keep_ratio = float(used_keep_ratio) if np.isfinite(float(used_keep_ratio)) else float("nan")

        if bool(got):
            poly_locked = True
            lock_used_uv = np.asarray(used_uv, dtype=np.float32).reshape(-1, 2) if used_uv is not None else None
            last_inliers = int(used_inl)
            last_n = int(used_n)
            last_reason = str(used_reason)
        else:
            poly_locked = False
            lock_used_uv = np.asarray(used_uv, dtype=np.float32).reshape(-1, 2) if used_uv is not None else None
            last_inliers = 0
            last_n = int(used_n)
            last_reason = str(used_reason) if str(used_reason) else "no_lock"
            # Decay-to-identity when lock fails.
            try:
                hl = float(corr_decay_half_life_s)
                if np.isfinite(hl) and float(hl) > 0.0:
                    decay = float(0.5) ** (float(dt_corr) / float(max(1e-6, hl)))
                    corr_params = np.asarray(corr_params, dtype=np.float64).reshape(4)
                    corr_params *= float(decay)
                    corr_params[2] = float(_wrap_pi(float(corr_params[2])))
            except Exception:
                pass

        # Final polygon = VO-projection transformed by current correction estimate.
        A_final = _sim_affine_from_params(
            tx=float(corr_params[0]), ty=float(corr_params[1]), theta_rad=float(corr_params[2]), log_s=float(corr_params[3])
        )
        poly_uv_out = cv2.transform(
            np.asarray(uv_poly_vo, dtype=np.float32).reshape(-1, 1, 2), np.asarray(A_final, dtype=np.float64)
        ).reshape(-1, 2)
        poly_uv = np.asarray(poly_uv_out, dtype=np.float32).reshape(-1, 2)
        poly_bbox2 = _bbox_from_poly_uv(poly_uv=np.asarray(poly_uv, dtype=np.float32), w=int(w_img), h=int(h_img), margin_px=0, min_size_px=2)
        if poly_bbox2 is not None:
            poly_bbox = tuple(poly_bbox2)

        # Propagate the refinement polygon (pre-confirm) using the tracked polygon motion (similarity).
        try:
            if bool(poly_active) and str(sel_kind) == "hole" and (not bool(pid_confirmed)) and ref_poly_uv is not None and trk_poly_prev is not None:
                src = np.asarray(trk_poly_prev, dtype=np.float32).reshape(-1, 2)
                dst = np.asarray(poly_uv, dtype=np.float32).reshape(-1, 2)
                if int(src.shape[0]) == int(dst.shape[0]) and int(src.shape[0]) >= 3:
                    A, _inl = cv2.estimateAffinePartial2D(src, dst, method=0)
                    if A is not None and np.asarray(A).shape == (2, 3):
                        ref_poly_uv = cv2.transform(np.asarray(ref_poly_uv, dtype=np.float32).reshape(-1, 1, 2), np.asarray(A, dtype=np.float64)).reshape(-1, 2)
                trk_poly_prev = np.asarray(poly_uv, dtype=np.float32).reshape(-1, 2).copy()
        except Exception:
            try:
                trk_poly_prev = np.asarray(poly_uv, dtype=np.float32).reshape(-1, 2).copy()
            except Exception:
                pass

        did_auto_confirm_publish = False
        # Auto stage-2 transition (optional): confirm when ROI bbox fills a large portion of the frame.
        try:
            if (
                bool(poly_active)
                and (not bool(pid_confirmed))
                and str(stage2_transition_mode) in ("roi_fill", "roi", "auto_roi_fill")
                and int(w_img) > 0
                and int(h_img) > 0
            ):
                ok_gate = True
                try:
                    if bool(stage2_require_ok):
                        if str(sel_kind) == "hole":
                            ok_gate = bool(hole_ok_pid)
                        elif str(sel_kind) == "plane":
                            ok_gate = bool(plane_ok_pid)
                except Exception:
                    ok_gate = True

                if bool(ok_gate):
                    poly_uv_chk = None
                    try:
                        if str(sel_kind) == "hole" and ref_poly_uv is not None and int(np.asarray(ref_poly_uv).shape[0]) >= 3:
                            poly_uv_chk = np.asarray(ref_poly_uv, dtype=np.float32).reshape(-1, 2)
                        elif poly_uv is not None and int(np.asarray(poly_uv).shape[0]) >= 3:
                            poly_uv_chk = np.asarray(poly_uv, dtype=np.float32).reshape(-1, 2)
                    except Exception:
                        poly_uv_chk = None

                    if poly_uv_chk is not None:
                        bb = _bbox_from_poly_uv(
                            poly_uv=np.asarray(poly_uv_chk, dtype=np.float32),
                            w=int(w_img),
                            h=int(h_img),
                            margin_px=0,
                            min_size_px=2,
                        )
                        if bb is not None:
                            try:
                                bw = int(bb[2])
                                bh = int(bb[3])
                            except Exception:
                                bw, bh = 0, 0
                            fill = float(max(0, int(bw))) * float(max(0, int(bh))) / float(max(1, int(w_img) * int(h_img)))
                            if np.isfinite(float(fill)) and float(fill) >= float(stage2_roi_fill_frac):
                                now_wall = float(time.time())
                                if float(now_wall - float(stage2_auto_last_t)) >= float(stage2_cooldown_s):
                                    stage2_auto_last_t = float(now_wall)
                                    if bool(dbg_log):
                                        try:
                                            print(
                                                f"[poly_vo_lk] stage2_auto_confirm kind={str(sel_kind)} fill={float(fill):.2f} thr={float(stage2_roi_fill_frac):.2f}",
                                                flush=True,
                                            )
                                        except Exception:
                                            pass
                                    if str(sel_kind) == "hole" and bool(hole_runtime_enabled):
                                        _handle_confirm_hole({"fi": int(fi), "ts": float(ts)})
                                        did_auto_confirm_publish = True
                                    elif str(sel_kind) == "plane" and bool(plane_runtime_enabled):
                                        _handle_confirm_plane({"fi": int(fi), "ts": float(ts)})
                                        did_auto_confirm_publish = True
        except Exception:
            did_auto_confirm_publish = False

        _fps_n += 1
        now = float(time.time())
        dt_fps = float(now - float(_fps_t0))
        if float(dt_fps) >= 0.5:
            fps = float(_fps_n) / float(max(1e-6, dt_fps))
            _fps_n = 0
            _fps_t0 = float(now)

        if bool(debug_log) and bool(dbg_log) and float(now - float(_last_dbg_t)) >= 1.0:
            _last_dbg_t = float(now)
            try:
                try:
                    alive_n = int(np.count_nonzero(np.asarray(edge_alive, dtype=bool).reshape(-1))) if edge_alive is not None else int(0)
                except Exception:
                    alive_n = int(0)
                print(
                    f"[poly_vo_lk] fi={int(fi)} gap={int(gap)} lock={int(bool(poly_locked))} try={int(last_try_n)} keep={float(last_keep_ratio):.2f} "
                    f"n={int(last_n)} inl={int(last_inliers)} alive={int(alive_n)}/{int(M)} reason={str(last_reason)}",
                    flush=True,
                )
            except Exception:
                pass

            if bool(perf_log) and float(now - float(_perf_t0)) >= float(1.0 / float(max(0.1, perf_rate_hz))):
                _perf_t0 = float(now)
                try:
                    avg_total = float(_perf_sum_total_ms) / float(max(1, int(_perf_n)))
                except Exception:
                    avg_total = float("nan")
                try:
                    avg_zncc = float(_perf_sum_zncc_ms) / float(max(1, int(_perf_zncc_n)))
                except Exception:
                    avg_zncc = float("nan")
                try:
                    avg_ref = float(_perf_sum_refine_apply_ms) / float(max(1, int(_perf_ref_n)))
                except Exception:
                    avg_ref = float("nan")
                try:
                    det_ms = float(_perf_last_refine_det_ms) if np.isfinite(float(_perf_last_refine_det_ms)) else float("nan")
                except Exception:
                    det_ms = float("nan")
                try:
                    print(
                        f"[poly_perf] total_ms={float(avg_total):.1f} zncc_ms={float(avg_zncc):.1f} "
                        f"ref_apply_ms={float(avg_ref):.1f} ref_det_ms={float(det_ms):.1f}",
                        flush=True,
                    )
                except Exception:
                    pass
                _perf_n = 0
                _perf_sum_total_ms = 0.0
                _perf_zncc_n = 0
                _perf_sum_zncc_ms = 0.0
                _perf_ref_n = 0
                _perf_sum_refine_apply_ms = 0.0

        # Movement-stage plane reseed: if too many LK points get disabled as off-plane, regenerate a fresh edge-point pool
        # on the fitted wall plane. This is designed to handle cases where background features "through" the opening were
        # accidentally used for locking while approaching the hole.
        try:
            if (
                bool(plane_reseed_enabled)
                and bool(pid_confirmed)
                and poly_uv is not None
                and edge_alive is not None
            ):
                plane_w_n = None
                plane_w_d = 0.0
                off_px = 0.0
                try:
                    sk0 = str(sel_kind)
                except Exception:
                    sk0 = "none"
                if sk0 == "hole" and bool(hole_plane_w_ok) and hole_plane_w_n is not None:
                    plane_w_n = np.asarray(hole_plane_w_n, dtype=np.float64).reshape(3)
                    plane_w_d = float(hole_plane_w_d)
                    off_px = float(plane_reseed_edge_offset_px)
                elif sk0 == "plane" and bool(plane_plane_w_ok) and plane_plane_w_n is not None:
                    plane_w_n = np.asarray(plane_plane_w_n, dtype=np.float64).reshape(3)
                    plane_w_d = float(plane_plane_w_d)
                    off_px = 0.0

                if plane_w_n is not None:
                    alive_n = int(np.count_nonzero(np.asarray(edge_alive, dtype=bool).reshape(-1)))
                    if int(alive_n) < int(max(4, int(plane_reseed_min_alive))):
                        now_wall = float(time.time())
                        if float(now_wall - float(_plane_reseed_last_t)) >= float(max(0.0, float(plane_reseed_cooldown_s))):
                            new_edge = _reseed_edge_points_on_plane(
                                poly_uv_img=np.asarray(poly_uv, dtype=np.float32).reshape(-1, 2),
                                Twc_now=Twc_now,
                                plane_w_n=np.asarray(plane_w_n, dtype=np.float64).reshape(3),
                                plane_w_d=float(plane_w_d),
                                edge_offset_px=float(off_px),
                            )
                            if new_edge is not None and int(np.asarray(new_edge).shape[0]) >= int(min_fit_inliers):
                                edge_world_pts_w = np.asarray(new_edge, dtype=np.float64).reshape(-1, 3)
                                edge_alive = np.ones((int(edge_world_pts_w.shape[0]),), dtype=np.bool_)
                                edge_plane_bad = np.zeros((int(edge_world_pts_w.shape[0]),), dtype=np.uint8)
                                pts_target_n = int(pts_min)
                                _plane_reseed_last_t = float(now_wall)
                                if bool(dbg_log):
                                    try:
                                        print(
                                            f"[poly_vo_lk] plane_reseed kind={str(sk0)} fi={int(fi)} alive={int(alive_n)} -> n={int(edge_world_pts_w.shape[0])} off_px={float(off_px):.1f}",
                                            flush=True,
                                        )
                                    except Exception:
                                        pass
        except Exception:
            pass

        if bool(perf_log) and float(_t_frame0) > 0.0:
            _perf_sum_total_ms += 1000.0 * float(time.perf_counter() - float(_t_frame0))
            _perf_n += 1

        # VO-agnostic feature acquisition (ORB/lines). This does not change tracking behavior by itself; it is
        # intended as a side-channel for future reseed/recovery strategies and debugging.
        try:
            if feature_acq is not None and bool(poly_active):
                stage_ok = (not bool(pid_confirmed) and bool(feature_acq_stage1)) or (bool(pid_confirmed) and bool(feature_acq_stage2))
                if bool(stage_ok) and float(feature_acq_rate_hz) > 0.0:
                    now_wall = float(time.time())
                    if float(now_wall - float(feat_last_t)) >= float(1.0 / float(max(1e-6, float(feature_acq_rate_hz)))):
                        poly_uv_feat = None
                        try:
                            if (not bool(pid_confirmed)) and str(sel_kind) == "hole" and ref_poly_uv is not None and int(np.asarray(ref_poly_uv).shape[0]) >= 3:
                                poly_uv_feat = np.asarray(ref_poly_uv, dtype=np.float32).reshape(-1, 2)
                            elif poly_uv is not None and int(np.asarray(poly_uv).shape[0]) >= 3:
                                poly_uv_feat = np.asarray(poly_uv, dtype=np.float32).reshape(-1, 2)
                        except Exception:
                            poly_uv_feat = None

                        if poly_uv_feat is not None:
                            feat_last = feature_acq.acquire(gray=np.asarray(gray, dtype=np.uint8), poly_uv=np.asarray(poly_uv_feat, dtype=np.float32).reshape(-1, 2))
                            feat_last_t = float(now_wall)

                            try:
                                feat_orb_n = int(0 if feat_last.orb_uv is None else np.asarray(feat_last.orb_uv).shape[0])
                            except Exception:
                                feat_orb_n = 0
                            try:
                                feat_lines_n = int(0 if feat_last.lines_xyxy is None else np.asarray(feat_last.lines_xyxy).shape[0])
                            except Exception:
                                feat_lines_n = 0

                            feat_orb_A = None
                            feat_orb_ok = False
                            feat_orb_inl_n = 0
                            try:
                                if feat_anchor is not None:
                                    est = feature_acq.estimate_similarity_from_orb(anchor=feat_anchor, current=feat_last)
                                    if est is not None:
                                        A0, inl0 = est
                                        feat_orb_A = np.asarray(A0, dtype=np.float64).reshape(2, 3)
                                        feat_orb_inl_n = int(np.count_nonzero(np.asarray(inl0, dtype=bool).reshape(-1)))
                                        feat_orb_ok = True
                            except Exception:
                                feat_orb_A = None
                                feat_orb_ok = False
                                feat_orb_inl_n = 0

                            feat_orb_uv_pub = None
                            feat_lines_pub = None
                            try:
                                if bool(feature_acq_publish_orb) and feat_last.orb_uv is not None:
                                    uv0 = np.asarray(feat_last.orb_uv, dtype=np.float32).reshape(-1, 2)
                                    nmax = int(max(0, int(feature_acq_publish_max_orb)))
                                    if int(nmax) > 0 and int(uv0.shape[0]) > int(nmax):
                                        uv0 = np.asarray(uv0[: int(nmax)], dtype=np.float32).reshape(-1, 2)
                                    feat_orb_uv_pub = uv0
                            except Exception:
                                feat_orb_uv_pub = None
                            try:
                                if bool(feature_acq_publish_lines) and feat_last.lines_xyxy is not None:
                                    L0 = np.asarray(feat_last.lines_xyxy, dtype=np.float32).reshape(-1, 4)
                                    nmax = int(max(0, int(feature_acq_publish_max_lines)))
                                    if int(nmax) > 0 and int(L0.shape[0]) > int(nmax):
                                        L0 = np.asarray(L0[: int(nmax)], dtype=np.float32).reshape(-1, 4)
                                    feat_lines_pub = L0
                            except Exception:
                                feat_lines_pub = None
        except Exception:
            pass

        # Operator debug: distance to the cached plane at polygon center.
        # - hole: annulus plane (stage1+stage2)
        # - plane: selected plane patch
        try:
            if (
                bool(poly_active)
                and str(sel_kind) == "hole"
                and bool(hole_plane_w_ok)
                and hole_plane_w_n is not None
                and intr_obj is not None
                and Twc_now is not None
            ):
                poly_uv_c = None
                try:
                    if (not bool(pid_confirmed)) and ref_poly_uv is not None and int(np.asarray(ref_poly_uv).shape[0]) >= 3:
                        poly_uv_c = np.asarray(ref_poly_uv, dtype=np.float32).reshape(-1, 2)
                    elif poly_uv is not None and int(np.asarray(poly_uv).shape[0]) >= 3:
                        poly_uv_c = np.asarray(poly_uv, dtype=np.float32).reshape(-1, 2)
                except Exception:
                    poly_uv_c = None
                if poly_uv_c is not None:
                    try:
                        poly_i = np.asarray(poly_uv_c, dtype=np.int32).reshape(-1, 1, 2)
                        m = cv2.moments(poly_i)
                        if abs(float(m.get("m00", 0.0))) > 1e-6:
                            u_c = float(m["m10"]) / float(m["m00"])
                            v_c = float(m["m01"]) / float(m["m00"])
                        else:
                            u_c = float(np.mean(np.asarray(poly_uv_c, dtype=np.float64)[:, 0]))
                            v_c = float(np.mean(np.asarray(poly_uv_c, dtype=np.float64)[:, 1]))
                    except Exception:
                        u_c = float(np.mean(np.asarray(poly_uv_c, dtype=np.float64)[:, 0]))
                        v_c = float(np.mean(np.asarray(poly_uv_c, dtype=np.float64)[:, 1]))

                    n_c, d_c = _plane_cam_from_world(
                        n_w=np.asarray(hole_plane_w_n, dtype=np.float64).reshape(3),
                        d_w=float(hole_plane_w_d),
                        Twc_now=np.asarray(Twc_now, dtype=np.float64).reshape(4, 4),
                    )

                    fx = float(intr_obj.fx)
                    fy = float(intr_obj.fy)
                    cx0 = float(intr_obj.cx)
                    cy0 = float(intr_obj.cy)
                    if float(fx) > 0.0 and float(fy) > 0.0:
                        x = (float(u_c) - float(cx0)) / float(fx)
                        y = (float(v_c) - float(cy0)) / float(fy)
                        denom = float(n_c[0]) * float(x) + float(n_c[1]) * float(y) + float(n_c[2])
                        if np.isfinite(float(denom)) and abs(float(denom)) > 1e-9:
                            z_plane = -float(d_c) / float(denom)
                            if np.isfinite(float(z_plane)) and float(z_plane) > 0.0:
                                hole_plane_center_range_m = float(z_plane) * float(np.sqrt(1.0 + float(x) * float(x) + float(y) * float(y)))
        except Exception:
            hole_plane_center_range_m = float("nan")

        try:
            if (
                bool(poly_active)
                and str(sel_kind) == "plane"
                and bool(plane_plane_w_ok)
                and plane_plane_w_n is not None
                and intr_obj is not None
                and Twc_now is not None
                and poly_uv is not None
                and int(np.asarray(poly_uv).shape[0]) >= 3
            ):
                poly_uv_c = np.asarray(poly_uv, dtype=np.float32).reshape(-1, 2)
                try:
                    poly_i = np.asarray(poly_uv_c, dtype=np.int32).reshape(-1, 1, 2)
                    m = cv2.moments(poly_i)
                    if abs(float(m.get("m00", 0.0))) > 1e-6:
                        u_c = float(m["m10"]) / float(m["m00"])
                        v_c = float(m["m01"]) / float(m["m00"])
                    else:
                        u_c = float(np.mean(np.asarray(poly_uv_c, dtype=np.float64)[:, 0]))
                        v_c = float(np.mean(np.asarray(poly_uv_c, dtype=np.float64)[:, 1]))
                except Exception:
                    u_c = float(np.mean(np.asarray(poly_uv_c, dtype=np.float64)[:, 0]))
                    v_c = float(np.mean(np.asarray(poly_uv_c, dtype=np.float64)[:, 1]))

                n_c, d_c = _plane_cam_from_world(
                    n_w=np.asarray(plane_plane_w_n, dtype=np.float64).reshape(3),
                    d_w=float(plane_plane_w_d),
                    Twc_now=np.asarray(Twc_now, dtype=np.float64).reshape(4, 4),
                )

                fx = float(intr_obj.fx)
                fy = float(intr_obj.fy)
                cx0 = float(intr_obj.cx)
                cy0 = float(intr_obj.cy)
                if float(fx) > 0.0 and float(fy) > 0.0:
                    x = (float(u_c) - float(cx0)) / float(fx)
                    y = (float(v_c) - float(cy0)) / float(fy)
                    denom = float(n_c[0]) * float(x) + float(n_c[1]) * float(y) + float(n_c[2])
                    if np.isfinite(float(denom)) and abs(float(denom)) > 1e-9:
                        z_plane = -float(d_c) / float(denom)
                        if np.isfinite(float(z_plane)) and float(z_plane) > 0.0:
                            plane_plane_center_range_m = float(z_plane) * float(np.sqrt(1.0 + float(x) * float(x) + float(y) * float(y)))
        except Exception:
            plane_plane_center_range_m = float("nan")

        if not bool(did_auto_confirm_publish):
            _publish(fi=int(fi), ts=float(ts), reason="tick")

        # Update prev state.
        prev_gray = np.asarray(gray, dtype=np.uint8).copy()
        prev_Twc = np.asarray(Twc_now, dtype=np.float64).reshape(4, 4).copy()
        prev_corr_params = np.asarray(corr_params, dtype=np.float64).copy()
