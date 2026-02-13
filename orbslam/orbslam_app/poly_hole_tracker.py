from __future__ import annotations

"""
Polygon / "hole" tracking subsystem designed for headless integration.

Key constraint: all VO is performed by ORB-SLAM3; this module only consumes:
  - Per-frame pose (Twc: camera->world)
  - Aligned depth (meters)
  - Pinhole intrinsics
"""

import math
import time
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class PinholeIntrinsics:
    w: int
    h: int
    fx: float
    fy: float
    cx: float
    cy: float

    @staticmethod
    def from_any(obj: Any) -> Optional["PinholeIntrinsics"]:
        if obj is None:
            return None

        if isinstance(obj, dict):
            try:
                w = int(obj.get("w", obj.get("width", 0)) or 0)
                h = int(obj.get("h", obj.get("height", 0)) or 0)
                fx = float(obj.get("fx", 0.0) or 0.0)
                fy = float(obj.get("fy", 0.0) or 0.0)
                cx = float(obj.get("cx", obj.get("ppx", 0.0)) or 0.0)
                cy = float(obj.get("cy", obj.get("ppy", 0.0)) or 0.0)
                if w > 0 and h > 0 and fx > 0.0 and fy > 0.0:
                    return PinholeIntrinsics(w=w, h=h, fx=fx, fy=fy, cx=cx, cy=cy)
            except Exception:
                return None

        # RealSense intrinsics-like object.
        try:
            w = int(getattr(obj, "width"))
            h = int(getattr(obj, "height"))
            fx = float(getattr(obj, "fx"))
            fy = float(getattr(obj, "fy"))
            # RealSense uses ppx/ppy (don't use eager-evaluated getattr defaults).
            cx0 = getattr(obj, "ppx", None)
            if cx0 is None:
                cx0 = getattr(obj, "cx", None)
            cy0 = getattr(obj, "ppy", None)
            if cy0 is None:
                cy0 = getattr(obj, "cy", None)
            if cx0 is None or cy0 is None:
                return None
            cx = float(cx0)
            cy = float(cy0)
            if w > 0 and h > 0 and fx > 0.0 and fy > 0.0:
                return PinholeIntrinsics(w=w, h=h, fx=fx, fy=fy, cx=cx, cy=cy)
        except Exception:
            return None


@dataclass(frozen=True)
class PolyHoleSettings:
    enabled: bool = True

    # Stage-0 hover preview detector (depth-based).
    hole_preview_enabled: bool = True
    hole_preview_rate_hz: float = 10.0
    hole_preview_hold_s: float = 0.60

    # Runtime toggle (cmd: hole_enable).
    hole_runtime_enabled_default: bool = True

    # Detection window around cursor.
    hole_window_wh: Tuple[int, int] = (0, 0)
    hole_min_area_px: float = 25.0
    hole_unknown_as_full: bool = True
    hole_seed_snap_px: int = 32
    hole_depth_r_px: int = 8

    # Plane-depth estimate around cursor (ring median).
    search_max_depth_m: float = 15.0
    plane_ring_r_in_px: int = 14
    plane_ring_r_out_px: int = 150
    plane_ring_step_px: int = 3

    # Hole classification behind plane.
    hole_margin_abs_m: float = 0.06
    hole_margin_rel: float = 0.02
    hole_margin_rms_k: float = 3.0

    # Polygon fit params.
    canny_lo: int = 60
    canny_hi: int = 140
    dilate_edges: bool = True
    poly_concavity_max: float = 0.15
    poly_approx_eps_frac: float = 0.03
    poly_max_vertices: int = 18

    # Stage-2 gate / auto-stop.
    hole_pid_min_radius_m: float = 0.55
    hole_done_enabled: bool = False
    hole_done_fill_ratio: float = 0.95


@dataclass
class HoleDetection:
    ok: bool
    reason: str
    poly_uv: Optional[np.ndarray] = None  # (N,2) float32 full-image coords
    center_uv: Optional[Tuple[float, float]] = None
    bbox: Optional[Tuple[int, int, int, int]] = None  # x0,y0,x1,y1
    touches_border: bool = False
    fill: float = 0.0
    plane_inliers: int = 0
    plane_rms_m: float = float("nan")
    plane_cov: float = float("nan")
    plane_range_m: float = float("nan")
    r_px: float = float("nan")
    r_m: float = float("nan")
    err: str = ""


def _clamp_int(v: int, lo: int, hi: int) -> int:
    if v < lo:
        return int(lo)
    if v > hi:
        return int(hi)
    return int(v)


def _window_bbox(w: int, h: int, *, x: int, y: int, win_w: int, win_h: int) -> Tuple[int, int, int, int]:
    if int(win_w) <= 0 or int(win_h) <= 0:
        return 0, 0, int(w), int(h)
    ww = int(max(16, min(int(w), int(win_w))))
    hh = int(max(16, min(int(h), int(win_h))))
    x0 = _clamp_int(int(x) - ww // 2, 0, max(0, int(w) - 1))
    y0 = _clamp_int(int(y) - hh // 2, 0, max(0, int(h) - 1))
    x1 = _clamp_int(int(x0) + ww, 1, int(w))
    y1 = _clamp_int(int(y0) + hh, 1, int(h))
    return int(x0), int(y0), int(x1), int(y1)


def _poly_center_uv(poly_uv: np.ndarray) -> Optional[Tuple[float, float]]:
    try:
        pts = np.asarray(poly_uv, dtype=np.float32).reshape(-1, 2)
    except Exception:
        return None
    if int(pts.shape[0]) < 3 or not np.isfinite(pts).all():
        return None
    try:
        cont = pts.reshape(-1, 1, 2)
        m = cv2.moments(cont)
        if abs(float(m.get("m00", 0.0))) > 1e-6:
            return float(m["m10"]) / float(m["m00"]), float(m["m01"]) / float(m["m00"])
    except Exception:
        pass
    try:
        return float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1]))
    except Exception:
        return None


def _bbox_from_poly(poly_uv: np.ndarray, *, w: int, h: int, margin_px: int = 0) -> Optional[Tuple[int, int, int, int]]:
    try:
        pts = np.asarray(poly_uv, dtype=np.float32).reshape(-1, 2)
    except Exception:
        return None
    if int(pts.shape[0]) < 3 or not np.isfinite(pts).all():
        return None
    x0 = float(np.min(pts[:, 0]))
    x1 = float(np.max(pts[:, 0]))
    y0 = float(np.min(pts[:, 1]))
    y1 = float(np.max(pts[:, 1]))
    m = int(max(0, int(margin_px)))
    x0i = _clamp_int(int(math.floor(x0)) - m, 0, max(0, int(w) - 1))
    y0i = _clamp_int(int(math.floor(y0)) - m, 0, max(0, int(h) - 1))
    x1i = _clamp_int(int(math.ceil(x1)) + m + 1, 1, int(w))
    y1i = _clamp_int(int(math.ceil(y1)) + m + 1, 1, int(h))
    if int(x1i - x0i) < 2 or int(y1i - y0i) < 2:
        return None
    return int(x0i), int(y0i), int(x1i), int(y1i)


def _sample_depth_m(depth_m: np.ndarray, *, u: int, v: int, r_px: int, max_depth_m: float) -> float:
    h, w = int(depth_m.shape[0]), int(depth_m.shape[1])
    r = int(max(0, int(r_px)))
    u0 = _clamp_int(int(u) - r, 0, max(0, w - 1))
    v0 = _clamp_int(int(v) - r, 0, max(0, h - 1))
    u1 = _clamp_int(int(u) + r + 1, 1, w)
    v1 = _clamp_int(int(v) + r + 1, 1, h)
    roi = np.asarray(depth_m[int(v0) : int(v1), int(u0) : int(u1)], dtype=np.float32).reshape(-1)
    if int(roi.size) <= 0:
        return float("nan")
    ok = (roi > 0.0) & (roi < float(max_depth_m))
    if not bool(np.any(ok)):
        return float("nan")
    try:
        return float(np.median(roi[ok]))
    except Exception:
        try:
            return float(np.mean(roi[ok]))
        except Exception:
            return float("nan")


def _estimate_plane_depth_ring(
    depth_m: np.ndarray,
    *,
    x: int,
    y: int,
    r_in_px: int,
    r_out_px: int,
    step_px: int,
    max_depth_m: float,
) -> tuple[float, int, float]:
    """
    Return (plane_z_median, inliers_n, rms_m).

    This is a lightweight approximation of a plane-guided detector: sample a ring around the seed pixel
    and use the median depth as a local plane proxy.
    """
    h, w = int(depth_m.shape[0]), int(depth_m.shape[1])
    r_out = int(max(4, int(r_out_px)))
    r_in = int(max(0, min(int(r_in_px), r_out - 1)))
    step = int(max(1, int(step_px)))

    x0 = _clamp_int(int(x) - r_out, 0, max(0, w - 1))
    y0 = _clamp_int(int(y) - r_out, 0, max(0, h - 1))
    x1 = _clamp_int(int(x) + r_out + 1, 1, w)
    y1 = _clamp_int(int(y) + r_out + 1, 1, h)
    roi = np.asarray(depth_m[int(y0) : int(y1) : int(step), int(x0) : int(x1) : int(step)], dtype=np.float32)
    if int(roi.size) <= 0:
        return float("nan"), 0, float("nan")

    ys = (np.arange(int(roi.shape[0]), dtype=np.int32) * int(step) + int(y0) - int(y)).astype(np.int32)
    xs = (np.arange(int(roi.shape[1]), dtype=np.int32) * int(step) + int(x0) - int(x)).astype(np.int32)
    dy2 = (ys.astype(np.int64) ** 2).reshape(-1, 1)
    dx2 = (xs.astype(np.int64) ** 2).reshape(1, -1)
    dist2 = dy2 + dx2
    m = (dist2 >= int(r_in * r_in)) & (dist2 <= int(r_out * r_out))

    vals = roi[m].reshape(-1)
    if int(vals.size) <= 0:
        return float("nan"), 0, float("nan")
    ok = (vals > 0.0) & (vals < float(max_depth_m))
    if not bool(np.any(ok)):
        return float("nan"), 0, float("nan")
    v_ok = vals[ok]
    try:
        med = float(np.median(v_ok))
    except Exception:
        med = float(np.mean(v_ok))

    try:
        mad = float(np.median(np.abs(v_ok - float(med))))
        sigma = 1.4826 * float(mad)
        rms = float(max(0.0, sigma))
    except Exception:
        rms = float("nan")

    return float(med), int(v_ok.size), float(rms)


def _mask_to_polygon(
    mask_u8: np.ndarray,
    *,
    x0: int,
    y0: int,
    concavity_max: float,
    approx_eps_frac: float,
    max_vertices: int,
) -> Optional[np.ndarray]:
    try:
        m = np.asarray(mask_u8, dtype=np.uint8)
    except Exception:
        return None
    if m.ndim != 2 or int(m.size) <= 0:
        return None
    try:
        fc = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if isinstance(fc, tuple) and int(len(fc)) == 3:
            _img, conts, _hier = fc
        else:
            conts, _hier = fc
    except Exception:
        conts = []
    if not conts:
        return None
    c = max(conts, key=lambda cc: float(cv2.contourArea(cc)))
    try:
        area = float(cv2.contourArea(c))
    except Exception:
        area = 0.0
    if float(area) < 1.0:
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
        use = hull if float(concavity) > float(concavity_max) else c_full

        peri = float(cv2.arcLength(use, True))
        eps = float(max(1.0, float(approx_eps_frac) * float(peri)))
        pts = None
        for _ in range(12):
            approx = cv2.approxPolyDP(use, float(eps), True)
            pts0 = np.asarray(approx, dtype=np.float32).reshape(-1, 2)
            pts = pts0
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


def detect_hole(
    *,
    gray_u8: np.ndarray,
    depth_m: np.ndarray,
    intr: PinholeIntrinsics,
    x: int,
    y: int,
    settings: PolyHoleSettings,
) -> HoleDetection:
    h, w = int(depth_m.shape[0]), int(depth_m.shape[1])
    if h <= 0 or w <= 0:
        return HoleDetection(ok=False, reason="hole_no_depth", err="no_depth")
    if not (0 <= int(x) < int(w) and 0 <= int(y) < int(h)):
        return HoleDetection(ok=False, reason="hole_oob", err="oob")

    win_w, win_h = int(settings.hole_window_wh[0]), int(settings.hole_window_wh[1])
    x0, y0, x1, y1 = _window_bbox(int(w), int(h), x=int(x), y=int(y), win_w=int(win_w), win_h=int(win_h))
    depth_roi = np.asarray(depth_m[int(y0) : int(y1), int(x0) : int(x1)], dtype=np.float32)
    gray_roi = np.asarray(gray_u8[int(y0) : int(y1), int(x0) : int(x1)], dtype=np.uint8)
    if int(depth_roi.size) <= 0 or int(depth_roi.shape[0]) < 16 or int(depth_roi.shape[1]) < 16:
        return HoleDetection(ok=False, reason="hole_roi_small", err="roi_small")

    sx = int(x) - int(x0)
    sy = int(y) - int(y0)

    plane_z, plane_in, plane_rms = _estimate_plane_depth_ring(
        depth_m,
        x=int(x),
        y=int(y),
        r_in_px=int(settings.plane_ring_r_in_px),
        r_out_px=int(settings.plane_ring_r_out_px),
        step_px=int(settings.plane_ring_step_px),
        max_depth_m=float(settings.search_max_depth_m),
    )
    if not (math.isfinite(float(plane_z)) and float(plane_z) > 0.0):
        plane_z = _sample_depth_m(
            depth_m,
            u=int(x),
            v=int(y),
            r_px=8,
            max_depth_m=float(settings.search_max_depth_m),
        )
        plane_in = 0
        plane_rms = float("nan")

    if not (math.isfinite(float(plane_z)) and float(plane_z) > 0.0):
        return HoleDetection(ok=False, reason="hole_no_plane", err="no_plane")

    margin = float(max(float(settings.hole_margin_abs_m), float(settings.hole_margin_rel) * float(plane_z)))
    if math.isfinite(float(plane_rms)):
        margin = float(max(float(margin), float(settings.hole_margin_rms_k) * float(plane_rms)))
    thr = float(plane_z) + float(margin)

    d = depth_roi
    if bool(settings.hole_unknown_as_full):
        hole_mask = (d <= 0.0) | (d >= float(thr))
    else:
        hole_mask = (d > 0.0) & (d >= float(thr))

    # Optional edge barrier to reduce leaks across strong texture edges.
    try:
        edges = cv2.Canny(gray_roi, int(settings.canny_lo), int(settings.canny_hi), L2gradient=True)
        if bool(settings.dilate_edges):
            k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            edges = cv2.dilate(edges, k, iterations=1)
        hole_mask = np.asarray(hole_mask, dtype=bool) & (np.asarray(edges, dtype=np.uint8) == 0)
    except Exception:
        hole_mask = np.asarray(hole_mask, dtype=bool)

    # Seed snap if needed.
    if not bool(hole_mask[int(sy), int(sx)]):
        r = int(max(0, int(settings.hole_seed_snap_px)))
        if int(r) <= 0:
            return HoleDetection(ok=False, reason="hole_seed_not_in_mask", err="seed_not_in_mask")
        u0 = _clamp_int(int(sx) - r, 0, max(0, int(hole_mask.shape[1]) - 1))
        v0 = _clamp_int(int(sy) - r, 0, max(0, int(hole_mask.shape[0]) - 1))
        u1 = _clamp_int(int(sx) + r + 1, 1, int(hole_mask.shape[1]))
        v1 = _clamp_int(int(sy) + r + 1, 1, int(hole_mask.shape[0]))
        sub = hole_mask[int(v0) : int(v1), int(u0) : int(u1)]
        ys, xs = np.nonzero(sub)
        if int(xs.size) <= 0:
            return HoleDetection(ok=False, reason="hole_no_component", err="no_component")
        dx = (xs.astype(np.int32) + int(u0) - int(sx)).astype(np.int32)
        dy = (ys.astype(np.int32) + int(v0) - int(sy)).astype(np.int32)
        d2 = dx.astype(np.int64) * dx.astype(np.int64) + dy.astype(np.int64) * dy.astype(np.int64)
        j = int(np.argmin(d2))
        sx = int(xs[j]) + int(u0)
        sy = int(ys[j]) + int(v0)

    # Extract connected component via flood fill on the binary mask.
    hole_u8 = (hole_mask.astype(np.uint8) * 255).copy()
    ff_img = hole_u8.copy()
    ff_mask = np.zeros((int(ff_img.shape[0]) + 2, int(ff_img.shape[1]) + 2), dtype=np.uint8)
    try:
        cv2.floodFill(ff_img, ff_mask, seedPoint=(int(sx), int(sy)), newVal=128, loDiff=0, upDiff=0)
    except Exception:
        return HoleDetection(ok=False, reason="hole_floodfill_err", err="floodfill_err")
    comp = (ff_img == 128)
    area = int(np.count_nonzero(comp))
    if float(area) < float(settings.hole_min_area_px):
        return HoleDetection(ok=False, reason="hole_small", err="small")

    touches_border = bool(
        bool(np.any(comp[0, :]))
        or bool(np.any(comp[-1, :]))
        or bool(np.any(comp[:, 0]))
        or bool(np.any(comp[:, -1]))
    )
    fill = float(area) / float(max(1, int(comp.size)))

    poly_uv = _mask_to_polygon(
        (comp.astype(np.uint8) * 255),
        x0=int(x0),
        y0=int(y0),
        concavity_max=float(settings.poly_concavity_max),
        approx_eps_frac=float(settings.poly_approx_eps_frac),
        max_vertices=int(settings.poly_max_vertices),
    )
    if poly_uv is None:
        return HoleDetection(ok=False, reason="hole_no_poly", err="no_poly")

    center_uv = _poly_center_uv(poly_uv)
    bbox = _bbox_from_poly(poly_uv, w=int(intr.w), h=int(intr.h), margin_px=0)

    # Inscribed radius (pixels) via distance transform.
    try:
        dist = cv2.distanceTransform(comp.astype(np.uint8), cv2.DIST_L2, 3)
        r_px = float(np.max(dist)) if dist is not None and int(dist.size) > 0 else float("nan")
    except Exception:
        r_px = float("nan")

    fx0 = float(max(1e-9, float(intr.fx)))
    fy0 = float(max(1e-9, float(intr.fy)))
    fx_avg = float(0.5 * (fx0 + fy0))
    r_m = float(r_px) * float(plane_z) / float(fx_avg) if math.isfinite(float(r_px)) else float("nan")

    return HoleDetection(
        ok=True,
        reason="hole_ok",
        poly_uv=np.asarray(poly_uv, dtype=np.float32).reshape(-1, 2),
        center_uv=(tuple(float(v) for v in center_uv) if center_uv is not None else None),
        bbox=bbox,
        touches_border=bool(touches_border),
        fill=float(fill),
        plane_inliers=int(plane_in),
        plane_rms_m=float(plane_rms),
        plane_cov=float("nan"),
        plane_range_m=float(plane_z),
        r_px=float(r_px),
        r_m=float(r_m),
        err="",
    )


class PolyHoleTracker:
    """
    Polygon/hole tracker that consumes Twc from ORB-SLAM3 and outputs headless-friendly state.
    """

    def __init__(self, *, settings: PolyHoleSettings, intr: PinholeIntrinsics) -> None:
        self.settings = settings
        self.intr = intr

        self.runtime_enabled = bool(settings.hole_runtime_enabled_default)
        self.hover_uv: Optional[Tuple[int, int]] = None

        self._preview_last_t = 0.0
        self._preview_cache: Optional[dict] = None

        self.active: bool = False
        self.sel_kind: str = "none"  # none|hole|bbox
        self.pid_confirmed: bool = False
        self.reason: str = ""

        self.poly_world: Optional[np.ndarray] = None  # (N,3) float64
        self.poly_uv: Optional[np.ndarray] = None  # (N,2) float32
        self.bbox: Optional[Tuple[int, int, int, int]] = None
        self.last_fi: int = -1
        self.last_ts: float = 0.0

        self.hole_center_uv: Optional[Tuple[float, float]] = None
        self.hole_r_m: float = float("nan")
        self.hole_fill: float = float("nan")
        self.hole_ok_pid: bool = False
        self.hole_plane_inliers: int = 0
        self.hole_plane_rms_m: float = float("nan")
        self.hole_plane_cov: float = float("nan")
        self.hole_plane_range_m: float = float("nan")
        self.hole_err: str = ""

        self._last_pid_fi: int = -1

    def clear(self) -> None:
        self.active = False
        self.sel_kind = "none"
        self.pid_confirmed = False
        self.reason = ""
        self.poly_world = None
        self.poly_uv = None
        self.bbox = None
        self.hole_center_uv = None
        self.hole_r_m = float("nan")
        self.hole_fill = float("nan")
        self.hole_ok_pid = False
        self.hole_plane_inliers = 0
        self.hole_plane_rms_m = float("nan")
        self.hole_plane_cov = float("nan")
        self.hole_plane_range_m = float("nan")
        self.hole_err = ""
        self._last_pid_fi = -1

    def set_hover(self, *, x: int, y: int) -> None:
        self.hover_uv = (int(x), int(y))

    def set_runtime_enabled(self, *, enable: bool) -> None:
        self.runtime_enabled = bool(enable)
        if not bool(self.runtime_enabled):
            self.hover_uv = None
            self._preview_cache = None

    def maybe_preview(self, *, fi: int, gray_u8: np.ndarray, depth_m: np.ndarray) -> Optional[dict]:
        if not bool(self.settings.enabled):
            return None
        if not bool(self.settings.hole_preview_enabled):
            return None
        if not bool(self.runtime_enabled):
            return None
        if self.active:
            return None
        if self.hover_uv is None:
            return None

        rate = float(max(0.0, float(self.settings.hole_preview_rate_hz)))
        min_dt = (1.0 / rate) if rate > 0.0 else 0.0
        now = float(time.time())
        if float(min_dt) > 0.0 and float(now - float(self._preview_last_t)) < float(min_dt):
            return None
        self._preview_last_t = float(now)

        x, y = int(self.hover_uv[0]), int(self.hover_uv[1])
        det = detect_hole(gray_u8=gray_u8, depth_m=depth_m, intr=self.intr, x=int(x), y=int(y), settings=self.settings)
        if not bool(det.ok) or det.poly_uv is None:
            return None

        verts_uv: list[Tuple[int, int]] = []
        try:
            for u, v in np.asarray(det.poly_uv, dtype=np.float32).reshape(-1, 2).tolist():
                verts_uv.append((int(round(float(u))), int(round(float(v)))))
        except Exception:
            verts_uv = []
        if len(verts_uv) < 3:
            return None

        if det.center_uv is not None:
            c_u, c_v = float(det.center_uv[0]), float(det.center_uv[1])
        else:
            c_uv = _poly_center_uv(det.poly_uv)
            if c_uv is not None:
                c_u, c_v = float(c_uv[0]), float(c_uv[1])
            else:
                c_u, c_v = float(self.intr.cx), float(self.intr.cy)

        ok_pid = (
            (not bool(det.touches_border))
            and math.isfinite(float(det.r_m))
            and float(det.r_m) >= float(self.settings.hole_pid_min_radius_m)
        )

        payload = {
            "fi": int(fi),
            "verts_uv": verts_uv,
            "center_uv": (float(c_u), float(c_v)),
            "ok_pid": int(bool(ok_pid)),
            "hole_stats": {
                "center_uv": (float(c_u), float(c_v)),
                "r_m": float(det.r_m) if math.isfinite(float(det.r_m)) else float("nan"),
                "fill": float(det.fill),
                "ok_pid": int(bool(ok_pid)),
                "plane_inliers": int(det.plane_inliers),
                "plane_rms_m": float(det.plane_rms_m) if math.isfinite(float(det.plane_rms_m)) else float("nan"),
                "plane_cov": float(det.plane_cov) if math.isfinite(float(det.plane_cov)) else float("nan"),
                "range_m": float(det.plane_range_m) if math.isfinite(float(det.plane_range_m)) else float("nan"),
                "err": str(det.err or ""),
            },
        }
        self._preview_cache = dict(payload)
        return payload

    def _lift_poly_to_world(
        self,
        *,
        poly_uv: np.ndarray,
        depth_m: np.ndarray,
        Twc: np.ndarray,
        plane_z: float,
    ) -> Optional[np.ndarray]:
        try:
            pts = np.asarray(poly_uv, dtype=np.float32).reshape(-1, 2)
        except Exception:
            return None
        if int(pts.shape[0]) < 3:
            return None
        Twc0 = np.asarray(Twc, dtype=np.float64).reshape(4, 4)
        Rwc = np.asarray(Twc0[:3, :3], dtype=np.float64).reshape(3, 3)
        twc = np.asarray(Twc0[:3, 3], dtype=np.float64).reshape(3)

        fx = float(max(1e-9, float(self.intr.fx)))
        fy = float(max(1e-9, float(self.intr.fy)))
        cx = float(self.intr.cx)
        cy = float(self.intr.cy)

        out = np.empty((int(pts.shape[0]), 3), dtype=np.float64)
        for i in range(int(pts.shape[0])):
            u = int(round(float(pts[i, 0])))
            v = int(round(float(pts[i, 1])))
            z = _sample_depth_m(
                depth_m,
                u=int(u),
                v=int(v),
                r_px=int(self.settings.hole_depth_r_px),
                max_depth_m=float(self.settings.search_max_depth_m),
            )
            if not (math.isfinite(float(z)) and float(z) > 0.0):
                z = float(plane_z)
            if not (math.isfinite(float(z)) and float(z) > 0.0):
                return None
            x = (float(u) - float(cx)) * float(z) / float(fx)
            y = (float(v) - float(cy)) * float(z) / float(fy)
            Pw = (Rwc @ np.asarray([x, y, z], dtype=np.float64).reshape(3)) + twc
            out[int(i), :] = Pw.reshape(3)
        return out

    def select_hole(
        self,
        *,
        fi: int,
        ts: float,
        x: int,
        y: int,
        gray_u8: np.ndarray,
        depth_m: np.ndarray,
        Twc: Optional[np.ndarray],
    ) -> bool:
        if not bool(self.settings.enabled):
            return False
        if Twc is None:
            self.reason = "hole_no_pose"
            self.hole_err = "no_pose"
            return False
        det = detect_hole(gray_u8=gray_u8, depth_m=depth_m, intr=self.intr, x=int(x), y=int(y), settings=self.settings)
        if not bool(det.ok) or det.poly_uv is None:
            self.reason = str(det.reason or "hole_fail")
            self.hole_err = str(det.err or det.reason or "hole_fail")
            return False

        plane_z = float(det.plane_range_m) if math.isfinite(float(det.plane_range_m)) else float("nan")
        verts_w = self._lift_poly_to_world(poly_uv=det.poly_uv, depth_m=depth_m, Twc=np.asarray(Twc), plane_z=float(plane_z))
        if verts_w is None:
            self.reason = "hole_lift_fail"
            self.hole_err = "lift_fail"
            return False

        self.active = True
        self.sel_kind = "hole"
        self.pid_confirmed = False
        self.reason = "select_ok"
        self.poly_world = np.asarray(verts_w, dtype=np.float64).reshape(-1, 3)
        self.poly_uv = np.asarray(det.poly_uv, dtype=np.float32).reshape(-1, 2)
        self.bbox = det.bbox
        self.last_fi = int(fi)
        self.last_ts = float(ts)

        self.hole_center_uv = det.center_uv
        self.hole_r_m = float(det.r_m)
        self.hole_fill = float(det.fill)
        self.hole_plane_inliers = int(det.plane_inliers)
        self.hole_plane_rms_m = float(det.plane_rms_m)
        self.hole_plane_cov = float(det.plane_cov)
        self.hole_plane_range_m = float(det.plane_range_m)
        self.hole_ok_pid = (
            (not bool(det.touches_border))
            and math.isfinite(float(det.r_m))
            and float(det.r_m) >= float(self.settings.hole_pid_min_radius_m)
        )
        self.hole_err = str(det.err or "")
        return True

    def select_bbox(
        self,
        *,
        fi: int,
        ts: float,
        bbox: Tuple[int, int, int, int],
        depth_m: np.ndarray,
        Twc: Optional[np.ndarray],
    ) -> bool:
        """
        Generic polygon selection from a bbox.

        For parity with Drone_client's headless protocol this supports the `select_bbox` command,
        but uses a simple rectangle polygon (no contour inference).
        """
        if not bool(self.settings.enabled):
            return False
        if Twc is None:
            self.reason = "bbox_no_pose"
            self.hole_err = "no_pose"
            return False
        try:
            x0, y0, x1, y1 = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
        except Exception:
            self.reason = "bbox_invalid"
            self.hole_err = "bbox_invalid"
            return False

        x0 = _clamp_int(int(x0), 0, max(0, int(self.intr.w) - 1))
        y0 = _clamp_int(int(y0), 0, max(0, int(self.intr.h) - 1))
        x1 = _clamp_int(int(x1), x0 + 1, int(self.intr.w))
        y1 = _clamp_int(int(y1), y0 + 1, int(self.intr.h))
        if int(x1 - x0) < 2 or int(y1 - y0) < 2:
            self.reason = "bbox_small"
            self.hole_err = "bbox_small"
            return False

        poly_uv = np.asarray(
            [[float(x0), float(y0)], [float(x1), float(y0)], [float(x1), float(y1)], [float(x0), float(y1)]],
            dtype=np.float32,
        ).reshape(-1, 2)
        cz = _sample_depth_m(
            depth_m,
            u=int((x0 + x1) // 2),
            v=int((y0 + y1) // 2),
            r_px=8,
            max_depth_m=float(self.settings.search_max_depth_m),
        )
        verts_w = self._lift_poly_to_world(poly_uv=poly_uv, depth_m=depth_m, Twc=np.asarray(Twc), plane_z=float(cz))
        if verts_w is None:
            self.reason = "bbox_lift_fail"
            self.hole_err = "lift_fail"
            return False

        self.active = True
        self.sel_kind = "bbox"
        self.pid_confirmed = False
        self.reason = "bbox_ok"
        self.poly_world = np.asarray(verts_w, dtype=np.float64).reshape(-1, 3)
        self.poly_uv = np.asarray(poly_uv, dtype=np.float32).reshape(-1, 2)
        self.bbox = (int(x0), int(y0), int(x1), int(y1))
        self.last_fi = int(fi)
        self.last_ts = float(ts)

        self.hole_center_uv = _poly_center_uv(poly_uv)
        self.hole_r_m = float("nan")
        self.hole_fill = float("nan")
        self.hole_ok_pid = False
        self.hole_plane_inliers = 0
        self.hole_plane_rms_m = float("nan")
        self.hole_plane_cov = float("nan")
        self.hole_plane_range_m = float("nan")
        self.hole_err = ""
        return True

    def confirm_hole(self) -> bool:
        if not (bool(self.active) and str(self.sel_kind) == "hole"):
            return False
        if not bool(self.hole_ok_pid):
            self.reason = "hole_pid_gate"
            self.hole_err = "pid_gate"
            return False
        self.pid_confirmed = True
        self.reason = "hole_confirmed"
        return True

    def update_track(self, *, fi: int, ts: float, Twc: Optional[np.ndarray]) -> None:
        self.last_fi = int(fi)
        self.last_ts = float(ts)
        if not bool(self.active):
            return
        if Twc is None or self.poly_world is None:
            self.poly_uv = None
            self.bbox = None
            self.reason = "vo_lost"
            return

        Twc0 = np.asarray(Twc, dtype=np.float64).reshape(4, 4)
        try:
            Tcw = np.linalg.inv(Twc0)
        except Exception:
            self.poly_uv = None
            self.bbox = None
            self.reason = "vo_lost"
            return

        Rcw = np.asarray(Tcw[:3, :3], dtype=np.float64).reshape(3, 3)
        tcw = np.asarray(Tcw[:3, 3], dtype=np.float64).reshape(3)
        Pw = np.asarray(self.poly_world, dtype=np.float64).reshape(-1, 3)
        Pc = (Rcw @ Pw.T).T + tcw.reshape(1, 3)
        z = Pc[:, 2]
        if not bool(np.any(z > 1e-6)):
            self.poly_uv = None
            self.bbox = None
            self.reason = "poly_behind_cam"
            return

        fx = float(self.intr.fx)
        fy = float(self.intr.fy)
        cx = float(self.intr.cx)
        cy = float(self.intr.cy)
        u = fx * (Pc[:, 0] / z) + cx
        v = fy * (Pc[:, 1] / z) + cy
        uv = np.stack([u, v], axis=1).astype(np.float32, copy=False)
        if not np.isfinite(uv).all():
            self.poly_uv = None
            self.bbox = None
            self.reason = "poly_nan"
            return

        self.poly_uv = uv
        self.bbox = _bbox_from_poly(uv, w=int(self.intr.w), h=int(self.intr.h), margin_px=0)

        if bool(self.pid_confirmed) and bool(self.settings.hole_done_enabled) and self.bbox is not None:
            try:
                x0, y0, x1, y1 = self.bbox
                span_x = float(x1 - x0)
                span_y = float(y1 - y0)
                thr_x = float(self.settings.hole_done_fill_ratio) * float(self.intr.w)
                thr_y = float(self.settings.hole_done_fill_ratio) * float(self.intr.h)
                if float(span_x) >= float(thr_x) or float(span_y) >= float(thr_y):
                    self.clear()
                    self.reason = "hole_done"
            except Exception:
                pass

    def poly_out(self) -> dict:
        poly_uv_list = None
        if self.poly_uv is not None:
            try:
                poly_uv_list = np.asarray(self.poly_uv, dtype=np.float32).reshape(-1, 2).tolist()
            except Exception:
                poly_uv_list = None

        hs = {
            "center_uv": (tuple(float(x) for x in self.hole_center_uv) if self.hole_center_uv is not None else None),
            "r_m": float(self.hole_r_m) if math.isfinite(float(self.hole_r_m)) else float("nan"),
            "fill": float(self.hole_fill) if math.isfinite(float(self.hole_fill)) else float("nan"),
            "ok_pid": int(bool(self.hole_ok_pid)),
            "plane_inliers": int(self.hole_plane_inliers),
            "plane_rms_m": float(self.hole_plane_rms_m) if math.isfinite(float(self.hole_plane_rms_m)) else float("nan"),
            "plane_cov": float(self.hole_plane_cov) if math.isfinite(float(self.hole_plane_cov)) else float("nan"),
            "err": str(self.hole_err or ""),
        }
        if math.isfinite(float(self.hole_plane_range_m)):
            hs["range_m"] = float(self.hole_plane_range_m)

        return {
            "active": int(bool(self.active)),
            "sel_kind": str(self.sel_kind),
            "pid_confirmed": int(bool(self.pid_confirmed)),
            "bbox": (tuple(int(v) for v in self.bbox) if self.bbox is not None else None),
            "verts_uv": poly_uv_list,
            "reason": str(self.reason or ""),
            "fi": int(self.last_fi),
            "hole_stats": hs,
        }

    def hole_out(self) -> Optional[dict]:
        if not (bool(self.active) and str(self.sel_kind) == "hole"):
            return None
        stage = 2 if bool(self.pid_confirmed) else 1

        center_uv = None
        if bool(self.pid_confirmed) and self.poly_uv is not None:
            center_uv = _poly_center_uv(self.poly_uv)
        if center_uv is None:
            center_uv = self.hole_center_uv
        if center_uv is None:
            center_uv = (float(self.intr.cx), float(self.intr.cy))
        u0, v0 = float(center_uv[0]), float(center_uv[1])

        fx = float(max(1e-9, float(self.intr.fx)))
        fy = float(max(1e-9, float(self.intr.fy)))
        cx = float(self.intr.cx)
        cy = float(self.intr.cy)
        h_deg = float(np.degrees(np.arctan2(float(u0) - float(cx), float(fx))))
        v_deg = float(np.degrees(np.arctan2(float(v0) - float(cy), float(fy))))

        plane = {
            "inliers": int(self.hole_plane_inliers),
            "rms_m": float(self.hole_plane_rms_m) if math.isfinite(float(self.hole_plane_rms_m)) else float("nan"),
            "cov": float(self.hole_plane_cov) if math.isfinite(float(self.hole_plane_cov)) else float("nan"),
            "range_m": float(self.hole_plane_range_m) if math.isfinite(float(self.hole_plane_range_m)) else float("nan"),
        }

        return {
            "stage": int(stage),
            "fi": int(self.last_fi),
            "center_uv": (float(u0), float(v0)),
            "dhv_deg": (float(h_deg), float(v_deg)),
            "r_m": float(self.hole_r_m) if math.isfinite(float(self.hole_r_m)) else float("nan"),
            "fill": float(self.hole_fill) if math.isfinite(float(self.hole_fill)) else float("nan"),
            "ok_pid": int(bool(self.hole_ok_pid)),
            "plane": plane,
            "err": str(self.hole_err or ""),
        }

    def maybe_pid_event(self) -> Optional[dict]:
        hole = self.hole_out()
        if hole is None:
            return None
        fi = int(hole.get("fi", -1))
        if int(fi) == int(self._last_pid_fi):
            return None
        self._last_pid_fi = int(fi)
        return dict(hole)


def settings_from_dict(d: dict) -> PolyHoleSettings:
    if not isinstance(d, dict):
        return PolyHoleSettings()

    def _g(key: str, default):
        return d.get(key, default)

    try:
        win = _g("hole_window_wh", (0, 0))
        if isinstance(win, (list, tuple)) and len(win) >= 2:
            hole_window_wh = (int(win[0]), int(win[1]))
        else:
            hole_window_wh = (0, 0)
    except Exception:
        hole_window_wh = (0, 0)

    return PolyHoleSettings(
        enabled=bool(_g("enabled", True)),
        hole_preview_enabled=bool(_g("hole_preview_enabled", True)),
        hole_preview_rate_hz=float(_g("hole_preview_rate_hz", 10.0) or 10.0),
        hole_preview_hold_s=float(_g("hole_preview_hold_s", 0.60) or 0.60),
        hole_runtime_enabled_default=bool(_g("hole_runtime_enabled_default", True)),
        hole_window_wh=hole_window_wh,
        hole_min_area_px=float(_g("hole_min_area_px", 25.0) or 25.0),
        hole_unknown_as_full=bool(_g("hole_unknown_as_full", True)),
        hole_seed_snap_px=int(_g("hole_select_snap_px", _g("hole_seed_snap_px", 32)) or 32),
        hole_depth_r_px=int(_g("hole_depth_r_px", 8) or 8),
        search_max_depth_m=float(_g("search_max_depth_m", 15.0) or 15.0),
        plane_ring_r_in_px=int(_g("plane_ring_r_in_px", 14) or 14),
        plane_ring_r_out_px=int(_g("plane_ring_r_out_px", 150) or 150),
        plane_ring_step_px=int(_g("plane_ring_step_px", 3) or 3),
        hole_margin_abs_m=float(_g("hole_margin_abs_m", 0.06) or 0.06),
        hole_margin_rel=float(_g("hole_margin_rel", 0.02) or 0.02),
        hole_margin_rms_k=float(_g("hole_margin_rms_k", 3.0) or 3.0),
        canny_lo=int(_g("canny_lo", 60) or 60),
        canny_hi=int(_g("canny_hi", 140) or 140),
        dilate_edges=bool(_g("dilate_edges", True)),
        poly_concavity_max=float(_g("hole_poly_concavity_max", _g("poly_concavity_max", 0.15)) or 0.15),
        poly_approx_eps_frac=float(_g("hole_poly_approx_eps_frac", _g("poly_approx_eps_frac", 0.03)) or 0.03),
        poly_max_vertices=int(_g("hole_poly_max_vertices", _g("poly_max_vertices", 18)) or 18),
        hole_pid_min_radius_m=float(_g("hole_pid_min_radius_m", 0.55) or 0.55),
        hole_done_enabled=bool(_g("hole_done_enabled", False)),
        hole_done_fill_ratio=float(_g("hole_done_fill_ratio", 0.95) or 0.95),
    )
