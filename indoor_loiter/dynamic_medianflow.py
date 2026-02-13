
"""
DynMedianFlowTracker - fast MedianFlow-style tracker with adaptive grid.

This is a practical, lightweight tracker designed for high FPS, but it also stays robust under
moderate frame-to-frame motion by using:
- LK forward-backward consistency
- robust median translation
- robust median-log scale (with reliability checks + damping)
- optional NCC point verification (Numba-accelerated if available)
- adaptive grid density (dynamic grid) with a tiny-target fallback ("track every pixel")

------------------------------------------------------------------------------
Quick start
------------------------------------------------------------------------------
import cv2
from tracker import DynMedianFlowTracker

cap = cv2.VideoCapture(0)
ok, frame = cap.read()
bbox = cv2.selectROI("init", frame, False)  # (x,y,w,h)

trk = DynMedianFlowTracker()
trk.init(frame, bbox)

while True:
    ok, frame = cap.read()
    if not ok:
        break
    ok_trk, bbox = trk.update(frame)  # bbox is (x,y,w,h) floats
    if ok_trk:
        x,y,w,h = bbox
        p1 = (int(x), int(y))
        p2 = (int(x+w), int(y+h))
        cv2.rectangle(frame, p1, p2, (0,255,255), 2)
    cv2.imshow("trk", frame)
    if cv2.waitKey(1) == 27:
        break

------------------------------------------------------------------------------
Notes on defaults (based on our experiments)
------------------------------------------------------------------------------
- Dynamic grid is ON and adapts to tracking quality:
  * grows grid when too many points are lost
  * shrinks grid when quality is very high (to save CPU)
- Grid floor is 10x10 in normal sizes, but for tiny objects we automatically cap
  each axis by bbox dimension so you never sample more points than bbox pixels.
- NCC is ON by default for robustness. If you need maximum speed and can accept
  more drift near distractors, set Params.ncc_enable=False.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import math
import numpy as np
import cv2

try:
    from numba import njit
    _HAVE_NUMBA = True
except Exception:
    _HAVE_NUMBA = False
    def njit(*args, **kwargs):  # type: ignore
        def wrap(fn):
            return fn
        return wrap


BBox = Tuple[float, float, float, float]


@njit(cache=True)
def _ncc_scores_uint8(prev: np.ndarray,
                     curr: np.ndarray,
                     px: np.ndarray, py: np.ndarray,
                     cx: np.ndarray, cy: np.ndarray,
                     half: int) -> np.ndarray:
    """
    Compute NCC scores for patches centered at (px,py) in prev and (cx,cy) in curr.
    Coordinates are float arrays; we round to nearest int.
    Returns score in [-1,1], or -2.0 if patch out of bounds or near-constant.
    """
    n = px.shape[0]
    out = np.empty(n, dtype=np.float32)
    h, w = prev.shape[0], prev.shape[1]
    win = 2 * half + 1
    area = win * win

    for i in range(n):
        x0 = int(px[i] + 0.5)
        y0 = int(py[i] + 0.5)
        x1 = int(cx[i] + 0.5)
        y1 = int(cy[i] + 0.5)

        if (x0 - half < 0) or (x0 + half >= w) or (y0 - half < 0) or (y0 + half >= h):
            out[i] = -2.0
            continue
        if (x1 - half < 0) or (x1 + half >= w) or (y1 - half < 0) or (y1 + half >= h):
            out[i] = -2.0
            continue

        # compute means
        s0 = 0.0
        s1 = 0.0
        for dy in range(-half, half + 1):
            yy0 = y0 + dy
            yy1 = y1 + dy
            for dx in range(-half, half + 1):
                s0 += float(prev[yy0, x0 + dx])
                s1 += float(curr[yy1, x1 + dx])
        m0 = s0 / area
        m1 = s1 / area

        # compute variance and covariance
        v0 = 0.0
        v1 = 0.0
        cov = 0.0
        for dy in range(-half, half + 1):
            yy0 = y0 + dy
            yy1 = y1 + dy
            for dx in range(-half, half + 1):
                a = float(prev[yy0, x0 + dx]) - m0
                b = float(curr[yy1, x1 + dx]) - m1
                v0 += a * a
                v1 += b * b
                cov += a * b

        if v0 < 1e-6 or v1 < 1e-6:
            out[i] = -2.0
            continue
        out[i] = cov / math.sqrt(v0 * v1 + 1e-12)
    return out


def _to_gray(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 2:
        return frame
    if frame.ndim == 3 and frame.shape[2] == 1:
        return frame[:, :, 0]
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def _clip_bbox(b: BBox, w: int, h: int) -> BBox:
    x, y, bw, bh = b
    bw = max(1.0, float(bw))
    bh = max(1.0, float(bh))
    x = float(x)
    y = float(y)
    # clamp to frame with minimal validity
    if x < 0:
        x = 0.0
    if y < 0:
        y = 0.0
    if x + bw > w:
        x = max(0.0, float(w) - bw)
    if y + bh > h:
        y = max(0.0, float(h) - bh)
    return (x, y, bw, bh)


def _bbox_center(b: BBox) -> Tuple[float, float]:
    x, y, w, h = b
    return (x + 0.5 * w, y + 0.5 * h)


def _bbox_diag(b: BBox) -> float:
    return math.hypot(float(b[2]), float(b[3]))


def _make_grid_points(b: BBox, cols: int, rows: int) -> np.ndarray:
    x, y, w, h = b
    cols = max(1, int(cols))
    rows = max(1, int(rows))
    # sample at cell centers
    xs = x + (np.arange(cols, dtype=np.float32) + 0.5) * (w / cols)
    ys = y + (np.arange(rows, dtype=np.float32) + 0.5) * (h / rows)
    xx, yy = np.meshgrid(xs, ys)
    pts = np.stack([xx.reshape(-1), yy.reshape(-1)], axis=1).astype(np.float32)
    return pts.reshape(-1, 1, 2)


def _patch_template(gray: np.ndarray, bbox: BBox, size: int) -> Optional[np.ndarray]:
    h, w = gray.shape[:2]
    x, y, bw, bh = _clip_bbox(bbox, w, h)
    x0 = int(x)
    y0 = int(y)
    x1 = int(x + bw)
    y1 = int(y + bh)
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None
    patch = gray[y0:y1, x0:x1]
    patch = cv2.resize(patch, (size, size), interpolation=cv2.INTER_AREA)
    v = patch.astype(np.float32).reshape(-1)
    v -= v.mean()
    n = float(np.linalg.norm(v) + 1e-6)
    v /= n
    return v


def _template_sim(gray: np.ndarray, bbox: BBox, tmpl: np.ndarray, size: int) -> float:
    v = _patch_template(gray, bbox, size)
    if v is None:
        return -1.0
    return float(np.clip(np.dot(v, tmpl), -1.0, 1.0))


@dataclass
class DynMFParams:
    # Dynamic grid
    dynamic_grid: bool = True
    grid_min: Tuple[int, int] = (10, 10)
    grid_max: Tuple[int, int] = (35, 35)
    grow_if_good_frac_below: float = 0.35
    shrink_if_good_frac_above: float = 0.85
    grid_step: float = 1.20  # multiplicative step when growing/shrinking

    # LK
    lk_win_size: int = 15
    lk_max_level: int = 5
    lk_max_iter: int = 20
    lk_eps: float = 0.03

    # FB filtering
    fb_factor: float = 2.5
    fb_min: float = 1.0

    # NCC verification
    ncc_enable: bool = True
    ncc_half_window: int = 8  # 17x17 patches
    ncc_min: float = 0.80
    ncc_keep_frac: float = 0.70  # fallback keep top frac if threshold too strict

    # Minimum points to accept update
    min_points: int = 10
    reseed_if_points_below: int = 20

    # Scale estimation (robust median-log)
    scale_enable: bool = True
    scale_clamp: float = 1.25  # per-frame clamp
    scale_alpha: float = 0.35  # damping
    scale_min_rad: float = 2.0  # ignore points too close to center
    scale_log_mad_max: float = 0.25  # if too spread, ignore scale update

    # Confidence (optional) - does NOT fail by default
    template_enable: bool = True
    template_size: int = 32
    template_min_sim: float = 0.15
    fail_on_low_template: bool = False


class DynMedianFlowTracker:
    """
    DynMedianFlowTracker: adaptive-grid MedianFlow-style tracker.

    - No CUDA, no IMM, no FlowGate.
    - Designed for performance. Debug info is disabled by default.
    """

    def __init__(self, params: Optional[DynMFParams] = None):
        self.p = params or DynMFParams()
        self._tmpl: Optional[np.ndarray] = None
        self._prev_gray: Optional[np.ndarray] = None
        self._bbox: Optional[BBox] = None
        self._pts: Optional[np.ndarray] = None   # Nx1x2 float32
        self._ages: Optional[np.ndarray] = None  # N int32
        self._grid_cols: int = self.p.grid_min[0]
        self._grid_rows: int = self.p.grid_min[1]
        self._expect_gray: bool = False
        self._last_conf: float = 1.0
        self._last_template_sim: float = 1.0

    @property
    def bbox(self) -> Optional[BBox]:
        return self._bbox

    @property
    def last_confidence(self) -> float:
        return float(self._last_conf)

    @property
    def last_template_similarity(self) -> float:
        return float(self._last_template_sim)

    def init(self, frame: np.ndarray, bbox: Tuple[float, float, float, float]) -> bool:
        gray = _to_gray(frame)
        self._expect_gray = (frame.ndim == 2) or (frame.ndim == 3 and frame.shape[2] == 1)
        h, w = gray.shape[:2]
        self._bbox = _clip_bbox(tuple(map(float, bbox)), w, h)
        self._prev_gray = gray.copy()

        if self.p.template_enable:
            self._tmpl = _patch_template(self._prev_gray, self._bbox, self.p.template_size)

        self._grid_cols, self._grid_rows = self._initial_grid(self._bbox)
        self._pts = _make_grid_points(self._bbox, self._grid_cols, self._grid_rows)
        self._ages = np.zeros((self._pts.shape[0],), dtype=np.int32)
        self._last_conf = 1.0
        self._last_template_sim = 1.0
        return True

    def _initial_grid(self, bbox: BBox) -> Tuple[int, int]:
        # Base grid aims for grid_min, but never exceeds bbox pixel dims.
        cols0, rows0 = self.p.grid_min
        w = int(max(1.0, bbox[2]))
        h = int(max(1.0, bbox[3]))
        cols = min(max(1, cols0), w)
        rows = min(max(1, rows0), h)
        # also clamp to grid_max
        cols = min(cols, self.p.grid_max[0])
        rows = min(rows, self.p.grid_max[1])
        return int(cols), int(rows)

    def _adapt_grid(self, good_frac: float, bbox: BBox) -> None:
        if not self.p.dynamic_grid:
            return
        cols, rows = self._grid_cols, self._grid_rows
        if good_frac < self.p.grow_if_good_frac_below:
            cols = int(math.ceil(cols * self.p.grid_step))
            rows = int(math.ceil(rows * self.p.grid_step))
        elif good_frac > self.p.shrink_if_good_frac_above:
            cols = int(math.floor(cols / self.p.grid_step))
            rows = int(math.floor(rows / self.p.grid_step))
        # clamp to configured min/max, but also to bbox dims
        w = int(max(1.0, bbox[2]))
        h = int(max(1.0, bbox[3]))
        cols = max(self.p.grid_min[0], cols)
        rows = max(self.p.grid_min[1], rows)
        cols = min(self.p.grid_max[0], cols, w)
        rows = min(self.p.grid_max[1], rows, h)
        self._grid_cols, self._grid_rows = int(cols), int(rows)

    def _maybe_reseed(self, bbox: BBox, reason: str) -> None:
        # Reseed a fresh grid if points too few or bbox moved/scale changed significantly.
        self._pts = _make_grid_points(bbox, self._grid_cols, self._grid_rows)
        self._ages = np.zeros((self._pts.shape[0],), dtype=np.int32)

    def update(self, frame: np.ndarray, return_info: bool = False):
        if self._bbox is None or self._prev_gray is None or self._pts is None:
            if return_info:
                return False, (0.0, 0.0, 0.0, 0.0), {"reason": "not_initialized"}
            return False, (0.0, 0.0, 0.0, 0.0)

        gray = frame if self._expect_gray else _to_gray(frame)
        if gray.ndim != 2:
            gray = _to_gray(frame)
        gray = gray if gray.dtype == np.uint8 else gray.astype(np.uint8)

        prev_gray = self._prev_gray
        bbox = self._bbox
        pts0 = self._pts

        h, w = gray.shape[:2]
        bbox = _clip_bbox(bbox, w, h)

        # Template similarity (optional confidence)
        tmpl_sim = 1.0
        if self.p.template_enable and self._tmpl is not None:
            tmpl_sim = _template_sim(gray, bbox, self._tmpl, self.p.template_size)
        self._last_template_sim = float(tmpl_sim)

        # LK forward
        win = (int(self.p.lk_win_size), int(self.p.lk_win_size))
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, int(self.p.lk_max_iter), float(self.p.lk_eps))
        pts1, st_f, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, pts0, None,
                                                 winSize=win, maxLevel=int(self.p.lk_max_level),
                                                 criteria=criteria, flags=0, minEigThreshold=1e-4)
        if pts1 is None or st_f is None:
            self._last_conf = 0.0
            if return_info:
                return False, bbox, {"reason": "lk_forward_failed"}
            return False, bbox

        mask = st_f.reshape(-1).astype(bool)
        if mask.sum() < self.p.min_points:
            self._last_conf = 0.0
            # try reseed once
            self._maybe_reseed(bbox, "lk_few_points")
            self._prev_gray = gray.copy()
            if return_info:
                return False, bbox, {"reason": "lk_few_points"}
            return False, bbox

        p0 = pts0.reshape(-1, 2)[mask]
        p1 = pts1.reshape(-1, 2)[mask]

        # LK backward
        pts0r, st_b, _ = cv2.calcOpticalFlowPyrLK(gray, prev_gray, p1.reshape(-1, 1, 2), None,
                                                  winSize=win, maxLevel=int(self.p.lk_max_level),
                                                  criteria=criteria, flags=0, minEigThreshold=1e-4)
        if pts0r is None or st_b is None:
            self._last_conf = 0.0
            self._maybe_reseed(bbox, "lk_backward_failed")
            self._prev_gray = gray.copy()
            if return_info:
                return False, bbox, {"reason": "lk_backward_failed"}
            return False, bbox

        mask_b = st_b.reshape(-1).astype(bool)
        if mask_b.sum() < self.p.min_points:
            self._last_conf = 0.0
            self._maybe_reseed(bbox, "lk_backward_few")
            self._prev_gray = gray.copy()
            if return_info:
                return False, bbox, {"reason": "lk_backward_few"}
            return False, bbox

        p0 = p0[mask_b]
        p1 = p1[mask_b]
        p0r = pts0r.reshape(-1, 2)[mask_b]

        # FB error gate
        fb = np.linalg.norm(p0 - p0r, axis=1)
        med_fb = float(np.median(fb)) if fb.size else 1e9
        thr_fb = max(self.p.fb_min, self.p.fb_factor * med_fb)
        mask_fb = fb <= thr_fb
        if mask_fb.sum() < self.p.min_points:
            self._last_conf = 0.0
            self._maybe_reseed(bbox, "fb_few")
            self._prev_gray = gray.copy()
            if return_info:
                return False, bbox, {"reason": "fb_few"}
            return False, bbox

        p0 = p0[mask_fb]
        p1 = p1[mask_fb]

        # NCC gate (optional, numba-accelerated)
        ncc_keep = None
        if self.p.ncc_enable and p0.shape[0] >= self.p.min_points:
            half = int(self.p.ncc_half_window)
            px = p0[:, 0].astype(np.float32)
            py = p0[:, 1].astype(np.float32)
            cx = p1[:, 0].astype(np.float32)
            cy = p1[:, 1].astype(np.float32)

            scores = _ncc_scores_uint8(prev_gray, gray, px, py, cx, cy, half)
            valid = scores > -1.5
            if valid.sum() >= self.p.min_points:
                scores2 = scores[valid]
                idx_valid = np.nonzero(valid)[0]
                keep = scores2 >= self.p.ncc_min
                if keep.sum() < self.p.min_points:
                    # keep top fraction instead
                    k = max(self.p.min_points, int(math.ceil(self.p.ncc_keep_frac * scores2.size)))
                    order = np.argsort(scores2)[::-1]
                    sel = order[:k]
                    keep_idx = idx_valid[sel]
                else:
                    keep_idx = idx_valid[np.nonzero(keep)[0]]
                if keep_idx.size >= self.p.min_points:
                    p0 = p0[keep_idx]
                    p1 = p1[keep_idx]
                    ncc_keep = float(keep_idx.size) / float(valid.sum())
            # else: NCC not informative; skip

        # robust translation (median)
        d = p1 - p0
        dx = float(np.median(d[:, 0]))
        dy = float(np.median(d[:, 1]))

        # robust scale (median log ratio)
        scale_raw = 1.0
        scale_used = 1.0
        scale_log_mad = 0.0
        if self.p.scale_enable and p0.shape[0] >= self.p.min_points:
            cx0, cy0 = _bbox_center(bbox)
            r0 = np.linalg.norm(p0 - np.array([cx0, cy0], dtype=np.float32), axis=1)
            r1 = np.linalg.norm(p1 - np.array([cx0 + dx, cy0 + dy], dtype=np.float32), axis=1)
            mask_r = (r0 >= self.p.scale_min_rad) & (r1 >= self.p.scale_min_rad)
            if mask_r.sum() >= self.p.min_points:
                ratios = r1[mask_r] / (r0[mask_r] + 1e-6)
                log_r = np.log(np.clip(ratios, 1e-3, 1e3))
                med_log = float(np.median(log_r))
                mad = float(np.median(np.abs(log_r - med_log))) + 1e-12
                scale_log_mad = mad
                scale_raw = float(math.exp(med_log))
                if mad <= self.p.scale_log_mad_max:
                    # damping and clamp
                    s = 1.0 + self.p.scale_alpha * (scale_raw - 1.0)
                    s = float(np.clip(s, 1.0 / self.p.scale_clamp, self.p.scale_clamp))
                    scale_used = s

        # update bbox
        x, y, bw, bh = bbox
        cx1 = x + 0.5 * bw + dx
        cy1 = y + 0.5 * bh + dy
        bw1 = bw * scale_used
        bh1 = bh * scale_used
        bbox1 = (cx1 - 0.5 * bw1, cy1 - 0.5 * bh1, bw1, bh1)
        bbox1 = _clip_bbox(bbox1, w, h)

        # update confidence
        final_n = int(p0.shape[0])
        req_n = int(self._grid_cols * self._grid_rows)
        good_frac = float(final_n) / float(max(1, req_n))
        conf = min(1.0, good_frac / max(1e-6, self.p.shrink_if_good_frac_above))
        if self.p.template_enable and tmpl_sim >= 0:
            # mix in template similarity weakly
            conf = float(np.clip(0.5 * conf + 0.5 * max(0.0, tmpl_sim), 0.0, 1.0))
        self._last_conf = conf

        # optional hard fail on low template similarity
        if self.p.template_enable and self.p.fail_on_low_template and tmpl_sim >= 0 and tmpl_sim < self.p.template_min_sim:
            ok = False
        else:
            ok = True

        # adapt grid (even on ok=false, to help next frame)
        self._adapt_grid(good_frac, bbox1)

        # reseed if points too low (or bbox became tiny and grid changed)
        if final_n < self.p.reseed_if_points_below:
            self._maybe_reseed(bbox1, "few_points")
        else:
            # keep current tracked points for next frame (faster than reseed)
            self._pts = p1.reshape(-1, 1, 2).astype(np.float32)
            self._ages = None  # ages unused in perf build

        # update state
        self._bbox = bbox1
        self._prev_gray = gray.copy()

        if return_info:
            return ok, bbox1, {
                "final_points": final_n,
                "grid_cols": self._grid_cols,
                "grid_rows": self._grid_rows,
                "good_frac": good_frac,
                "dx": dx,
                "dy": dy,
                "scale_raw": scale_raw,
                "scale_used": scale_used,
                "scale_log_mad": scale_log_mad,
                "template_sim": tmpl_sim,
                "ncc_keep_frac": ncc_keep,
                "conf": conf,
            }
        return ok, bbox1


__all__ = ["DynMedianFlowTracker", "DynMFParams"]
