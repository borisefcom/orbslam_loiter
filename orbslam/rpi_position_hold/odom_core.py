"""Depth sampling helpers (subset).

This file intentionally contains only the small helpers needed by the polygon
tracking subprocess (`processes/poly_vo_lk_track.py`).
"""

from __future__ import annotations

import numpy as np

try:
    from numba import njit  # type: ignore

    _HAVE_NUMBA = True
except Exception:
    njit = None  # type: ignore
    _HAVE_NUMBA = False


if _HAVE_NUMBA:

    @njit(cache=True)  # type: ignore[misc]
    def _nb_sample_depth_stats_u16(depth_raw: np.ndarray, x: int, y: int, r: int) -> tuple[int, float, int]:
        h = int(depth_raw.shape[0])
        w = int(depth_raw.shape[1])
        x0 = x
        y0 = y
        if x0 < 0:
            x0 = 0
        elif x0 >= w:
            x0 = w - 1
        if y0 < 0:
            y0 = 0
        elif y0 >= h:
            y0 = h - 1
        rr = int(r)
        x_start = x0 - rr
        if x_start < 0:
            x_start = 0
        x_end = x0 + rr + 1
        if x_end > w:
            x_end = w
        y_start = y0 - rr
        if y_start < 0:
            y_start = 0
        y_end = y0 + rr + 1
        if y_end > h:
            y_end = h
        max_n = int(x_end - x_start) * int(y_end - y_start)
        if max_n <= 0:
            return 0, 0.0, 0
        vals = np.empty(max_n, dtype=np.int64)
        n = 0
        for yy in range(y_start, y_end):
            for xx in range(x_start, x_end):
                v = int(depth_raw[int(yy), int(xx)])
                if v <= 0:
                    continue
                vals[int(n)] = int(v)
                n += 1
        if n <= 0:
            return 0, 0.0, 0
        arr = vals[:n].astype(np.float64)
        med = float(np.median(arr))
        mad = float(np.median(np.abs(arr - float(med))))
        return int(round(med)), float(mad), int(n)


def _sample_depth_stats_u16(depth_raw: np.ndarray, x: int, y: int, r: int = 1) -> tuple[int, float, int]:
    """Return (median_depth_u16, MAD_u16, n_valid) in a radius-r patch."""
    if depth_raw is None or not isinstance(depth_raw, np.ndarray) or depth_raw.ndim != 2:
        return 0, 0.0, 0
    if _HAVE_NUMBA:
        try:
            return _nb_sample_depth_stats_u16(depth_raw, int(x), int(y), int(r))  # type: ignore[name-defined]
        except Exception:
            pass
    h, w = depth_raw.shape[:2]
    x0 = max(0, min(w - 1, int(x)))
    y0 = max(0, min(h - 1, int(y)))
    rr = int(max(0, int(r)))
    x1 = max(0, min(w, x0 + rr + 1))
    y1 = max(0, min(h, y0 + rr + 1))
    x0 = max(0, x0 - rr)
    y0 = max(0, y0 - rr)
    patch = depth_raw[y0:y1, x0:x1].reshape(-1)
    if patch.size == 0:
        return 0, 0.0, 0
    patch = patch[patch > 0]
    if patch.size == 0:
        return 0, 0.0, 0
    try:
        med = float(np.median(patch))
        mad = float(np.median(np.abs(patch.astype(np.float64) - float(med))))
        return int(round(med)), float(mad), int(patch.size)
    except Exception:
        return 0, 0.0, int(patch.size)

