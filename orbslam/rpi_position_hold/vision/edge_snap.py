from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


@dataclass(frozen=True)
class EdgeSnapConfig:
    enabled: bool = False
    search_px: int = 12
    max_step_px: float = 2.0
    canny_lo: int = 60
    canny_hi: int = 140
    canny_L2gradient: bool = True
    blur_ksize: int = 5
    sobel_ksize: int = 3
    edge_bonus: float = 2000.0
    grad_weight: float = 1.0
    min_score_improve: float = 0.0


def _odd_ksize(k: int) -> int:
    try:
        k = int(k)
    except Exception:
        return 0
    if k <= 1:
        return 0
    if (k % 2) == 0:
        k += 1
    return int(max(3, k))


def edge_normal_snap_poly_uv(
    *,
    gray: np.ndarray,
    poly_uv: np.ndarray,
    cfg: EdgeSnapConfig,
) -> np.ndarray:
    """
    Micro-refine polygon vertices by searching along the local edge normal for a stronger edge response.

    This is VO-agnostic and safe to use in stage-1 refinement.

    Returns a new (N,2) float32 polygon.
    """
    if not bool(cfg.enabled):
        return np.asarray(poly_uv, dtype=np.float32).reshape(-1, 2)

    try:
        gray_u8 = np.asarray(gray, dtype=np.uint8)
    except Exception:
        return np.asarray(poly_uv, dtype=np.float32).reshape(-1, 2)
    if int(gray_u8.ndim) != 2 or int(gray_u8.size) <= 0:
        return np.asarray(poly_uv, dtype=np.float32).reshape(-1, 2)

    try:
        P = np.asarray(poly_uv, dtype=np.float32).reshape(-1, 2)
    except Exception:
        return np.asarray(poly_uv, dtype=np.float32).reshape(-1, 2)
    if int(P.shape[0]) < 3:
        return np.asarray(poly_uv, dtype=np.float32).reshape(-1, 2)

    h = int(gray_u8.shape[0])
    w = int(gray_u8.shape[1])
    if int(h) <= 1 or int(w) <= 1:
        return np.asarray(poly_uv, dtype=np.float32).reshape(-1, 2)

    search_px = int(max(0, int(cfg.search_px)))
    if int(search_px) <= 0:
        return np.asarray(poly_uv, dtype=np.float32).reshape(-1, 2)
    max_step = float(cfg.max_step_px)
    if (not np.isfinite(float(max_step))) or float(max_step) <= 0.0:
        return np.asarray(poly_uv, dtype=np.float32).reshape(-1, 2)

    blur_k = _odd_ksize(int(cfg.blur_ksize))
    sob_k = int(cfg.sobel_ksize)
    if sob_k not in (1, 3, 5, 7):
        sob_k = 3

    # Score map = gradient magnitude + bonus if Canny edge is present.
    try:
        g_in = gray_u8
        if int(blur_k) > 0:
            g_in = cv2.GaussianBlur(gray_u8, (int(blur_k), int(blur_k)), 0)
        gx = cv2.Sobel(g_in, cv2.CV_32F, 1, 0, ksize=int(sob_k))
        gy = cv2.Sobel(g_in, cv2.CV_32F, 0, 1, ksize=int(sob_k))
        gmag = (np.abs(np.asarray(gx, dtype=np.float32)) + np.abs(np.asarray(gy, dtype=np.float32))).astype(np.float32, copy=False)
    except Exception:
        gmag = None

    try:
        edges = cv2.Canny(
            g_in if "g_in" in locals() else gray_u8,
            int(cfg.canny_lo),
            int(cfg.canny_hi),
            L2gradient=bool(cfg.canny_L2gradient),
        )
        edge_on = (np.asarray(edges, dtype=np.uint8) != 0).astype(np.float32, copy=False)
    except Exception:
        edge_on = None

    if gmag is None and edge_on is None:
        return np.asarray(poly_uv, dtype=np.float32).reshape(-1, 2)

    if gmag is None:
        gmag = np.zeros((int(h), int(w)), dtype=np.float32)
    if edge_on is None:
        edge_on = np.zeros((int(h), int(w)), dtype=np.float32)

    score = float(cfg.grad_weight) * gmag + float(cfg.edge_bonus) * edge_on
    min_improve = float(cfg.min_score_improve) if np.isfinite(float(cfg.min_score_improve)) else 0.0

    out = np.asarray(P, dtype=np.float32).copy()
    n = int(out.shape[0])

    def _score_at(u: float, v: float) -> Optional[float]:
        try:
            x = int(round(float(u)))
            y = int(round(float(v)))
        except Exception:
            return None
        if x < 0 or y < 0 or x >= int(w) or y >= int(h):
            return None
        try:
            return float(score[int(y), int(x)])
        except Exception:
            return None

    for i in range(int(n)):
        p_prev = out[(i - 1) % n]
        p0 = out[i]
        p_next = out[(i + 1) % n]

        t = np.asarray(p_next - p_prev, dtype=np.float32).reshape(2)
        tn = float(np.hypot(float(t[0]), float(t[1])))
        if (not np.isfinite(tn)) or tn <= 1e-6:
            continue
        # Unit normal (direction ambiguity is fine because we search both sides).
        nx = -float(t[1]) / float(tn)
        ny = float(t[0]) / float(tn)
        if not (np.isfinite(nx) and np.isfinite(ny)):
            continue

        s0 = _score_at(float(p0[0]), float(p0[1]))
        if s0 is None:
            continue

        best_s = float(s0)
        best_t = 0.0
        # Search integer offsets along the normal.
        for dt in range(-int(search_px), int(search_px) + 1):
            if dt == 0:
                continue
            u = float(p0[0]) + float(dt) * float(nx)
            v = float(p0[1]) + float(dt) * float(ny)
            s = _score_at(u, v)
            if s is None:
                continue
            if float(s) > float(best_s):
                best_s = float(s)
                best_t = float(dt)

        if float(best_s) < float(s0) + float(min_improve):
            continue

        # Strict per-update displacement limit.
        if float(best_t) > float(max_step):
            best_t = float(max_step)
        elif float(best_t) < -float(max_step):
            best_t = -float(max_step)

        out[i, 0] = float(p0[0]) + float(best_t) * float(nx)
        out[i, 1] = float(p0[1]) + float(best_t) * float(ny)

    # Clamp to image bounds.
    try:
        out[:, 0] = np.clip(out[:, 0], 0.0, float(max(0, int(w - 1))))
        out[:, 1] = np.clip(out[:, 1], 0.0, float(max(0, int(h - 1))))
    except Exception:
        pass
    return np.asarray(out, dtype=np.float32).reshape(-1, 2)

