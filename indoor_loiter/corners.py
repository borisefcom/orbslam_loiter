#!/usr/bin/env python3
# corners.py - simple background VO using LK + homography.

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, List

import numpy as np
import cv2

from tracker_types import FramePacket, bbox_from_quad


@dataclass
class BgStats:
    n_pts: int = 0
    inliers: int = 0
    inlier_ratio: float = 0.0
    spread: float = 0.0
    solver: str = "NONE"
    healthy: bool = False
    reproj_err: float = 0.0
    health_score: float = 0.0
    adapt_budget: float = 0.0


class CornerHandler:
    """
    Background corner tracker:
      - detect ORB corners
      - track with LK
      - fit homography via RANSAC
      - mark health based on points/inliers/spread
      - maintain a pool of tracked corners across frames
    """

    def __init__(self, cfg) -> None:
        bc = cfg.get("background", {}) or {}

        self.max_corners = int(bc.get("orb_nfeatures", bc.get("max_corners", 800)))

        self.lk_win = int(bc.get("lk_win", 21))
        self.lk_levels = int(bc.get("lk_levels", 3))
        self.lk_iters = int(bc.get("lk_iters", 30))
        self.lk_eps = float(bc.get("lk_eps", 0.01))
        self.fb_enable = bool(bc.get("fb_enable", True))
        self.fb_thresh = float(bc.get("fb_thresh", 3.0))

        self.h_ransac_thresh_cfg = float(bc.get("h_ransac_thresh", 4.0))
        self.h_confidence_cfg = float(bc.get("h_confidence", 0.995))
        self.h_max_iters_cfg = int(bc.get("h_max_iters", 2500))

        self.min_inliers = int(bc.get("h_min_inliers", 30))
        self.min_inlier_ratio = float(bc.get("h_min_inlier_ratio", 0.5))
        self.min_spread = float(bc.get("h_min_spread", 0.2))
        self.health_alpha = float(bc.get("health_alpha", 0.2))
        self.health_err_max = float(bc.get("health_err_max", 3.0))  # pixels
        self.health_thresh = float(bc.get("health_thresh", 0.5))
        self.health_score = 0.0

        # Corner pool (ecosystem)
        self.pool_max = int(bc.get("corner_pool_size", 600))
        self.pool_min_dist = float(bc.get("corner_pool_min_dist", 4.0))
        self.pool_pts: Optional[np.ndarray] = None  # Nx2 float32, positions in prev frame
        self.pool_age: Optional[np.ndarray] = None  # N int, age in frames
        self.pool_score: Optional[np.ndarray] = None  # N float, detection score

        # Adaptive throttle settings
        self.adaptive_enable = bool(bc.get("adaptive_enable", True))
        self.adapt_floor = float(bc.get("adaptive_min_budget", 0.0))
        self.adapt_ceil = float(bc.get("adaptive_max_budget", 1.0))
        self.adapt_rate = float(bc.get("adaptive_rate", 0.1))  # max change per frame
        # Start adaptive VO in the mid-range budget so we ramp up/down quickly.
        self._budget = 0.5 if self.adaptive_enable else 1.0
        self._prev_budget = self._budget
        self._profile_label = self._profile_from_budget(self._budget)

        # Baseline (High) settings from config
        self.base_orb = self.max_corners
        self.base_pool = self.pool_max
        self.base_lk_win = self.lk_win
        self.base_lk_levels = self.lk_levels
        self.base_lk_iters = self.lk_iters
        self.base_h_iters = self.h_max_iters_cfg
        self.base_h_conf = self.h_confidence_cfg

        # Low/Med/High anchors
        self.low_orb = int(bc.get("adaptive_low_orb", 300))
        self.med_orb = int(bc.get("adaptive_med_orb", 600))
        self.high_orb = self.base_orb

        self.low_pool = int(bc.get("adaptive_low_pool", 350))
        self.med_pool = int(bc.get("adaptive_med_pool", 500))
        self.high_pool = self.base_pool

        self.low_lk_win = int(bc.get("adaptive_low_lk_win", 15))
        self.med_lk_win = int(bc.get("adaptive_med_lk_win", 19))
        self.high_lk_win = self.base_lk_win

        self.low_lk_levels = int(bc.get("adaptive_low_lk_levels", 2))
        self.med_lk_levels = int(bc.get("adaptive_med_lk_levels", 2))
        self.high_lk_levels = self.base_lk_levels

        self.low_lk_iters = int(bc.get("adaptive_low_lk_iters", 15))
        self.med_lk_iters = int(bc.get("adaptive_med_lk_iters", 25))
        self.high_lk_iters = self.base_lk_iters

        self.low_h_iters = int(bc.get("adaptive_low_h_iters", 500))
        self.med_h_iters = int(bc.get("adaptive_med_h_iters", 1200))
        self.high_h_iters = self.base_h_iters

        self.low_h_conf = float(bc.get("adaptive_low_h_conf", 0.93))
        self.med_h_conf = float(bc.get("adaptive_med_h_conf", 0.98))
        self.high_h_conf = self.base_h_conf

        # Preprocess and ORB detector
        self.sharpen_enable = bool(bc.get("sharpen_enable", False))
        self.sharpen_amount = float(bc.get("sharpen_amount", 0.3))
        self.sharpen_sigma = float(bc.get("sharpen_sigma", 1.0))
        self.obj_feat_min_dist = float(bc.get("object_feature_min_dist", 2.0))
        self.obj_feat_max = int(bc.get("object_feature_max", 60))
        try:
            self.orb = cv2.ORB_create(nfeatures=self.max_corners)
        except Exception:
            self.orb = None

    # -----------------------------------------------------------------

    def process(
        self,
        prev_fp: Optional[FramePacket],
        fp: FramePacket,
        ghost_quad: Optional[np.ndarray] = None,
    ):
        """
        Compute background homography from prev_fp -> fp.

        Returns:
          H_vo      : 3x3 homography or None
          stats     : BgStats
          tracks    : []  (placeholder)
          sfrs_mask : None
          bg_points : list of [x, y] in current frame
        """
        if prev_fp is None:
            # No previous frame; seed pool from current frame for next iteration.
            self._seed_pool(fp.gray, ghost_quad, fp.bgr if hasattr(fp, "bgr") else None)
            return None, BgStats(), [], None, []

        prev_gray = prev_fp.gray
        curr_gray = fp.gray
        H_img, W_img = prev_gray.shape[:2]

        # Ensure pool exists; if empty, seed from prev frame.
        if self.pool_pts is None or self.pool_pts.shape[0] == 0:
            self._seed_pool(prev_gray, ghost_quad, prev_fp.bgr if hasattr(prev_fp, "bgr") else None)
        if self.pool_pts is None or self.pool_pts.shape[0] == 0:
            self._seed_pool(curr_gray, ghost_quad, fp.bgr if hasattr(fp, "bgr") else None)
            return None, BgStats(), [], None, []

        # Track pooled corners forward; drop any corners that fall inside the current object mask.
        p0_pool = self.pool_pts.astype(np.float32)
        n_pool = p0_pool.shape[0]
        ages = self.pool_age.astype(np.int32) if (self.pool_age is not None and self.pool_age.shape[0] == n_pool) else np.zeros((n_pool,), dtype=np.int32)
        scores = self.pool_score.astype(np.float32) if (self.pool_score is not None and self.pool_score.shape[0] == n_pool) else np.zeros((n_pool,), dtype=np.float32)
        if ghost_quad is not None and p0_pool.shape[0] > 0:
            x, y, w, h = bbox_from_quad(ghost_quad)
            inside = (
                (p0_pool[:, 0] >= x)
                & (p0_pool[:, 0] <= (x + w))
                & (p0_pool[:, 1] >= y)
                & (p0_pool[:, 1] <= (y + h))
            )
            if np.any(inside):
                keep_mask = ~inside
                p0_pool = p0_pool[keep_mask]
                ages = ages[keep_mask]
                scores = scores[keep_mask]
                if p0_pool.shape[0] == 0:
                    self.pool_pts = np.zeros((0, 2), dtype=np.float32)
                    self.pool_age = np.zeros((0,), dtype=np.int32)
                    self.pool_score = np.zeros((0,), dtype=np.float32)
                    self._top_up_pool(prev_gray, ghost_quad, prev_fp.bgr if hasattr(prev_fp, "bgr") else None)
                    self._top_up_pool(curr_gray, ghost_quad, fp.bgr if hasattr(fp, "bgr") else None)
                    return None, BgStats(), [], None, []

        lk_criteria = (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            self.lk_iters,
            self.lk_eps,
        )

        # Track all pooled points to update state; homography uses mature ones (age > 3).
        if p0_pool.shape[0] == 0:
            self._top_up_pool(prev_gray, ghost_quad, prev_fp.bgr if hasattr(prev_fp, "bgr") else None)
            self._top_up_pool(curr_gray, ghost_quad, fp.bgr if hasattr(fp, "bgr") else None)
            return None, BgStats(), [], None, []

        p1_pool, st_pool, err_pool = cv2.calcOpticalFlowPyrLK(
            prev_gray,
            curr_gray,
            p0_pool.reshape(-1, 1, 2),
            None,
            winSize=(self.lk_win, self.lk_win),
            maxLevel=self.lk_levels,
            criteria=lk_criteria,
            flags=cv2.OPTFLOW_LK_GET_MIN_EIGENVALS,
            minEigThreshold=1e-4,
        )
        if p1_pool is None or st_pool is None:
            self._top_up_pool(prev_gray, ghost_quad, prev_fp.bgr if hasattr(prev_fp, "bgr") else None)
            self._top_up_pool(curr_gray, ghost_quad, fp.bgr if hasattr(fp, "bgr") else None)
            return None, BgStats(), [], None, []
        st_pool = st_pool.reshape(-1)
        p1_pool = p1_pool.reshape(-1, 2)

        # Filter out failed tracks or out-of-bounds
        in_bounds = (
            (st_pool == 1)
            & np.isfinite(p1_pool).all(axis=1)
            & (p1_pool[:, 0] >= 0)
            & (p1_pool[:, 0] < W_img)
            & (p1_pool[:, 1] >= 0)
            & (p1_pool[:, 1] < H_img)
        )

        p0_good = p0_pool[in_bounds]
        p1_good = p1_pool[in_bounds]
        age_good = ages[in_bounds] + 1  # age forward
        score_good = scores[in_bounds]

        # Update pool with surviving tracks
        self.pool_pts = p1_good.copy()
        self.pool_age = age_good.copy()
        self.pool_score = score_good.copy()

        # Select mature points for homography (age > 3)
        mature_mask = age_good > 3
        p0_mature = p0_good[mature_mask]
        p1_mature = p1_good[mature_mask]
        n_pts = int(p0_mature.shape[0])
        if n_pts < 8:
            # Even if not enough points, keep pool updated and try to reseed
            self._top_up_pool(curr_gray, ghost_quad, fp.bgr if hasattr(fp, "bgr") else None)
            return None, BgStats(n_pts=n_pts), [], None, []

        def _run_homography(max_iters, conf):
            try:
                H, mask = cv2.findHomography(
                    p0_mature,
                    p1_mature,
                    method=getattr(cv2, "USAC_MAGSAC", cv2.RANSAC),
                    ransacReprojThreshold=self.h_ransac_thresh_cfg,
                    maxIters=max_iters,
                    confidence=conf,
                )
            except Exception:
                H = None
                mask = None
            if H is None or mask is None:
                try:
                    H, mask = cv2.findHomography(
                        p0_mature,
                        p1_mature,
                        method=cv2.RANSAC,
                        ransacReprojThreshold=self.h_ransac_thresh_cfg,
                        maxIters=max_iters,
                        confidence=conf,
                    )
                except Exception:
                    H = None
                    mask = None
            return H, mask

        H_vo, inlier_mask = _run_homography(self.h_max_iters_cfg, self.h_confidence_cfg)

        # If homography failed and we just escalated budget, retry with high settings.
        budget_jump = (self._budget - getattr(self, "_prev_budget", self._budget)) > 1e-3
        if (H_vo is None or inlier_mask is None) and self.adaptive_enable and budget_jump:
            H_vo, inlier_mask = _run_homography(self.high_h_iters, self.high_h_conf)

        if H_vo is None or inlier_mask is None:
            self._top_up_pool(curr_gray, ghost_quad, fp.bgr if hasattr(fp, "bgr") else None)
            return None, BgStats(n_pts=n_pts, solver="H_FAIL"), [], None, []

        inlier_mask = inlier_mask.reshape(-1).astype(bool)
        inliers = int(inlier_mask.sum())
        inlier_ratio = float(inliers) / float(max(1, n_pts))

        if inliers > 0:
            pts_in = p0_mature[inlier_mask]
            xs = pts_in[:, 0]
            ys = pts_in[:, 1]
            span_x = (xs.max() - xs.min()) / max(1.0, float(W_img))
            span_y = (ys.max() - ys.min()) / max(1.0, float(H_img))
            spread = float(max(span_x, span_y))
        else:
            spread = 0.0

        # Reprojection error on inliers
        if inliers > 0:
            try:
                p0_h = p0_mature[inlier_mask].reshape(-1, 1, 2).astype(np.float32)
                p1_est = cv2.perspectiveTransform(p0_h, H_vo).reshape(-1, 2)
                err = np.linalg.norm(p1_est - p1_mature[inlier_mask], axis=1)
                reproj_err = float(np.median(err)) if err.size else float("inf")
            except Exception:
                reproj_err = float("inf")
        else:
            reproj_err = float("inf")

        # Instantaneous health score (0..1) from reprojection error and spread
        counts_ok = (inliers >= self.min_inliers) and (inlier_ratio >= self.min_inlier_ratio)
        spread_score = float(np.clip(spread / max(1e-6, self.min_spread), 0.0, 1.0))
        if not np.isfinite(reproj_err) or reproj_err < 0.0:
            err_score = 0.0
        else:
            err_score = float(np.clip(1.0 - reproj_err / max(1e-6, self.health_err_max), 0.0, 1.0))
        inst_score = (spread_score * err_score) if counts_ok else 0.0

        # EMA over time for stability; hard fail on invalid reprojection
        alpha = float(np.clip(self.health_alpha, 0.0, 1.0))
        if err_score == 0.0:
            self.health_score = 0.0
        else:
            self.health_score = (1.0 - alpha) * self.health_score + alpha * inst_score
        healthy = bool(self.health_score >= self.health_thresh)

        # Adaptive budget update (continuous, rate-limited)
        if self.adaptive_enable:
            target_budget = float(np.clip(self.health_score, self.adapt_floor, self.adapt_ceil))
            delta = target_budget - self._budget
            max_step = float(np.clip(self.adapt_rate, 0.0, 1.0))
            delta = float(np.clip(delta, -max_step, max_step))
            self._budget = float(np.clip(self._budget + delta, self.adapt_floor, self.adapt_ceil))
            b = self._budget
            # Piecewise-linear between low/med/high anchors
            def lerp(a, b0, b1, t0=0.0, t1=1.0):
                t = (b - t0) / max(1e-6, (t1 - t0))
                t = float(np.clip(t, 0.0, 1.0))
                return a + (b1 - a) * t
            # Two segments: [0,0.5]->low->med, [0.5,1]->med->high
            seg = 0.5
            def blend(lo, me, hi):
                if b <= seg:
                    return lerp(lo, 0.0, seg)
                return lerp(me, seg, 1.0) + (hi - me) * max(0.0, b - seg) / max(1e-6, 1.0 - seg)

            self.max_corners = int(blend(self.low_orb, self.med_orb, self.high_orb))
            self.pool_max = int(blend(self.low_pool, self.med_pool, self.high_pool))
            self.lk_win = int(blend(self.low_lk_win, self.med_lk_win, self.high_lk_win))
            self.lk_levels = int(blend(self.low_lk_levels, self.med_lk_levels, self.high_lk_levels))
            self.lk_iters = int(blend(self.low_lk_iters, self.med_lk_iters, self.high_lk_iters))
            self.h_max_iters_cfg = int(blend(self.low_h_iters, self.med_h_iters, self.high_h_iters))
            self.h_confidence_cfg = float(blend(self.low_h_conf, self.med_h_conf, self.high_h_conf))
            profile = self._profile_from_budget(b)
            if profile != self._profile_label:
                self._profile_label = profile
                try:
                    print(f"[VO][adaptive] profile -> {profile} (budget={b:.2f})", flush=True)
                except Exception:
                    pass

        stats = BgStats(
            n_pts=n_pts,
            inliers=inliers,
            inlier_ratio=inlier_ratio,
            spread=spread,
            solver="H_RANSAC",
            healthy=healthy,
            reproj_err=reproj_err if np.isfinite(reproj_err) else 0.0,
            health_score=float(self.health_score),
            adapt_budget=float(self._budget),
        )

        # Track budget for next-frame comparison (used for retries on escalation).
        self._prev_budget = self._budget

        bg_points = p1_mature[inlier_mask].tolist()
        tracks: List = []
        sfrs_mask = None

        # Top up pool with new detections on current frame for next iteration
        self._top_up_pool(curr_gray, ghost_quad, fp.bgr if hasattr(fp, "bgr") else None)

        return H_vo, stats, tracks, sfrs_mask, bg_points

    # -----------------------------------------------------------------
    # Pool management helpers
    # -----------------------------------------------------------------

    def _seed_pool(self, gray: np.ndarray, ghost_quad: Optional[np.ndarray], bgr: Optional[np.ndarray] = None) -> None:
        """Initial populate of the corner pool from a single frame."""
        cand, scores = self._detect_corners(gray, ghost_quad, bgr)
        if cand is None or len(cand) == 0:
            self.pool_pts = np.zeros((0, 2), dtype=np.float32)
            self.pool_age = np.zeros((0,), dtype=np.int32)
            self.pool_score = np.zeros((0,), dtype=np.float32)
            return
        take = min(self.pool_max, cand.shape[0])
        self.pool_pts = cand[:take].astype(np.float32)
        self.pool_age = np.zeros((take,), dtype=np.int32)
        self.pool_score = scores[:take].astype(np.float32)

    def _top_up_pool(self, gray: np.ndarray, ghost_quad: Optional[np.ndarray], bgr: Optional[np.ndarray] = None) -> None:
        """Add new corners to the pool, enforcing min distance and max size."""
        if self.pool_pts is None:
            self._seed_pool(gray, ghost_quad, bgr)
            return
        cand, scores = self._detect_corners(gray, ghost_quad, bgr)
        if cand is None or len(cand) == 0:
            return

        existing = self.pool_pts
        keep_pts = []
        keep_scores = []
        for pt, sc in zip(cand, scores):
            if existing.shape[0] > 0:
                d2 = np.sum((existing - pt) ** 2, axis=1)
                if np.any(d2 < (self.pool_min_dist ** 2)):
                    continue
            keep_pts.append(pt)
            keep_scores.append(sc)
            if (existing.shape[0] + len(keep_pts)) >= self.pool_max:
                break
        if not keep_pts:
            return
        new_pts = np.asarray(keep_pts, dtype=np.float32)
        new_scores = np.asarray(keep_scores, dtype=np.float32)
        new_age = np.zeros((new_pts.shape[0],), dtype=np.int32)

        self.pool_pts = np.vstack([existing, new_pts]) if existing.size else new_pts
        self.pool_score = np.concatenate([self.pool_score, new_scores]) if self.pool_score is not None else new_scores
        self.pool_age = np.concatenate([self.pool_age, new_age]) if self.pool_age is not None else new_age

        # If we exceed capacity, keep highest-score corners
        if self.pool_pts.shape[0] > self.pool_max:
            order = np.argsort(self.pool_score)[::-1]
            order = order[: self.pool_max]
            self.pool_pts = self.pool_pts[order]
            self.pool_score = self.pool_score[order]
            self.pool_age = self.pool_age[order]

    def _detect_corners(
        self, gray: np.ndarray, ghost_quad: Optional[np.ndarray], bgr: Optional[np.ndarray] = None
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Detect candidates using ORB responses (masked) and return points + scores."""
        gray_proc = self._preprocess_gray(bgr, gray)
        H_img, W_img = gray.shape[:2]
        mask = np.ones_like(gray, dtype=np.uint8)
        if ghost_quad is not None:
            x, y, w, h = bbox_from_quad(ghost_quad)
            x0 = max(0, int(np.floor(x)))
            y0 = max(0, int(np.floor(y)))
            x1 = min(W_img - 1, int(np.ceil(x + w)))
            y1 = min(H_img - 1, int(np.ceil(y + h)))
            mask[y0:y1 + 1, x0:x1 + 1] = 0

        pts, scores = self._detect_features(gray_proc, mask)
        return pts, scores

    def _preprocess_gray(self, bgr: Optional[np.ndarray], gray_fallback: np.ndarray) -> np.ndarray:
        """Sharpen and convert to gray; falls back to provided gray."""
        gray = gray_fallback
        if bgr is None or not self.sharpen_enable:
            return gray
        try:
            lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
            L, A, B = cv2.split(lab)
            blur = cv2.GaussianBlur(L, ksize=(0, 0), sigmaX=self.sharpen_sigma)
            sharp_L = cv2.addWeighted(L, 1.0 + self.sharpen_amount, blur, -self.sharpen_amount, 0)
            sharp_L = np.clip(sharp_L, 0, 255).astype(np.uint8)
            lab_sharp = cv2.merge([sharp_L, A, B])
            bgr_sharp = cv2.cvtColor(lab_sharp, cv2.COLOR_LAB2BGR)
            gray = cv2.cvtColor(bgr_sharp, cv2.COLOR_BGR2GRAY)
        except Exception:
            gray = gray_fallback
        return gray

    def _detect_features(self, gray: np.ndarray, mask: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Detect ORB features only."""
        if self.orb is not None:
            try:
                kps = self.orb.detect(gray, mask)
            except Exception:
                kps = None
            if kps:
                pts = np.array([[kp.pt[0], kp.pt[1]] for kp in kps], dtype=np.float32)
                scores = np.array([kp.response for kp in kps], dtype=np.float32)
                order = np.argsort(scores)[::-1]
                return pts[order], scores[order]

        return None, None

    # -----------------------------------------------------------------
    # Object feature helper (ORB only)
    # -----------------------------------------------------------------
    def object_features(self, gray: np.ndarray, ghost_quad: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """Detect object features within ghost_quad using ORB only."""
        if ghost_quad is None or self.orb is None:
            return None
        H_img, W_img = gray.shape[:2]
        mask = np.zeros((H_img, W_img), dtype=np.uint8)
        x, y, w, h = bbox_from_quad(ghost_quad)
        x0 = max(0, int(np.floor(x)))
        y0 = max(0, int(np.floor(y)))
        x1 = min(W_img - 1, int(np.ceil(x + w)))
        y1 = min(H_img - 1, int(np.ceil(y + h)))
        mask[y0:y1 + 1, x0:x1 + 1] = 255

        gray_proc = gray
        try:
            if self.sharpen_enable:
                blur = cv2.GaussianBlur(gray, ksize=(0, 0), sigmaX=self.sharpen_sigma)
                gray_proc = cv2.addWeighted(gray, 1.0 + self.sharpen_amount, blur, -self.sharpen_amount, 0)
                gray_proc = np.clip(gray_proc, 0, 255).astype(np.uint8)
        except Exception:
            gray_proc = gray

        try:
            kps = self.orb.detect(gray_proc, mask)
        except Exception:
            return None
        if not kps:
            return None

        pts = np.array([[kp.pt[0], kp.pt[1]] for kp in kps], dtype=np.float32)
        scores = np.array([kp.response for kp in kps], dtype=np.float32)
        order = np.argsort(scores)[::-1]
        pts = pts[order]
        selected = []
        for pt in pts:
            if len(selected) >= self.obj_feat_max:
                break
            if selected:
                d2 = np.sum((np.array(selected) - pt) ** 2, axis=1)
                if np.any(d2 < (self.obj_feat_min_dist ** 2)):
                    continue
            selected.append(pt)
        if not selected:
            return None
        return np.array(selected, dtype=np.float32)

    def _profile_from_budget(self, budget: float) -> str:
        """Map budget in [0,1] to a coarse profile label for debug logs."""
        b = float(np.clip(budget, 0.0, 1.0))
        if b < 0.33:
            return "low"
        if b < 0.67:
            return "med"
        return "high"
