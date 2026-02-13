from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


@dataclass(frozen=True)
class BandMaskConfig:
    # How to construct the band relative to the polygon interior.
    # - "both": ring straddling the boundary (dilate - erode)
    # - "outside": outside-only band (dilate - fill)
    # - "inside": inside-only band (fill - erode)
    mode: str = "both"
    outer_px: int = 16
    inner_px: int = 8


@dataclass(frozen=True)
class OrbAcquireConfig:
    enabled: bool = False

    # ORB parameters
    nfeatures: int = 600
    scaleFactor: float = 1.2
    nlevels: int = 8
    edgeThreshold: int = 31
    firstLevel: int = 0
    WTA_K: int = 2
    scoreType: int = 0  # 0=HARRIS_SCORE, 1=FAST_SCORE
    patchSize: int = 31
    fastThreshold: int = 20

    # Matching / pose estimate parameters
    max_matches: int = 200
    ratio_test: float = 0.75
    ransac_reproj_thresh_px: float = 3.0
    ransac_confidence: float = 0.99
    min_inliers: int = 12


@dataclass(frozen=True)
class LineAcquireConfig:
    enabled: bool = False
    min_length_px: float = 25.0
    # LSD refine mode: 0=none, 1=std, 2=adv (OpenCV constants differ by build; keep int)
    refine: int = 1


@dataclass(frozen=True)
class FeatureAcquireConfig:
    enabled: bool = False
    band: BandMaskConfig = BandMaskConfig()
    orb: OrbAcquireConfig = OrbAcquireConfig()
    lines: LineAcquireConfig = LineAcquireConfig()


@dataclass(frozen=True)
class FeatureSnapshot:
    poly_uv: np.ndarray  # (N,2) float32
    band_mask_u8: Optional[np.ndarray]  # (H,W) uint8 {0,255}

    orb_uv: Optional[np.ndarray]  # (K,2) float32
    orb_desc: Optional[np.ndarray]  # (K,32) uint8

    lines_xyxy: Optional[np.ndarray]  # (M,4) float32


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


def poly_band_mask_u8(*, gray: np.ndarray, poly_uv: np.ndarray, cfg: BandMaskConfig) -> Optional[np.ndarray]:
    try:
        g = np.asarray(gray, dtype=np.uint8)
    except Exception:
        return None
    if int(g.ndim) != 2 or int(g.size) <= 0:
        return None
    h = int(g.shape[0])
    w = int(g.shape[1])
    if int(h) <= 1 or int(w) <= 1:
        return None

    try:
        P = np.asarray(poly_uv, dtype=np.float32).reshape(-1, 2)
    except Exception:
        return None
    if int(P.shape[0]) < 3:
        return None

    try:
        poly_i = np.round(P).astype(np.int32).reshape(-1, 1, 2)
    except Exception:
        return None

    fill = np.zeros((int(h), int(w)), dtype=np.uint8)
    try:
        cv2.fillPoly(fill, [poly_i], 255)
    except Exception:
        return None

    outer_px = int(max(0, int(cfg.outer_px)))
    inner_px = int(max(0, int(cfg.inner_px)))
    if int(outer_px) <= 0 and int(inner_px) <= 0:
        return fill

    def _dilate(mask: np.ndarray, r: int) -> np.ndarray:
        if int(r) <= 0:
            return mask
        k = int(2 * int(r) + 1)
        k = int(min(255, max(3, k)))
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(k), int(k)))
        return cv2.dilate(mask, ker, iterations=1)

    def _erode(mask: np.ndarray, r: int) -> np.ndarray:
        if int(r) <= 0:
            return mask
        k = int(2 * int(r) + 1)
        k = int(min(255, max(3, k)))
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(k), int(k)))
        return cv2.erode(mask, ker, iterations=1)

    fill_d = _dilate(fill, int(outer_px))
    fill_e = _erode(fill, int(inner_px))
    mode = str(cfg.mode or "both").strip().lower()
    if mode == "outside":
        band = cv2.subtract(fill_d, fill)
    elif mode == "inside":
        band = cv2.subtract(fill, fill_e)
    else:
        band = cv2.subtract(fill_d, fill_e)
    return np.asarray(band, dtype=np.uint8)


def _orb_create(cfg: OrbAcquireConfig):
    try:
        return cv2.ORB_create(
            nfeatures=int(max(0, int(cfg.nfeatures))),
            scaleFactor=float(cfg.scaleFactor),
            nlevels=int(max(1, int(cfg.nlevels))),
            edgeThreshold=int(max(0, int(cfg.edgeThreshold))),
            firstLevel=int(max(0, int(cfg.firstLevel))),
            WTA_K=int(max(2, int(cfg.WTA_K))),
            scoreType=int(cfg.scoreType),
            patchSize=int(max(0, int(cfg.patchSize))),
            fastThreshold=int(max(0, int(cfg.fastThreshold))),
        )
    except Exception:
        return None


def _lsd_create(cfg: LineAcquireConfig):
    try:
        return cv2.createLineSegmentDetector(int(cfg.refine))
    except Exception:
        try:
            return cv2.createLineSegmentDetector()
        except Exception:
            return None


class PolyFeatureAcquirer:
    def __init__(self, cfg: FeatureAcquireConfig):
        self.cfg = cfg
        self._orb = _orb_create(cfg.orb) if bool(cfg.enabled and cfg.orb.enabled) else None
        self._lsd = _lsd_create(cfg.lines) if bool(cfg.enabled and cfg.lines.enabled) else None
        self._bf = None
        if self._orb is not None:
            try:
                self._bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
            except Exception:
                self._bf = None

    def acquire(self, *, gray: np.ndarray, poly_uv: np.ndarray) -> FeatureSnapshot:
        P = np.asarray(poly_uv, dtype=np.float32).reshape(-1, 2)
        band = poly_band_mask_u8(gray=np.asarray(gray, dtype=np.uint8), poly_uv=P, cfg=self.cfg.band) if bool(self.cfg.enabled) else None

        orb_uv = None
        orb_desc = None
        if self._orb is not None:
            try:
                kps, desc = self._orb.detectAndCompute(np.asarray(gray, dtype=np.uint8), band)
            except Exception:
                kps, desc = None, None
            if kps is not None and desc is not None and int(len(kps)) > 0 and int(getattr(desc, "shape", (0,))[0]) == int(len(kps)):
                try:
                    uv = np.asarray([kp.pt for kp in kps], dtype=np.float32).reshape(-1, 2)
                except Exception:
                    uv = None
                if uv is not None:
                    orb_uv = uv
                    orb_desc = np.asarray(desc, dtype=np.uint8)

        lines_xyxy = None
        if self._lsd is not None:
            try:
                lines = self._lsd.detect(np.asarray(gray, dtype=np.uint8))[0]
            except Exception:
                lines = None
            if lines is not None and int(np.asarray(lines).size) > 0:
                try:
                    L = np.asarray(lines, dtype=np.float32).reshape(-1, 4)
                    x1, y1, x2, y2 = L[:, 0], L[:, 1], L[:, 2], L[:, 3]
                    ll = np.hypot(x2 - x1, y2 - y1)
                    keep = np.isfinite(ll) & (ll >= float(self.cfg.lines.min_length_px))
                    L = L[keep]
                    lines_xyxy = np.asarray(L, dtype=np.float32).reshape(-1, 4) if int(L.shape[0]) > 0 else None
                except Exception:
                    lines_xyxy = None

        return FeatureSnapshot(
            poly_uv=P,
            band_mask_u8=band,
            orb_uv=orb_uv,
            orb_desc=orb_desc,
            lines_xyxy=lines_xyxy,
        )

    def estimate_similarity_from_orb(
        self,
        *,
        anchor: FeatureSnapshot,
        current: FeatureSnapshot,
    ) -> Optional[tuple[np.ndarray, np.ndarray]]:
        """
        Estimate a 2D similarity (as 2x3 affine) from ORB matches between snapshots.

        Returns (A_2x3, inlier_mask_bool) or None.
        """
        if self._bf is None:
            return None
        if anchor.orb_uv is None or anchor.orb_desc is None or current.orb_uv is None or current.orb_desc is None:
            return None
        if int(anchor.orb_uv.shape[0]) < 4 or int(current.orb_uv.shape[0]) < 4:
            return None

        try:
            matches_knn = self._bf.knnMatch(anchor.orb_desc, current.orb_desc, k=2)
        except Exception:
            return None
        if not matches_knn:
            return None

        good = []
        rt = float(self.cfg.orb.ratio_test)
        for m12 in matches_knn:
            try:
                m1 = m12[0]
                m2 = m12[1] if int(len(m12)) > 1 else None
            except Exception:
                continue
            if m2 is not None:
                if float(m1.distance) >= float(rt) * float(m2.distance):
                    continue
            good.append(m1)
        if int(len(good)) <= 0:
            return None

        try:
            good = sorted(good, key=lambda m: float(getattr(m, "distance", 0.0)))
        except Exception:
            pass
        max_m = int(max(0, int(self.cfg.orb.max_matches)))
        if int(max_m) > 0 and int(len(good)) > int(max_m):
            good = list(good[: int(max_m)])

        try:
            src = np.asarray([anchor.orb_uv[int(m.queryIdx)] for m in good], dtype=np.float32).reshape(-1, 2)
            dst = np.asarray([current.orb_uv[int(m.trainIdx)] for m in good], dtype=np.float32).reshape(-1, 2)
        except Exception:
            return None
        if int(src.shape[0]) < 4 or int(dst.shape[0]) < 4:
            return None

        try:
            A, inl = cv2.estimateAffinePartial2D(
                src,
                dst,
                method=cv2.RANSAC,
                ransacReprojThreshold=float(self.cfg.orb.ransac_reproj_thresh_px),
                confidence=float(self.cfg.orb.ransac_confidence),
                refineIters=0,
            )
        except Exception:
            A, inl = None, None
        if A is None or np.asarray(A).shape != (2, 3) or inl is None:
            return None
        try:
            inl_b = np.asarray(inl, dtype=np.uint8).reshape(-1) != 0
        except Exception:
            return None
        if int(np.count_nonzero(inl_b)) < int(self.cfg.orb.min_inliers):
            return None

        return np.asarray(A, dtype=np.float64).reshape(2, 3), np.asarray(inl_b, dtype=bool).reshape(-1)

