from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np


def _as_T44(v: Any) -> Optional[np.ndarray]:
    try:
        T = np.asarray(v, dtype=np.float64).reshape(4, 4)
    except Exception:
        return None
    if not bool(np.isfinite(T).all()):
        return None
    return T


def _rot_angle_deg(R: np.ndarray) -> float:
    try:
        M = np.asarray(R, dtype=np.float64).reshape(3, 3)
    except Exception:
        return float("inf")
    tr = float(M[0, 0] + M[1, 1] + M[2, 2])
    c = max(-1.0, min(1.0, (tr - 1.0) * 0.5))
    try:
        return float(np.degrees(np.arccos(c)))
    except Exception:
        return float("inf")


@dataclass
class PoseFrames:
    """Both ORB-SLAM3 frames we care about.

    - `Twc_odom`: continuous local odom frame (used for control + polygon tracking).
    - `Twc_map`: ORB map/world frame (can jump after BA / loop closure).
    """

    Twc_odom: Optional[np.ndarray]
    Twc_map: Optional[np.ndarray]
    ok: bool
    map_id: Optional[int] = None


class OrbslamFrameResolver:
    """Maintain a stable odometry frame from ORB-SLAM3's map/world poses.

    We maintain a transform `T_map_odom` (starts as identity) and compute the continuous
    local pose as:

      T_odom_cam = inv(T_map_odom) * T_map_cam

    When ORB-SLAM3 applies a map correction (loop closure / BA / merge) and the map/world
    frame jumps, we update `T_map_odom` so `T_odom_cam` stays continuous:

      T_map_odom = T_map_cam * inv(T_odom_cam_prev)

    This matches the proven continuity scheme used in `apps/realsense_rgbd_inertial_topview.py`.
    """

    def __init__(self, *, jump_trans_m: float = 0.10, jump_rot_deg: float = 7.0) -> None:
        self._lock = threading.Lock()
        self._T_map_odom = np.eye(4, dtype=np.float64)
        self._Twc_odom_prev: Optional[np.ndarray] = None
        self._last_big_change_idx: Optional[int] = None
        self._pending_map_change = False
        self._jump_trans_m = float(jump_trans_m)
        self._jump_rot_deg = float(jump_rot_deg)

    def reset(self) -> None:
        with self._lock:
            self._T_map_odom = np.eye(4, dtype=np.float64)
            self._Twc_odom_prev = None
            self._last_big_change_idx = None
            self._pending_map_change = False

    def T_map_odom(self) -> np.ndarray:
        with self._lock:
            return np.asarray(self._T_map_odom, dtype=np.float64).copy()

    def T_odom_map(self) -> np.ndarray:
        with self._lock:
            try:
                return np.linalg.inv(np.asarray(self._T_map_odom, dtype=np.float64))
            except Exception:
                return np.eye(4, dtype=np.float64)

    def observe_slam(self, *, slam: Any) -> bool:
        """Observe ORB-SLAM3 for map changes; returns True if a new change was detected."""
        try:
            cur = int(slam.GetCurrentMapBigChangeIndex())
        except Exception:
            return False
        with self._lock:
            if self._last_big_change_idx is None:
                self._last_big_change_idx = int(cur)
                return False
            if int(cur) == int(self._last_big_change_idx):
                return False
            self._last_big_change_idx = int(cur)
            self._pending_map_change = True
            return True

    def _jump_detected(self, *, Twc_odom: np.ndarray, Twc_odom_prev: np.ndarray) -> bool:
        try:
            delta = np.linalg.inv(np.asarray(Twc_odom_prev, dtype=np.float64)) @ np.asarray(Twc_odom, dtype=np.float64)
        except Exception:
            return False
        try:
            trans_norm = float(np.linalg.norm(np.asarray(delta[:3, 3], dtype=np.float64).reshape(3)))
        except Exception:
            trans_norm = float("inf")
        try:
            rot_deg = float(_rot_angle_deg(np.asarray(delta[:3, :3], dtype=np.float64)))
        except Exception:
            rot_deg = float("inf")
        return bool(trans_norm > float(self._jump_trans_m) or rot_deg > float(self._jump_rot_deg))

    def resolve(self, *, Twc_map: Any, tracking_ok: bool, map_id: Optional[int] = None) -> PoseFrames:
        """Resolve stable odom pose + raw map pose.

        If `tracking_ok` is False, returns the last stable odom pose (hold) with ok=False.
        """
        T_map_cam = _as_T44(Twc_map)
        if not bool(tracking_ok) or T_map_cam is None:
            with self._lock:
                hold = np.asarray(self._Twc_odom_prev, dtype=np.float64).copy() if self._Twc_odom_prev is not None else None
            return PoseFrames(Twc_odom=hold, Twc_map=T_map_cam, ok=False, map_id=map_id)

        with self._lock:
            T_map_odom = np.asarray(self._T_map_odom, dtype=np.float64).reshape(4, 4)
            Twc_odom_prev = (
                np.asarray(self._Twc_odom_prev, dtype=np.float64).reshape(4, 4).copy()
                if self._Twc_odom_prev is not None
                else None
            )
            pending_map_change = bool(self._pending_map_change)

        # Compute current odom pose using the current map->odom relation.
        try:
            Twc_odom = np.linalg.inv(T_map_odom) @ T_map_cam
        except Exception:
            Twc_odom = None

        if Twc_odom is None:
            with self._lock:
                hold = np.asarray(self._Twc_odom_prev, dtype=np.float64).copy() if self._Twc_odom_prev is not None else None
            return PoseFrames(Twc_odom=hold, Twc_map=T_map_cam, ok=False, map_id=map_id)

        # Detect jumps in the *odom* frame (should be steady).
        jump = False
        if Twc_odom_prev is not None:
            try:
                jump = bool(self._jump_detected(Twc_odom=Twc_odom, Twc_odom_prev=Twc_odom_prev))
            except Exception:
                jump = False

        # If map jumped (loop closure) or odom jumped, absorb correction into T_map_odom so odom stays continuous.
        if (pending_map_change or jump) and Twc_odom_prev is not None:
            try:
                T_map_odom_new = (T_map_cam @ np.linalg.inv(Twc_odom_prev)).reshape(4, 4)
                Twc_odom = (np.linalg.inv(T_map_odom_new) @ T_map_cam).reshape(4, 4)
                with self._lock:
                    self._T_map_odom = np.asarray(T_map_odom_new, dtype=np.float64).reshape(4, 4)
                    self._pending_map_change = False
            except Exception:
                pass
        else:
            if pending_map_change:
                # If we don't have a previous odom pose, we can't enforce continuity yet, but we should still clear the flag.
                with self._lock:
                    self._pending_map_change = False

        with self._lock:
            self._Twc_odom_prev = np.asarray(Twc_odom, dtype=np.float64).reshape(4, 4).copy()
            out_odom = np.asarray(self._Twc_odom_prev, dtype=np.float64).copy()

        return PoseFrames(Twc_odom=out_odom, Twc_map=T_map_cam, ok=True, map_id=map_id)
