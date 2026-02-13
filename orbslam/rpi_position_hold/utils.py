"""Utilities (subset) for the polygon tracking subprocess.

This is a deliberately small shim that provides just the helpers used by
`rpi_position_hold.processes.poly_vo_lk_track`.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class _Intr:
    fx: float
    fy: float
    cx: float
    cy: float
    w: int
    h: int


def _intrinsics_from_dict(d: dict, *, fallback_w: int = 0, fallback_h: int = 0) -> Optional[_Intr]:
    if not isinstance(d, dict):
        return None
    try:
        w = int(d.get("w", d.get("width", fallback_w)) or fallback_w or 0)
        h = int(d.get("h", d.get("height", fallback_h)) or fallback_h or 0)
        fx = float(d.get("fx", 0.0) or 0.0)
        fy = float(d.get("fy", 0.0) or 0.0)
        # RealSense uses ppx/ppy; fall back to cx/cy.
        cx0 = d.get("cx", None)
        if cx0 is None:
            cx0 = d.get("ppx", None)
        cy0 = d.get("cy", None)
        if cy0 is None:
            cy0 = d.get("ppy", None)
        cx = float(cx0) if cx0 is not None else 0.0
        cy = float(cy0) if cy0 is not None else 0.0
        if w <= 0 or h <= 0 or fx <= 0.0 or fy <= 0.0:
            return None
        return _Intr(fx=float(fx), fy=float(fy), cx=float(cx), cy=float(cy), w=int(w), h=int(h))
    except Exception:
        return None


def _setup_stdio_logging(*, log_spec: dict, role: str) -> None:
    """Best-effort logging shim.

    The original pipeline can tee stdout/stderr into per-run log files. For the ORB-SLAM3
    integration we keep this as a no-op unless explicitly enabled.
    """

    try:
        enabled = bool((log_spec or {}).get("enabled", False))
    except Exception:
        enabled = False
    if not bool(enabled):
        return
    try:
        path = str((log_spec or {}).get("latest_path", "") or (log_spec or {}).get("path", "") or "").strip()
    except Exception:
        path = ""
    if not path:
        return
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        f = open(path, "a", encoding="utf-8", buffering=1)
    except Exception:
        return
    try:
        sys.stdout = f  # type: ignore[assignment]
        sys.stderr = f  # type: ignore[assignment]
    except Exception:
        try:
            f.close()
        except Exception:
            pass

