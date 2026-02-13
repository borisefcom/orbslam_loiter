from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYBIND_ROOT = PROJECT_ROOT / "third_party" / "ORB_SLAM3_pybind"
sys.path.insert(0, str(PYBIND_ROOT))

from python_wrapper.orb_slam3 import ORB_SLAM3  # type: ignore


def _make_base_texture(h: int, w: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    base = rng.integers(0, 256, size=(h, w), dtype=np.uint8)
    # Add a few stable, high-contrast shapes to help feature matching.
    for k in range(80):
        cx = int(rng.integers(0, w))
        cy = int(rng.integers(0, h))
        rr = int(rng.integers(6, 40))
        val = int(rng.integers(0, 256))
        yy, xx = np.ogrid[:h, :w]
        mask = (xx - cx) * (xx - cx) + (yy - cy) * (yy - cy) <= rr * rr
        base[mask] = np.uint8(val)
    return base


def _crop_view(base: np.ndarray, out_h: int, out_w: int, cx: float, cy: float) -> np.ndarray:
    h, w = base.shape
    x0 = int(round(cx - out_w * 0.5))
    y0 = int(round(cy - out_h * 0.5))
    x1 = x0 + out_w
    y1 = y0 + out_h
    if x0 < 0 or y0 < 0 or x1 > w or y1 > h:
        raise ValueError("Crop out of bounds; reduce motion amplitude or increase base texture size.")
    return base[y0:y1, x0:x1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=600, help="Number of synthetic frames to feed.")
    ap.add_argument("--fps", type=float, default=30.0, help="Synthetic frame rate used for timestamps.")
    ap.add_argument("--seed", type=int, default=1, help="RNG seed for the synthetic scene.")
    ap.add_argument("--no-vis", action="store_true", help="Disable ORB viewer (recommended).")
    ap.add_argument(
        "--inject-alignment",
        action="store_true",
        help="Debug: inject a synthetic alignment event (tests Python->C++ plumbing even if no loop/merge happens).",
    )
    ap.add_argument(
        "--inject-frame",
        type=int,
        default=-1,
        help="Frame index to inject the debug alignment event (default: mid-sequence when --inject-alignment is set).",
    )
    ap.add_argument(
        "--config",
        type=str,
        default=str(PROJECT_ROOT / "apps" / "synth_rgbd.yaml"),
        help="ORB-SLAM3 settings YAML.",
    )
    ap.add_argument(
        "--vocab",
        type=str,
        default=str(PYBIND_ROOT / "Vocabulary" / "ORBvoc.txt"),
        help="ORB vocabulary path.",
    )
    args = ap.parse_args()

    H, W = 480, 640
    base_h, base_w = 2048, 2048
    base = _make_base_texture(base_h, base_w, seed=int(args.seed))
    depth = np.full((H, W), 2.0, dtype=np.float32)

    slam = ORB_SLAM3(str(args.vocab), str(args.config), "RGBD", (not bool(args.no_vis)))

    inject_at = int(args.inject_frame)
    if bool(args.inject_alignment) and inject_at < 0:
        inject_at = int(max(0, int(args.frames) // 2))

    last_info = None
    t0 = time.perf_counter()
    for i in range(int(args.frames)):
        theta = 2.0 * math.pi * (float(i) / max(1.0, float(args.frames)))
        # A smooth loop (ends close to start) to encourage loop closure.
        amp = 520.0
        cx = 0.5 * base_w + amp * math.cos(theta)
        cy = 0.5 * base_h + amp * math.sin(theta)

        gray = _crop_view(base, H, W, cx, cy)
        bgr = np.repeat(gray[:, :, None], 3, axis=2)
        bgr = np.ascontiguousarray(bgr)

        ts = float(i) / float(max(1e-6, float(args.fps)))
        slam.TrackRGBD(image=bgr, depthmap=depth, timestamp=ts)

        if bool(args.inject_alignment) and int(i) == int(inject_at):
            # Simulate a rigid correction "new <- old" (10 deg yaw + 0.25m x-translation).
            yaw = math.radians(10.0)
            c, s = math.cos(yaw), math.sin(yaw)
            T = np.array(
                [
                    [c, -s, 0.0, 0.25],
                    [s, c, 0.0, 0.00],
                    [0.0, 0.0, 1.0, 0.00],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            )
            try:
                slam.DebugPushAlignmentEvent(
                    type=1,
                    t_s=float(ts),
                    map_a=int(slam.GetCurrentMapId()),
                    map_b=int(slam.GetCurrentMapId()),
                    kf_a=0,
                    kf_b=0,
                    scale=1.0,
                    T_new_old=T,
                )
                print(f"[inject] alignment event injected at frame={int(i)} t={ts:.3f}s", flush=True)
            except Exception as e:
                print(f"[inject] failed: {e}", flush=True)

        # Map hashes / telemetry.
        info = (
            int(slam.GetCurrentMapId()),
            int(slam.GetAtlasMapCount()),
            int(slam.GetCurrentMapBigChangeIndex()),
            int(slam.GetCurrentMapChangeIndex()),
        )
        if last_info is None or info != last_info:
            print(f"[telemetry] map_id={info[0]} maps={info[1]} big={info[2]} chg={info[3]}", flush=True)
            last_info = info

        evs = slam.PopAlignmentEvents()
        if evs:
            for ev in evs:
                et = int(ev.get("type", -1))
                tag = "LOOP" if et == 0 else ("MERGE" if et == 1 else str(et))
                print(
                    f"[align] {tag} t={ev.get('t_s', 0.0):.3f}s "
                    f"map_a={ev.get('map_a')} map_b={ev.get('map_b')} "
                    f"kf_a={ev.get('kf_a')} kf_b={ev.get('kf_b')} "
                    f"scale={ev.get('scale', 1.0):.6f}",
                    flush=True,
                )

    slam.Shutdown()
    dt = float(time.perf_counter() - t0)
    print(f"[done] frames={int(args.frames)} wall_s={dt:.2f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
