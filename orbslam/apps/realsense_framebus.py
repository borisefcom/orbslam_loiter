from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from orbslam_app.framebus import FrameBusConfig, FrameBusWriter
from orbslam_app.modes import MODE_BY_ID, resolve_mode, mode_id as mode_id_for


def _import_realsense() -> object:
    try:
        import pyrealsense2 as rs  # type: ignore
    except Exception as exc:
        raise RuntimeError(f"pyrealsense2 is required: {exc}") from exc
    return rs


def _start_realsense(
    rs, mode, *, deadline: float | None, fixed_streams: bool
) -> tuple[object, object, object | None, bool]:
    while True:
        if deadline is not None and time.time() >= float(deadline):
            raise TimeoutError("Auto-exit deadline reached while starting RealSense.")
        pipe = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
        if fixed_streams:
            cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            cfg.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8, 30)
            cfg.enable_stream(rs.stream.infrared, 2, 640, 480, rs.format.y8, 30)
            cfg.enable_stream(rs.stream.gyro, rs.format.motion_xyz32f, 200)
            cfg.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, 200)
        else:
            if mode.use_depth:
                cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            if mode.use_ir_left:
                cfg.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8, 30)
            if mode.use_ir_right:
                cfg.enable_stream(rs.stream.infrared, 2, 640, 480, rs.format.y8, 30)
            if mode.use_imu:
                cfg.enable_stream(rs.stream.gyro, rs.format.motion_xyz32f, 200)
                cfg.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, 200)

        q = rs.frame_queue(512)
        try:
            profile = pipe.start(cfg, q)
        except RuntimeError as exc:
            print(f"[framebus] start failed ({mode.label}): {exc}", flush=True)
            try:
                pipe.stop()
            except Exception:
                pass
            if fixed_streams:
                print("[framebus] Falling back to mode-specific streams.", flush=True)
                fixed_streams = False
            time.sleep(1.0)
            continue

        try:
            dev = profile.get_device()
            try:
                serial = dev.get_info(rs.camera_info.serial_number)
                usb = dev.get_info(rs.camera_info.usb_type_descriptor)
                print(f"[framebus] serial={serial} usb={usb}", flush=True)
            except Exception:
                pass
            for sensor in dev.query_sensors():
                if sensor.supports(rs.option.global_time_enabled):
                    sensor.set_option(rs.option.global_time_enabled, 1.0)
        except Exception:
            pass

        align = rs.align(rs.stream.color) if fixed_streams or mode.use_depth else None
        return pipe, q, align, fixed_streams


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--mode",
        type=str,
        default="rgbd+imu",
        help="Initial mode: rgbd, rgbd+imu, stereo, stereo+imu, mono, mono+imu",
    )
    ap.add_argument("--framebus-name", type=str, default=FrameBusConfig().name, help="Shared memory name prefix")
    ap.add_argument("--force", action="store_true", help="Force recreate shared memory if it exists")
    ap.add_argument(
        "--auto-exit-s",
        type=float,
        default=0.0,
        help="Auto-exit after N seconds (0 disables)",
    )
    ap.add_argument(
        "--fixed-streams",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep a fixed stream set for fast mode switching (default: enabled)",
    )
    args = ap.parse_args()

    rs = _import_realsense()

    config = FrameBusConfig(name=str(args.framebus_name))
    writer = FrameBusWriter(config, force=bool(args.force))
    writer.set_status(0)

    current_mode = resolve_mode(str(args.mode))
    current_mode_id = mode_id_for(current_mode)
    writer.set_active_mode(current_mode_id)
    writer.set_request_mode(current_mode_id)

    auto_exit_s = float(args.auto_exit_s)
    deadline = float(time.time() + auto_exit_s) if auto_exit_s > 0 else None
    fixed_streams = bool(args.fixed_streams)
    try:
        pipe, q, align, fixed_streams = _start_realsense(
            rs, current_mode, deadline=deadline, fixed_streams=fixed_streams
        )
    except TimeoutError as exc:
        print(f"[framebus] {exc}", flush=True)
        writer.close(unlink=True)
        return 0
    t_base_s: float | None = None
    timeout_streak = 0
    video_cache: dict[str, tuple[float, np.ndarray]] = {}
    last_emit_ts: float | None = None
    last_emit_wall_s: float | None = None

    def _cache_frame(key: str, frame) -> None:
        nonlocal t_base_s
        if frame is None:
            return
        ts_raw = float(frame.get_timestamp()) * 1e-3
        if t_base_s is None:
            # Anchor timestamps using the preferred time base.
            # For IMU modes we prefer color timestamps (best aligned with motion data).
            if current_mode.use_imu:
                if key == "color":
                    t_base_s = float(ts_raw)
            else:
                if key in ("ir_left", "color"):
                    t_base_s = float(ts_raw)
        data = np.ascontiguousarray(np.asanyarray(frame.get_data()))
        video_cache[str(key)] = (float(ts_raw), data)

    def _maybe_emit_frame() -> None:
        nonlocal t_base_s, last_emit_ts, last_emit_wall_s
        if current_mode.uses_rgbd() and "color" not in video_cache:
            return
        if current_mode.use_depth and "depth" not in video_cache:
            return
        if current_mode.use_ir_left and "ir_left" not in video_cache:
            return
        if current_mode.use_ir_right and "ir_right" not in video_cache:
            return

        if current_mode.use_imu and "color" in video_cache:
            ts_raw = float(video_cache["color"][0])
        elif current_mode.use_ir_left and "ir_left" in video_cache:
            ts_raw = float(video_cache["ir_left"][0])
        elif current_mode.uses_rgbd() and "color" in video_cache:
            ts_raw = float(video_cache["color"][0])
        elif "color" in video_cache:
            ts_raw = float(video_cache["color"][0])
        elif "ir_left" in video_cache:
            ts_raw = float(video_cache["ir_left"][0])
        else:
            return

        if last_emit_ts is not None and float(ts_raw) <= float(last_emit_ts):
            return
        last_emit_ts = float(ts_raw)

        if t_base_s is None:
            t_base_s = float(ts_raw)
        ts = float(ts_raw - t_base_s)

        rgb = video_cache.get("color", (0.0, None))[1]
        depth_raw = video_cache.get("depth", (0.0, None))[1] if current_mode.use_depth else None
        ir_left = video_cache.get("ir_left", (0.0, None))[1] if current_mode.use_ir_left else None
        ir_right = video_cache.get("ir_right", (0.0, None))[1] if current_mode.use_ir_right else None

        writer.write_frame(
            mode_id=int(current_mode_id),
            ts_s=float(ts),
            rgb=rgb,
            depth=depth_raw,
            ir_left=ir_left,
            ir_right=ir_right,
        )
        last_emit_wall_s = float(time.time())

    def _restart_pipeline(reason: str) -> bool:
        nonlocal pipe, q, align, fixed_streams
        nonlocal t_base_s, video_cache, last_emit_ts, last_emit_wall_s
        print(f"[framebus] {reason} (restarting pipeline...)", flush=True)
        try:
            pipe.stop()
        except Exception:
            pass
        try:
            pipe, q, align, fixed_streams = _start_realsense(
                rs, current_mode, deadline=deadline, fixed_streams=fixed_streams
            )
        except TimeoutError as exc:
            print(f"[framebus] {exc}", flush=True)
            return False
        writer.set_active_mode(current_mode_id)
        t_base_s = None
        video_cache.clear()
        last_emit_ts = None
        last_emit_wall_s = float(time.time())
        return True

    print(
        f"[framebus] Running ({current_mode.label}). Shared memory prefix='{config.name}'.",
        flush=True,
    )
    if fixed_streams:
        print("[framebus] Fixed streams enabled for fast mode switching.", flush=True)
    try:
        while True:
            if writer.stop_requested():
                print("[framebus] Stop requested", flush=True)
                break
            if deadline is not None and time.time() >= float(deadline):
                print("[framebus] Auto-exit", flush=True)
                break
            req_id = writer.request_mode_id()
            if req_id != current_mode_id:
                new_mode = MODE_BY_ID.get(int(req_id))
                if new_mode is not None:
                    print(f"[framebus] Switching mode -> {new_mode.label}", flush=True)
                    current_mode = new_mode
                    current_mode_id = int(req_id)
                    if fixed_streams:
                        writer.set_active_mode(current_mode_id)
                        t_base_s = None
                        video_cache.clear()
                        last_emit_ts = None
                        last_emit_wall_s = float(time.time())
                    else:
                        if not _restart_pipeline("mode switch"):
                            break

            if last_emit_wall_s is not None and (time.time() - last_emit_wall_s) > 2.0:
                if not _restart_pipeline("video stall"):
                    break

            try:
                f = q.wait_for_frame(5000)
            except RuntimeError as exc:
                msg = str(exc)
                if "did not arrive in time" in msg:
                    timeout_streak += 1
                    if timeout_streak >= 2:
                        if not _restart_pipeline("timeout streak"):
                            break
                        t_base_s = None
                        timeout_streak = 0
                    continue
                if ("disconnected" in msg.lower()) or ("no device" in msg.lower()):
                    if not _restart_pipeline(msg):
                        break
                    t_base_s = None
                    timeout_streak = 0
                    continue
                raise

            if f is None:
                continue
            timeout_streak = 0

            if f.is_motion_frame():
                if not current_mode.use_imu:
                    continue
                mf = f.as_motion_frame()
                data = mf.get_motion_data()
                t_s = float(mf.get_timestamp()) * 1e-3
                if t_base_s is None:
                    continue
                t_s = float(t_s - t_base_s)
                st = mf.get_profile().stream_type()
                if st == rs.stream.gyro:
                    writer.write_gyro(t_s=t_s, gx=float(data.x), gy=float(data.y), gz=float(data.z))
                elif st == rs.stream.accel:
                    writer.write_accel(t_s=t_s, ax=float(data.x), ay=float(data.y), az=float(data.z))
                continue

            if f.is_frameset():
                fs = f.as_frameset()
                if align is not None and current_mode.use_depth:
                    fs = align.process(fs)

                color = fs.get_color_frame()
                if color:
                    _cache_frame("color", color)
                if current_mode.use_depth:
                    depth = fs.get_depth_frame()
                    if depth:
                        _cache_frame("depth", depth)
                if current_mode.use_ir_left:
                    left = fs.get_infrared_frame(1)
                    if left:
                        _cache_frame("ir_left", left)
                if current_mode.use_ir_right:
                    right = fs.get_infrared_frame(2)
                    if right:
                        _cache_frame("ir_right", right)
                _maybe_emit_frame()
                continue

            profile = f.get_profile()
            st = profile.stream_type()
            if st == rs.stream.color:
                _cache_frame("color", f)
                _maybe_emit_frame()
                continue
            if st == rs.stream.depth:
                if current_mode.use_depth:
                    _cache_frame("depth", f)
                    _maybe_emit_frame()
                continue
            if st == rs.stream.infrared:
                idx = int(profile.stream_index())
                if idx == 1 and current_mode.use_ir_left:
                    _cache_frame("ir_left", f)
                    _maybe_emit_frame()
                elif idx == 2 and current_mode.use_ir_right:
                    _cache_frame("ir_right", f)
                    _maybe_emit_frame()
                continue

    finally:
        try:
            pipe.stop()
        except Exception:
            pass
        writer.close(unlink=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
