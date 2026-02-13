# Indoor Loiter — Drone Server (RPi companion computer)

This repo is the **drone-side server**. It’s designed to run on a Raspberry Pi 5 as a companion computer and talk over Wi‑Fi to the ground-station GUI in `indoor-osd-app/`.

## Quick start

- Primary entrypoint: `server.py`
- Main config: `drone.yaml`
- Run: `python3 server.py`
  - If a local `.venv/` exists, `server.py` will re-exec into it automatically (set `LOITER_NO_VENV_REEXEC=1` to disable).

## Python deps (RPi)

The supported runtime on the Pi is a local venv (typically `~/vo_loiter/.venv/`, symlinked to `indoor_loiter/.venv`)
created with `python3 -m venv --system-site-packages`.

Required pip packages in that venv:
- `pymavlink` (PX4 serial + MAVLink utilities)
- `PyYAML` (tracker YAML configs)

The external ORB-SLAM3 tracker additionally requires:
- `pyrealsense2` (RealSense capture)
- `cv2` (OpenCV)
- the custom ORB-SLAM3 Python binding in `orbslam/third_party/ORB_SLAM3_pybind/` (build on target OS/arch)

## RPi sync (Windows staging -> RPi)

All performance experiments are run on the RPi, so keep the RPi code in sync with this Windows staging tree.

- Deploy both `indoor_loiter/` and `orbslam/` to the RPi into `~/vo_loiter/`:
  - `.\deploy_rpi.ps1 -TargetHost 10.92.44.50 -TargetUser esp710 -JumpHost 100.86.78.11 -JumpUser xtend_m2`
  - If you have direct SSH access (no hop), add `-NoJump`.
  - The script uses PuTTY tools (`plink`, `pscp`) and prompts for passwords.
- Fetch RPi logs back into `._rpi_logs/` (tarballs `indoor_loiter/logs/` on the RPi and extracts locally):
  - `.\fetch_rpi_logs.ps1 -TargetHost 10.92.44.50 -TargetUser esp710 -JumpHost 100.86.78.11 -JumpUser xtend_m2`

## High-level architecture

`server.py` wires together:

- Video capture + streaming: `gstreamer.py` (`VideoController`)
  - RealSense-only capture (color + optional IMU)
  - streams via **RTSP** or **RTP/UDP**
- PX4 bridge: `mavlink_bridge.py` (`MavlinkBridge`)
  - serial link to FCU + optional UDP mirror of raw MAVLink
- JSON telemetry + control routing: `telemetry.py` (`TelemetryManager`)
- Tracker: `tracker.py` (`Tracker` + `FrameBus`)
  - tracker runs in a **separate process** and publishes JSON overlays (`detect`, `track_status`, …)
- Optional PID loop: `pid.py` (emits `pid_debug` JSON)
- Optional IMU worker: `imu_fusion.py`

## Networking (default ports)

All values are configurable in `drone.yaml`.

- **Control (client → drone)**: UDP JSON in on `control.json_in` (default `0.0.0.0:6020`)
- **Telemetry/overlay (drone → client)**: UDP JSON out to `telemetry.json.udp` (default `…:6021`)
- **MAVLink mirror (drone → client)**: UDP out to `px4.mirror.udp` (default `…:14550`) when `px4.mirror.enabled=true`
- **Video**
  - RTSP: binds `video.rtsp.bind:video.rtsp.port` (default `0.0.0.0:8554`), path `video.rtsp.path` (default `/stream`)
  - UDP: sends RTP to `video.udp_sink` (default `…:5600`)

Wi‑Fi setup helpers (NetworkManager):

- Hotspot: `wifi_ap.sh` (typically assigns `10.42.0.1/24`)
- Client mode: `wifi_client.sh`

## UDP JSON protocol (server ⇄ client)

### Telemetry/overlay messages (server → client)

One JSON object per UDP datagram. Most messages include:

- `type`: message kind (string)
- `ts` or `t_ms`: timestamp in milliseconds

Core message types:

- `pose` (from `telemetry.py`): VO pose snapshot (sourced from `track3d` when enabled)
  - `{"type":"pose","ts":…,"vo_pose":{...},"zed_pose":{...}}`
  - `vo_pose.position_m.{x,y,z}` is in meters (currently `track3d` `pose.pos_rel_w_m` if available).
  - `zed_pose` is kept as a back-compat alias for older clients; new code should use `vo_pose`.
- `cellular` (from `telemetry.py`): normalized cellular status snapshot
- `sys` (from `telemetry.py` + `sys_stats.py`): system utilization snapshot
  - `cpu_pct: [core1..coreN]`, `mem_pct`, `swap_pct`
- `detect` (from `tracker.py`): detections + `img_size` + `seq`
  - `tracker.py` truncates `detections` to stay under `telemetry.json.max_bytes` (default `1200`) to reduce UDP fragmentation loss.
- `track_status` / `track_lost` (from `tracker.py`): tracking state and error terms
- `pid_debug` (from `pid.py`): PID state, gates, measurement + output
- `track2d` (from `track2d_shim.py`): lightweight 2D ROI tracker status + bbox (used when the GUI enables the 2D tracker)
- `camera` (from `server.py`): active video feed settings (so the GUI can gate overlays to the correct camera)
  - `{"type":"camera","ts":…,"active":"ir|depth|v4l","source":"track3d|v4l","img_size":[W,H],"fps":…,"intrinsics":{...}}`
- `vo_features` (from `server.py` track3d pump): feature overlay points for the GUI
  - `img_size: [W,H]`
  - `pts: [[u,v,group], ...]` where `group` maps to display colors (1=green, 2=yellow, 3=red, else gray)

### Control messages (client → server)

Control is **best-effort UDP JSON** (no acknowledgements). The server applies any keys it recognizes (see `TelemetryManager.handle_control` in `telemetry.py`).

Supported keys:

- Telemetry retarget: `telemetry_ip`, `telemetry_port`
- Video routing: `video_to_sender` (set UDP sink IP to sender), or `dest_ip`, `dest_port`
- Video encoder controls (currently mostly no-op unless pipeline restart/live update is implemented): `bitrate_kbps`, `rc_mode`, `gop`, `idr`, `stream_method`
- MAVLink mirror: `mav_mirror_ip` (port is taken from current config / defaults to `14550`)
- Tracking: `track_select` (int or `{track_id}`), `roi_select` (bbox), `track_cancel`
- Mouse (future): `type="mouse_move"|"mouse_stop"|"mouse_click"` with cursor position over the main OSD video
- 3D hole acquisition toggle: `{"type":"hole_enable","enable":0|1}` (disables/enables the expensive depth-based stage-0 preview + click-to-lock flow in the external tracker)
  - Stream behavior: stays on the **regular IR (bw)** feed (`active="ir"`) unless the 2D tracker is enabled.
  - Note: acquisition uses depth internally; a processed depth visualization exists in the SHM ring but is not the default streamed feed.
- 3D plane acquisition toggle: `{"type":"plane_enable","enable":0|1}` (exclusive with hole acquisition; uses the same depth-based stage-0 preview + click-to-lock flow)
  - Stream behavior: stays on the **regular IR (bw)** feed (`active="ir"`) unless the 2D tracker is enabled.
- 2D ROI tracker (DynMedianFlow) control:
  - `{"type":"track2d_enable","enable":0|1}` (exclusive with the 3D hole acquisition mouse flow)
  - `{"type":"track2d_rect","surface":"osd_main","x0_norm":0..1,"y0_norm":0..1,"x1_norm":0..1,"y1_norm":0..1}` (select ROI and start tracking)
  - `{"type":"track2d_cancel"}` (stop tracking)
  - Uses the **non-RealSense** tracking camera configured under `camera.v4l` in `drone.yaml` (OpenCV V4L2 MJPEG capture at high FPS).
  - When enabled, the server switches the stream to the V4L camera and emits `type="camera"` with `active="v4l"`.
  - When disabled, the server switches back to `active="ir"`.

Note: `indoor-osd-app` sends “startup routing” control messages on launch to ask the drone to route telemetry/video/MAVLink mirror to the GUI’s IP. This updates in-memory state only (it does not persist back to `drone.yaml`).

## Tracking pipeline (YOLO → DynMedianFlow → PID angles)

### Track3d debug logs (on the drone)

Configured in `drone.yaml` under `debug.*`:

- `logs/fps_detail.log`: per-second perf counters (includes mouse rates, preview rate, poly stage/verts, VO rates).
- `logs/track3d_hole.log`: hole detector preview/select failures (reason + detector error string).
- `logs/track3d_plane.log`: plane detector preview/select failures (reason + plane stats like radius, inliers, RMS, range).

Tracking runs inside `tracker.py` in a **separate process**. The vision engine it wraps is `homography.py:HomographyTracker`.

### State machine (engine)

`HomographyTracker` has two main modes:

- **ACQUIRE**: (re)detect targets (YOLO) and emit `type="detect"` JSON for the GUI to click/select.
- **TRACKING**: once a target is chosen, track it frame-to-frame using `dynamic_medianflow.py:DynMedianFlowTracker` (MedianFlow-style LK) and output a live target center point.

Key control entrypoints:

- `HomographyTracker.start_from_bbox(bbox_xywh)` transitions to TRACKING and defers `DynMedianFlowTracker.init()` to the next frame (`_pending_init_bbox`).
- `HomographyTracker.cancel()` transitions back to ACQUIRE and resets the object tracker.

### MedianFlow output → tracker signals

In TRACKING mode:

1. `DynMedianFlowTracker.update(gray, return_info=True)` returns:
   - `ok` (bool)
   - `bbox` (x, y, w, h) in **image pixels**
   - `info` dict (includes robust per-frame translation `dx/dy` in pixels, `conf`, `good_frac`, grid size, etc.)
2. `HomographyTracker` converts `bbox` → target center:
   - `raw_pt = (bbox_cx, bbox_cy)`
   - currently `filt_pt` == `raw_pt` (no Kalman filtering in this build)
3. `TrackerWorker` (in `tracker.py`) converts the engine point into **FrameBus pixel coordinates** and applies `mav_control.axis_map` if set:
   - `used_xy_bus_int = _bus_scale_point(raw_pt, frame_shape, W_bus, H_bus)`
   - `used_xy_bus_int = _apply_axis_map(...)`
4. The worker emits two different paths:
   - **To the GUI** (UDP JSON via `_EVENT_JSON`): `type="detect"`, `type="track_status"`, `type="track_lost"`.
   - **To the PID loop** (IPC via `_EVENT_PID`): a measurement payload including `cx_px/cy_px` (chosen center) and `W/H`.

### Horizontal/vertical angle deltas (where they come from)

DynMedianFlow is pixel-domain. The **angular error** is computed later for PID:

- `tracker.py` emits `cx_px/cy_px` (FrameBus pixel coords) via `_EVENT_PID`.
- `server.py` receives that and computes camera-ray angle deltas using intrinsics from `VideoController.get_intrinsics()`:
  - `x_norm = (cx_px - cx) / fx`
  - `y_norm = (cy_px - cy) / fy`
  - `yaw_err_rad   = atan(x_norm)`  (horizontal angle delta)
  - `pitch_err_rad = atan(y_norm)` (vertical angle delta; +down in image)
- Those angles drive `pid.py` and are visible to the client as `type="pid_debug"` (`measurement.yaw_err_*`, `measurement.pitch_err_*`).

Important: this assumes `cx_px/cy_px` are in the **same pixel coordinate space** as the intrinsics (`fx/fy/cx/cy`). Today the canonical space is the FrameBus size.

Related (non-angle) signal:

- `type="track_status"` carries `error.ex/error.ey` as normalized pixel-center error:
  - `ex = (cx_px - W/2) / (W/2)`
  - `ey = (cy_px - H/2) / (H/2)`

DynMedianFlow also exposes **per-frame pixel deltas** in `info["dx"]` / `info["dy"]`. If you ever need a per-frame angular delta (motion) for a new backend, you can approximate:

- `d_yaw_rad   ≈ atan(dx / fx)` (or small-angle `dx/fx`)
- `d_pitch_rad ≈ atan(dy / fy)` (or small-angle `dy/fy`)

The current PID path uses the absolute center (`cx_px/cy_px`), not `dx/dy`.

### PID stage 1 vs stage 2 (PID_ALIGN vs PID_MOVE)

The “two stages” are selected by an RC 3‑position gate in `pid.py`:

- `OFF` → PID fully suppressed
- `PID_ALIGN` (stage 1) → alignment profile (`mav_control.pid_align`)
- `PID_MOVE` (stage 2) → movement profile (`mav_control.pid_move`)

Selection uses hysteresis on `mav_control.rc_gate.channel` (normalized to 0..1 with center ≈ 0.5):

- `rc01 < rc_gate.align_min` → OFF
- `rc_gate.off_max ≤ rc01 < rc_gate.align_max` → PID_ALIGN
- `rc01 ≥ rc_gate.align_max` → PID_MOVE

Behavior differences:

- **PID_ALIGN** has an optional auto-exit: `server.py` checks the *angle* error against `pid_align.auto_exit.delta_horizontal_deg` / `delta_vertical_deg`. When aligned, it calls `pm.mark_align_done()` and cancels the tracker (`tracker_obj.cancel()`), then stays silent until the switch changes.
- **PID_MOVE** can apply a special thrust-only “hold max” when the tracker stops due to `stop_reason="max_bbox_area"`: the tracker emits a PID event, `server.py` calls `pm.start_thrust_hold_max(...)`, and `pid.py` keeps sending forward thrust for `pid_move.thrust.down_ramp_time_s` even when not in `TRACKING` state.
- Both stages are also mode-gated (`pid_align.mode_gate` / `pid_move.mode_gate`) and only send full outputs when the tracker measurement `state == "TRACKING"`.

### PID → MAVLink MANUAL_CONTROL mapping (non-standard scaling)

`MavlinkBridge.send_manual_control(x_norm, y_norm, z_norm, r_norm)` sends MAVLink `MANUAL_CONTROL` where PX4 interprets:

- `x`: pitch (range -1000..+1000)
- `y`: roll  (range -1000..+1000)
- `z`: throttle (range 0..1000; **center/neutral is ~500** in ALTCTL/POSCTL)
- `r`: yaw   (range -1000..+1000)

Our PID output is mapped in a deliberately non-obvious way (see the “DO NOT CHANGE THIS ORDER” note in `pid.py`):

- `x_norm = -thrust_cmd` (this is the *forward/back* command; it’s named “thrust” internally but is sent on the **pitch/x** axis)
- `y_norm = roll_cmd`
- `z_norm = pitch_cmd + 0.5` (this is the *vertical* correction; it’s named “pitch” internally but is sent on the **throttle/z** axis, shifted so 0 means “neutral”)
- `r_norm = yaw_cmd`

Then `mavlink_bridge.py` clamps/scales to ints:

- `x/y/r`: clamp [-1..+1] and multiply by 1000
- `z`: clamp [0..1] and multiply by 1000

This mapping is intentional for this airframe/mode mix and is easy to break if “cleaned up”.

### Client↔server signals tied to tracking flow

Client → server (control UDP JSON on `:6020`, via `TelemetryManager.handle_control()`):

- `{"track_select": {"track_id": ID}}` → `tracker.select(ID)` → worker applies selection when a recent `detect` list exists, then calls `HomographyTracker.start_from_bbox(...)`.
- `{"roi_select": {"x":..,"y":..,"w":..,"h":..}}` → `tracker.roi_select(...)` → worker calls `HomographyTracker.start_from_bbox(...)` directly from ROI.
- `{"track_cancel": true}` → `tracker.cancel()` → worker calls `HomographyTracker.cancel()` and returns to ACQUIRE.

Server → client (telemetry UDP JSON on `:6021`):

- `type="detect"`: ACQUIRE (YOLO detections) and sometimes a synthetic overlay while tracking.
- `type="track_status"`: during tracking; includes normalized error `error.ex/error.ey` in [-1,+1] and `pid_source` (raw/kf/center).
- `type="track_lost"`: on tracker loss or auto-stop (includes diagnostic `mf` info for MedianFlow failures).
- `type="pid_debug"`: PID loop output; contains the **angle deltas** and the resulting manual-control commands.

### Preparing for future MedianFlow replacements (switchable backends)

The “swap point” for different MedianFlow-like backends is `homography.py` (the object tracker used in TRACKING mode).

Suggested config shape (future):

- `object_tracker.backend`: backend name (e.g. `DynMedianFlow`, `MedianFlowV2`, …)
- `object_tracker.params`: backend-agnostic knobs (or nest per-backend, e.g. `object_tracker.backends.<name>.*`)

Current behavior: `HomographyTracker` always instantiates `DynMedianFlowTracker(DynMFParams())`, and most DynMF tuning lives in `dynamic_medianflow.py` defaults; only a small part of `drone.yaml:object_tracker` is used today (enable/logging).

To add a new backend and switch it via config, keep this contract:

- Backend API shape (matches `DynMedianFlowTracker`):
  - `init(frame_gray_or_bgr, bbox_xywh) -> bool`
  - `update(frame_gray_or_bgr, return_info=True) -> (ok: bool, bbox_xywh, info: dict)`
- `bbox_xywh` must be in the same pixel space as the frames used for tracking (so `raw_pt` stays consistent).
- `info` should include (at minimum) diagnostic keys like `conf`, `good_frac`, and optionally `dx/dy` (pixels) so we can compare backends apples-to-apples.
- `homography.py:HomographyTracker.step(...)` must keep returning the fields `tracker.py` consumes:
  - ACQUIRE: `detect_tick` + `detections` (to populate `type="detect"` for the GUI)
  - TRACKING: `raw_pt`/`filt_pt` (target center), plus optional `ghost_bbox`/`viz.ghost` for overlays
  - Failure: `track_failed=True` and (optionally) `stop_reason` / `bbox_area_ratio` / a diagnostic dict (currently `mf`)
- As long as `HomographyTracker` keeps emitting `raw_pt/filt_pt` in image pixels, the rest of the pipeline (`tracker.py` → PID angles → GUI overlays) does not need to change.

## Performance & latency (RPi5)

Prioritize predictable latency over perfect fidelity.

- Prefer **hardware video encode** where possible; avoid high-res/high-fps **software x264** on the Pi.
- Keep UDP JSON payloads under MTU; preserve `telemetry.json.max_bytes` (~1200B) if operating over lossy/VPN/cellular links.
- Don’t add blocking I/O on hot paths (tracker/IMU/control). `UdpJsonTx` is intentionally non-blocking and drops on congestion.
- Be conservative with CPU threading (OpenCV thread count is set in `server.py`). Consider making this tunable if you change workloads.

## Change guidelines

- Treat `drone.yaml` as the API surface. If you add config keys, keep defaults safe for the Pi.
- If you add/modify a JSON message type or control key:
  - update the emitter and the parser (`indoor-osd-app/mavlink.py`)
  - document the change in both AGENTS files
  - keep backward compatibility when feasible (alias old key names)

## 3D tracker integration (track3d)

This staging tree uses an external ORB-SLAM3 RGB-D-Inertial tracker in `../orbslam/`
(spawned as a subprocess by `server.py`).

High level:
- `server.py` spawns `apps/realsense_orbslam3_rgbd_inertial_minimal.py --headless --config ...` (see `drone.yaml:track3d.*`).
- The 3D tracker owns the RealSense device via its own capture loop and publishes frames to shared memory.
- The server attaches to the shared-memory ring and streams video without opening RealSense (`video.external_capture: true`).
- Mouse-driven acquisition is handled via UDP JSON mouse messages from the client.

### Server <-> 3D tracker transport (JSONL)

The 3D tracker runs headless with a JSON shim (stdin/stdout JSONL):
- Server -> 3D tracker commands (stdin JSONL):
  - `{"cmd":"hover","x":px,"y":px}` (Stage 0 candidate search; sent from mouse_move)
  - `{"cmd":"select_hole","x":px,"y":px}` (Stage 1 begin)
  - `{"cmd":"confirm_hole","x":px,"y":px}` (Stage 2 begin)
  - `{"cmd":"clear"}` (cancel/clear)
- 3D tracker -> server events (stdout JSONL):
  - `type="init"` (ring spec + intrinsics; server uses this to attach streaming)
  - `type="preview"` (Stage 0 candidate polygon)
  - `type="telemetry"` (polygon state: active/verts/pid_confirmed)
  - `type="pid"` (dhv_deg stream; used for PID->MAVLink)

Implementation: `track3d_bridge.py` (subprocess + mouse->cmd mapping).

### Shared-memory streaming

When `track3d.enabled=true`:
- RealSense capture is owned by the 3D tracker.
- The server streams from shared memory using `shm_ring_source.py` (attaches using the `init.payload.ring_spec` dict).
- `gstreamer.py:VideoController` runs in `video.external_capture=true` mode and does not open RealSense.

### Acquisition overlay message (server -> client)

New UDP JSON message:
- `type="acq_poly"`:
  - `stage`: `0|1|2`
  - `img_size`: `[W,H]`
  - `verts_uv`: `[[u,v], ...]` (ints) or null (clear)
  - `center_uv`: `[u,v]` (floats) or null

Stage semantics (client colors):
- Stage 0: yellow polygon, red X, no fill
- Stage 1: purple polygon, yellow X, no fill
- Stage 2: purple polygon, purple fill (alpha=0.33), green X

### 3D tracker dhv_deg -> existing PID -> MAVLink

The server keeps the existing PID manager, RC gates, and MANUAL_CONTROL scaling/mapping.

Wiring:
- 3D tracker emits `type="pid"` with `payload.dhv_deg=(h_deg,v_deg)` and `payload.stage`.
- `server.py` uses only `stage==2` and converts degrees to radians:
  - `yaw_err_rad = radians(h_deg)`
  - `pitch_err_rad = radians(v_deg)`
- These are fed into `pid.Measurement(state="TRACKING", ...)` so the rest of the PID->MAVLink path is unchanged.

Stop/reset:
- 3D tracker internal "hole_done" auto-stop is disabled in its YAML; stop/reset is managed by the server (via `cmd=clear`).
