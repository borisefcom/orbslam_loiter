# Integration: indoor_loiter + ORB-SLAM3 external tracker (staging)

This staging tree replaces the legacy multiprocess VO pipeline with ORB-SLAM3 as the only VO source.

Folder layout:

- `orbslam_intoor_loiter/indoor_loiter/` (drone-side server + GUI app)
- `orbslam_intoor_loiter/orbslam/` (external ORB-SLAM3 RGB-D-Inertial tracker + custom Python binding)

## How it works

1. `indoor_loiter/server.py` spawns the external tracker configured by `indoor_loiter/drone.yaml:track3d.*`.
2. The tracker (`orbslam/apps/realsense_orbslam3_rgbd_inertial_minimal.py`) owns the RealSense device, runs ORB-SLAM3 VO, and publishes:
   - Shared-memory frame ring (`ring_spec` in the `init` JSON event)
   - Shared-memory VO pose state (`ipc_vo` shm_state compatible with Drone_client)
   - Headless JSONL events (`init`, `preview`, `telemetry`, `pid`, `features`)
3. The server attaches to the SHM ring for streaming (`video.external_capture: true`) and forwards tracker events into the existing UDP JSON overlays + PID flow.

## Config files

- Server config: `indoor_loiter/drone.yaml`
  - `track3d.workdir: ../orbslam`
  - `track3d.script: apps/realsense_orbslam3_rgbd_inertial_minimal.py`
  - `track3d.config: apps/orbslam_track3d.yaml`
- Tracker config: `orbslam/apps/orbslam_track3d.yaml`
  - Enables headless JSON I/O + SHM rings + polygon/hole tracker
  - Disables PX4 serial output in the tracker (the server handles PX4 injection if enabled)

## Headless protocol (JSONL)

Server -> tracker (stdin JSONL):
- `{"cmd":"hover","x":px,"y":px}`
- `{"cmd":"select_hole","x":px,"y":px}`
- `{"cmd":"confirm_hole","x":px,"y":px}`
- `{"cmd":"select_bbox","bbox":[x0,y0,x1,y1]}`
- `{"cmd":"clear"}`

Tracker -> server (stdout JSONL):
- `type="init"`: intrinsics + SHM ring spec
- `type="preview"`: stage-0 candidate polygon
- `type="telemetry"`: pose + polygon/hole state
- `type="pid"`: `dhv_deg` stream for PID stages
- `type="features"`: keypoint overlay points

### Stage 2: confirmed (PID active)

- User left-clicks the refined polygon:
  - server sends `cmd=confirm_hole`.
- The server begins feeding the 3D tracker's `pid`/`dhv_deg` into the existing PID loop and MAVLink output path (still RC-gated for safety, unless we decide otherwise).
- GUI draws:
  - purple polygon outline
  - purple filled overlay with alpha blend (alpha: 0.33)
  - green "X" at polygon center

### Cancel/clear gesture

- User right-click:
  - server sends `cmd=clear`
  - GUI clears any polygon overlays and returns to Stage 0.

## Message mapping

### Client -> server (already present)

Mouse messages (only sent over the main OSD video, not over PiP/config screens):

- `{"type":"mouse_move","surface":"osd_main","x_norm":0..1,"y_norm":0..1, ...}`
- `{"type":"mouse_click","button":"left|right","surface":"osd_main","x_norm":0..1,"y_norm":0..1, ...}`
- `{"type":"mouse_stop", ...}` (optional; likely not required for the new flow)

### Client -> server (2D ROI tracker)

The 2D ROI tracker is exclusive with the 3D mouse-driven acquisition flow. When enabled, the server
switches the streamed video feed to the V4L camera (non-RealSense) so the GUI ROI selection matches
the tracker camera.

- Enable/disable: `{"type":"track2d_enable","enable":0|1}`
- Start tracking from a rectangle (normalized coords):  
  `{"type":"track2d_rect","surface":"osd_main","x0_norm":0..1,"y0_norm":0..1,"x1_norm":0..1,"y1_norm":0..1}`
- Cancel/stop: `{"type":"track2d_cancel"}`

### Server -> client (new)

Add a new UDP JSON message type for polygon overlays, example:

`type="acq_poly"`

Fields (proposed):
- `stage`: 0|1|2 (UI stage)
- `img_size`: `[W,H]` (pixel basis for `verts_uv`)
- `verts_uv`: `[[u,v], ...]` polygon vertices in pixels (same basis as `img_size`)
- `center_uv`: `[u,v]` (polygon center in pixels)
- `source`: `"3d_track"`
- optional: `reason`, `ok_pid`, `metrics` (kept small to avoid UDP fragmentation)

The client should treat this as an overlay that is separate from legacy `detect/track_status`.

Additional server -> client message used for multi-camera overlay gating:

`type="camera"`

Fields:
- `active`: `"ir"` | `"depth"` | `"v4l"`
- `img_size`: `[W,H]` pixels
- optional: `device`, `fps`, `format`, `intrinsics`

Client behavior:
- If `active="v4l"`, hide `acq_poly` and `vo_features` overlays (they belong to the track3d stream).
- If `active!="v4l"` (IR/depth), hide the `track2d` bbox overlay (it belongs to the V4L stream).

## Bridging to the 3D tracker headless commands/events

### Server -> 3D tracker commands (stdin JSONL)

Coordinate conversion:
- 3D tracker expects `x/y` in pixels of its capture/output resolution (`capture.out_width/out_height`).
- Server uses `mouse_move.x_norm/y_norm` and multiplies by `[W,H]` to compute `x/y`.

Command mapping:
- Stage 0 hover: `{"cmd":"hover","x":<px>,"y":<px>}`
- Stage 1 select: `{"cmd":"select_hole","x":<px>,"y":<px>}` (or `select_bbox`)
- Stage 2 confirm: `{"cmd":"confirm_hole"}`
- Cancel: `{"cmd":"clear"}`

### 3D tracker -> server events (stdout JSONL)

Consume:
- `type="telemetry"`:
  - `payload.poly.verts_uv` and `payload.poly.pid_confirmed` are the main overlay sources.
- `type="pid"`:
  - `payload.dhv_deg` provides (horizontal_deg, vertical_deg).
  - `payload.stage` is 1 or 2 (selection vs confirmed).
  - `payload.ok_pid` can be used as an additional safety gate.

Server responsibilities:
- Translate 3D tracker polygon state into `type=acq_poly` for the GUI.
- Translate 3D tracker `dhv_deg` into the existing server PID measurement path.

## PID integration details (keep existing MAVLink mapping)

The 3D tracker emits `dhv_deg=(h_deg, v_deg)` computed from camera intrinsics:
- `h_deg = atan2(u - cx, fx)` in degrees
- `v_deg = atan2(v - cy, fy)` in degrees

Sign conventions: the 3D tracker `dhv_deg` is computed using the same convention as the current server:
- `h_deg > 0` means the target is to the right of center.
- `v_deg > 0` means the target is below center (positive down in image).
This matches the server's `yaw_err=atan((cx_px-cx)/fx)` and `pitch_cam=atan((cy_px-cy)/fy)` path in `server.py`.

Stage behavior mismatch to resolve:
- 3D tracker stage (1/2) is "selected vs confirmed".
- Server PID stages (PID_ALIGN vs PID_MOVE) are RC-gated and intentionally different.

Proposed rule (safe default):
- Only allow PID measurements from the 3D tracker when it is confirmed (stage 2 / pid_confirmed=1).
- Keep RC gating in place for whether commands are sent to PX4.

## Implementation phases (recommended order)

### Phase 1: run 3D tracker headless under server control (no video changes yet)

Goal: validate command/event bridging in isolation.

- Add a server-side "3d_track bridge" module:
  - spawns `3d_track.py --headless --config <yaml>`
  - writes stdin JSONL commands
  - reads stdout JSONL events
  - exposes latest polygon + latest `dhv_deg`
- Implement a minimal synthetic overlay message (`acq_poly`) to the GUI from 3D tracker `telemetry`.
- No MAVLink/PID actions yet; just telemetry visualization.

### Phase 2: integrate video ownership (choose Option A or B)

If Option A:
- 3D tracker capture process owns RealSense.
- Server streams from the 3D tracker's shared-memory ring (no direct RealSense access).
- Use a deterministic shared-memory name prefix in the 3D tracker capture config so the server can attach reliably.

If Option B:
- Extend server capture to also produce aligned depth + IMU to the tracking backend.
- Feed 3D tracking code from server capture.

### Phase 3: implement hover preview polygons in headless mode

Two possible approaches:

1) Extend the 3D tracker headless output:
   - When `cmd=hover` arrives and no polygon is active, compute a hole preview polygon (same logic as GUI mode) and emit it as a headless event (new type, or include in `telemetry`).

2) Implement preview in the server:
   - Port/borrow the 3D tracker's `HoleDetector` logic into the server and compute preview there using depth.
   - This is only feasible if the server has access to aligned depth frames.

Recommendation: extend the 3D tracker headless output (keeps depth+intrinsics local to that codebase).

### Phase 4: connect confirmation to the existing PID loop

- When 3D tracker stage becomes confirmed:
  - feed its `dhv_deg` into the existing PID manager as the measurement (instead of MedianFlow-derived center pixel).
- Keep:
  - RC gate behavior
  - MANUAL_CONTROL scaling/mapping
  - PID stage split behavior (PID_ALIGN vs PID_MOVE) unless explicitly changed later

### Phase 5: config-driven backend selection + cleanup

- Add a `tracker.backend` (or similar) config key to select:
  - `medianflow` (existing)
  - `3d_track` (new)
- Keep both paths for rollback until flight-tested.

## Debug plan

3D tracker GUI mode (for debugging):
- Run `3d_track.py --config configs/3d_track_current.yaml --gui`

3D tracker headless mode (integration shim):
- Run `3d_track.py --config configs/3d_track_current.yaml --headless`
- Send commands as JSONL on stdin and observe JSON events on stdout.

## Risks / likely issues

- RealSense device contention if both the server and the 3D tracker try to open it.
- Shared-memory naming/discovery between independent processes (if Option A).
- UDP packet size if polygon vertex lists get large (keep overlays compact).
- Sign/axis mismatches between `dhv_deg` and the existing PID expectations.
- Operator confusion if stage colors or click semantics are inconsistent; keep UI state explicit.

## Open questions (need answers before coding integration)

Answered (current decisions):
- Capture is owned by a dedicated shared ringbuffer process (3D tracker capture); server streams from the ring (no direct RealSense access).
- Keep support for bbox->MedianFlow->PID, but do not wire it into the GUI client yet.
- Wire 3D tracker `dhv_deg` into the existing server PID->MAVLink mechanism; keep RC gates and PID stages/config logic unchanged.
- Match resolution and grayscale (IR1).
- Extend headless output to include preview polygons.
- Mouse clicks: left=select/confirm, right=clear.
- Overlay styling:
  - Stage 0: yellow polygon, red X, no fill
  - Stage 1: purple polygon, yellow X, no fill
  - Stage 2: purple polygon, purple fill alpha=0.33, green X
- Server spawns/manages `3d_track.py`.
- Deterministic SHM prefix is allowed.
- Disable 3D tracker stop logic in its YAML; stop is managed by the server.
