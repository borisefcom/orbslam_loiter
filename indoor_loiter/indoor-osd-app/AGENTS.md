# Indoor OSD App — Ground station GUI

This folder contains the **client GUI** that connects over Wi‑Fi to the drone server in the repo root.

It renders:

- Video (RTSP or RTP/UDP via GStreamer)
- MAVLink telemetry (UDP, via `pymavlink`)
- Drone JSON telemetry/overlays (UDP JSON: pose/sys/detect/track_status/pid_debug/acq_poly/vo_features/camera/track2d)

## Quick start

- Entrypoint: `main.py`
- Config:
  - `app_config.json` (networking, video settings, control defaults, debug logging)
  - `osd_config.json` (OSD layout)
  - `map_config.json` (tile providers; optional internet access)
- Run: `python main.py`

Notes:

- This app uses `gi`/GStreamer, GLFW, OpenGL, and ImGui bindings. On Windows/Linux you’ll need a working GStreamer install and the corresponding Python packages.

## Networking (default ports)

All values are configurable via `app_config.json`.

- **MAVLink in** (from drone mirror): `mavlink.endpoint` (default `udpin:0.0.0.0:14550`)
- **JSON telemetry in**: `control.telemetry_ip`/`control.telemetry_port` (default bind `0.0.0.0:6021`)
- **Control out** (to drone server): `control.ctrl_ip`/`control.ctrl_port` (default `:6020`)
- **Video**
  - RTSP: `video.rtsp_url` (e.g. `rtsp://<drone-ip>:8554/stream`)
  - UDP: `video.source="udp"` and `video.udp_port` (default `5600`)

## Startup routing behavior (important)

On launch, `main.py`:

1. Starts MAVLink (`TelemetryWorker`) and JSON telemetry (`JsonTelemetryWorker`) listeners.
2. Detects the local IP used to reach the drone (`_local_ip_for_peer`).
3. Sends UDP JSON control messages to the drone (retry loop) to request routing:
   - `{"telemetry_ip": <this_gui_ip>, "telemetry_port": 6021}`
   - `{"mav_mirror_ip": <this_gui_ip>}` (PX4 MAVLink mirror)
   - `{"dest_ip": <this_gui_ip>, "dest_port": <video.udp_port>}` (only relevant if drone streams UDP)

These messages are handled by the drone’s `TelemetryManager.handle_control()` in `telemetry.py`.

Additional control message (future functionality):

- `{"type":"mouse_move", ...}` sent while the cursor moves over the main fullscreen video in OSD mode (not over PiP widgets or UI).
- `{"type":"mouse_click","button":"left|right", ...}` sent on click over the main fullscreen video in OSD mode (not over PiP widgets or UI).
- `{"type":"hole_enable","enable":0|1}` runtime toggle for the drone's 3D hole acquisition (stage-0 preview + click-to-lock)
- `{"type":"plane_enable","enable":0|1}` runtime toggle for the drone's 3D plane acquisition (stage-0 preview + click-to-lock; mutually exclusive with hole)
- `{"type":"track2d_enable","enable":0|1}` runtime toggle for the 2D ROI tracker (server selects the V4L stream when enabled)
- `{"type":"track2d_rect","surface":"osd_main","x0_norm":...,"y0_norm":...,"x1_norm":...,"y1_norm":...}` start 2D ROI tracking (sent once on LMB drag release)
- `{"type":"track2d_cancel"}` cancel 2D ROI tracking

## UDP JSON protocol expectations

Parsing lives in `mavlink.py` (`JsonTelemetryWorker._apply_json`).

Incoming `type` values currently handled:

- `pose`
- `sys`
- `cellular` (may be disabled server-side; UI currently repurposes the widget area for `sys`)
- `detect` (detections + image size, used for clicking/ROI selection)
- `track_status`
- `pid_debug`
- `acq_poly`
- `vo_features`
- `camera`
- `track2d`

Control UI (`gui.py:ControlView`) sends key-based control dicts (no `type` field). Not all keys currently have a live effect on the server (see “Known gotchas” below).

## Tracking flow (signals + coordinate spaces)

The GUI does not implement MedianFlow itself. It drives the server-side tracker via UDP control messages and renders the server’s `detect/track_status/pid_debug` outputs.

### What the server sends (and how the UI uses it)

Parsed in `mavlink.py:JsonTelemetryWorker._apply_json`:

- `type="detect"`
  - Contains `img_size: [W,H]` and `detections: [...]` with `bbox_px` and `id` fields.
  - `osd.py:DetectionsOverlay` uses `telem.detect_img_size` to map between:
    - framebuffer coords (mouse + drawing) and
    - detection image coords (bbox pixels the server expects for ROI selection).
- `type="track_status"`
  - Contains `track_id`, `pid_source`, and normalized error `error.ex/error.ey` in [-1,+1].
  - This is not an angle; it’s a normalized pixel-center error around the image center.
- `type="pid_debug"`
  - Used by the PID Tune screen.
  - Contains the **horizontal/vertical angle deltas** (`measurement.yaw_err_*`, `measurement.pitch_err_*`) computed on the server from the tracked center pixel + camera intrinsics.
  - `gate_state` indicates which PID stage is active (`PID_ALIGN` vs `PID_MOVE`).
  - Note: `result.pitch` and `result.thrust` are internal normalized values and are mapped to MAVLink `MANUAL_CONTROL` in a non-obvious way on the server (throttle is centered at 0.5; forward “thrust” is sent on the pitch/x axis).
- `type="acq_poly"`
  - Polygon overlay for the 3D acquisition flow (preview/selected/confirmed).
  - Contains `stage: 0|1|2`, `img_size: [W,H]`, and `verts_uv: [[u,v], ...]` (or null to clear).
- `type="vo_features"`
  - Lightweight feature overlay points from the 3D tracker (for debugging VO/LK quality).
  - Contains `img_size: [W,H]` and `pts: [[u,v,group], ...]` where `group` maps to a display color.
  - Group semantics are backend-dependent:
    - OpenVINS: currently buckets by track depth (near/mid/far)
    - Legacy depth tracker: `1`=tracked keypoints, `2/3`=polygon LK state
- `type="camera"`
  - Sent on startup + whenever the server switches the active video stream.
  - Contains `active: "ir"|"depth"|"v4l"`, optional `img_size: [W,H]`, and optional `intrinsics` (`fx/fy/cx/cy`).
- `type="track2d"`
  - 2D ROI tracker bbox (drawn when `camera.active=="v4l"`).
  - Carries bbox in normalized coords (`x0_norm/y0_norm/x1_norm/y1_norm`) or equivalent fields.

### What the user does (and what gets sent to the server)

For the 3D acquisition flow, the GUI uses mouse events over the main fullscreen video (OSD tab only):

- While moving (only when `hole_enable=1` or `plane_enable=1`): `{"type":"mouse_move","surface":"osd_main","x_norm":0..1,"y_norm":0..1,"ts":...}`
- Commit/cancel clicks (only when `hole_enable=1` or `plane_enable=1`): `{"type":"mouse_click","button":"left|right","surface":"osd_main","x_norm":0..1,"y_norm":0..1,"ts":...}`

For the 2D ROI flow (only when `track2d_enable=1`):

- Start ROI once (on LMB drag release): `{"type":"track2d_rect","surface":"osd_main","x0_norm":...,"y0_norm":...,"x1_norm":...,"y1_norm":...}`
- Cancel (RMB): `{"type":"track2d_cancel"}`

Legacy click-to-track + ROI selection messages (`track_select`, `roi_select`, `track_cancel`) are intentionally disabled in the GUI for now.

## Video decoding & performance

Video receive/parse/decode is implemented in `stream.py`:

- Pipelines are built for low latency (leaky queues, `sync=false`, jitterbuffer `latency=0` for UDP).
- If CPU-bound:
  - reduce stream resolution/FPS/bitrate on the drone
  - prefer H.264 over H.265 (often cheaper to decode in software)
  - consider using platform hardware decoders (GStreamer elements differ by OS; `uridecodebin`/`decodebin` can help auto-select)

## Known gotchas (server/client mismatches)

- Some controls exposed in the GUI (bitrate/GOP/RC mode/stream method, “structural” fps/size) are currently **printed by the server** but not fully applied live because `gstreamer.VideoController.apply_control()` only updates `udp_sink`.
- `stream_method` naming differs:
  - server expects `"udp"` or `"rtsp"` (see `gstreamer.py`)
  - GUI currently uses `"UDP_UNICAST"` in places (treated as an unknown value unless normalized).
- GUI can send `mavlink_port`, but the server control handler currently only supports changing mirror IP (port is taken from config/defaults).

## Change guidelines

- Keep the render loop non-blocking; do all network/video work in background threads.
- If you add/modify a JSON message type or control key:
  - update both sides (server emitter + client parser)
  - keep payloads small (avoid UDP fragmentation)
  - update both `AGENTS.md` files with the new schema/behavior
