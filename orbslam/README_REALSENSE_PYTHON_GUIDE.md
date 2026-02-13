# ORB-SLAM3 (pybind) + RealSense RGB-D-Inertial Python Integration Guide

This guide documents the crash-resistant integration pattern used in this repo for:

- Intel RealSense D435I (video + depth + gyro + accel)
- ORB-SLAM3 via `orbslam/third_party/ORB_SLAM3_pybind`
- A downstream consumer (local voxel map, obstacle costmap, etc.)

It focuses on the practical issues that cause silent tracking failures or native crashes when calling ORB-SLAM3 from Python.

If you just want to run something:

- Minimal example: `orbslam/apps/realsense_orbslam3_rgbd_inertial_minimal.py`
- Full multi-process app (RealSense + voxels + ORB process isolation): `realsence_3d.py`

## 0) Mental model

You are connecting 4 streams:

1. Video frames (~30Hz) from color or IR-left
2. Depth frames (~30Hz), aligned to the chosen video stream
3. Gyro (~200Hz)
4. Accel (~200Hz)

ORB-SLAM3 IMU-RGBD expects, for each image timestamp `t_img`, a batch of IMU measurements covering `(t_prev, t_img]`
and (in practice) at least one measurement at/after `t_img` so it can integrate to `t_img` robustly.

From Python, you must also ensure the memory backing the input images is stable long enough for ORB-SLAM3's internal threads.

## 1) What ORB-SLAM3 expects (Python side)

### 1.1 Image

- `image`: `np.ndarray` uint8, either `(H,W)` or `(H,W,3)`
- C-contiguous (`image.flags['C_CONTIGUOUS'] == True`)
- Must not be overwritten/reused immediately after `Track*()` returns (see ring buffer rule below)

### 1.2 Depth

Recommended:

- `depthmap`: `np.ndarray` float32 `(H,W)` in meters
- Set `RGBD.DepthMapFactor: 1.0` in YAML

### 1.3 IMU batches

`vImuMeas` is float64 with shape `(N,7)`:

```
ax, ay, az, gx, gy, gz, t_s
```

Rules:

- `t_s` strictly increasing within the batch
- values are finite (no NaNs/Infs)
- units: accel in `m/s^2`, gyro in `rad/s`
- include at least one measurement with `t_s >= t_img`

If these rules are violated ORB-SLAM3 often prints `not IMU meas`, repeatedly resets, and in some cases crashes
inside IMU preintegration due to NaNs.

## 2) The two crash classes (and the fixes)

### 2.1 cv::Mat lifetime / buffer reuse (native crashes)

The pybind layer converts numpy arrays to OpenCV `cv::Mat` by pointer (zero-copy). ORB-SLAM3 is multi-threaded,
and some builds appear to keep references to those `cv::Mat` buffers beyond the `TrackRGBD()` call.

If you pass numpy views backed by buffers that get reused/overwritten (very common when using
`np.asanyarray(frame.get_data())` directly), you can trigger memory corruption later.

Fix:

- Use a ring buffer of numpy arrays and copy each frame into a slot before calling ORB.
- Do not reuse the same slot for at least `ring_n` frames.

This is implemented in:

- `realsence_3d.py` (ORB input ring in the ORB process)
- `orbslam/apps/realsense_orbslam3_rgbd_inertial_minimal.py`

### 2.2 Bad IMU batches (resets, "not IMU meas", Sophus NaNs)

Common failure modes:

- IMU timestamps not strictly increasing
- IMU lags the camera frames (`max(t_imu) < t_img`)
- missing "post-frame" sample (`t_s >= t_img`)

Fix:

- batch IMU between frames using a synchronizer (see `orbslam/orbslam_app/imu_sync.py`)
- enforce monotonic timestamps (drop samples that go backwards)
- ensure at least one sample at/after the image timestamp:
  - best: use the first real gyro sample after the frame timestamp
  - fallback: synthesize a sample by duplicating the last one at `t_img + eps`

## 3) RealSense timestamps: choose a single time domain

RealSense can provide:

- device/hardware timestamps (recommended for stable sensor sync)
- "global time" mapped to system clock (can introduce jitter)

In `realsence_3d.py` we use device timestamps (global time disabled) and align IMU to the camera timestamps via a
small constant offset:

```
t_imu_aligned = t_imu_raw + RS_IMU_TIME_OFFSET_S
```

The sign convention used in this repo is:

- `RS_IMU_TIME_OFFSET_S` is added to IMU timestamps
- after this: `t_imu_aligned ~= t_img` in the same domain

Example calibration output:

- `dt_cam_to_imu_s = -0.013` meaning "IMU lags camera by 13ms"
- therefore set `RS_IMU_TIME_OFFSET_S = +0.013`

## 4) Undistortion strategy (factory intrinsics)

RealSense distortion models include:

- `brown_conrady` / `modified_brown_conrady` (OpenCV undistort supported)
- `inverse_brown_conrady` (OpenCV does not accept the coeffs directly)

This repo uses a consistent policy:

1. Undistort the incoming video + aligned depth using factory intrinsics.
2. Write ORB-SLAM3 YAML with distortion = 0 (pinhole).

Important:

- If ORB runs in another process (multi-process mode), the voxel/map process must also backproject using the same
  undistorted depth model, otherwise the world map will be warped relative to SLAM poses.

See:

- `realsence_3d.py` (undistort map creation and remap)

## 5) Recommended architecture (multi-process)

Native crashes are hard to debug and can kill the whole Python process. This repo's default is to isolate ORB-SLAM3
into its own process.

`realsence_3d.py` runs:

- Capture thread: RealSense pipeline + IMU ring buffer
- ORB process: undistort + `TrackRGBD()` + publishes pose + visualization + alignment events
- Voxel process: per-frame voxelizer + surface/column detectors + sparse hashed voxel submaps (keyed by map_id)
- Open3D viewer process: renders a point cloud view (WASD + Q/E + +/-)

Data exchange:

- `MPSharedLatest`: image/depth frames (shared memory)
- `SharedIMURing`: IMU samples (shared memory ring)
- `MPSharedOrbOut`: pose + small visualization panes (shared memory)
- `align_q`: alignment events from ORB (`PopAlignmentEvents()`) for submap merge/warp

## 6) ORB-SLAM3 loop closures / map merges: keeping your voxel map consistent

ORB-SLAM3 can change its world frame on:

- loop closures (Sim3 correction within the same map)
- atlas map merges (two map_ids become one)

This repo exposes those events via `PopAlignmentEvents()` and uses them to:

- warp a voxel submap in-place on loop closure
- merge voxel submaps on map merge

See:

- ORB process: `mp_orbslam_process` sends events to `align_q`
- Voxel process: `ProcessingThread._drain_external_alignment_events()` applies them

## 7) Programmer checklist (RealSense + IMU + RGB-D)

When writing your own RealSense + IMU consumer, keep this checklist:

1. Choose a time domain and stay in it (prefer device timestamps).
2. Align depth to the video stream you will use for VO (color or IR-left).
3. Convert depth units to meters using `depth_scale`.
4. Use ring buffers for any numpy arrays passed into ORB-SLAM3.
5. Collect IMU samples between frame timestamps and enforce monotonic time.
6. Ensure at least one IMU sample at/after each frame timestamp (real or synthesized).
7. Avoid feeding duplicate or backwards frame timestamps into ORB-SLAM3.
8. If using a distortion-free pinhole YAML, feed undistorted frames consistently.
9. Avoid unstable APIs touching `MapPoint*` unless you have validated your build.
10. Prefer isolating ORB-SLAM3 in its own process during development.

## 8) Debugging knobs (realsence_3d.py)

Useful env vars / flags:

- `RS_IMU_TIME_OFFSET_S`: IMU timestamp offset to add (seconds)
- `ORBSLAM3_INPUT_RING_SIZE`: ring buffer size for ORB inputs
- `ORBSLAM3_POP_ALIGNMENT_EVENTS_ENABLED`: enable/disable alignment event polling
- `OPEN3D_VIEWER_PROCESS_ENABLED`: enable/disable Open3D viewer process

ORB debug log:

- Run `realsence_3d.py --orb-debug-log`
- Output: `D:/DroneServer/_tmp/orb_process_debug.log`

## 9) ORB startup/tracking state machine (realsence_3d.py)

`realsence_3d.py` runs ORB-SLAM3 in a separate process and publishes a small state machine to the main GUI so you
can see what ORB is doing during startup/initialization.

States (see `realsence_3d.py` constants `ORB_STATE_*`):

- `LOADING_VOCAB`: ORB process is alive and constructing the `System` (vocabulary loading can take 20-60s).
- `WAIT_FRAMES`: waiting for the first/second frame (or the camera stream is stalled).
- `WAIT_IMU`: camera frames are arriving but IMU batches are missing/invalid for the current frame.
- `WAIT_MOTION`: IMU is not initialized yet and the motion gate is enabled; move/rotate the camera.
- `INITIALIZING`: ORB is running but has not produced a valid pose yet.
- `TRACKING`: ORB is tracking (pose_valid=1).
- `LOST`: ORB previously tracked but is currently lost.
- `RESETTING`: a timestamp jump/reset was detected and ORB is resetting its internal integration/map.
- `ERROR`: a Python-side exception occurred while calling ORB APIs.

The GUI overlays also show the key numbers that drive this state:

- `pose_seq` / `pose_valid` and `map_id`
- `imu_batch_n`, `imu_lag_ms`, `imu_post_synth`, `imu_initialized`
- tracked keypoint counts (safe) and alignment event count (low-rate polling)
