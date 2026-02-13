# Agent Notes (ORB-SLAM3 + RealSense + Python)

These notes are for humans and coding agents working in `orbslam/` and `orbslam/third_party/ORB_SLAM3_pybind/`.

## Primary references

- Integration guide: `orbslam/README_REALSENSE_PYTHON_GUIDE.md`
- Minimal example: `orbslam/apps/realsense_orbslam3_rgbd_inertial_minimal.py`
- Production multi-process app: `realsence_3d.py`

## Golden rules (avoid the crashes we already hit)

1. **Never pass ephemeral numpy views into ORB-SLAM3.**
   - The pybind layer converts numpy -> `cv::Mat` by pointer (zero-copy).
   - Some ORB-SLAM3 builds keep references after `Track*()` returns.
   - Use a **ring buffer** for image + depth arrays (or deep-copy) before calling `Track*()`.

2. **IMU batches must be valid and monotonic.**
   - `vImuMeas` is `(N,7)` float64: `[ax,ay,az,gx,gy,gz,t_s]`.
   - `t_s` must be strictly increasing.
   - Include at least one IMU measurement at/after the image timestamp (real or synthesized).

3. **Undistort consistently.**
   - We feed ORB-SLAM3 a distortion-free pinhole model (distortion set to 0 in YAML).
   - Therefore the image+depth we feed ORB must be undistorted with the same intrinsics.
   - If ORB runs in a different process, the voxel/map process must also backproject using the same undistorted depth model.

4. **Avoid unstable ORB APIs unless validated.**
   - `GetTrackedMapPoints()` can be unstable on some custom builds.
   - Prefer `GetTrackedKeyPointsUn()` for visualization.
   - Poll `PopAlignmentEvents()` at low rate (e.g. 1Hz), not per-frame.

5. **Write runtime YAML atomically.**
   - Use write-to-temp then `os.replace()` to prevent corrupted YAML reads.

## If you modify the C++ pybind code

Files:
- `orbslam/third_party/ORB_SLAM3_pybind/python_wrapper/orb_slam3_pybind.cpp`
- `orbslam/third_party/ORB_SLAM3_pybind/src/System.cc` (custom APIs)

Rebuild (Windows):
`cmake --build orbslam/third_party/ORB_SLAM3_pybind/build --config Release --target orb_slam_pybind`

Rebuild (Linux/WSL):
`cmake --build orbslam/third_party/ORB_SLAM3_pybind/build -j`

## Debugging tips

- Enable ORB debug log: run `realsence_3d.py --orb-debug-log`
- Logs are written/rotated under `D:/DroneServer/_tmp/` (see console for exact paths).

