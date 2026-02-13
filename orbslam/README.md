# ORB-SLAM3 (pybind) + RealSense (RGB-D-Inertial)

This repo vendors ORB-SLAM3 + a Python binding and provides RealSense-focused integration glue used by:

- Production multi-process app: `realsence_3d.py`
- Minimal RealSense example: `orbslam/apps/realsense_orbslam3_rgbd_inertial_minimal.py`
- Synthetic sanity test (no camera): `orbslam/apps/synth_rgbd_telemetry_test.py`

## Practical Python Guide (Recommended)

For the full integration details (timestamps, IMU batching, buffer lifetimes, undistortion, and the crash fixes we learned):

- `orbslam/README_REALSENSE_PYTHON_GUIDE.md`

Repo-specific binding notes:

- `orbslam/third_party/ORB_SLAM3_pybind/python_wrapper/README_PYTHON_API.md`

## Build

See:

- `orbslam/BUILDING.md`

Vocabulary extraction:

- `orbslam/third_party/ORB_SLAM3_pybind/Vocabulary/ORBvoc.txt.tar.gz` -> `ORBvoc.txt`

## License

ORB-SLAM3 is GPLv3; accordingly any redistributed binaries that include it must be GPLv3-compatible.
