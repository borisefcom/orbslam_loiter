# ORB_SLAM3_pybind - Python API Notes (Repo-Specific)

This repo uses the vendored ORB-SLAM3 pybind wrapper under:

- `orbslam/third_party/ORB_SLAM3_pybind/python_wrapper/`

This document focuses on Python binding gotchas that matter for stability.

For the full RealSense + IMU + timing guide, see:

- `orbslam/README_REALSENSE_PYTHON_GUIDE.md`

## 1) What the binding does (important for safety)

The C++ binding converts numpy arrays to OpenCV `cv::Mat` like this:

- it requires C-contiguous arrays
- it creates a `cv::Mat` that points at the numpy buffer (`mutable_data()`), i.e. zero-copy

Therefore:

- if the numpy buffer is overwritten/reused while ORB-SLAM3 still holds a `cv::Mat` reference, you can get native
  crashes later (access violations) or random behavior

Practical rule: treat the image/depth arrays passed to `Track*()` as owned by ORB-SLAM3 for a short time.
Use a ring buffer or deep copy.

## 2) IMU measurement format

`vImuMeas` is a float64 array with shape `(N,7)`:

```
[ax, ay, az, gx, gy, gz, t_s]
```

- units: accel in `m/s^2`, gyro in `rad/s`
- `t_s` must be strictly increasing
- include at least one measurement at/after the image timestamp (`t_s >= t_img`)

If these rules are violated, ORB-SLAM3 can:

- print `not IMU meas`
- reset repeatedly ("not enough motion" / "bad imu flag")
- in worst cases crash inside IMU preintegration due to NaNs

## 3) Safe visualization APIs

This repo adds a safe API:

- `GetTrackedKeyPointsUn()` -> `(N,2)` float32 (undistorted keypoints)

This is safe because it returns value types (no pointer lifetimes).

Some builds have been observed to be unstable when frequently calling APIs that touch `MapPoint*` lifetimes
(e.g. tracked map points across map resets/merges). Prefer keypoints unless you validated stability.

## 4) Alignment / loop-closure events

This repo exposes `PopAlignmentEvents()` which returns a list of alignment events (Sim3 transforms) when ORB-SLAM3:

- performs a loop closure correction (same map)
- merges maps (Atlas multi-map merge)

Recommended:

- poll at low rate (e.g. ~1Hz), not every frame

## 5) Wrapper file to read

High-level Python wrapper:

- `orbslam/third_party/ORB_SLAM3_pybind/python_wrapper/orb_slam3.py`
