# Building Notes (ORB-SLAM3 + pybind)

This project vendors `ORB_SLAM3_pybind` under `third_party/ORB_SLAM3_pybind` and adds a small Python app on top.

The Python app needs a compiled `orb_slam_pybind` extension from the third-party tree.

## Linux / WSL2 (recommended)

### 1) System deps (Ubuntu example)
```bash
sudo apt update
sudo apt install -y \
  build-essential cmake git pkg-config \
  libeigen3-dev libopencv-dev \
  libboost-serialization-dev libssl-dev
```

### 2) Pangolin
ORB-SLAM3 uses Pangolin for its viewer (even if you run with `--viewer` off, it is often still a build dependency).

```bash
cd /tmp
git clone --depth 1 https://github.com/stevenlovegrove/Pangolin.git
cd Pangolin
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j"$(nproc)"
sudo cmake --install .
```

### 3) Python deps
```bash
python3 -m pip install -U pip
python3 -m pip install -U pybind11 numpy opencv-python pyrealsense2
```

### 4) Build ORB-SLAM3 + pybind module
```bash
cd third_party/ORB_SLAM3_pybind
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j"$(nproc)"
```

### 5) Run
```bash
cd /mnt/d/DroneServer/orbslam   # (WSL2 example)
python3 apps/realsense_rgbd_inertial_topview.py
```

## Native Windows (MSVC + vcpkg)
This repo now builds on Windows using **MSVC** + **vcpkg** (OpenCV/Eigen/Boost/OpenSSL/Pangolin).

### 1) Install C++ deps via vcpkg
```powershell
D:\DroneServer\vcpkg\vcpkg.exe install opencv:x64-windows pangolin:x64-windows openssl:x64-windows boost-serialization:x64-windows
```

### 2) Configure + build ORB-SLAM3 + pybind
```powershell
cd D:\DroneServer\orbslam
cmake -S third_party\ORB_SLAM3_pybind -B third_party\ORB_SLAM3_pybind\build -G "Visual Studio 17 2022" -A x64 `
  -DCMAKE_TOOLCHAIN_FILE=D:\DroneServer\vcpkg\scripts\buildsystems\vcpkg.cmake -DVCPKG_TARGET_TRIPLET=x64-windows `
  -DPython3_EXECUTABLE=D:\DroneServer\python\Scripts\python.exe -DPython3_ROOT_DIR=D:\DroneServer\python
cmake --build third_party\ORB_SLAM3_pybind\build --config Release --target orb_slam_pybind
```

### 3) Run
```powershell
D:\DroneServer\python\Scripts\python.exe apps\realsense_rgbd_inertial_topview.py --auto-exit-s 10
```
