#!/usr/bin/env bash
set -euo pipefail

# setup_pi5.sh
#
# Bring-up helper for the Pi5 test image used for this project.
# This script installs ONLY packages we actually installed during bring-up and
# rebuilds librealsense (RSUSB backend) + python bindings the same way.
#
# Expected folder layout on the Pi:
#   ~/vo_loiter/
#     indoor_loiter/          (this repo; script lives here)
#     orbslam/                (external ORB-SLAM3 tracker)
#     _deps/
#     .venv/
#
# Usage:
#   cd ~/vo_loiter/indoor_loiter
#   bash setup_pi5.sh

log() { echo "[setup] $*"; }

if [[ "${EUID:-0}" -eq 0 ]]; then
  log "Run as a normal user (pi5) with passwordless sudo, not as root."
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

DEPS_DIR="$VO_ROOT/_deps"
RS_DIR="$DEPS_DIR/librealsense"

VENV_DIR="$VO_ROOT/.venv"
VENV_PY="$VENV_DIR/bin/python"

log "vo_root=$VO_ROOT"
log "venv=$VENV_DIR"
log "librealsense=$RS_DIR"

log "Installing apt deps (sudo)..."
sudo apt-get update

# From /var/log/apt/history.log on the Pi bring-up:
# - build deps for librealsense + python wrappers
# - GI typelibs for gstreamer python bindings (gi.repository Gst + GstRtspServer)
# - ORB-SLAM3 pybind deps (OpenCV, Eigen, Boost serialization, Pangolin)
sudo apt-get install -y \
  git \
  cmake \
  build-essential \
  pkg-config \
  libusb-1.0-0-dev \
  libssl-dev \
  libudev-dev \
  python3-dev \
  python3-venv \
  dkms \
  libopencv-dev \
  libeigen3-dev \
  libboost-serialization-dev \
  libgl1-mesa-dev \
  libglu1-mesa-dev \
  libglew-dev \
  libglfw3-dev \
  libepoxy-dev \
  libx11-dev \
  libxext-dev \
  libxi-dev \
  libxrandr-dev \
  libxinerama-dev \
  libxcursor-dev \
  libxxf86vm-dev \
  gir1.2-gstreamer-1.0 \
  gir1.2-gst-plugins-base-1.0 \
  gir1.2-gst-rtsp-server-1.0

log "Checking OpenCV python binding (cv2) ..."
python3 - <<'PY'
import importlib
import sys
try:
    importlib.import_module("cv2")
    print("cv2 OK")
except Exception as e:
    print("cv2 FAIL", type(e).__name__, e, file=sys.stderr)
    raise SystemExit(2)
PY

if systemctl list-unit-files 2>/dev/null | grep -q '^motion\.service'; then
  log "Disabling motion.service (can grab /dev/video0 and block RealSense)..."
  sudo systemctl stop motion || true
  sudo systemctl disable motion || true
fi

if [[ ! -x "$VENV_PY" ]]; then
  log "Creating venv (system-site-packages) at $VENV_DIR"
  python3 -m venv --system-site-packages "$VENV_DIR"
fi

if [[ ! -e "$VO_ROOT/indoor_loiter/.venv" ]]; then
  log "Linking indoor_loiter/.venv -> ../.venv (lets server.py auto re-exec into the venv)..."
  ln -s "$VENV_DIR" "$VO_ROOT/indoor_loiter/.venv"
fi

log "Ensuring pymavlink is installed in the venv..."
"$VENV_PY" - <<'PY'
import importlib
import sys
try:
    importlib.import_module("pymavlink")
    print("pymavlink OK")
except Exception as e:
    print("pymavlink MISSING", type(e).__name__, e, file=sys.stderr)
    raise SystemExit(2)
PY || "$VENV_PY" -m pip install "pymavlink==2.4.49"

log "Ensuring PyYAML is installed in the venv..."
"$VENV_PY" - <<'PY'
import importlib
import sys
try:
    importlib.import_module("yaml")
    print("PyYAML OK")
except Exception as e:
    print("PyYAML MISSING", type(e).__name__, e, file=sys.stderr)
    raise SystemExit(2)
PY || "$VENV_PY" -m pip install "PyYAML==6.0.2"

if [[ ! -d "$RS_DIR/.git" ]]; then
  log "Cloning librealsense into $RS_DIR"
  mkdir -p "$DEPS_DIR"
  git clone https://github.com/IntelRealSense/librealsense.git "$RS_DIR"
  # Pin to the commit used during bring-up (Jan 29, 2026).
  (cd "$RS_DIR" && git checkout 3d264cad1da6503364e214613eaa01d6ab92b62e) || true
fi

if [[ -f "$RS_DIR/config/99-realsense-libusb.rules" ]]; then
  log "Installing RealSense udev rules..."
  sudo cp -f "$RS_DIR/config/99-realsense-libusb.rules" /etc/udev/rules.d/99-realsense-libusb.rules
  sudo udevadm control --reload-rules || true
  sudo udevadm trigger || true
fi

BUILD_DIR="$RS_DIR/build_rsusb"
mkdir -p "$BUILD_DIR"

log "Configuring librealsense (RSUSB backend, python bindings)..."
cmake -S "$RS_DIR" -B "$BUILD_DIR" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=/usr/local \
  -DFORCE_RSUSB_BACKEND=ON \
  -DFORCE_LIBUVC=OFF \
  -DBUILD_PYTHON_BINDINGS=ON \
  -DPYTHON_EXECUTABLE="$VENV_PY" \
  -DBUILD_EXAMPLES=OFF \
  -DBUILD_GRAPHICAL_EXAMPLES=OFF \
  -DBUILD_TOOLS=ON \
  -DBUILD_UNIT_TESTS=OFF \
  -DBUILD_WITH_OPENMP=OFF

log "Building librealsense..."
cmake --build "$BUILD_DIR" -- -j"$(nproc)"

log "Installing librealsense..."
sudo cmake --install "$BUILD_DIR"
sudo /sbin/ldconfig || true

log "Smoke test: python imports (system python)"
python3 - <<'PY'
import importlib
mods = ["cv2", "numpy", "pyrealsense2", "gi"]
for m in mods:
    try:
        importlib.import_module(m)
        print(m, "OK")
    except Exception as e:
        print(m, "FAIL", type(e).__name__, e)
PY

log "Smoke test: python imports (venv python)"
"$VENV_PY" - <<'PY'
import importlib
mods = ["cv2", "numpy", "pyrealsense2", "gi", "pymavlink", "yaml"]
for m in mods:
    try:
        importlib.import_module(m)
        print(m, "OK")
    except Exception as e:
        print(m, "FAIL", type(e).__name__, e)
PY

if command -v rs-enumerate-devices >/dev/null 2>&1; then
  log "RealSense enumerate:"
  rs-enumerate-devices | sed -n '1,120p' || true
else
  log "rs-enumerate-devices not found on PATH."
fi

PANG_DIR="$DEPS_DIR/Pangolin"
PANG_BUILD="$PANG_DIR/build"
PANG_CMAKE="/usr/local/lib/cmake/Pangolin/PangolinConfig.cmake"

if [[ ! -f "$PANG_CMAKE" ]]; then
  if [[ ! -d "$PANG_DIR/.git" ]]; then
    log "Cloning Pangolin into $PANG_DIR"
    mkdir -p "$DEPS_DIR"
    git clone --depth 1 https://github.com/stevenlovegrove/Pangolin.git "$PANG_DIR"
  fi

  log "Building + installing Pangolin (needed by ORB-SLAM3 Viewer build)..."
  mkdir -p "$PANG_BUILD"
  cmake -S "$PANG_DIR" -B "$PANG_BUILD" \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_EXAMPLES=OFF \
    -DBUILD_TOOLS=OFF \
    -DBUILD_PANGOLIN_PYTHON=OFF
  cmake --build "$PANG_BUILD" -- -j"$(nproc)"
  sudo cmake --install "$PANG_BUILD"
  sudo /sbin/ldconfig || true
else
  log "Pangolin already installed ($PANG_CMAKE)"
fi

ORB_PYBIND_DIR="$VO_ROOT/orbslam/third_party/ORB_SLAM3_pybind"
ORB_PYBIND_BUILD="$ORB_PYBIND_DIR/build_rpi"

if [[ -d "$ORB_PYBIND_DIR" ]]; then
  VOC_DIR="$ORB_PYBIND_DIR/Vocabulary"
  VOC_TXT="$VOC_DIR/ORBvoc.txt"
  VOC_TGZ="$VOC_DIR/ORBvoc.txt.tar.gz"
  if [[ -d "$VOC_DIR" ]]; then
    if [[ ! -f "$VOC_TXT" ]]; then
      if [[ -f "$VOC_TGZ" ]]; then
        log "Extracting ORBvoc.txt from $VOC_TGZ (GitHub repo stores the compressed vocab; raw file is >100MB)..."
        tar -xzf "$VOC_TGZ" -C "$VOC_DIR"
      else
        log "WARN: ORB vocabulary missing ($VOC_TXT) and no archive found ($VOC_TGZ). ORB-SLAM3 will not start."
      fi
    else
      log "ORB vocabulary present ($VOC_TXT)"
    fi
  fi

  if ! ls "$ORB_PYBIND_DIR/python_wrapper/"orb_slam_pybind*.so >/dev/null 2>&1; then
    log "Building ORB_SLAM3_pybind (pybind11 module) ..."
    mkdir -p "$ORB_PYBIND_BUILD"
    cmake -S "$ORB_PYBIND_DIR" -B "$ORB_PYBIND_BUILD" \
      -DCMAKE_BUILD_TYPE=Release \
      -DPYTHON_EXECUTABLE="$VENV_PY"
    cmake --build "$ORB_PYBIND_BUILD" -- -j"$(nproc)" --target orb_slam_pybind
  else
    log "ORB_SLAM3_pybind already built (python_wrapper/orb_slam_pybind*.so exists)"
  fi

  log "Smoke test: ORB-SLAM3 python binding import (venv python)"
  (cd "$ORB_PYBIND_DIR" && "$VENV_PY" -c 'from python_wrapper.orb_slam3 import ORB_SLAM3; print(1)')
else
  log "WARN: ORB_SLAM3_pybind not found at $ORB_PYBIND_DIR (skipping build)"
fi

log "Done."
