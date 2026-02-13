from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
THIRD_PARTY_ORB = REPO_ROOT / "third_party" / "ORB_SLAM3_pybind"

DEFAULT_VOCAB_TAR_GZ = THIRD_PARTY_ORB / "Vocabulary" / "ORBvoc.txt.tar.gz"
DEFAULT_VOCAB_TXT = THIRD_PARTY_ORB / "Vocabulary" / "ORBvoc.txt"
DEFAULT_SETTINGS_RGBD_INERTIAL = THIRD_PARTY_ORB / "Examples" / "RGB-D-Inertial" / "RealSense_D435i.yaml"
DEFAULT_SETTINGS_RGBD = THIRD_PARTY_ORB / "Examples" / "RGB-D" / "RealSense_D435i.yaml"
DEFAULT_SETTINGS_STEREO = THIRD_PARTY_ORB / "Examples" / "Stereo" / "RealSense_D435i.yaml"
DEFAULT_SETTINGS_STEREO_INERTIAL = THIRD_PARTY_ORB / "Examples" / "Stereo-Inertial" / "RealSense_D435i.yaml"
DEFAULT_SETTINGS_MONO = THIRD_PARTY_ORB / "Examples" / "Monocular" / "RealSense_D435i.yaml"
DEFAULT_SETTINGS_MONO_INERTIAL = THIRD_PARTY_ORB / "Examples" / "Monocular-Inertial" / "RealSense_D435i.yaml"
