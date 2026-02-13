from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from orbslam_app.paths import (
    DEFAULT_SETTINGS_MONO,
    DEFAULT_SETTINGS_MONO_INERTIAL,
    DEFAULT_SETTINGS_RGBD,
    DEFAULT_SETTINGS_RGBD_INERTIAL,
    DEFAULT_SETTINGS_STEREO,
    DEFAULT_SETTINGS_STEREO_INERTIAL,
)


@dataclass(frozen=True)
class ModeSpec:
    key: str
    label: str
    sensor: str
    settings: Path
    use_color: bool
    use_depth: bool
    use_ir_left: bool
    use_ir_right: bool
    use_imu: bool

    def uses_rgbd(self) -> bool:
        return bool(self.use_color and self.use_depth)

    def uses_stereo(self) -> bool:
        return bool(self.use_ir_left and self.use_ir_right)

    def uses_mono(self) -> bool:
        return bool(self.use_ir_left and not self.use_ir_right)


MODE_SPECS = [
    ModeSpec(
        key="rgbd",
        label="RGBD",
        sensor="RGBD",
        settings=DEFAULT_SETTINGS_RGBD,
        use_color=True,
        use_depth=True,
        use_ir_left=False,
        use_ir_right=False,
        use_imu=False,
    ),
    ModeSpec(
        key="rgbd_imu",
        label="RGBD+IMU",
        sensor="IMU_RGBD",
        settings=DEFAULT_SETTINGS_RGBD_INERTIAL,
        use_color=True,
        use_depth=True,
        use_ir_left=False,
        use_ir_right=False,
        use_imu=True,
    ),
    ModeSpec(
        key="stereo",
        label="STEREO",
        sensor="STEREO",
        settings=DEFAULT_SETTINGS_STEREO,
        use_color=True,
        use_depth=False,
        use_ir_left=True,
        use_ir_right=True,
        use_imu=False,
    ),
    ModeSpec(
        key="stereo_imu",
        label="STEREO+IMU",
        sensor="IMU_STEREO",
        settings=DEFAULT_SETTINGS_STEREO_INERTIAL,
        use_color=True,
        use_depth=False,
        use_ir_left=True,
        use_ir_right=True,
        use_imu=True,
    ),
    ModeSpec(
        key="mono",
        label="MONO",
        sensor="MONOCULAR",
        settings=DEFAULT_SETTINGS_MONO,
        use_color=True,
        use_depth=False,
        use_ir_left=True,
        use_ir_right=False,
        use_imu=False,
    ),
    ModeSpec(
        key="mono_imu",
        label="MONO+IMU",
        sensor="IMU_MONOCULAR",
        settings=DEFAULT_SETTINGS_MONO_INERTIAL,
        use_color=True,
        use_depth=False,
        use_ir_left=True,
        use_ir_right=False,
        use_imu=True,
    ),
]

MODE_BY_KEY = {mode.key: mode for mode in MODE_SPECS}
MODE_BY_LABEL = {mode.label: mode for mode in MODE_SPECS}
MODE_BY_ID = {idx: mode for idx, mode in enumerate(MODE_SPECS)}
MODE_ID_BY_KEY = {mode.key: idx for idx, mode in enumerate(MODE_SPECS)}
MODE_ID_BY_LABEL = {mode.label: idx for idx, mode in enumerate(MODE_SPECS)}
MODE_ALIASES = {
    "rgbd": "rgbd",
    "rgbdimu": "rgbd_imu",
    "imurgbd": "rgbd_imu",
    "stereo": "stereo",
    "stereoimu": "stereo_imu",
    "imustereo": "stereo_imu",
    "mono": "mono",
    "monoimu": "mono_imu",
    "imumono": "mono_imu",
}


def _normalize_mode_name(name: str) -> str:
    key = str(name).strip().lower()
    for ch in (" ", "_", "-", "+"):
        key = key.replace(ch, "")
    return key


def resolve_mode(name: str) -> ModeSpec:
    key = _normalize_mode_name(name)
    if key in MODE_ALIASES:
        return MODE_BY_KEY[MODE_ALIASES[key]]
    raise ValueError(f"Unknown mode '{name}'. Expected one of: {', '.join(sorted(MODE_ALIASES))}")


def mode_id(mode: ModeSpec) -> int:
    return int(MODE_ID_BY_KEY[mode.key])
