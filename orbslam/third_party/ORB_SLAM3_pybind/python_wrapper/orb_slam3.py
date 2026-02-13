from __future__ import annotations

import numpy as np

from python_wrapper import orb_slam_pybind as orb


class ORB_SLAM3:
    def __init__(self, vocabulary: str, config: str, sensor: str, vis: bool):

        """
        Initializes the ORB-SLAM3 pipeline (Python wrapper over the pybind `_System`).

        IMPORTANT (stability):
        The pybind layer converts numpy arrays to OpenCV `cv::Mat` by *pointer* (zero-copy).
        Some ORB-SLAM3 builds keep references to those `cv::Mat` buffers longer than a single `Track*()` call.
        If you pass arrays backed by buffers that get overwritten/reused immediately (common with
        `np.asanyarray(frame.get_data())` from RealSense), you can hit native crashes later.

        Practical rule:
        - Use a ring buffer (or deep copies) for the image/depth arrays you pass to `Track*()`.
        - Ensure arrays are C-contiguous.

        Args:
            vocabulary: path to vocabulary
            config: path to config file
            sensor: sensor type case-insensitive (eg: "monocular", "stereo"...)
            vis: on/off visualisation

        """
        self.supported_sensors = {
            "MONOCULAR": orb._System.eSensor.MONOCULAR,
            "STEREO": orb._System.eSensor.STEREO,
            "RGBD": orb._System.eSensor.RGBD,
            "IMU_MONOCULAR": orb._System.eSensor.IMU_MONOCULAR,
            "IMU_STEREO": orb._System.eSensor.IMU_STEREO,
            "IMU_RGBD": orb._System.eSensor.IMU_RGBD,
        }
        eSensor = self.supported_sensors[sensor.upper()]
        # initialise orb slam here
        self.slam = orb._System(vocabulary, config, eSensor, vis)

    @staticmethod
    def _require_contiguous(arr: np.ndarray, *, name: str) -> None:
        if not isinstance(arr, np.ndarray):
            raise TypeError(f"{name} must be a numpy array, got {type(arr)!r}")
        if not bool(arr.flags["C_CONTIGUOUS"]):
            raise ValueError(
                f"{name} must be C-contiguous (got non-contiguous view). "
                "Use np.ascontiguousarray(...) or copy into a preallocated ring buffer."
            )

    @staticmethod
    def _coerce_imu_batch(vImuMeas: np.ndarray) -> np.ndarray:
        """
        ORB-SLAM3 inertial APIs expect a float64 Nx7 array:
          [ax, ay, az, gx, gy, gz, t_s]

        Rules:
        - strictly increasing timestamps within the batch
        - include at least one measurement at/after the image timestamp

        This wrapper coerces dtype/order to improve error messages and reduce pybind cast failures.
        """
        if not isinstance(vImuMeas, np.ndarray):
            raise TypeError(f"vImuMeas must be a numpy array, got {type(vImuMeas)!r}")
        if vImuMeas.ndim != 2 or vImuMeas.shape[1] != 7:
            raise ValueError(f"vImuMeas must have shape (N,7); got {vImuMeas.shape!r}")
        v = np.asarray(vImuMeas, dtype=np.float64, order="C")
        if v.size == 0:
            return v.reshape(0, 7)
        if not bool(np.all(np.isfinite(v))):
            raise ValueError("vImuMeas contains NaN/Inf")
        dt = np.diff(v[:, 6])
        if bool(np.any(dt <= 0.0)):
            raise ValueError("vImuMeas timestamps must be strictly increasing")
        return v

    def TrackRGBD(
        self,
        image: np.ndarray,
        depthmap: np.ndarray,
        timestamp: float,
        vImuMeas: np.ndarray = None,
        # np.array([acc_x,acc_y,acc_z,ang_vel_x,ang_vel_y,ang_vel_z,timestamp])(nx7)
        filename: str = "",
    ) -> None:
        """
        Track one RGB-D frame (optionally inertial).

        - `image` must match your YAML `Camera.RGB` setting (0=BGR, 1=RGB).
        - `depthmap` must match `RGBD.DepthMapFactor` (commonly float32 meters with factor=1.0).
        - `timestamp` should be in seconds and monotonic (prefer starting near 0.0).
        - `vImuMeas` (when provided) must be float64 Nx7: [ax,ay,az,gx,gy,gz,t_s].
        """
        self._require_contiguous(image, name="image")
        self._require_contiguous(depthmap, name="depthmap")
        if timestamp is None:
            raise ValueError("timestamp must not be None")
        if vImuMeas is not None:
            v = self._coerce_imu_batch(vImuMeas)
            return self.slam._TrackRGBD(image, depthmap, float(timestamp), v, filename)
        if vImuMeas is None:
            return self.slam._TrackRGBD(image, depthmap, float(timestamp), filename=filename)

    def TrackMonocular(
        self,
        image: np.ndarray,
        timestamp: float,
        vImuMeas: np.ndarray = None,
        # np.array([acc_x,acc_y,acc_z,ang_vel_x,ang_vel_y,ang_vel_z,timestamp])(nx7)
        filename: str = "",
    ) -> None:
        self._require_contiguous(image, name="image")
        if timestamp is None:
            raise ValueError("timestamp must not be None")
        if vImuMeas is not None:
            v = self._coerce_imu_batch(vImuMeas)
            return self.slam._TrackMonocular(image, float(timestamp), v, filename)
        if vImuMeas is None:
            return self.slam._TrackMonocular(image, float(timestamp), filename=filename)

    def TrackStereo(
        self,
        imLeft: np.ndarray,
        imRight: np.ndarray,
        timestamp: float,
        vImuMeas: np.ndarray = None,
        # np.array([acc_x,acc_y,acc_z,ang_vel_x,ang_vel_y,ang_vel_z,timestamp])(nx7)
        filename: str = "",
    ) -> None:
        self._require_contiguous(imLeft, name="imLeft")
        self._require_contiguous(imRight, name="imRight")
        if timestamp is None:
            raise ValueError("timestamp must not be None")
        if vImuMeas is not None:
            v = self._coerce_imu_batch(vImuMeas)
            return self.slam._TrackStereo(
                imLeft, imRight, float(timestamp), v, filename
            )
        if vImuMeas is None:
            return self.slam._TrackStereo(imLeft, imRight, float(timestamp), filename=filename)

    def ActivateLocalizationMode(self):
        self.slam._ActivateLocalizationMode()

    def DeactivateLocalizationMode(self):
        self.slam._DeactivateLocalizationMode()

    def SwitchSensor(self, config: str, sensor: str, localization_only: bool = True) -> bool:
        eSensor = self.supported_sensors[sensor.upper()]
        return bool(self.slam._SwitchSensor(config, eSensor, localization_only))

    def GetMapPoints(self) -> np.ndarray:
        return self.slam._GetMapPoints()

    def GetTrackedMapPoints(self) -> np.ndarray:
        """
        Return tracked MapPoints (Nx3 float32 world coords) from the last tracking step.

        WARNING:
        On some custom ORB-SLAM3 builds, calling MapPoint-related APIs frequently (especially across resets/merges)
        has been associated with instability. Prefer `GetTrackedKeyPointsUn()` for visualization unless you have
        validated this on your build.
        """
        return self.slam._GetTrackedMapPoints()

    def GetTrackedKeyPointsUn(self) -> np.ndarray:
        """
        Return undistorted tracked keypoints (Nx2 float32, u/v pixels) from the last tracking step.

        This is generally safer than MapPoint-based APIs since it returns value types, not pointers.
        """
        return self.slam._GetTrackedKeyPointsUn()

    def IsAtlasInertial(self) -> bool:
        return bool(self.slam._IsAtlasInertial())

    def IsImuInitialized(self) -> bool:
        return bool(self.slam._IsImuInitialized())

    def GetImageScale(self):
        return self.slam._GetImageScale()

    def MapChanged(self):
        return self.slam._MapChanged()

    def GetCurrentMapId(self) -> int:
        return int(self.slam._GetCurrentMapId())

    def GetCurrentMapChangeIndex(self) -> int:
        return int(self.slam._GetCurrentMapChangeIndex())

    def GetCurrentMapBigChangeIndex(self) -> int:
        return int(self.slam._GetCurrentMapBigChangeIndex())

    def GetAtlasMapCount(self) -> int:
        return int(self.slam._GetAtlasMapCount())

    def GetAtlasMapIds(self) -> np.ndarray:
        return self.slam._GetAtlasMapIds()

    def PopAlignmentEvents(self):
        return self.slam._PopAlignmentEvents()

    def DebugPushAlignmentEvent(
        self,
        *,
        type: int,
        T_new_old: np.ndarray,
        t_s: float = 0.0,
        map_a: int = 0,
        map_b: int = 0,
        kf_a: int = 0,
        kf_b: int = 0,
        scale: float = 1.0,
    ):
        assert isinstance(T_new_old, np.ndarray)
        return self.slam._DebugPushAlignmentEvent(
            int(type),
            float(t_s),
            int(map_a),
            int(map_b),
            int(kf_a),
            int(kf_b),
            float(scale),
            T_new_old,
        )

    def Reset(self):
        self.slam._Reset()

    def ResetActiveMap(self):
        self.slam._ResetActiveMap()

    def SaveKeyFrameTrajectoryEuroC(self, filename):
        self.slam._SaveKeyFrameTrajectoryEuroC(filename)

    def SaveTrajectoryEuroC(self, filename):
        self.slam._SaveTrajectoryEuroC(filename)

    def SaveKeyFrameTrajectoryTUM(self, filename):
        self.slam._SaveKeyFrameTrajectoryTUM(filename)

    def SaveTrajectoryKITTI(self, filename):
        self.slam._SaveTrajectoryKITTI(filename)

    def SaveTrajectoryTUM(self, filename):
        self.slam._SaveTrajectoryTUM(filename)

    def Shutdown(self):
        self.slam._Shutdown()

    def isFinished(self):
        return self.slam._isFinished()

    def isLost(self):
        return self.slam._isLost()

    def isShutDown(self):
        return self.slam._isShutDown()
