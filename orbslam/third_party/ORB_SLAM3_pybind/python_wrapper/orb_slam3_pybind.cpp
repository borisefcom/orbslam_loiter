//#include "orb_slam3_pybind.h"

#include <System.h>
#include <MapPoint.h>
#include <pybind11/detail/common.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>

#include <string>
#include <vector>
#include <map>
#include <opencv2/opencv.hpp>

#include "ImuTypes.h"
#include "sophus/se3.hpp"

namespace py = pybind11;
using namespace py::literals;
namespace ORB_SLAM3 {

std::map<std::string, int> np_cv{
    {py::format_descriptor<uint8_t>::format(), CV_8U},
    {py::format_descriptor<int8_t>::format(), CV_8S},
    {py::format_descriptor<uint16_t>::format(), CV_16U},
    {py::format_descriptor<int16_t>::format(), CV_16S},
    {py::format_descriptor<int32_t>::format(), CV_32S},
    {py::format_descriptor<float>::format(), CV_32F},
    {py::format_descriptor<double>::format(), CV_64F}};

bool is_array_contiguous(const pybind11::array &a) {
  py::ssize_t expected_stride = a.itemsize();
  for (int i = a.ndim() - 1; i >= 0; --i) {
    pybind11::ssize_t current_stride = a.strides()[i];
    if (current_stride != expected_stride) {
      return false;
    }
    expected_stride = expected_stride * a.shape()[i];
  }
  return true;
}

int determine_cv_type(const py::array &np_array) {
  const int ndim = np_array.ndim();
  std::string np_type;
  np_type += np_array.dtype().char_();
  int cv_type = 0;

  if (auto search = np_cv.find(np_type); search != np_cv.end()) {
    cv_type = search->second;
  } else {
    cv_type = -1;
  }
  if (ndim < 2) {
    throw std::invalid_argument(
        "determine_cv_type needs at least two dimensions");
  }
  if (ndim > 3) {
    throw std::invalid_argument(
        "determine_cv_type needs at most three dimensions");
  }
  if (ndim == 2) {
    return CV_MAKETYPE(cv_type, 1);
  }
  return CV_MAKETYPE(cv_type, np_array.shape(2));
}

cv::Mat py_array_to_mat(py::array &np_array) {
  // IMPORTANT (Python safety):
  // This conversion is intentionally *zero-copy*: the resulting cv::Mat points directly at the numpy buffer.
  // If the caller overwrites/reuses that numpy buffer while ORB-SLAM3 still holds a reference (ORB-SLAM3 is
  // multi-threaded and may keep cv::Mat references beyond a single Track* call on some builds), this can
  // lead to use-after-free / memory corruption / native crashes.
  //
  // Practical guidance for Python callers:
  // - Always pass C-contiguous arrays (enforced here).
  // - Prefer copying frames into a ring buffer of numpy arrays and pass those stable buffers to Track*().
  bool is_contiguous = is_array_contiguous(np_array);
  bool is_empty = np_array.size() == 0;
  if (!is_contiguous) {
    throw std::invalid_argument("is not contiguous array; try np.contiguous");
  }
  if (is_empty) {
    throw std::invalid_argument("numpy array is empty");
  }
  int mat_type = determine_cv_type(np_array);
  cv::Mat im(np_array.shape(0), np_array.shape(1), mat_type,
             np_array.mutable_data(), np_array.strides(0));
  return im;
}

std::vector<IMU::Point> py_array_to_vector_imu_points(
    py::array_t<double> &array) {
  // IMU batch format expected by ORB-SLAM3:
  //   Nx7 float64: [ax, ay, az, gx, gy, gz, timestamp_s]
  // Timestamps must be strictly increasing within the batch.
  if (array.ndim() != 2 || array.shape(1) != 7) {
    throw py::cast_error();
  }
  std::vector<IMU::Point> imu_vectors{};
  imu_vectors.reserve(array.shape(0));
  auto array_unchecked = array.unchecked<2>();
  for (auto i = 0; i < array_unchecked.shape(0); ++i) {
    auto acc_x = static_cast<float>(array_unchecked(i, 0));
    auto acc_y = static_cast<float>(array_unchecked(i, 1));
    auto acc_z = static_cast<float>(array_unchecked(i, 2));
    auto ang_vel_x = static_cast<float>(array_unchecked(i, 3));
    auto ang_vel_y = static_cast<float>(array_unchecked(i, 4));
    auto ang_vel_z = static_cast<float>(array_unchecked(i, 5));
    double timestamp = array_unchecked(i, 6);
    imu_vectors.emplace_back(IMU::Point(acc_x, acc_y, acc_z, ang_vel_x,
                                        ang_vel_y, ang_vel_z, timestamp));
  }
  return imu_vectors;
}
PYBIND11_MODULE(orb_slam_pybind, m) {
  py::class_<System> system(m, "_System",
        "This is the low level C++ bindings, all the methods and "
        "constructor defined within this module (starting with a ``_`` "
        "should not be used. Please reffer to the python Procesor class to "
        "check how to use the API");
  system
      .def(py::init<const std::string &, const std::string &,
                    const System::eSensor, bool, const int,
                    const std::string &>(),
           "strVocFile"_a, "strSettingsFile"_a, "sensor"_a,
           "bUseViewer"_a = true, "initFr"_a = 0, "strSequence"_a = "")
      .def("_ActivateLocalizationMode", &System::ActivateLocalizationMode)
      .def("_DeactivateLocalizationMode", &System::DeactivateLocalizationMode)
      .def("_SwitchSensor", &System::SwitchSensor, "settings"_a, "sensor"_a,
           "localization_only"_a = true)
      .def("_IsAtlasInertial", &System::IsAtlasInertial)
      .def("_IsImuInitialized", &System::IsImuInitialized)
      .def("_GetImageScale", &System::GetImageScale)
      .def("_MapChanged", &System::MapChanged)
      .def("_Reset", &System::Reset)
      .def("_ResetActiveMap", &System::ResetActiveMap)
      .def("_GetCurrentMapId", &System::GetCurrentMapId,
           "Return the current active Map::GetId().")
      .def("_GetCurrentMapChangeIndex", &System::GetCurrentMapChangeIndex,
           "Return the current Map::GetMapChangeIndex().")
      .def("_GetCurrentMapBigChangeIndex", &System::GetCurrentMapBigChangeIndex,
           "Return the current Map::GetLastBigChangeIdx().")
      .def("_GetAtlasMapCount", &System::GetAtlasMapCount,
           "Return Atlas::CountMaps().")
      .def(
          "_GetAtlasMapIds",
          [](System &self) {
            std::vector<long unsigned int> ids = self.GetAtlasMapIds();
            py::array_t<std::uint64_t> out({static_cast<py::ssize_t>(ids.size())});
            auto o = out.mutable_unchecked<1>();
            for (py::ssize_t i = 0; i < o.shape(0); ++i) {
              o(i) = static_cast<std::uint64_t>(ids[static_cast<std::size_t>(i)]);
            }
            return out;
          },
          "Return a 1D uint64 array of atlas Map ids.")
       .def(
           "_PopAlignmentEvents",
           [](System &self) {
             std::vector<AlignmentEvent> evs = self.PopAlignmentEvents();
            py::list out;
            for (const auto &ev : evs) {
              py::dict d;
              d["type"] = py::int_(ev.type);
              d["t_s"] = py::float_(ev.t_s);
              d["map_a"] = py::int_(static_cast<std::uint64_t>(ev.map_a));
              d["map_b"] = py::int_(static_cast<std::uint64_t>(ev.map_b));
              d["kf_a"] = py::int_(static_cast<std::uint64_t>(ev.kf_a));
              d["kf_b"] = py::int_(static_cast<std::uint64_t>(ev.kf_b));
              d["scale"] = py::float_(ev.scale);

              py::array_t<double> T({4, 4});
              auto m = T.mutable_unchecked<2>();
              for (int r = 0; r < 4; ++r) {
                for (int c = 0; c < 4; ++c) {
                  m(r, c) = ev.T_new_old[static_cast<std::size_t>(r * 4 + c)];
                }
              }
              d["T_new_old"] = T;
              out.append(d);
            }
            return out;
           },
           "Pop loop/merge map alignment events as a list of dicts.")
      .def(
          "_DebugPushAlignmentEvent",
          [](System &self, const int type, const double t_s, const std::uint64_t map_a,
             const std::uint64_t map_b, const std::uint64_t kf_a, const std::uint64_t kf_b,
             const double scale, py::array_t<double> T_new_old) {
            if (T_new_old.ndim() != 2 || T_new_old.shape(0) != 4 || T_new_old.shape(1) != 4) {
              throw std::invalid_argument("T_new_old must be a 4x4 float64 array");
            }
            AlignmentEvent ev;
            ev.type = static_cast<int>(type);
            ev.t_s = static_cast<double>(t_s);
            ev.map_a = static_cast<std::uint64_t>(map_a);
            ev.map_b = static_cast<std::uint64_t>(map_b);
            ev.kf_a = static_cast<std::uint64_t>(kf_a);
            ev.kf_b = static_cast<std::uint64_t>(kf_b);
            ev.scale = static_cast<double>(scale);
            auto m = T_new_old.unchecked<2>();
            for (int r = 0; r < 4; ++r) {
              for (int c = 0; c < 4; ++c) {
                ev.T_new_old[static_cast<std::size_t>(r * 4 + c)] =
                    static_cast<double>(m(r, c));
              }
            }
            self.DebugPushAlignmentEvent(ev);
          },
          "type"_a, "t_s"_a = 0.0, "map_a"_a = 0ULL, "map_b"_a = 0ULL, "kf_a"_a = 0ULL,
          "kf_b"_a = 0ULL, "scale"_a = 1.0, "T_new_old"_a = py::array_t<double>(),
          "Debug: push a synthetic alignment event into the queue.")
      .def(
          "_SaveKeyFrameTrajectoryEuroC",
          [](System &self, const std::string &filename) {
            self.SaveKeyFrameTrajectoryEuRoC(filename);
          },
          "filename"_a)
      .def(
          "_SaveTrajectoryEuroC",
          [](System &self, const std::string &filename) {
            self.SaveTrajectoryEuRoC(filename);
          },
          "filename"_a)
      .def("_SaveKeyFrameTrajectoryTUM", &System::SaveKeyFrameTrajectoryTUM,
           "filename"_a)
      .def("_SaveTrajectoryKITTI", &System::SaveTrajectoryKITTI, "filename"_a)
      .def("_SaveTrajectoryTUM", &System::SaveTrajectoryTUM, "filename"_a)
      .def("_Shutdown", &System::Shutdown)
      .def("_isFinished", &System::isFinished)
      .def("_isLost", &System::isLost)
      .def("_isShutDown", &System::isShutDown)
       .def(
           "_GetTrackedMapPoints",
           [](System &self) {
             std::vector<MapPoint *> vpts = self.GetTrackedMapPoints();
             std::size_t n = 0;
            for (auto *p : vpts) {
              if (!p)
                continue;
              if (p->isBad())
                continue;
              n++;
            }

            py::array_t<float> out({static_cast<py::ssize_t>(n),
                                   static_cast<py::ssize_t>(3)});
            auto out_m = out.mutable_unchecked<2>();
            std::size_t i = 0;
            for (auto *p : vpts) {
              if (!p)
                continue;
              if (p->isBad())
                continue;
              Eigen::Vector3f X = p->GetWorldPos();
              out_m(i, 0) = X[0];
              out_m(i, 1) = X[1];
              out_m(i, 2) = X[2];
              i++;
            }
            return out;
           },
           "Return Nx3 float32 world points tracked in the last frame.")
       .def(
           "_GetTrackedKeyPointsUn",
           [](System &self) {
             std::vector<cv::KeyPoint> kps = self.GetTrackedKeyPointsUn();
             py::array_t<float> out({static_cast<py::ssize_t>(kps.size()),
                                    static_cast<py::ssize_t>(2)});
             auto m = out.mutable_unchecked<2>();
             for (py::ssize_t i = 0; i < m.shape(0); ++i) {
               const auto &kp = kps[static_cast<std::size_t>(i)];
               m(i, 0) = static_cast<float>(kp.pt.x);
               m(i, 1) = static_cast<float>(kp.pt.y);
             }
             return out;
           },
           "Return Nx2 float32 undistorted keypoints (u,v) from the last frame.")
       .def(
           "_GetMapPoints",
           [](System &self) {
             std::vector<MapPoint *> vpts = self.GetCurrentMapPoints();
             std::size_t n = 0;
            for (auto *p : vpts) {
              if (!p)
                continue;
              if (p->isBad())
                continue;
              n++;
            }

            py::array_t<float> out({static_cast<py::ssize_t>(n),
                                   static_cast<py::ssize_t>(3)});
            auto out_m = out.mutable_unchecked<2>();
            std::size_t i = 0;
            for (auto *p : vpts) {
              if (!p)
                continue;
              if (p->isBad())
                continue;
              Eigen::Vector3f X = p->GetWorldPos();
              out_m(i, 0) = X[0];
              out_m(i, 1) = X[1];
              out_m(i, 2) = X[2];
              i++;
            }
            return out;
          },
          "Return Nx3 float32 world points from the current map.")
      .def(
          "_TrackMonocular",
          [](System &self, py::array &image, const double timestamp,
             py::array_t<double> &vImuMeas, const std::string &filename) {
            cv::Mat im = py_array_to_mat(image);
            std::vector<IMU::Point> vector_imu{};
            if (vImuMeas.size() != 0) {
              vector_imu = py_array_to_vector_imu_points(vImuMeas);
            }
            return self.TrackMonocular(im, timestamp, vector_imu, filename)
                .matrix();
          },
          "im"_a, "timestamp"_a, "vImuMeas"_a = py::array_t<double>(),
          "filename"_a = "")

      .def(
          "_TrackRGBD",
          [](System &self, py::array &image, py::array &depth,
             const double &timestamp, py::array_t<double> vImuMeas,
             const std::string &filename) {
            cv::Mat im = py_array_to_mat(image);
            cv::Mat depthmap = py_array_to_mat(depth);
            std::vector<IMU::Point> vector_imu{};
            if (vImuMeas.size() != 0) {
              vector_imu = py_array_to_vector_imu_points(vImuMeas);
            }

            return self.TrackRGBD(im, depthmap, timestamp, vector_imu, filename)
                .matrix();
          },
          "im"_a, "depthmap"_a, "timestamp"_a,
          "vImuMeas"_a = py::array_t<double>(), "filename"_a = "")

      .def(
          "_TrackStereo",
          [](System &self, py::array &imLeft, py::array &imRight,
             const double &timestamp, py::array_t<double> vImuMeas,
             const std::string &filename) {
            cv::Mat imL = py_array_to_mat(imLeft);
            cv::Mat imR = py_array_to_mat(imRight);
            std::vector<IMU::Point> vector_imu{};
            if (vImuMeas.size() != 0) {
              vector_imu = py_array_to_vector_imu_points(vImuMeas);
            }

            return self.TrackStereo(imL, imR, timestamp, vector_imu, filename)
                .matrix();
          },
          "imLeft"_a, "imRight"_a, "timestamp"_a,
          "vImuMeas"_a = py::array_t<double>(), "filename"_a = "");

  py::enum_<System::eSensor>(system, "eSensor")
      .value("MONOCULAR", System::eSensor::MONOCULAR)
      .value("STEREO", System::eSensor::STEREO)
      .value("RGBD", System::RGBD)
      .value("IMU_MONOCULAR", System::eSensor::IMU_MONOCULAR)
      .value("IMU_STEREO", System::eSensor::IMU_STEREO)
      .value("IMU_RGBD", System::eSensor::IMU_RGBD);

  py::class_<IMU::Point> imu_point(m, "Point");
  imu_point.def(
      py::init<const float &, const float &, const float &, const float &,
               const float &, const float &, const double &>(),
      "acc_x"_a, "acc_y"_a, "acc_z"_a, "ang_vel_x"_a, "ang_vel_y"_a,
      "ang_vel_z"_a, "timestamp"_a);
}
}  // namespace ORB_SLAM3
