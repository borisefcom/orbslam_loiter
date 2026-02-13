/**
 * Minimal telemetry struct for exposing ORB-SLAM3 loop/merge map alignment
 * transforms to external mapping code (e.g., voxel submaps).
 */

#ifndef ORB_SLAM3_ALIGNMENT_EVENT_H
#define ORB_SLAM3_ALIGNMENT_EVENT_H

#include <array>
#include <cstdint>

namespace ORB_SLAM3
{

struct AlignmentEvent
{
    // 0 = loop closure correction, 1 = map merge correction
    int type = 0;
    // keyframe timestamp (seconds)
    double t_s = 0.0;
    // map ids (as seen by Atlas/Map::GetId())
    std::uint64_t map_a = 0;
    std::uint64_t map_b = 0;
    // keyframe ids (KeyFrame::mnId)
    std::uint64_t kf_a = 0;
    std::uint64_t kf_b = 0;
    // Similarity scale (1.0 for RGB-D / inertial)
    double scale = 1.0;
    // Row-major 4x4 matrix with rotation+translation (SE3). Apply as:
    //   p_new = scale * (R * p_old) + t
    // where R,t are taken from this matrix.
    std::array<double, 16> T_new_old = {0};
};

} // namespace ORB_SLAM3

#endif // ORB_SLAM3_ALIGNMENT_EVENT_H
