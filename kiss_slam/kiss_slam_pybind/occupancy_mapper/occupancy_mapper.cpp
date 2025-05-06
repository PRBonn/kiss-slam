// MIT License

// Copyright (c) 2025 Tiziano Guadagnino, Benedikt Mersch, Saurabh Gupta,
// Meher Malladi Cyrill Stachniss.

// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
#include "occupancy_mapper.hpp"

#include <Eigen/Core>
#include <algorithm>
#include <bonxai/bonxai.hpp>
#include <cmath>
#include <tuple>
#include <vector>

namespace {
static constexpr float logodds_free = -0.8473f;
static constexpr float logodds_occ = 2.1972f;
inline float ProbabilityOccupied(const float logodds) {
    return 1.0f - (1.0f / (1.0f + std::exp(logodds)));
}
inline float LogOddsOccupied(const float probability) {
    return std::log((1.0f / (1.0f - probability)) - 1.0f);
}
}  // namespace

namespace occupancy_mapper {
OccupancyMapper::OccupancyMapper(const float resolution, const float max_range)
    : max_range_(max_range), map_(resolution), accessor_(map_.createAccessor()) {}

void OccupancyMapper::IntegrateFrame(const Vector3fVector &pointcloud,
                                     const Eigen::Matrix4f &pose) {
    const Eigen::Matrix3f &R = pose.block<3, 3>(0, 0);
    const Eigen::Vector3f &t = pose.block<3, 1>(0, 3);
    const auto start_coord = map_.posToCoord(t);
    std::for_each(pointcloud.cbegin(), pointcloud.cend(), [&](const Eigen::Vector3f &point) {
        const auto point_range = point.norm();
        if (point_range < max_range_) {
            const Eigen::Vector3f point_tf = R * point + t;
            const auto end_coord = map_.posToCoord(point_tf);
            Bresenham3DLine(start_coord, end_coord);
        }
    });
}

std::tuple<Vector3iVector, std::vector<float>> OccupancyMapper::GetOccupancyInformation() const {
    Vector3iVector voxel_indices;
    std::vector<float> voxel_occupancies;
    const auto num_of_active_voxels = map_.activeCellsCount();
    voxel_indices.reserve(num_of_active_voxels);
    voxel_occupancies.reserve(num_of_active_voxels);
    map_.forEachCell([&](const float &logodds, const Bonxai::CoordT &voxel) {
        voxel_indices.emplace_back(Bonxai::ConvertPoint<Eigen::Vector3i>(voxel));
        voxel_occupancies.emplace_back(ProbabilityOccupied(logodds));
    });
    return std::make_tuple(voxel_indices, voxel_occupancies);
}

Vector3iVector OccupancyMapper::GetOccupiedVoxels(
    const float occupancy_probability_threshold) const {
    Vector3iVector voxel_indices;
    const float logodds_threshold = LogOddsOccupied(occupancy_probability_threshold);
    const auto num_of_active_voxels = map_.activeCellsCount();
    voxel_indices.reserve(num_of_active_voxels);
    map_.forEachCell([&](const float &logodds, const Bonxai::CoordT &voxel) {
        if (logodds > logodds_threshold) {
            voxel_indices.emplace_back(Bonxai::ConvertPoint<Eigen::Vector3i>(voxel));
        }
    });
    voxel_indices.shrink_to_fit();
    return voxel_indices;
}

void OccupancyMapper::UpdateVoxelOccupancy(const Bonxai::CoordT &coord, const float value) {
    // tg exploit Vdb caching
    accessor_.setCellOn(coord, 0.0);
    float *logodds = accessor_.value(coord);
    *logodds += value;
}

void OccupancyMapper::Bresenham3DLine(const Bonxai::CoordT &start_coord,
                                      const Bonxai::CoordT &end_coord) {
    const auto ray = end_coord - start_coord;
    const int x_sign = ray.x > 0 ? 1 : -1;
    const int y_sign = ray.y > 0 ? 1 : -1;
    const int z_sign = ray.z > 0 ? 1 : -1;

    const int ray_x_abs = x_sign * ray.x;
    const int ray_y_abs = y_sign * ray.y;
    const int ray_z_abs = z_sign * ray.z;

    int p1 = 0;
    int p2 = 0;
    Bonxai::CoordT delta{0, 0, 0};
    if (ray_x_abs >= ray_y_abs && ray_x_abs > ray_z_abs) {
        p1 = 2 * ray_y_abs - ray_x_abs;
        p2 = 2 * ray_z_abs - ray_x_abs;
        while (std::abs(delta.x) < ray_x_abs) {
            const auto voxel = start_coord + delta;
            UpdateVoxelOccupancy(voxel, logodds_free);
            if (p1 >= 0) {
                delta.y += y_sign;
                p1 -= (2 * ray_x_abs);
            }
            if (p2 >= 0) {
                delta.z += z_sign;
                p2 -= (2 * ray_x_abs);
            }
            p1 += (2 * ray_y_abs);
            p2 += (2 * ray_z_abs);
            delta.x += x_sign;
        }
    } else if (ray_y_abs >= ray_x_abs && ray_y_abs > ray_z_abs) {
        p1 = 2 * ray_x_abs - ray_y_abs;
        p2 = 2 * ray_z_abs - ray_y_abs;
        while (std::abs(delta.y) < ray_y_abs) {
            const auto voxel = start_coord + delta;
            UpdateVoxelOccupancy(voxel, logodds_free);
            if (p1 >= 0) {
                delta.x += x_sign;
                p1 -= (2 * ray_y_abs);
            }
            if (p2 >= 0) {
                delta.z += z_sign;
                p2 -= (2 * ray_y_abs);
            }
            p1 += (2 * ray_x_abs);
            p2 += (2 * ray_z_abs);
            delta.y += y_sign;
        }
    } else {
        p1 = 2 * ray_y_abs - ray_z_abs;
        p2 = 2 * ray_x_abs - ray_z_abs;
        while (std::abs(delta.z) < ray_z_abs) {
            const auto voxel = start_coord + delta;
            UpdateVoxelOccupancy(voxel, logodds_free);
            if (p1 >= 0) {
                delta.y += y_sign;
                p1 -= (2 * ray_z_abs);
            }
            if (p2 >= 0) {
                delta.x += x_sign;
                p2 -= (2 * ray_z_abs);
            }
            p1 += (2 * ray_y_abs);
            p2 += (2 * ray_x_abs);
            delta.z += z_sign;
        }
    }
    UpdateVoxelOccupancy(end_coord, logodds_occ);
}
}  // namespace occupancy_mapper
