// MIT License

// Copyright (c) 2025 Tiziano Guadagnino, Benedikt Mersch, Saurabh Gupta, Cyrill
// Stachniss.

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
#include "voxel_map.hpp"

#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <tuple>
#include <vector>

namespace {

inline Voxel ToVoxelCoordinates(const Eigen::Vector3f &point, const float voxel_size) {
    return Voxel(static_cast<int>(std::floor(point.x() / voxel_size)),
                 static_cast<int>(std::floor(point.y() / voxel_size)),
                 static_cast<int>(std::floor(point.z() / voxel_size)));
}

static constexpr unsigned int min_points_for_covariance_computation = 3;

std::tuple<Eigen::Vector3f, Eigen::Vector3f> ComputeCentroidAndNormal(
    const voxel_map::VoxelBlock &coordinates) {
    const float num_points = static_cast<float>(coordinates.size());
    const Eigen::Vector3f &mean =
        std::reduce(coordinates.cbegin(), coordinates.cend(), Eigen::Vector3f().setZero()) /
        num_points;

    const Eigen::Matrix3f &covariance =
        std::transform_reduce(coordinates.cbegin(), coordinates.cend(), Eigen::Matrix3f().setZero(),
                              std::plus<Eigen::Matrix3f>(),
                              [&mean](const Eigen::Vector3f &point) {
                                  const Eigen::Vector3f &centered = point - mean;
                                  const Eigen::Matrix3f S = centered * centered.transpose();
                                  return S;
                              }) /
        (num_points - 1);
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> solver(covariance);
    const Eigen::Vector3f normal = solver.eigenvectors().col(0);
    return std::make_tuple(mean, normal);
}

}  // namespace

namespace voxel_map {

void VoxelBlock::emplace_back(const Eigen::Vector3f &p) {
    if (size() < max_points_per_normal_computation) {
        points.at(num_points) = p;
        ++num_points;
    }
}

VoxelMap::VoxelMap(const float voxel_size)
    : voxel_size_(voxel_size),
      map_resolution_(voxel_size /
                      static_cast<float>(std::sqrt(max_points_per_normal_computation))) {}

std::vector<Eigen::Vector3f> VoxelMap::Pointcloud() const {
    std::vector<Eigen::Vector3f> points;
    points.reserve(map_.size() * max_points_per_normal_computation);
    std::for_each(map_.cbegin(), map_.cend(), [&](const auto &map_element) {
        const auto &voxel_points = map_element.second;
        std::for_each(voxel_points.cbegin(), voxel_points.cend(),
                      [&](const auto &p) { points.emplace_back(p.template cast<float>()); });
    });
    points.shrink_to_fit();
    return points;
}

void VoxelMap::IntegrateFrame(const std::vector<Eigen::Vector3f> &points,
                              const Eigen::Matrix4f &pose) {
    std::vector<Eigen::Vector3f> points_transformed(points.size());
    const auto &R = pose.block<3, 3>(0, 0);
    const auto &t = pose.block<3, 1>(0, 3);
    std::transform(points.cbegin(), points.cend(), points_transformed.begin(),
                   [&](const auto &point) { return R * point + t; });
    AddPoints(points_transformed);
}

void VoxelMap::AddPoints(const std::vector<Eigen::Vector3f> &points) {
    std::for_each(points.cbegin(), points.cend(), [&](const auto &point) {
        const auto voxel = ToVoxelCoordinates(point, voxel_size_);
        const auto &[it, inserted] = map_.insert({voxel, VoxelBlock()});
        if (!inserted) {
            auto &voxel_points = it.value();
            if (voxel_points.size() == max_points_per_normal_computation ||
                std::any_of(voxel_points.cbegin(), voxel_points.cend(),
                            [&](const auto &voxel_point) {
                                return (voxel_point - point).norm() < map_resolution_;
                            })) {
                return;
            }
        }
        it.value().emplace_back(point);
    });
}

std::tuple<Vector3fVector, Vector3fVector> VoxelMap::PerVoxelPointAndNormal() const {
    Vector3fVector points;
    points.reserve(map_.size());
    Vector3fVector normals;
    normals.reserve(map_.size());
    std::for_each(map_.cbegin(), map_.cend(), [&](const auto &inner_block) {
        const auto &voxel_block = inner_block.second;
        if (voxel_block.size() >= min_points_for_covariance_computation) {
            const auto &[mean, normal] = ComputeCentroidAndNormal(voxel_block);
            points.emplace_back(mean);
            normals.emplace_back(normal);
        }
    });
    points.shrink_to_fit();
    normals.shrink_to_fit();
    return std::make_tuple(points, normals);
}

}  // namespace voxel_map
