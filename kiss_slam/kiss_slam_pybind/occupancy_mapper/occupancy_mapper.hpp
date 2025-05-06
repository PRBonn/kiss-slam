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
#pragma once
#include <Eigen/Core>
#include <bonxai/bonxai.hpp>
#include <bonxai/serialization.hpp>
#include <fstream>
#include <string>
#include <tuple>
#include <vector>

using Vector3fVector = std::vector<Eigen::Vector3f>;
using Vector3iVector = std::vector<Eigen::Vector3i>;

namespace occupancy_mapper {
class OccupancyMapper {
public:
    OccupancyMapper(const float resolution, const float max_range);
    ~OccupancyMapper() = default;

    void IntegrateFrame(const Vector3fVector &pointcloud, const Eigen::Matrix4f &pose);
    std::tuple<Vector3iVector, std::vector<float>> GetOccupancyInformation() const;
    Vector3iVector GetOccupiedVoxels(const float occupancy_probability_threshold) const;
    void SaveOccupancyVolume(const std::string &filename) const {
        std::ofstream data(filename, std::ios::binary);
        Bonxai::Serialize(data, map_);
    };

private:
    void Bresenham3DLine(const Bonxai::CoordT &start_coord, const Bonxai::CoordT &end_coord);
    void UpdateVoxelOccupancy(const Bonxai::CoordT &coord, const float value);

    float max_range_ = 0.0f;
    Bonxai::VoxelGrid<float> map_;
    using AccessorType = typename Bonxai::VoxelGrid<float>::Accessor;
    AccessorType accessor_;
};
}  // namespace occupancy_mapper
