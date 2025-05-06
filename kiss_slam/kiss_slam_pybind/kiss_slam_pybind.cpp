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
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include <Eigen/Core>
#include <memory>
#include <pose_graph_optimizer.hpp>
#include <vector>

#include "occupancy_mapper.hpp"
#include "stl_vector_eigen.h"
#include "voxel_map.hpp"

namespace py = pybind11;
using namespace py::literals;
PYBIND11_MAKE_OPAQUE(pgo::PoseGraphOptimizer::PoseIDMap);
PYBIND11_MAKE_OPAQUE(std::vector<Eigen::Vector3i>);
PYBIND11_MAKE_OPAQUE(std::vector<Eigen::Vector3f>);

PYBIND11_MODULE(kiss_slam_pybind, m) {
    using namespace pgo;
    using namespace voxel_map;
    using namespace occupancy_mapper;
    auto vector3fvector = pybind_eigen_vector_of_vector<Eigen::Vector3f>(
        m, "_Vector3fVector", "std::vector<Eigen::Vector3f>",
        py::py_array_to_vectors<Eigen::Vector3f>);
    auto vector3ivector = pybind_eigen_vector_of_vector<Eigen::Vector3i>(
        m, "_Vector3iVector", "std::vector<Eigen::Vector3i>",
        py::py_array_to_vectors<Eigen::Vector3i>);

    py::bind_map<pgo::PoseGraphOptimizer::PoseIDMap>(m, "PoseIDMap");
    py::class_<PoseGraphOptimizer> pgo(m, "_PoseGraphOptimizer", "Don't use this");
    pgo.def(py::init<int>(), "max_iterations"_a)
        .def("_add_variable", &PoseGraphOptimizer::addVariable, "id"_a, "T"_a)
        .def("_fix_variable", &PoseGraphOptimizer::fixVariable, "id"_a)
        .def("_add_factor", &PoseGraphOptimizer::addFactor, "id_source"_a, "id_target"_a, "T"_a,
             "omega"_a)
        .def("_optimize", &PoseGraphOptimizer::optimize)
        .def("_estimates", &PoseGraphOptimizer::estimates)
        .def("_read_graph", &PoseGraphOptimizer::readGraph, "filename"_a)
        .def("_write_graph", &PoseGraphOptimizer::writeGraph, "filename"_a);

    py::class_<VoxelMap> internal_map(m, "_VoxelMap", "Don't use this");
    internal_map.def(py::init<float>(), "voxel_size"_a)
        .def("_integrate_frame", &VoxelMap::IntegrateFrame, "points"_a, "pose"_a)
        .def("_add_points", &VoxelMap::AddPoints, "points"_a)
        .def("_point_cloud", &VoxelMap::Pointcloud)
        .def("_clear", &VoxelMap::Clear)
        .def("_num_voxels", &VoxelMap::NumVoxels)
        .def("_per_voxel_point_and_normal", &VoxelMap::PerVoxelPointAndNormal);

    py::class_<OccupancyMapper> grid_mapper(m, "_OccupancyMapper", "Don't use this");
    grid_mapper.def(py::init<float, float>(), "resolution"_a, "max_range"_a)
        .def("_integrate_frame", &OccupancyMapper::IntegrateFrame, "pointcloud"_a, "pose"_a)
        .def("_get_active_voxels", &OccupancyMapper::GetOccupancyInformation)
        .def("_get_occupied_voxels", &OccupancyMapper::GetOccupiedVoxels, "probability_threshold"_a)
        .def("_save_occupancy_volume", &OccupancyMapper::SaveOccupancyVolume, "filename"_a);
}
