# MIT License
#
# Copyright (c) 2025 Nathan Hewitt, Tiziano Guadagnino, Benedikt Mersch,
# Saurabh Gupta, Cyrill Stachniss.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import os

import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from kiss_slam_ros.conversions import (
    build_map,
    build_odometry,
    build_transform,
    matrix_to_pose,
    pc2_to_numpy,
    slam_config_from_params,
    slam_params_from_config,
)
from map_closures import map_closures
from nav_msgs.msg import OccupancyGrid, Odometry
from rcl_interfaces.msg import ParameterDescriptor, SetParametersResult
from rclpy.node import Node
from rclpy.parameter import Parameter
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Header
from tf2_ros import TransformBroadcaster

from kiss_slam.occupancy_mapper import OccupancyGridMapper
from kiss_slam.slam import KissSLAM


class KissSLAMNode(Node):
    def __init__(self):
        """Create parameters, subscriptions, publishers, and set up SLAM"""
        super().__init__("kiss_slam_node")

        # Create all the KISS-SLAM parameters in one go
        self.declare_parameters("", slam_params_from_config(None))

        # Params specific to ROS wrapper
        points_topic_desc = ParameterDescriptor(
            description="What topic to listen on for PointCloud2 messages."
        )
        self.declare_parameter("points_topic", "/points", points_topic_desc)
        self.points_topic = self.get_parameter("points_topic").value

        map_frame_desc = ParameterDescriptor(
            description="Frame ID to use as parent for SLAM estimated transform. Dynamic."
        )
        self.declare_parameter("map_frame", "map", map_frame_desc)
        self.map_frame = self.get_parameter("map_frame").value

        position_covariance_desc = ParameterDescriptor(
            description="Covariance value for position terms of odometry. Dynamic."
        )
        self.declare_parameter("position_covariance", 0.1, position_covariance_desc)
        self.position_covariance = self.get_parameter("position_covariance").value

        orientation_covariance_desc = ParameterDescriptor(
            description="Covariance value for orientation terms of odometry. Dynamic."
        )
        self.declare_parameter("orientation_covariance", 0.1, orientation_covariance_desc)
        self.orientation_covariance = self.get_parameter("orientation_covariance").value

        # Main subscription to drive the SLAM process
        self.subscription = self.create_subscription(
            PointCloud2, self.points_topic, self.cloud_callback, 10
        )

        # Publishers
        self.odom_publisher = self.create_publisher(Odometry, "odom", 10)
        self.map_publisher = self.create_publisher(OccupancyGrid, "map", 10)
        self.tf_broadcaster = TransformBroadcaster(self)

        # Pull slam config params and initialize
        self.slam_config = slam_config_from_params(self)
        self.slam = KissSLAM(self.slam_config)
        self.mapper = OccupancyGridMapper(self.slam_config.occupancy_mapper)

        # Initial variables for mapping
        self.ref_ground_alignment = None
        self.min_voxel_idx = (0, 0)

        self.add_on_set_parameters_callback(self.parameter_callback)

    def parameter_callback(self, params: list):
        """Callback to handle dynamic parameters for runtime reconfiguration

        :param params: List of params changed that triggered this callback
        :type params: list
        :return: Success regardless of whether a param was updated
        :rtype: SetParametersResult
        """
        for param in params:
            if param.name == "map_frame" and param.type_ == Parameter.Type.STRING:
                self.map_frame = param.value
            if param.name == "position_covariance" and param.type_ == Parameter.Type.DOUBLE:
                self.position_covariance = param.value
            if param.name == "orientation_covariance" and param.type_ == Parameter.Type.DOUBLE:
                self.orientation_covariance = param.value
        return SetParametersResult(successful=True)

    def cloud_callback(self, in_msg: PointCloud2):
        """When receiving a new point cloud, run KISS-SLAM and produce a tf and odometry
        estimate in the same frame as the point source

        :param in_msg: New points to injest
        :type in_msg: PointCloud2
        """
        pcd = pc2_to_numpy(in_msg)
        self.slam.process_scan(pcd, np.empty((0,)))

        pose = matrix_to_pose(self.slam.poses[-1])
        header = Header()
        header.stamp = in_msg.header.stamp
        header.frame_id = self.map_frame

        t = build_transform(header, pose, in_msg.header.frame_id)
        self.tf_broadcaster.sendTransform(t)

        odom = build_odometry(header, pose, self.position_covariance, self.orientation_covariance)
        self.odom_publisher.publish(odom)

        self.update_mapper(pcd)
        map_msg = build_map(
            header,
            self.occupancy_2d,
            self.min_voxel_idx,
            self.slam_config.occupancy_mapper.resolution,
        )
        self.map_publisher.publish(map_msg)

    def update_mapper(self, pcd: np.ndarray):
        """Use a point cloud to update the node's OccupancyGridMapper and store the new 2d grid.
        Also update the minimum active voxel xy idx so the map origin can be placed.

        :param pcd: Latest point cloud received
        :type pcd: np.ndarray
        """
        if self.ref_ground_alignment is None:
            self.ref_ground_alignment = map_closures.align_map_to_local_ground(
                self.slam.voxel_grid.open3d_pcd_with_normals().point.positions.cpu().numpy(),
                self.slam_config.odometry.mapping.voxel_size,
            )

        self.mapper.integrate_frame(pcd, self.ref_ground_alignment @ self.slam.poses[-1])
        self.mapper.compute_3d_occupancy_information()
        self.mapper.compute_2d_occupancy_information()
        self.occupancy_2d = self.mapper.occupancy_grid
        self.min_voxel_idx = (self.mapper.lower_bound[0], self.mapper.lower_bound[1])


def main(args=None):
    rclpy.init(args=args)

    kiss = KissSLAMNode()

    rclpy.spin(kiss)

    kiss.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
