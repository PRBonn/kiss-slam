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
import numpy as np
from geometry_msgs.msg import Pose, TransformStamped
from nav_msgs.msg import OccupancyGrid, Odometry
from rclpy.exceptions import (
    ParameterNotDeclaredException,
    ParameterUninitializedException,
)
from rclpy.node import Node

# Scipy is a really heavy dependency just for rot/quat conversion
# but it's being included anyway from KISS
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header

from kiss_slam.config import KissSLAMConfig, load_config


def pc2_to_numpy(msg: PointCloud2):
    """Convert a PointCloud2 to an (n, 3) numpy array for injestion by KISS-SLAM.
    This is distinct from point_cloud2.read_points_numpy() so we can handle clouds that
    have a different dtype between xyz and other fields.

    :param msg: Point source to be converted
    :type msg: PointCloud2
    :return: An array of size (n,3) with the same point information
    :rtype: np.ndarray
    """
    fields = ["x", "y", "z"]
    structured = point_cloud2.read_points(msg, field_names=fields)
    unstructured = point_cloud2.structured_to_unstructured(structured)
    return unstructured


def matrix_to_pose(mtx: np.ndarray):
    """Convert a homogenous transformation matrix into a geometry_msgs Pose.
    Welcome to suggestions of existing libraries for this.

    :param mtx: Homogenous transformation matrix to be converted of size (4,4)
    :type mtx: np.ndarray
    :return: A Pose message representing the same transformation
    :rtype: Pose
    """
    pose = Pose()

    pose.position.x = mtx[0, 3]
    pose.position.y = mtx[1, 3]
    pose.position.z = mtx[2, 3]

    R = Rotation.from_matrix(mtx[:3, :3])
    q = R.as_quat()
    pose.orientation.x = q[0]
    pose.orientation.y = q[1]
    pose.orientation.z = q[2]
    pose.orientation.w = q[3]

    return pose


def build_transform(header: Header, pose: Pose, child_frame_id: str):
    """Convenience function for packing a transform

    :param header: A Header to provide timestamp and frame_id
    :type header: Header
    :param pose: The Pose of the child frame relative to the base frame
    :type pose: Pose
    :param child_frame_id: Frame ID of the child frame
    :type child_frame_id: str
    :return: A tf containing input information
    :rtype: TransformStamped
    """
    t = TransformStamped()
    t.header = header
    t.transform.translation.x = pose.position.x
    t.transform.translation.y = pose.position.y
    t.transform.translation.z = pose.position.z
    t.transform.rotation = pose.orientation
    t.child_frame_id = child_frame_id
    return t


def build_odometry(header: Header, pose: Pose, position_cov: float, orientation_cov: float):
    """Convenience function for packing an odom message.
    Position and orientation variables are assumed to be independent.
    Twist not currently implemented.

    :param header: A Header to provide timestamp and frame_id
    :type header: Header
    :param pose: The Pose of the robot
    :type pose: Pose
    :param position_cov: Covariance value to use for x, y, and z
    :type position_cov: float
    :param orientation_cov: Covariance value to use for r_x, r_y, and r_z
    :type orientation_cov: float
    :return: An odom message containing input information
    :rtype: Odometry
    """
    odom = Odometry()
    odom.header = header
    odom.pose.pose = pose
    odom.pose.covariance[0] = position_cov
    odom.pose.covariance[7] = position_cov
    odom.pose.covariance[14] = position_cov
    odom.pose.covariance[21] = orientation_cov
    odom.pose.covariance[28] = orientation_cov
    odom.pose.covariance[35] = orientation_cov
    return odom


def build_map(header: Header, occupancy_2d: np.ndarray, min_voxel_idx: tuple, resolution: float):
    """Convenience function for packing a map message.

    :param header: A header to provide timestamp and frame_id
    :type header: Header
    :param occupancy_2d: An occupancy grid supplied by an OccupancyGridMapper
    :type occupancy_2d: np.ndarray
    :param min_voxel_idx: The lowest active voxel index, used to calculate map origin
    :type min_voxel_idx: tuple
    :param resolution: Size of a grid cell in m
    :type resolution: float
    :return: An occupancy grid message containing input information
    :rtype: OccupancyGrid
    """
    map_msg = OccupancyGrid()
    map_msg.header = header
    map_msg.info.resolution = resolution
    map_msg.info.width = occupancy_2d.shape[0]
    map_msg.info.height = occupancy_2d.shape[1]
    map_msg.info.origin.position.x = min_voxel_idx[0] * resolution
    map_msg.info.origin.position.y = min_voxel_idx[1] * resolution

    # Rotate, invert probabilities, and scale to [0, 100]
    occupancy_image = np.rint(100 * (1 - occupancy_2d)).astype(int)
    occupancy_image = np.rot90(occupancy_image)
    occupancy_image = np.flip(occupancy_image, axis=0)
    map_msg.data = occupancy_image.ravel().tolist()

    return map_msg


def slam_params_from_config(config_file: str | None, param_ns: str = "slam_config"):
    """Generate a list of params with default values from a KISS-SLAM config

    :param config_file: Path to a config yaml. Use None for default.
    :type config_file: str | None
    :param param_ns: An optional leading namespace, defaults to "slam_config"
    :type param_ns: str, optional
    :return: Params to be ingested by declare_parameters
    :rtype: List[Tuple[str, Any]]
    """
    default_dict = load_config(config_file).model_dump()

    def recurse_dict(name, value):
        params = []

        # Handle deeper levels of dict by suffixing keys (level0.level1.leveln)
        # and extending list with new entries
        if isinstance(value, dict):
            for k, v in value.items():
                params.extend(recurse_dict(name + "." + str(k), v))

        # At lowest level, create a new list containing only full name and its value
        else:
            params = [(name, value)]

        return params

    return recurse_dict(param_ns, default_dict)


def slam_config_from_params(node: Node, param_ns: str = "slam_config"):
    """Create a KissSLAMConfig by looking up a node's params

    :param node: ROS2 node which should have KISS-SLAM params
    :type node: Node
    :param param_ns: An optional leading namespace, defaults to "slam_config"
    :type param_ns: str, optional
    :return: A new config containing information found in the parameters
    :rtype: KissSLAMConfig
    """
    template = load_config(None)
    template_dict = template.model_dump()

    def recurse_dict(name, value):
        updated_dict = {}

        # Handle deeper levels of dict by suffixing keys and adding unsuffixed
        # key to dict if its value is valid
        if isinstance(value, dict):
            for k, v in value.items():
                new_v = recurse_dict(name + "." + str(k), v)
                if new_v is None:
                    pass
                updated_dict[k] = new_v

        # Try to retrieve param value using full key name, return None if it
        # was not initialized
        else:
            try:
                param_value = node.get_parameter(name).value
                return param_value
            except ParameterNotDeclaredException:
                return None
            except ParameterUninitializedException:
                return None

        return updated_dict

    # Fetch params and incorporate to a new config object
    new_dict = recurse_dict(param_ns, template_dict)
    return KissSLAMConfig.model_validate(new_dict)
