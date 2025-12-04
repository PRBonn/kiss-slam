# KISS-SLAM ROS 2 Wrapper
This directory provides a ROS 2 wrapper for some of KISS-SLAM's functionality.
The kiss_slam_node receives point cloud messages and publishes odometry and a transform between the map frame and laser frame.
An occupancy grid is also published, but this is currently incomplete and does not incorporate loop closures.

Currently tested on Humble.

### How to build
Copy this directory into your workspace, then build, source, and run as normal provided that `kiss-slam` is already installed in your environment.

### How to run
By default, the node will listen for the topic `/points`.
This can be changed at runtime with a parameter:
```
ros2 run kiss_slam_ros kiss_slam_node --ros-args -p points_topic:=your_pointcloud_topic
```
