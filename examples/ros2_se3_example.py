#!/usr/bin/env python3
"""
Practical ROS2 to SE3 conversion examples.

This shows how to use the converter with actual ROS2 message types:
- geometry_msgs/Transform and TransformStamped
- geometry_msgs/Pose and PoseStamped  
- tf2_msgs/TFMessage
"""

import torch
import sys
import os
from ros2_to_se3_converter import ROS2ToSE3Converter
