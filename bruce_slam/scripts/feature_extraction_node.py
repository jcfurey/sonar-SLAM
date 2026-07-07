#!/usr/bin/env python3
import sys
import rclpy
from rclpy.utilities import remove_ros_args

from bruce_slam.utils.io import *
from bruce_slam.feature_extraction import FeatureExtraction


def main(args=None):
    rclpy.init(args=args)

    # call class constructor and configure the node
    node = FeatureExtraction()
    node.init_node("feature_extraction_node")

    # parse the (non-ROS) CLI args
    parser = common_parser()
    parsed, _ = parser.parse_known_args(remove_ros_args(args=sys.argv)[1:])

    try:
        if not parsed.file:
            loginfo("Start online sonar feature extraction...")
        else:
            loginfo("Offline mode is driven by slam_node; running online here...")
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
