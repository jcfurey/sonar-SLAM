#!/usr/bin/env python3
import sys
import rclpy
from rclpy.utilities import remove_ros_args

# pull in the kalman filter code
from bruce_slam.utils.io import *
from bruce_slam.kalman import KalmanNode


def main(args=None):
    rclpy.init(args=args)

    node = KalmanNode()
    node.init_node("kalman")

    parser = common_parser()
    parsed, _ = parser.parse_known_args(remove_ros_args(args=sys.argv)[1:])

    try:
        if not parsed.file:
            loginfo("Start online Kalman...")
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
