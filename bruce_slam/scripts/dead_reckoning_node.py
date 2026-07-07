#!/usr/bin/env python3
import sys
import rclpy
from rclpy.utilities import remove_ros_args

# pull in the dead reckoning code
from bruce_slam.utils.io import *
from bruce_slam.dead_reckoning import DeadReckoningNode


def main(args=None):
    rclpy.init(args=args)

    node = DeadReckoningNode()
    node.init_node("localization")

    parser = common_parser()
    parsed, _ = parser.parse_known_args(remove_ros_args(args=sys.argv)[1:])

    try:
        if not parsed.file:
            loginfo("Start online localization...")
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
