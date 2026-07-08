#!/usr/bin/env python3
import rclpy

# pull in the dead reckoning code
from bruce_slam.utils.io import *
from bruce_slam.dead_reckoning import DeadReckoningNode


def main(args=None):
    rclpy.init(args=args)

    node = DeadReckoningNode()
    node.init_node("localization")

    try:
        loginfo("Start online localization...")
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
