#!/usr/bin/env python3
import rclpy

from bruce_slam.utils.io import *
from bruce_slam.gyro import GyroFilter


def main(args=None):
    rclpy.init(args=args)

    node = GyroFilter()
    node.init_node("gyro_fusion")

    try:
        loginfo("Start gyro_fusion...")
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
