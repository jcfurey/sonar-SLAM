#!/usr/bin/env python3
import rclpy

# pull in the kalman filter code
from bruce_slam.utils.io import *
from bruce_slam.kalman import KalmanNode


def main(args=None):
    rclpy.init(args=args)

    node = KalmanNode()
    node.init_node("kalman")

    try:
        loginfo("Start online Kalman...")
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
