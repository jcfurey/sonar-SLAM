#!/usr/bin/env python3
import rclpy

from bruce_slam.utils.io import *
from bruce_slam.feature_extraction import FeatureExtraction


def main(args=None):
    rclpy.init(args=args)

    # call class constructor and configure the node
    node = FeatureExtraction()
    node.init_node("feature_extraction_node")

    try:
        loginfo("Start online sonar feature extraction...")
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
