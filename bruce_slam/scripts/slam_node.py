#!/usr/bin/env python3

import sys
import threading

import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.utilities import remove_ros_args

from bruce_slam.utils.io import *
from bruce_slam.slam_ros import SLAMNode
from bruce_slam.utils.topics import *
from bruce_slam.utils.conversions import make_transform


def offline(node, args) -> None:
    """Run the SLAM system offline from a rosbag2 bag.

    All nodes are hosted in a single process; a background executor delivers the
    inter-node messages (features, odometry) while the bag is replayed from the
    main thread. This is a best-effort port of the ROS 1 offline mode and expects
    a rosbag2 directory rather than a legacy ``.bag`` file.

    Args:
        node (SLAMNode): the already-initialised SLAM node
        args: parsed CLI arguments (file / start / duration)
    """

    # pull in the extra imports required
    from rosgraph_msgs.msg import Clock
    from bruce_slam.dead_reckoning import DeadReckoningNode
    from bruce_slam.feature_extraction import FeatureExtraction
    from bruce_slam.gyro import GyroFilter
    from bruce_slam.utils import io

    # set some params
    io.offline = True
    node.save_fig = False
    node.save_data = False

    # instantiate the nodes required
    dead_reckoning_node = DeadReckoningNode()
    dead_reckoning_node.init_node("localization")
    feature_extraction_node = FeatureExtraction()
    feature_extraction_node.init_node("feature_extraction")
    gyro_node = GyroFilter()
    gyro_node.init_node("gyro")
    clock_pub = node.create_publisher(Clock, "/clock", 100)

    # spin every node in the background so inter-node topics are delivered
    executor = MultiThreadedExecutor()
    for n in (node, dead_reckoning_node, feature_extraction_node, gyro_node):
        executor.add_node(n)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    # loop over the entire rosbag
    for topic, msg in read_bag(args.file, args.start, args.duration, progress=True):
        while rclpy.ok():
            if callback_lock_event.wait(1.0):
                break

        if not rclpy.ok():
            break

        if topic == IMU_TOPIC or topic == IMU_TOPIC_MK_II:
            dead_reckoning_node.imu_sub.signalMessage(msg)
        elif topic == DVL_TOPIC:
            dead_reckoning_node.dvl_sub.signalMessage(msg)
        elif topic == DEPTH_TOPIC:
            dead_reckoning_node.depth_sub.signalMessage(msg)
        elif topic in (SONAR_TOPIC, SONAR_TOPIC_UNCOMPRESSED):
            feature_extraction_node.callback(msg)
        elif topic == GYRO_TOPIC:
            gyro_node.callback(msg)

        # use the IMU to drive the clock
        if topic == IMU_TOPIC or topic == IMU_TOPIC_MK_II:
            clock_pub.publish(Clock(clock=msg.header.stamp))

            # Publish map to world so we can visualize all in a z-down frame in rviz.
            node.tf.sendTransform(
                make_transform((0, 0, 0), (1, 0, 0, 0), msg.header.stamp, "world", "map")
            )

    executor.shutdown()
    for n in (dead_reckoning_node, feature_extraction_node, gyro_node):
        n.destroy_node()


def main(args=None):
    rclpy.init(args=args)

    # call the class constructor
    node = SLAMNode()
    node.init_node("slam")

    # parse and start
    parser = common_parser()
    parsed, _ = parser.parse_known_args(remove_ros_args(args=sys.argv)[1:])

    try:
        if not parsed.file:
            loginfo("Start online slam...")
            rclpy.spin(node)
        else:
            loginfo("Start offline slam...")
            offline(node, parsed)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
