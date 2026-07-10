#!/usr/bin/env python3

import os
import sys
import threading

import yaml as pyyaml

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.executors import SingleThreadedExecutor
from rclpy.parameter import Parameter
from rclpy.utilities import remove_ros_args
from ament_index_python.packages import get_package_share_directory

from bruce_slam.utils.io import *
from bruce_slam.slam_ros import SLAMNode
from bruce_slam.utils.topics import *
from bruce_slam.utils.conversions import make_transform


def load_config_overrides(config_name):
    """Load a bruce_slam config YAML as rclpy parameter overrides.

    Reads the ``/**: ros__parameters:`` document, flattens nested keys to the
    dotted ROS 2 form, and returns a list of rclpy Parameters. Used by the
    offline mode to give each in-process node exactly its own config file —
    restoring the per-node parameter isolation the online launch has (the
    ``/**`` wildcard would otherwise apply every file to every node, with
    colliding keys silently taking the last file's value).

    Args:
        config_name (str): file name inside the installed config directory

    Returns:
        list[Parameter]: overrides for the node constructor
    """

    path = os.path.join(
        get_package_share_directory("bruce_slam"), "config", config_name
    )
    with open(path) as f:
        data = pyyaml.safe_load(f)
    params = data.get("/**", {}).get("ros__parameters", {})

    flat = {}

    def _flatten(prefix, d):
        for k, v in d.items():
            key = "{}.{}".format(prefix, k) if prefix else k
            if isinstance(v, dict):
                _flatten(key, v)
            else:
                flat[key] = v

    _flatten("", params)
    return [Parameter(k, value=v) for k, v in flat.items()]


def offline(node, args) -> None:
    """Run the SLAM system offline from a rosbag2 bag.

    All nodes are hosted in a single process. Each sub-node is constructed with
    its own config file as parameter overrides and with use_global_arguments
    disabled, so the process-level params (slam.yaml) cannot bleed into it.
    Inter-node topics are delivered by a single-threaded executor pumped from
    a background thread; bag messages are injected from the main thread on the
    nodes' actual configured topics.

    Args:
        node (SLAMNode): the already-initialised SLAM node
        args: parsed CLI arguments (file / start / duration)
    """

    # pull in the extra imports required
    from rosgraph_msgs.msg import Clock
    from tf2_ros import StaticTransformBroadcaster
    from bruce_slam.dead_reckoning import DeadReckoningNode
    from bruce_slam.feature_extraction import FeatureExtraction
    from bruce_slam.gyro import GyroFilter
    from bruce_slam.utils import io

    # set some params
    io.offline = True
    node.save_fig = False
    node.save_data = False

    # instantiate the sub-nodes, each isolated to its own config file
    dead_reckoning_node = DeadReckoningNode()
    dead_reckoning_node.init_node(
        "localization",
        parameter_overrides=load_config_overrides("dead_reckoning.yaml"),
        use_global_arguments=False,
    )
    feature_extraction_node = FeatureExtraction()
    feature_extraction_node.init_node(
        "feature_extraction",
        parameter_overrides=load_config_overrides("feature.yaml"),
        use_global_arguments=False,
    )
    gyro_node = GyroFilter()
    gyro_node.init_node(
        "gyro",
        parameter_overrides=load_config_overrides("gyro.yaml"),
        use_global_arguments=False,
    )
    clock_pub = node.create_publisher(Clock, "/clock", 100)

    # world -> map is a fixed transform; broadcast it once (latched)
    static_tf = StaticTransformBroadcaster(node)
    static_tf.sendTransform(
        make_transform((0, 0, 0), (1, 0, 0, 0), Clock().clock, "world", "map")
    )

    # spin every node in the background so inter-node topics are delivered.
    # A single-threaded executor keeps callback concurrency close to the ROS 1
    # behaviour (one executor callback at a time).
    executor = SingleThreadedExecutor()
    for n in (node, dead_reckoning_node, feature_extraction_node, gyro_node):
        executor.add_node(n)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    # route on the topics the nodes are ACTUALLY configured with (the
    # dvl/depth/gyro/sonar topic params), falling back to the historic
    # defaults so old bags keep working.
    imu_topics = {IMU_TOPIC, IMU_TOPIC_MK_II}
    if dead_reckoning_node.imu_topic:
        imu_topics.add(dead_reckoning_node.imu_topic)
    dvl_topics = {DVL_TOPIC, dead_reckoning_node.dvl_topic}
    depth_topics = {DEPTH_TOPIC, dead_reckoning_node.depth_topic}
    sonar_topics = {SONAR_TOPIC, SONAR_TOPIC_UNCOMPRESSED, feature_extraction_node.sonar_topic}
    gyro_topics = {GYRO_TOPIC, gyro_node.gyro_topic}

    # loop over the entire rosbag
    for topic, msg in read_bag(args.file, args.start, args.duration, progress=True):
        while rclpy.ok():
            if callback_lock_event.wait(1.0):
                break

        if not rclpy.ok():
            break

        if topic in imu_topics:
            # imu_sub only exists when the localization node uses the IMU
            if hasattr(dead_reckoning_node, "imu_sub"):
                dead_reckoning_node.imu_sub.signalMessage(msg)
        elif topic in dvl_topics:
            dead_reckoning_node.dvl_sub.signalMessage(msg)
        elif topic in depth_topics:
            dead_reckoning_node.depth_sub.signalMessage(msg)
        elif topic in sonar_topics:
            # configure the SLAM node's sonar geometry (idempotent) and
            # extract features
            node.sonar_callback(msg)
            feature_extraction_node.callback(msg)
        elif topic in gyro_topics:
            gyro_node.callback(msg)

        # drive /clock from every message so sim time advances even for bags
        # with no IMU (e.g. the DVL+depth only mode)
        if hasattr(msg, "header"):
            clock_pub.publish(Clock(clock=msg.header.stamp))

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
    except (KeyboardInterrupt, ExternalShutdownException):
        # normal Ctrl-C / launch-initiated shutdown — not an error
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
