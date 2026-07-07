#!/usr/bin/env python3
import threading
import numpy as np

import rclpy
from message_filters import TimeSynchronizer, Subscriber
from sensor_msgs.msg import PointCloud2
from nav_msgs.msg import OccupancyGrid
from bruce_msgs.srv import GetOccupancyMap

from bruce_slam.utils.topics import *
from bruce_slam.utils.conversions import *
from bruce_slam.utils.io import *
from bruce_slam.mapping import Mapping


class MappingNode(Mapping, BruceNode):
    def __init__(self):
        Mapping.__init__(self)

        self.lock = threading.RLock()
        self.use_slam_traj = True

    def init_node(self, node_name="mapping"):
        # initialise the underlying rclpy node
        BruceNode.__init__(self, node_name)

        self.use_slam_traj = self.get_param("use_slam_traj", True)

        self.x0, self.y0 = self.get_param("origin")
        self.width, self.height = self.get_param("size")
        self.resolution = self.get_param("resolution")
        self.inc = self.get_param("inc")

        self.pub_occupancy1 = self.get_param("pub_occupancy1")
        self.hit_prob = self.get_param("hit_prob")
        self.miss_prob = self.get_param("miss_prob")
        self.inflation_angle = self.get_param("inflation_angle")
        self.inflation_radius = self.get_param("inflation_range")

        self.pub_occupancy2 = self.get_param("pub_occupancy2")
        self.inflation_radius = self.get_param("inflation_radius")
        self.outlier_filter_radius = self.get_param("outlier_filter_radius")
        self.outlier_filter_min_points = self.get_param(
            "outlier_filter_min_points"
        )

        self.pub_intensity = self.get_param("pub_intensity")

        # Only update keyframe that has significant movement
        self.min_translation = self.get_param("min_translation")
        self.min_rotation = self.get_param("min_rotation")

        self.sonar_sub = Subscriber(self, OculusPing, SONAR_TOPIC)
        if self.use_slam_traj:
            self.traj_sub = Subscriber(self, PointCloud2, SLAM_TRAJ_TOPIC)
        else:
            self.traj_sub = Subscriber(self, PointCloud2, LOCALIZATION_TRAJ_TOPIC)
        # Method 1
        if self.pub_occupancy1:
            self.feature_sub = Subscriber(self, PointCloud2, SONAR_FEATURE_TOPIC)
        # Method 2
        if self.pub_occupancy2:
            self.feature_sub = Subscriber(self, PointCloud2, SLAM_CLOUD_TOPIC)

        # The time stamps for trajectory and ping have to be exactly the same
        # A big queue_size is required to assure no keyframe is missed especially
        # for offline playing.
        self.ts = TimeSynchronizer(
            [self.traj_sub, self.sonar_sub, self.feature_sub], 100
        )
        self.ts.registerCallback(self.tpf_callback)

        self.intensity_map_pub = self.create_publisher(
            OccupancyGrid, MAPPING_INTENSITY_TOPIC, latched_qos()
        )
        self.occupancy_map_pub = self.create_publisher(
            OccupancyGrid, MAPPING_OCCUPANCY_TOPIC, latched_qos()
        )

        self.get_map_srv = self.create_service(GetOccupancyMap, "get_map", self.get_map)

        self.configure()
        loginfo("Mapping node is initialized")

    def get_map(self, request, response):
        with self.lock:
            occ_msg = self.get_occupancy_grid(request.frames, request.resolution)
            response.occ = occ_msg

        return response

    @add_lock
    def tpf_callback(self, traj_msg, ping, feature_msg):
        self.lock.acquire()
        with CodeTimer("Mapping - add keyframe"):
            traj = r2n(traj_msg)
            pose = pose322(n2g(traj[-1, :6], "Pose3"))
            points = r2n(feature_msg)
            self.add_keyframe(len(traj) - 1, pose, ping, points)

        with CodeTimer("Mapping - update keyframe"):
            for x in range(len(traj) - 1):
                pose = pose322(n2g(traj[x, :6], "Pose3"))
                self.update_pose(x, pose)

        if self.pub_intensity:
            with CodeTimer("Mapping - publish intensity map"):
                intensity_msg = self.get_intensity_grid()
                intensity_msg.header.stamp = ping.header.stamp
                if not self.use_slam_traj:
                    intensity_msg.header.frame_id = "odom"
                self.intensity_map_pub.publish(intensity_msg)

        if self.pub_occupancy1:
            with CodeTimer("Mapping - publish occupancy map"):
                occupancy_msg = self.get_occupancy_grid1()
                occupancy_msg.header.stamp = ping.header.stamp
                if not self.use_slam_traj:
                    occupancy_msg.header.frame_id = "odom"
                self.occupancy_map_pub.publish(occupancy_msg)

        if self.pub_occupancy2:
            with CodeTimer("Mapping - publish occupancy map"):
                occupancy_msg = self.get_occupancy_grid2()
                occupancy_msg.header.stamp = ping.header.stamp
                if not self.use_slam_traj:
                    occupancy_msg.header.frame_id = "odom"
                self.occupancy_map_pub.publish(occupancy_msg)

        self.lock.release()
        if self.save_fig:
            self.save_submaps()

    def save_submaps(self):
        submaps = []
        for keyframe in self.keyframes:
            submap = (
                g2n(keyframe.pose),
                keyframe.r,
                keyframe.c,
                keyframe.l,
                keyframe.i,
                keyframe.cimg,
                keyframe.limg,
            )
            submaps.append(submap)
        np.savez(
            "step-{}-submaps.npz".format(len(self.keyframes) - 1),
            submaps=submaps,
            map_size=(self.x0, self.y0, self.width, self.height, self.resolution),
        )


def main(args=None):
    rclpy.init(args=args)

    node = MappingNode()
    node.init_node("mapping")

    try:
        loginfo("Start online mapping...")
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
