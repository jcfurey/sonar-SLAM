#!/usr/bin/env python3
"""Relay an ENU (REP-105) odometry topic into bruce's z-down convention.

The bruce SLAM back-end consumes exactly one odometry input
(/bruce/slam/localization/odom) and uses relative between-keyframe deltas,
so an external EKF can drive it — but bruce's pipeline is 3-DOF z-down
while our robot_localization EKF publishes ENU z-up. This relay conjugates
the pose by the roll-pi transform (x, -y, -z; quaternion x, -y, -z, w),
flips the twist the same way, and passes header/stamp through untouched
(the back-end ApproximateTimeSynchronizes odom against feature clouds).

Replaces bruce's own dead_reckoning/kalman front-ends, which would fight
the EKF for odom->base_link.
"""
import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from nav_msgs.msg import Odometry

from bruce_slam.utils.topics import LOCALIZATION_ODOM_TOPIC

# sign signature of the roll-pi conjugation on (x, y, z, rot_x, rot_y, rot_z)
FLIP = np.array([1.0, -1.0, -1.0, 1.0, -1.0, -1.0])


class EnuOdomRelay(Node):
    def __init__(self):
        super().__init__("enu_odom_relay")
        self.declare_parameter("input_topic", "/localization/odometry/odom")
        input_topic = self.get_parameter("input_topic").value
        self.pub = self.create_publisher(Odometry, LOCALIZATION_ODOM_TOPIC, 10)
        self.sub = self.create_subscription(Odometry, input_topic, self.callback, 10)

    @staticmethod
    def flip_covariance(cov):
        c = np.asarray(cov, dtype=np.float64).reshape(6, 6)
        return (np.outer(FLIP, FLIP) * c).reshape(-1)

    def callback(self, msg: Odometry):
        out = Odometry()
        out.header = msg.header
        out.child_frame_id = msg.child_frame_id

        p = msg.pose.pose.position
        out.pose.pose.position.x = p.x
        out.pose.pose.position.y = -p.y
        out.pose.pose.position.z = -p.z
        q = msg.pose.pose.orientation
        out.pose.pose.orientation.x = q.x
        out.pose.pose.orientation.y = -q.y
        out.pose.pose.orientation.z = -q.z
        out.pose.pose.orientation.w = q.w
        out.pose.covariance = self.flip_covariance(msg.pose.covariance)

        lin = msg.twist.twist.linear
        ang = msg.twist.twist.angular
        out.twist.twist.linear.x = lin.x
        out.twist.twist.linear.y = -lin.y
        out.twist.twist.linear.z = -lin.z
        out.twist.twist.angular.x = ang.x
        out.twist.twist.angular.y = -ang.y
        out.twist.twist.angular.z = -ang.z
        out.twist.covariance = self.flip_covariance(msg.twist.covariance)

        self.pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = EnuOdomRelay()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        # normal Ctrl-C / launch-initiated shutdown — not an error
        pass
    finally:
        node.destroy_node()


if __name__ == "__main__":
    main()
