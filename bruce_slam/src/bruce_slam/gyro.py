import numpy as np
from nav_msgs.msg import Odometry
from std_msgs.msg import Header
import gtsam
from scipy.spatial.transform import Rotation

from bruce_slam.utils.topics import *
from bruce_slam.utils.conversions import *
from bruce_slam.utils.io import *
from bruce_slam.sensors import GYRO_ADAPTERS, make_adapter


class GyroFilter(BruceNode):
	'''A class to support dead reckoning using DVL and IMU readings
	'''
	def __init__(self):
		# start the euler angles
		self.roll, self.yaw, self.pitch = 90.,0.,0.

	def init_node(self, node_name:str="gyro_fusion", **node_kwargs)->None:
		"""Node init, get all the relevant params etc.

		Args:
			node_name (str, optional): The ROS 2 node name. Defaults to "gyro_fusion".
			**node_kwargs: extra rclpy Node kwargs (e.g. parameter_overrides).
		"""

		# initialise the underlying rclpy node
		BruceNode.__init__(self, node_name, **node_kwargs)

		# define the rotation offset matrix for the gyro, this makes the gyro frame align with the sonar frame
		x = self.get_param("offset/x")
		y = self.get_param("offset/y")
		z = self.get_param("offset/z")
		self.offset_matrix = Rotation.from_euler("xyz",[x,y,z],degrees=True).as_matrix()

		# the speed the earth is rotating
		self.latitude = np.radians(self.get_param("latitude"))
		self.earth_rate = -15.04107 * np.sin(self.latitude) / 3600.0
		self.sensor_rate = self.get_param("sensor_rate")

		# gyro driver (pluggable adapter + configurable topic)
		self.gyro_adapter, gyro_type = make_adapter(
			GYRO_ADAPTERS, self.get_param("gyro/driver", "kvh_gyro"), self)
		self.gyro_topic = self.get_param("gyro/topic", GYRO_TOPIC)

		# define tf transformer and gyro sub. QoS depth must be an int (a float
		# sensor_rate in the YAML would otherwise make rclpy raise).
		queue_depth = int(self.sensor_rate) + 50
		self.odom_pub = self.create_publisher(Odometry, GYRO_INTEGRATION_TOPIC, queue_depth)
		self.gyro_sub = self.create_subscription(
			gyro_type, self.gyro_topic, self.callback, queue_depth)

		loginfo("Gyro filtering node is initialized")


	def callback(self, gyro_msg)->None:
		"""Callback function, takes in the raw gyro readings (delta angles) and
		updates the estimate of euler angles. Publishes these angles as a ROS odometry message.

		Args:
			gyro_msg: the raw gyro driver message (normalized via the gyro adapter); the
				delta values are delta angles not rotation rates.
		"""

		# parse message and apply the offset matrix
		reading = self.gyro_adapter(gyro_msg)
		dx,dy,dz = reading.delta
		arr = np.array([dx,dy,dz])
		arr = arr.dot(self.offset_matrix)
		delta_yaw, delta_pitch, delta_roll = arr

		# subtract the rotation of the eath
		delta_roll += (self.earth_rate / self.sensor_rate)

		# perform the integration, note this is in radians
		self.pitch += delta_pitch
		self.yaw += delta_yaw
		self.roll += delta_roll

		#package as a gtsam object
		rot = gtsam.Rot3.Ypr(self.yaw,self.pitch,self.roll)
		pose = gtsam.Pose3(rot, gtsam.Point3(0,0,0))

		# publish an odom message
		header = Header()
		header.stamp = reading.header.stamp
		header.frame_id = "odom"
		odom_msg = Odometry()
		odom_msg.header = header
		odom_msg.pose.pose = g2r(pose)
		odom_msg.child_frame_id = "base_link"
		odom_msg.twist.twist.linear.x = 0.0
		odom_msg.twist.twist.linear.y = 0.0
		odom_msg.twist.twist.linear.z = 0.0
		odom_msg.twist.twist.angular.x = 0.0
		odom_msg.twist.twist.angular.y = 0.0
		odom_msg.twist.twist.angular.z = 0.0
		self.odom_pub.publish(odom_msg)
