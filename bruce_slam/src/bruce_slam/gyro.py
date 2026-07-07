import numpy as np
from nav_msgs.msg import Odometry
from std_msgs.msg import Header
import gtsam
from scipy.spatial.transform import Rotation

from kvh_gyro.msg import gyro

from bruce_slam.utils.topics import *
from bruce_slam.utils.conversions import *
from bruce_slam.utils.io import *


class GyroFilter(BruceNode):
	'''A class to support dead reckoning using DVL and IMU readings
	'''
	def __init__(self):
		# start the euler angles
		self.roll, self.yaw, self.pitch = 90.,0.,0.

	def init_node(self, node_name:str="gyro_fusion")->None:
		"""Node init, get all the relevant params etc.

		Args:
			node_name (str, optional): The ROS 2 node name. Defaults to "gyro_fusion".
		"""

		# initialise the underlying rclpy node
		BruceNode.__init__(self, node_name)

		# define the rotation offset matrix for the gyro, this makes the gyro frame align with the sonar frame
		x = self.get_param("offset/x")
		y = self.get_param("offset/y")
		z = self.get_param("offset/z")
		self.offset_matrix = Rotation.from_euler("xyz",[x,y,z],degrees=True).as_matrix()

		# the speed the earth is rotating
		self.latitude = np.radians(self.get_param("latitude"))
		self.earth_rate = -15.04107 * np.sin(self.latitude) / 3600.0
		self.sensor_rate = self.get_param("sensor_rate")

		# define tf transformer and gyro sub
		self.odom_pub = self.create_publisher(Odometry, GYRO_INTEGRATION_TOPIC, self.sensor_rate+50)
		self.gyro_sub = self.create_subscription(gyro, GYRO_TOPIC, self.callback, self.sensor_rate+50)

		loginfo("Gyro filtering node is initialized")


	def callback(self, gyro_msg:gyro)->None:
		"""Callback function, takes in the raw gyro readings (delta angles) and
		updates the estimate of euler angles. Publishes these angles as a ROS odometry message.

		Args:
			gyro_msg (gyro): the incoming gyro message, these are delta angles not rotation rates.
		"""

		# parse message and apply the offset matrix
		dx,dy,dz = list(gyro_msg.delta)
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
		header.stamp = gyro_msg.header.stamp
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
