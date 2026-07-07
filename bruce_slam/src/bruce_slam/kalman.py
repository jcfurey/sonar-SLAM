#python imports
import gtsam
import numpy as np
from scipy.spatial.transform import Rotation

# ros-python imports
from tf2_ros import TransformBroadcaster

# standard ros message imports
from nav_msgs.msg import Odometry
from std_msgs.msg import Header
from sensor_msgs.msg import Imu

# import custom messages
from rti_dvl.msg import DVL
from bar30_depth.msg import Depth
from kvh_gyro.msg import gyro

# bruce imports
from bruce_slam.utils.topics import *
from bruce_slam.utils.conversions import *
from bruce_slam.utils.io import *


def euler_from_quaternion(quat):
	"""Return (roll, pitch, yaw) from a quaternion given as (x, y, z, w).

	Replaces tf.transformations.euler_from_quaternion (unavailable in ROS 2)
	using scipy (default axis order 'xyz' == tf's 'sxyz').
	"""
	return Rotation.from_quat([quat[0], quat[1], quat[2], quat[3]]).as_euler("xyz")


class KalmanNode(BruceNode):
	'''A class to support Kalman filtering using DVL, IMU, FOG and Depth readings.
	'''

	def __init__(self):

		#state vector = (x,y,z,roll, pitch, yaw, x_dot,y_dot,z_dot,roll_dot,pitch_dot,yaw_dot)
		self.state_vector= np.array([[0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0]])
		self.cov_matrix= np.diag([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
		self.yaw_gyro = 0.
		self.imu_yaw0 = None


	def init_node(self, node_name="kalman")->None:
		"""Init the node, fetch all paramaters.

		Args:
			node_name (str, optional): The ROS 2 node name. Defaults to "kalman".
		"""

		# initialise the underlying rclpy node
		BruceNode.__init__(self, node_name)

		# ROS 2 parameters cannot hold 2-D arrays, so the matrices below are stored
		# flattened (row-major) in the YAML and reshaped to their true dimensions here.
		self.state_vector = np.array(self.get_param("state_vector"), dtype=float).reshape(12, 1)
		self.cov_matrix = np.array(self.get_param("cov_matrix"), dtype=float).reshape(12, 12)
		self.R_dvl = np.array(self.get_param("R_dvl"), dtype=float).reshape(3, 3)
		self.dt_dvl = self.get_param("dt_dvl")
		self.H_dvl = np.array(self.get_param("H_dvl"), dtype=float).reshape(3, 12)
		self.R_imu = np.array(self.get_param("R_imu"), dtype=float).reshape(3, 3)
		self.dt_imu = self.get_param("dt_imu")
		self.H_imu = np.array(self.get_param("H_imu"), dtype=float).reshape(3, 12)
		self.H_gyro = np.array(self.get_param("H_gyro"), dtype=float).reshape(3, 12)
		self.R_gyro = np.array(self.get_param("R_gyro"), dtype=float).reshape(3, 3)
		self.dt_gyro = self.get_param("dt_gyro")
		self.H_depth = np.array(self.get_param("H_depth"), dtype=float).reshape(3, 12)
		self.R_depth = np.array(self.get_param("R_depth"), dtype=float).reshape(3, 3)
		self.dt_depth = self.get_param("dt_depth")
		self.Q = np.array(self.get_param("Q"), dtype=float).reshape(12, 12)  # Process Noise Uncertainty
		self.A_imu = np.array(self.get_param("A_imu"), dtype=float).reshape(12, 12)  # State Transition Matrix
		x = self.get_param("offset/x")  # gyroscope offset matrix
		y = self.get_param("offset/y")
		z = self.get_param("offset/z")
		self.offset_matrix = Rotation.from_euler("xyz",[x,y,z],degrees=True).as_matrix()
		self.dvl_max_velocity = self.get_param("dvl_max_velocity")
		self.use_gyro = self.get_param("use_gyro")
		self.imu_offset = np.radians(self.get_param("imu_offset"))

		# check which version of the imu we are using
		if self.get_param("imu_version") == 1:
			self.imu_sub = self.create_subscription(Imu, IMU_TOPIC, self.imu_callback, 250)
		elif self.get_param("imu_version") == 2:
			self.imu_sub = self.create_subscription(Imu, IMU_TOPIC_MK_II, self.imu_callback, 250)

		# define the other subcribers
		self.dvl_sub = self.create_subscription(DVL, DVL_TOPIC, self.dvl_callback, 250)
		self.depth_sub = self.create_subscription(Depth, DEPTH_TOPIC, self.pressure_callback, 250)
		self.odom_pub_kalman = self.create_publisher(Odometry, LOCALIZATION_ODOM_TOPIC, 250)

		# define the transfor broadcaster
		self.tf1 = TransformBroadcaster(self)

		# if we are using the gyroscope set up the subscribers
		if self.use_gyro:
			self.gyro_sub = self.create_subscription(gyro, GYRO_TOPIC, self.gyro_callback, 250)

		# define the initial pose, all zeros
		R_init = gtsam.Rot3.Ypr(0.,0.,0.)
		self.pose = gtsam.Pose3(R_init, gtsam.Point3(0, 0, 0))

		# log at the roslevel that we are done with init
		loginfo("Kalman Node is initialized")


	def kalman_predict(self,previous_x:np.array,previous_P:np.array,A:np.array):
		"""Propagate the state and the error covariance ahead.

		Args:
			previous_x (np.array): value of the previous state vector
			previous_P (np.array): value of the previous covariance matrix
			A (np.array): State Transition Matrix

		Returns:
			predicted_x (np.array): predicted estimation
			predicted_P (np.array): predicted covariance matrix
		"""

		A = np.array(A)
		predicted_P = A @ previous_P @ A.T + self.Q
		predicted_x = A @ previous_x

		return predicted_x, predicted_P


	def kalman_correct(self, predicted_x:np.array, predicted_P:np.array, z:np.array, H:np.array, R:np.array):
		"""Measurement Update.

		Args:
			predicted_x (np.array): predicted state vector with kalman_predict()
			predicted_P (np.array): predicted covariance matrix with kalman_predict()
			z (np.array): Output Vector (measurement)
			H (np.array): Observation Matrix (H_dvl, H_imu, H_gyro, H_depth)
			R (np.array): Measurement Uncertainty (R_dvl, R_imu, R_gyro, R_depth)

		Returns:
			corrected_x (np.array): corrected estimation
			corrected_P (np.array): corrected covariance matrix

		"""

		K = predicted_P @ H.T @ np.linalg.inv(H @ predicted_P @ H.T + R)
		corrected_x = predicted_x + K @ (z - H @ predicted_x)
		corrected_P = predicted_P - K @ H @ predicted_P

		return corrected_x, corrected_P


	def gyro_callback(self,gyro_msg:gyro)->None:
		"""Handle the Kalman Filter using the FOG only.
		Args:
			gyro_msg (gyro): the euler angles from the gyro
		"""

		# parse message and apply the offset matrix
		arr = np.array(list(gyro_msg.delta))
		arr = arr.dot(self.offset_matrix)
		delta_yaw_meas = np.array([[arr[0]],[0],[0]]) #Measurement of shape(3,1) to apply Kalman
		self.state_vector,self.cov_matrix = self.kalman_correct(self.state_vector, self.cov_matrix, delta_yaw_meas, self.H_gyro, self.R_gyro)
		self.yaw_gyro += self.state_vector[11][0]

	def dvl_callback(self, dvl_msg:DVL)->None:
		"""Handle the Kalman Filter using the DVL only.

		Args:
			dvl_msg (DVL): the message from the DVL
		"""

		# parse the dvl velocites
		dvl_measurement = np.array([[dvl_msg.velocity.x], [dvl_msg.velocity.y], [dvl_msg.velocity.z]])

		# We do not do a kalman correction if the speed is high.
		if np.any(np.abs(dvl_measurement) > self.dvl_max_velocity):
			return
		else:
			self.state_vector,self.cov_matrix  = self.kalman_correct(self.state_vector, self.cov_matrix, dvl_measurement, self.H_dvl, self.R_dvl)


	def pressure_callback(self,depth_msg:Depth):
		"""Handle the Kalman Filter using the Depth.
		Args:
			depth_msg (Depth): pressure
		"""

		depth = np.array([[depth_msg.depth],[0],[0]]) # We need the shape(3,1) for the correction
		self.state_vector,self.cov_matrix = self.kalman_correct(self.state_vector, self.cov_matrix, depth, self.H_depth, self.R_depth)

	def imu_callback(self, imu_msg:Imu)->None:
		"""Handle the Kalman Filter using the VN100 only. Publish the state vector.

		Args:
			imu_msg (Imu): the message from VN100
		"""

		# Kalman prediction
		predicted_x, predicted_P = self.kalman_predict(self.state_vector, self.cov_matrix, self.A_imu)

		# parse the IMU measurnment
		roll_x, pitch_y, yaw_z = euler_from_quaternion((imu_msg.orientation.x,imu_msg.orientation.y,imu_msg.orientation.z,imu_msg.orientation.w))
		euler_angle = np.array([[self.imu_offset+roll_x], [pitch_y], [yaw_z]])

		#if we have no yaw yet, set this one as zero
		if self.imu_yaw0 is None:
			self.imu_yaw0 = yaw_z

		# make yaw relative to the first meas
		euler_angle[2] -= self.imu_yaw0

		# Kalman correction
		self.state_vector,self.cov_matrix = self.kalman_correct(predicted_x, predicted_P, euler_angle, self.H_imu, self.R_imu)

		# Use filtered velocity to update our x and y estimates
		trans_x = self.state_vector[6][0]*self.dt_imu # x update
		trans_y = self.state_vector[7][0]*self.dt_imu # y update
		local_point = gtsam.Point2(trans_x, trans_y)

		# check if we are using the FOG
		if self.use_gyro:
			R = gtsam.Rot3.Ypr(self.yaw_gyro,self.state_vector[4][0], self.state_vector[3][0])
			pose2 = gtsam.Pose2(self.pose.x(), self.pose.y(), self.yaw_gyro)
		else: # We are not using the gyro
			R = gtsam.Rot3.Ypr(self.state_vector[5][0], self.state_vector[4][0], self.state_vector[3][0])
			pose2 = gtsam.Pose2(self.pose.x(), self.pose.y(), self.pose.rotation().yaw())

		# update our pose estimate and send out the odometry message
		point = pose2.transformFrom(local_point)
		self.pose = gtsam.Pose3(R, gtsam.Point3(point[0], point[1], 0))
		self.send_odometry(imu_msg.header.stamp)

	def send_odometry(self,t):
		"""Publish the pose.
		Args:
			t: time stamp from imu_msg (builtin_interfaces/Time)
		"""

		header = Header()
		header.stamp = t
		header.frame_id = "odom"
		odom_msg = Odometry()
		odom_msg.header = header
		odom_msg.pose.pose = g2r(self.pose)
		odom_msg.child_frame_id = "base_link"
		odom_msg.twist.twist.linear.x = 0.
		odom_msg.twist.twist.linear.y = 0.
		odom_msg.twist.twist.linear.z = 0.
		odom_msg.twist.twist.angular.x = 0.
		odom_msg.twist.twist.angular.y = 0.
		odom_msg.twist.twist.angular.z = 0.
		self.odom_pub_kalman.publish(odom_msg)

		p = odom_msg.pose.pose.position
		q = odom_msg.pose.pose.orientation
		self.tf1.sendTransform(
			make_transform((p.x, p.y, p.z), (q.x, q.y, q.z, q.w), header.stamp, "odom", "base_link")
		)
