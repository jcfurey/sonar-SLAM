# python imports
import gtsam
import numpy as np

# ros-python imports
from tf2_ros import TransformBroadcaster
from nav_msgs.msg import Odometry
from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2, Imu
from message_filters import ApproximateTimeSynchronizer, Cache, Subscriber

# import custom messages
from kvh_gyro.msg import gyro as GyroMsg
from rti_dvl.msg import DVL
from bar30_depth.msg import Depth

# bruce imports
from bruce_slam.utils.topics import *
from bruce_slam.utils.conversions import *
from bruce_slam.utils.io import *
from bruce_slam.utils.visualization import ros_colorline_trajectory


class DeadReckoningNode(BruceNode):
	'''A class to support dead reckoning using DVL and IMU readings
	'''
	def __init__(self):
		self.pose = None #vehicle pose
		self.prev_time = None #previous reading time
		self.prev_vel = None #previous reading velocity
		self.keyframes = [] #keyframe list

		# Force yaw at origin to be aligned with x axis
		self.imu_yaw0 = None
		self.imu_pose = [0, 0, 0, -np.pi / 2, 0, 0]
		self.imu_rot = None
		self.dvl_max_velocity = 0.3

		# Create a new key pose when
		# - |ti - tj| > min_duration and
		# - |xi - xj| > max_translation or
		# - |ri - rj| > max_rotation
		self.keyframe_duration = None
		self.keyframe_translation = None
		self.keyframe_rotation = None
		self.dvl_error_timer = 0.0

		# latest heading estimate fed back from the SLAM scan matcher, used by the
		# DVL+depth only mode (no IMU/FOG) to orient the DVL velocities
		self.slam_yaw = 0.0

		# place holder for multi-robot SLAM
		self.rov_id = ""


	def init_node(self, node_name="localization")->None:
		"""Init the node, fetch all paramaters from ROS

		Args:
			node_name (str, optional): The ROS 2 node name. Defaults to "localization".
		"""

		# initialise the underlying rclpy node
		BruceNode.__init__(self, node_name)

		# Parameters for Node
		self.imu_pose = self.get_param("imu_pose")
		self.imu_pose = n2g(self.imu_pose, "Pose3")
		self.imu_rot = self.imu_pose.rotation()
		self.dvl_max_velocity = self.get_param("dvl_max_velocity")
		self.keyframe_duration = self.get_param("keyframe_duration")
		self.keyframe_translation = self.get_param("keyframe_translation")
		self.keyframe_rotation = self.get_param("keyframe_rotation")

		# Subscribers and caches
		self.dvl_sub = Subscriber(self, DVL, DVL_TOPIC)
		self.depth_sub = Subscriber(self, Depth, DEPTH_TOPIC)
		self.depth_cache = Cache(self.depth_sub, 1)

		# Use point cloud for visualization
		self.traj_pub = self.create_publisher(
			PointCloud2, "traj_dead_reck", 10)

		self.odom_pub = self.create_publisher(
			Odometry, LOCALIZATION_ODOM_TOPIC, 10)

		# which orientation sources are available?
		self.use_gyro = self.get_param("use_gyro")        # FOG gyroscope
		self.use_imu = self.get_param("use_imu", True)    # VN100 MEMS IMU

		# only subscribe to the IMU if we intend to use it
		if self.use_imu:
			if self.get_param("imu_version") == 1:
				self.imu_sub = Subscriber(self, Imu, IMU_TOPIC)
			else:
				self.imu_sub = Subscriber(self, Imu, IMU_TOPIC_MK_II)

		# define the callback based on the available orientation sources
		if self.use_imu and self.use_gyro:
			# VN100 (roll/pitch) + FOG (yaw) + DVL
			self.gyro_sub = Subscriber(self, Odometry, GYRO_INTEGRATION_TOPIC)
			self.ts = ApproximateTimeSynchronizer([self.imu_sub, self.dvl_sub, self.gyro_sub], 300, .1)
			self.ts.registerCallback(self.callback_with_gyro)
		elif self.use_imu:
			# VN100 (roll/pitch/yaw) + DVL
			self.ts = ApproximateTimeSynchronizer([self.imu_sub, self.dvl_sub], 200, .1)
			self.ts.registerCallback(self.callback)
		else:
			# No IMU and no FOG: dead reckon from the DVL and depth only, taking the
			# heading from the SLAM scan matcher (fed back on SLAM_ODOM_TOPIC).
			self.dvl_sub.registerCallback(self.callback_dvl_only)
			self.slam_sub = self.create_subscription(
				Odometry, SLAM_ODOM_TOPIC, self.slam_heading_callback, 10)
			logwarn(
				"Localization running in DVL+depth only mode (no IMU/FOG); "
				"heading is taken from the SLAM scan matcher.")

		self.tf = TransformBroadcaster(self)

		loginfo("Localization node is initialized")


	def callback(self, imu_msg:Imu, dvl_msg:DVL)->None:
		"""Handle the dead reckoning using the VN100 and DVL only. Fuse and publish an odometry message.

		Args:
			imu_msg (Imu): the message from VN100
			dvl_msg (DVL): the message from the DVL
		"""
		#get the previous depth message
		depth_msg = self.depth_cache.getLast()
		#if there is no depth message, then skip this time step
		if depth_msg is None:
			return

		#check the delay between the depth message and the DVL
		dd_delay = to_sec(depth_msg.header.stamp) - to_sec(dvl_msg.header.stamp)
		#print(dd_delay)
		if abs(dd_delay) > 1.0:
			logdebug("Missing depth message for {}".format(dd_delay))

		#convert the imu message from msg to gtsam rotation object
		rot = r2g(imu_msg.orientation)
		rot = rot.compose(self.imu_rot.inverse())

		#if we have no yaw yet, set this one as zero
		if self.imu_yaw0 is None:
			self.imu_yaw0 = rot.yaw()

		# Get a rotation matrix
		# if use_gyro has the same value in Kalman and DeadReck, use this line
		rot = gtsam.Rot3.Ypr(rot.yaw()-self.imu_yaw0, rot.pitch(), np.radians(90)+rot.roll())
		# if use_gyro = True in Kalman and use_gyro = False in DeadReck, use this line:
		# rot = gtsam.Rot3.Ypr(rot.yaw()-self.imu_yaw0, rot.pitch(), np.radians(90)+rot.roll())

		# parse the DVL message into an array of velocites
		vel = np.array([dvl_msg.velocity.x, dvl_msg.velocity.y, dvl_msg.velocity.z])

		# package the odom message and publish it
		self.send_odometry(vel,rot,dvl_msg.header.stamp,depth_msg.depth)


	def callback_with_gyro(self, imu_msg:Imu, dvl_msg:DVL, gyro_msg:Odometry)->None:
		"""Handle the dead reckoning state estimate using the fiber optic gyro. Here we use the
		Gyro as a means of getting the yaw estimate, roll and pitch are still VN100.

		Args:
			imu_msg (Imu): the vn100 imu message
			dvl_msg (DVL): the DVL message
			gyro_msg (Odometry): the integrated gyro odometry (from the gyro node)
		"""
		# decode the gyro message
		gyro_yaw = r2g(gyro_msg.pose.pose).rotation().yaw()

		#get the previous depth message
		depth_msg = self.depth_cache.getLast()

		#if there is no depth message, then skip this time step
		if depth_msg is None:
			return

		#check the delay between the depth message and the DVL
		dd_delay = to_sec(depth_msg.header.stamp) - to_sec(dvl_msg.header.stamp)
		#print(dd_delay)
		if abs(dd_delay) > 1.0:
			logdebug("Missing depth message for {}".format(dd_delay))

		#convert the imu message from msg to gtsam rotation object
		rot = r2g(imu_msg.orientation)
		rot = rot.compose(self.imu_rot.inverse())


		# Get a rotation matrix
		rot = gtsam.Rot3.Ypr(gyro_yaw, rot.pitch(), rot.roll())

		#parse the DVL message into an array of velocites
		vel = np.array([dvl_msg.velocity.x, dvl_msg.velocity.y, dvl_msg.velocity.z])

		# package the odom message and publish it
		self.send_odometry(vel,rot,dvl_msg.header.stamp,depth_msg.depth)


	def slam_heading_callback(self, odom_msg:Odometry)->None:
		"""Cache the latest heading estimated by the SLAM scan matcher.

		Used by the DVL+depth only mode to orient the DVL velocities without an
		IMU/FOG. Runs at keyframe rate; the value is held between updates.

		Args:
			odom_msg (Odometry): the SLAM pose estimate (SLAM_ODOM_TOPIC)
		"""
		self.slam_yaw = r2g(odom_msg.pose.pose).rotation().yaw()


	def callback_dvl_only(self, dvl_msg:DVL)->None:
		"""Dead reckon from the DVL and depth only, with no IMU or FOG.

		When neither the VN100 IMU nor the KVH FOG is available there is no inertial
		orientation source. Instead the heading is taken from the SLAM scan matcher
		(fed back on SLAM_ODOM_TOPIC) and the DVL body-frame velocities are rotated
		into the world frame with it. Roll and pitch are assumed level, consistent
		with the fixed-depth 3-DOF motion model. Before the first SLAM estimate
		arrives the heading bootstraps at zero.

		Args:
			dvl_msg (DVL): the message from the DVL
		"""
		# get the most recent depth measurement
		depth_msg = self.depth_cache.getLast()
		if depth_msg is None:
			return

		# heading from the SLAM scan matcher (0 until the first SLAM estimate), level attitude
		rot = gtsam.Rot3.Yaw(self.slam_yaw)

		# parse the DVL message into an array of velocities
		vel = np.array([dvl_msg.velocity.x, dvl_msg.velocity.y, dvl_msg.velocity.z])

		# package the odom message and publish it
		self.send_odometry(vel, rot, dvl_msg.header.stamp, depth_msg.depth)


	def send_odometry(self,vel:np.array,rot:gtsam.Rot3,dvl_time,depth:float)->None:
		"""Package the odometry given all the DVL, rotation matrix, and depth

		Args:
			vel (np.array): a numpy array (1D) of the DVL velocities
			rot (gtsam.Rot3): the rotation matrix of the vehicle
			dvl_time: the time stamp for the DVL message (builtin_interfaces/Time)
			depth (float): vehicle depth
		"""

		#if the DVL message has any velocity above the max threhold do some error handling
		if np.any(np.abs(vel) > self.dvl_max_velocity):
			if self.pose:

				self.dvl_error_timer += (to_sec(dvl_time) - to_sec(self.prev_time))
				if self.dvl_error_timer > 5.0:
					logwarn(
						"DVL velocity ({:.1f}, {:.1f}, {:.1f}) exceeds max velocity {:.1f} for {:.1f} secs.".format(
							vel[0],
							vel[1],
							vel[2],
							self.dvl_max_velocity,
							self.dvl_error_timer,
						)
					)
				vel = self.prev_vel
			else:
				return
		else:
			self.dvl_error_timer = 0.0

		if self.pose:
			# figure out how far we moved in the body frame using the DVL message
			dt = to_sec(dvl_time) - to_sec(self.prev_time)
			dv = (vel + self.prev_vel) * 0.5
			trans = dv * dt

			# get a rotation matrix with only roll and pitch
			rotation_flat = gtsam.Rot3.Ypr(0, rot.pitch(), rot.roll())

			# transform our movement to the global frame
			#trans[2] = -trans[2]
			#trans = trans.dot(rotation_flat.matrix())

			# propagate our movement forward using the GTSAM utilities
			local_point = gtsam.Point2(trans[0], trans[1])

			pose2 = gtsam.Pose2(
				self.pose.x(), self.pose.y(), self.pose.rotation().yaw()
			)
			point = pose2.transformFrom(local_point)

			self.pose = gtsam.Pose3(
				rot, gtsam.Point3(point[0], point[1], depth)
			)

		else:
			# init the pose
			self.pose = gtsam.Pose3(rot, gtsam.Point3(0, 0, depth))

		# log the this timesteps messages for next time
		self.prev_time = dvl_time
		self.prev_vel = vel

		new_keyframe = False
		if not self.keyframes:
			new_keyframe = True
		else:
			duration = to_sec(self.prev_time) - self.keyframes[-1][0]
			if duration > self.keyframe_duration:
				odom = self.keyframes[-1][1].between(self.pose)
				odom = g2n(odom)
				translation = np.linalg.norm(odom[:3])
				rotation = abs(odom[-1])

				if (
					translation > self.keyframe_translation
					or rotation > self.keyframe_rotation
				):
					new_keyframe = True

		if new_keyframe:
			self.keyframes.append((to_sec(self.prev_time), self.pose))
		self.publish_pose(new_keyframe)


	def publish_pose(self, publish_traj:bool=False)->None:
		"""Publish the pose

		Args:
			publish_traj (bool, optional): Are we publishing the whole set of keyframes?. Defaults to False.

		"""
		if self.pose is None:
			return

		header = Header()
		header.stamp = self.prev_time
		header.frame_id = "odom"

		odom_msg = Odometry()
		odom_msg.header = header
		# pose in odom frame
		odom_msg.pose.pose = g2r(self.pose)
		# twist in local frame
		odom_msg.child_frame_id = "base_link"
		# Local planer behaves worse
		# odom_msg.twist.twist.linear.x = self.prev_vel[0]
		# odom_msg.twist.twist.linear.y = self.prev_vel[1]
		# odom_msg.twist.twist.linear.z = self.prev_vel[2]
		# odom_msg.twist.twist.angular.x = self.prev_omega[0]
		# odom_msg.twist.twist.angular.y = self.prev_omega[1]
		# odom_msg.twist.twist.angular.z = self.prev_omega[2]
		odom_msg.twist.twist.linear.x = 0.0
		odom_msg.twist.twist.linear.y = 0.0
		odom_msg.twist.twist.linear.z = 0.0
		odom_msg.twist.twist.angular.x = 0.0
		odom_msg.twist.twist.angular.y = 0.0
		odom_msg.twist.twist.angular.z = 0.0
		self.odom_pub.publish(odom_msg)

		p = odom_msg.pose.pose.position
		q = odom_msg.pose.pose.orientation
		self.tf.sendTransform(
			make_transform((p.x, p.y, p.z), (q.x, q.y, q.z, q.w), header.stamp, "odom", "base_link")
		)
		if publish_traj:
			traj = np.array([g2n(pose) for _, pose in self.keyframes])
			traj_msg = ros_colorline_trajectory(traj)
			traj_msg.header = header
			self.traj_pub.publish(traj_msg)
