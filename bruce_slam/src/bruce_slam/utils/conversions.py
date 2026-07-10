from typing import Any
import numpy as np
import gtsam
import cv2
import cv_bridge
import struct

from std_msgs.msg import Header
from sensor_msgs.msg import Image, PointCloud2, PointField
from geometry_msgs.msg import Pose, PoseStamped, Quaternion, TransformStamped
from sensor_msgs_py import point_cloud2 as pc2

# The Oculus driver messages are optional: deployments using only generic sonar
# drivers (see bruce_slam.sensors) need not have the sonar_oculus package built.
try:
    from sonar_oculus.msg import OculusPing, OculusPingUncompressed
    _OCULUS_PING_TYPES = (OculusPing, OculusPingUncompressed)
except ImportError:  # pragma: no cover - depends on installed driver packages
    _OCULUS_PING_TYPES = ()

from .topics import *


def to_sec(stamp) -> float:
    """Return floating point seconds from a ROS 2 time-like object.

    Handles builtin_interfaces/Time (``.sec`` / ``.nanosec``), rclpy ``Time`` and
    ``Duration`` (``.nanoseconds``) as well as plain numbers.

    Args:
        stamp: a time-like object

    Returns:
        float: the time in seconds
    """

    if hasattr(stamp, "nanoseconds"):  # rclpy Time / Duration
        return stamp.nanoseconds * 1e-9
    if hasattr(stamp, "sec") and hasattr(stamp, "nanosec"):  # builtin_interfaces/Time
        return stamp.sec + stamp.nanosec * 1e-9
    return float(stamp)


def make_transform(translation, quaternion, stamp, parent_frame: str, child_frame: str) -> TransformStamped:
    """Build a geometry_msgs/TransformStamped for a tf2 broadcaster.

    Args:
        translation: (x, y, z) translation
        quaternion: (x, y, z, w) rotation
        stamp: the message time stamp (builtin_interfaces/Time)
        parent_frame (str): the parent (reference) frame id
        child_frame (str): the child frame id

    Returns:
        TransformStamped: the populated transform message
    """

    t = TransformStamped()
    t.header.stamp = stamp
    t.header.frame_id = parent_frame
    t.child_frame_id = child_frame
    t.transform.translation.x = float(translation[0])
    t.transform.translation.y = float(translation[1])
    t.transform.translation.z = float(translation[2])
    t.transform.rotation.x = float(quaternion[0])
    t.transform.rotation.y = float(quaternion[1])
    t.transform.rotation.z = float(quaternion[2])
    t.transform.rotation.w = float(quaternion[3])
    return t


def X(x:int) -> gtsam.symbol:
    """convert an integer to a gtsam symbol

    Args:
        x (int): the index of the symbol

    Returns:
        gtsam.symbol: gtsam symbol x_n
    """

    return gtsam.symbol("x", x)

def pose322(pose:gtsam.Pose3) -> gtsam.Pose2:
    """Convert a gtsam.Pose3 to a gtsam.Pose2

    Args:
        pose (gtsam.Pose3): the input 3D pose

    Returns:
        gtsam.Pose2: the 2D pose
    """

    return gtsam.Pose2(pose.x(), pose.y(), pose.rotation().yaw())


def pose223(pose:gtsam.Pose2) -> gtsam.Pose3:
    """convert a gtsam.Pose2 to a gtsam.Pose3

    Args:
        pose (gtsam.Pose2): the input 2D pose

    Returns:
        gtsam.Pose3: the 3D pose with zeros for the unkown values
    """

    return gtsam.Pose3(
        gtsam.Rot3.Yaw(pose.theta()), gtsam.Point3(pose.x(), pose.y(), 0)
    )


def n2g(numpy_arr:np.array, obj:str) -> any:
    """convert a numpy array to gtsam

    Args:
        numpy_arr (np.array): the input numpy array
        obj (str): the output object

    Raises:
        NotImplementedError: _description_

    Returns:
        any: the desired gtsam object
    """

    if obj == "Quaternion":
        x, y, z, w = numpy_arr
        return gtsam.Rot3.Quaternion(w, x, y, z)
    elif obj == "Euler":
        roll, pitch, yaw = numpy_arr
        return gtsam.Rot3.Ypr(yaw, pitch, roll)
    elif obj == "Point2":
        x, y = numpy_arr
        return gtsam.Point2(x, y)
    elif obj == "Pose2":
        x, y, yaw = numpy_arr
        return gtsam.Pose2(x, y, yaw)
    elif obj == "Point3":
        x, y, z = numpy_arr
        return gtsam.Point3(x, y, z)
    elif obj == "Pose3":
        x, y, z, roll, pitch, yaw = numpy_arr
        return gtsam.Pose3(gtsam.Rot3.Ypr(yaw, pitch, roll), gtsam.Point3(x, y, z))
    elif obj == "imuBiasConstantBias":
        imu_bias = numpy_arr
        return gtsam.imuBias_ConstantBias(
            np.array(imu_bias[:3]), np.array(imu_bias[3:])
        )
    elif obj == "Vector":
        return np.array(numpy_arr)
    else:
        raise NotImplementedError("Not implemented from numpy to " + obj)


def g2n(gtsam_obj:gtsam) -> np.array:
    """converts a gtsam object to a numpy array

    Args:
        gtsam_obj (gtsam): the input gtsam object, we can accept several types

    Raises:
        NotImplementedError: if we don't have a conversion for that object raise an exception

    Returns:
        np.array: the input data in numpy format
    """

    if isinstance(gtsam_obj, type(gtsam.Point2())) and gtsam_obj.shape == (2,):
        point = gtsam_obj
        return np.array([point[0], point[1]])
    elif isinstance(gtsam_obj, type(gtsam.Point3())) and gtsam_obj.shape == (3,):
        point = gtsam_obj
        return np.array([point[0], point[1], point[2]])
    elif isinstance(gtsam_obj, gtsam.Rot3):
        rot = gtsam_obj
        return np.array([rot.roll(), rot.pitch(), rot.yaw()])
    elif isinstance(gtsam_obj, gtsam.Pose2):
        pose = gtsam_obj
        return np.array([pose.x(), pose.y(), pose.theta()])
    elif isinstance(gtsam_obj, gtsam.Pose3):
        pose = gtsam_obj
        return np.array(
            [
                pose.x(),
                pose.y(),
                pose.z(),
                pose.rotation().roll(),
                pose.rotation().pitch(),
                pose.rotation().yaw(),
            ]
        )
    elif isinstance(gtsam_obj, gtsam.imuBias_ConstantBias):
        bias = gtsam_obj
        return np.r_[bias.accelerometer(), bias.gyroscope()]
    elif isinstance(gtsam_obj, np.ndarray):
        return gtsam_obj
    else:
        raise NotImplementedError(
            "Not implemented from {} to numpy".format(str(type(gtsam_obj)))
        )


def r2g(ros_msg) -> gtsam.Pose3:
    """convert a ros message to a 3D pose in gtsam

    Args:
        ros_msg (geometry_msgs.msg ): the input geometry message

    Raises:
        NotImplementedError: if unknown type raise exception

    Returns:
        gtsam.Pose3: the input data packaged as a gtsam 3D pose
    """

    if isinstance(ros_msg, Pose):
        x = ros_msg.position.x
        y = ros_msg.position.y
        z = ros_msg.position.z
        qx = ros_msg.orientation.x
        qy = ros_msg.orientation.y
        qz = ros_msg.orientation.z
        qw = ros_msg.orientation.w
        return gtsam.Pose3(
            n2g([qx, qy, qz, qw], "Quaternion"), n2g([x, y, z], "Point3")
        )
    elif isinstance(ros_msg, PoseStamped):
        return r2g(ros_msg.pose)
    elif isinstance(ros_msg, Quaternion):
        return n2g([ros_msg.x, ros_msg.y, ros_msg.z, ros_msg.w], "Quaternion")
    else:
        raise NotImplementedError(
            "Not implemented from {} to gtsam".format(str(type(ros_msg)))
        )


def g2r(gtsam_obj:gtsam.Pose3) -> Pose:
    """convert a gtsam.Pose3 to a ros pose message

    Args:
        gtsam_obj (gtsam.Pose3): the input pose

    Raises:
        NotImplementedError: if not a gtsam.Pose3 raise an execption

    Returns:
        Pose: the poise message in ros
    """

    if isinstance(gtsam_obj, gtsam.Pose3):
        pose = gtsam_obj
        pose_msg = Pose()
        pose_msg.position.x = pose.x()
        pose_msg.position.y = pose.y()
        pose_msg.position.z = pose.z()
        # gtsam >= 4.2 removed Rot3.quaternion() (returned [w,x,y,z]);
        # toQuaternion() returns an Eigen quaternion with accessor methods
        q = pose.rotation().toQuaternion()
        pose_msg.orientation.x = q.x()
        pose_msg.orientation.y = q.y()
        pose_msg.orientation.z = q.z()
        pose_msg.orientation.w = q.w()
        return pose_msg
    else:
        raise NotImplementedError(
            "Not implemented from {} to ros".format(str(type(gtsam_obj)))
        )


bridge = cv_bridge.CvBridge()


def apply_gamma(img: np.ndarray, gamma_byte: float) -> np.ndarray:
    """Undo the Oculus gamma correction on a sonar image.

    Args:
        img (np.ndarray): the raw image (0-255)
        gamma_byte (float): the raw gamma byte from the fire message; per the
            Oculus spec 0 (and 0xff) mean "gamma correction = 1.0", so 0 is
            treated as 255 rather than dividing by zero.

    Returns:
        np.ndarray: the gamma-corrected image (float, 0-255)
    """

    gamma = gamma_byte if gamma_byte else 255.0
    return np.clip(cv2.pow(img / 255.0, 255.0 / gamma) * 255.0, 0, 255)


def r2n(ros_msg) -> np.array:
    """Convert a ros message of type OculusPing (or Image/PointCloud2) to a numpy array

    Args:
        ros_msg: the input sonar message

    Raises:
        NotImplementedError: catch the wrong types

    Returns:
        np.array: the image data in numpy array form
    """

    if _OCULUS_PING_TYPES and isinstance(ros_msg, _OCULUS_PING_TYPES):

        img = r2n(ros_msg.ping)
        img = apply_gamma(img, ros_msg.fire_msg.gamma)
        return np.float32(img)
    elif isinstance(ros_msg, Image):
        img = bridge.imgmsg_to_cv2(ros_msg, desired_encoding="passthrough")
        return np.array(img, "uint8")
    elif isinstance(ros_msg, PointCloud2):
        rows = ros_msg.width
        cols = sum(f.count for f in ros_msg.fields)
        field_names = [f.name for f in ros_msg.fields]
        points = pc2.read_points(ros_msg, field_names=field_names, skip_nans=False)
        if isinstance(points, np.ndarray) and points.dtype.names is not None:
            # vectorized: loop over the handful of fields, not every point
            arr = np.column_stack([points[name] for name in field_names])
        else:
            arr = np.array([[p[name] for name in field_names] for p in points])
        return arr.reshape(rows, cols)
    else:
        raise NotImplementedError(
            "Not implemented from {} to numpy".format(str(type(ros_msg)))
        )


def pointcloud2_to_xyz_array(msg: PointCloud2) -> np.ndarray:
    """Extract the (x, y, z) channels of a PointCloud2 as an Nx3 float array.

    Replaces ros_numpy.point_cloud2.pointcloud2_to_xyz_array (unavailable in ROS 2).
    NaN points are preserved (skip_nans=False) so downstream logic that flags empty
    frames continues to work.

    Args:
        msg (PointCloud2): the input point cloud

    Returns:
        np.ndarray: an Nx3 array of xyz coordinates
    """

    points = pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=False)
    if isinstance(points, np.ndarray) and points.dtype.names is not None:
        return np.column_stack((points["x"], points["y"], points["z"])).astype(np.float32)
    return np.array([[p[0], p[1], p[2]] for p in points], dtype=np.float32)


def build_rgb_cloud(arr:np.array) -> PointCloud2:
    """Convert an array of [xyz,rgb] to a ROS point cloud with colors

    Args:
        arr (np.array): the input array

    Returns:
        PointCloud2: a ROS point cloud 2
    """

    # define the point cloud fields and header
    header = Header()
    fields = [
        PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        PointField(name='rgb', offset=12, datatype=PointField.UINT32, count=1),
    ]

    # parse out and convert the RGB values to RGBA
    points = []
    for row in arr:
        r, g, b = int(row[2]), int(row[3]), int(row[4])
        rgb = struct.unpack('I', struct.pack('BBBB', b, g, r, 255))[0]
        points.append( [row[0], row[1], 0., rgb] )
    return pc2.create_cloud(header, fields, points)


def n2r(numpy_arr:np.array, msg:any) -> any:
    """Package a nump array as the target ros message type in msg

    Args:
        numpy_arr (np.array): the data to be entered into a message
        msg (any): the target message type

    Raises:
        NotImplementedError: catch an unkown type

    Returns:
        any: the output message
    """

    if msg == "Image":
        if numpy_arr.ndim == 2 or numpy_arr.shape[2] == 1:
            return bridge.cv2_to_imgmsg(numpy_arr, encoding="mono8")
        else:
            return bridge.cv2_to_imgmsg(numpy_arr, encoding="rgb8")
    elif msg == "ImageBGR":
        return bridge.cv2_to_imgmsg(numpy_arr, encoding="bgr8")
    elif msg == "PointCloudXYZ":
        header = Header()
        return pc2.create_cloud_xyz32(header, np.array(numpy_arr))
    elif msg == "PointCloudXYZI":
        header = Header()
        fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name="i", offset=12, datatype=PointField.FLOAT32, count=1),
        ]
        return pc2.create_cloud(header, fields, np.array(numpy_arr))
    elif msg == "PointCloudXYZRGB":
        return build_rgb_cloud(numpy_arr)
    else:
        raise NotImplementedError("Not implemented from numpy array to {}".format(msg))
