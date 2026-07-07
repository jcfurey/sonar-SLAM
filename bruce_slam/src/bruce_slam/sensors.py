"""Pluggable sensor adapters.

`bruce_slam` decouples the SLAM/localization algorithms from any particular
hardware driver through a small adapter registry. Each sensor kind (``dvl``,
``depth``, ``gyro``, ``sonar``) has named driver adapters; a node selects one by
parameter, the adapter declares its ROS 2 message type (resolved at runtime, so
unused driver packages need not be installed), and normalizes each message into
a small reading object the algorithm consumes.

The IMU is intentionally not adapted: it already uses the standard
``sensor_msgs/Imu`` message, so only its topic is configurable.

Adding support for a new sensor is a matter of writing a small adapter class and
registering it in the appropriate ``*_ADAPTERS`` dict below.
"""

import numpy as np
import cv2
import cv_bridge
from rosidl_runtime_py.utilities import get_message


_bridge = cv_bridge.CvBridge()


# ---------------------------------------------------------------------------
# Normalized readings
# ---------------------------------------------------------------------------
class DvlReading(object):
    """Normalized DVL reading."""

    __slots__ = ("header", "velocity", "altitude")

    def __init__(self, header, velocity, altitude=0.0):
        self.header = header
        self.velocity = np.asarray(velocity, dtype=float)  # [vx, vy, vz] (m/s, body frame)
        self.altitude = float(altitude)                    # height above bottom (m)


class DepthReading(object):
    """Normalized depth reading."""

    __slots__ = ("header", "depth")

    def __init__(self, header, depth):
        self.header = header
        self.depth = float(depth)  # depth below surface (m)


class GyroReading(object):
    """Normalized gyro reading (integrated delta angles about x, y, z)."""

    __slots__ = ("header", "delta")

    def __init__(self, header, delta):
        self.header = header
        self.delta = np.asarray(delta, dtype=float)  # [dx, dy, dz] (rad)


class FireMsg(object):
    """Sonar acquisition metadata (mirrors the Oculus fire message)."""

    __slots__ = ("mode", "gamma", "flags", "range", "gain", "speed_of_sound", "salinity")

    def __init__(self, mode=1, gamma=255, flags=0, range=0.0, gain=0.0,
                 speed_of_sound=1500.0, salinity=0.0):
        self.mode = mode
        self.gamma = gamma  # raw gamma byte (0-255); OculusProperty divides by 255
        self.flags = flags
        self.range = range
        self.gain = gain
        self.speed_of_sound = speed_of_sound
        self.salinity = salinity


class SonarPing(object):
    """Normalized sonar ping.

    Exposes the same attribute names the pipeline historically read off the
    Oculus message (so ``sonar.py`` / ``mapping.py`` need minimal change) plus a
    pre-decoded grayscale ``image``. Bearings are always in RADIANS.
    """

    __slots__ = ("header", "image", "bearings", "range_resolution", "num_ranges",
                 "ping_id", "fire_msg", "part_number", "raw")

    def __init__(self, header, image, bearings, range_resolution, num_ranges,
                 ping_id=0, fire_msg=None, part_number=None, raw=None):
        self.header = header
        self.image = image                       # 2D uint8 polar image (rows=ranges, cols=beams)
        self.bearings = np.asarray(bearings, dtype=np.float32)  # radians
        self.range_resolution = range_resolution
        self.num_ranges = num_ranges
        self.ping_id = ping_id
        self.fire_msg = fire_msg if fire_msg is not None else FireMsg()
        self.part_number = part_number
        self.raw = raw


# ---------------------------------------------------------------------------
# Adapter base
# ---------------------------------------------------------------------------
class SensorAdapter(object):
    """Base class for sensor adapters.

    Subclasses set ``msg_type`` to a ROS 2 type string (``pkg/msg/Type``) and
    implement ``__call__(msg) -> reading``. ``__init__`` receives the owning node
    so adapters that need parameters (e.g. a plain image sonar) can read them.
    """

    msg_type = None

    def __init__(self, node):
        self.node = node

    def __call__(self, msg):
        raise NotImplementedError

    @classmethod
    def message_class(cls):
        return get_message(cls.msg_type)


# ---------------------------------------------------------------------------
# DVL adapters
# ---------------------------------------------------------------------------
class RtiDvlAdapter(SensorAdapter):
    msg_type = "rti_dvl/msg/DVL"

    def __call__(self, msg):
        return DvlReading(
            msg.header,
            [msg.velocity.x, msg.velocity.y, msg.velocity.z],
            getattr(msg, "altitude", 0.0),
        )


class TwistStampedDvlAdapter(SensorAdapter):
    msg_type = "geometry_msgs/msg/TwistStamped"

    def __call__(self, msg):
        v = msg.twist.linear
        return DvlReading(msg.header, [v.x, v.y, v.z], 0.0)


class TwistCovDvlAdapter(SensorAdapter):
    msg_type = "geometry_msgs/msg/TwistWithCovarianceStamped"

    def __call__(self, msg):
        v = msg.twist.twist.linear
        return DvlReading(msg.header, [v.x, v.y, v.z], 0.0)


DVL_ADAPTERS = {
    "rti_dvl": RtiDvlAdapter,
    "twist_stamped": TwistStampedDvlAdapter,
    "twist_cov": TwistCovDvlAdapter,
}


# ---------------------------------------------------------------------------
# Depth adapters
# ---------------------------------------------------------------------------
class Bar30DepthAdapter(SensorAdapter):
    msg_type = "bar30_depth/msg/Depth"

    def __call__(self, msg):
        return DepthReading(msg.header, msg.depth)


class FluidPressureDepthAdapter(SensorAdapter):
    """sensor_msgs/FluidPressure -> depth.

    Assumes gauge pressure in Pascals; depth = P / (rho * g). Water density and g
    are read from the ``depth/water_density`` and ``depth/gravity`` parameters
    (defaults: 1025 kg/m^3 saltwater, 9.80665 m/s^2).
    """

    msg_type = "sensor_msgs/msg/FluidPressure"

    def __init__(self, node):
        super().__init__(node)
        self.rho = float(node.get_param("depth/water_density", 1025.0))
        self.g = float(node.get_param("depth/gravity", 9.80665))

    def __call__(self, msg):
        return DepthReading(msg.header, msg.fluid_pressure / (self.rho * self.g))


DEPTH_ADAPTERS = {
    "bar30": Bar30DepthAdapter,
    "fluid_pressure": FluidPressureDepthAdapter,
}


# ---------------------------------------------------------------------------
# Gyro adapters (raw integrated delta angles)
# ---------------------------------------------------------------------------
class KvhGyroAdapter(SensorAdapter):
    msg_type = "kvh_gyro/msg/Gyro"

    def __call__(self, msg):
        return GyroReading(msg.header, list(msg.delta))


class Vector3StampedGyroAdapter(SensorAdapter):
    """geometry_msgs/Vector3Stamped carrying per-sample delta angles (rad)."""

    msg_type = "geometry_msgs/msg/Vector3Stamped"

    def __call__(self, msg):
        v = msg.vector
        return GyroReading(msg.header, [v.x, v.y, v.z])


GYRO_ADAPTERS = {
    "kvh_gyro": KvhGyroAdapter,
    "vector3_stamped": Vector3StampedGyroAdapter,
}


# ---------------------------------------------------------------------------
# Sonar adapters
# ---------------------------------------------------------------------------
def _oculus_fire(msg):
    fm = msg.fire_msg
    return FireMsg(
        mode=fm.mode, gamma=fm.gamma, flags=fm.flags, range=fm.range,
        gain=fm.gain, speed_of_sound=fm.speed_of_sound, salinity=fm.salinity,
    )


def _oculus_ping(msg, image):
    # Oculus bearings are int16 hundredths of a degree -> radians
    bearings = np.deg2rad(np.asarray(msg.bearings, dtype=np.float32) / 100.0)
    return SonarPing(
        header=msg.header,
        image=image,
        bearings=bearings,
        range_resolution=msg.range_resolution,
        num_ranges=msg.num_ranges,
        ping_id=msg.ping_id,
        fire_msg=_oculus_fire(msg),
        part_number=getattr(msg, "part_number", None),
        raw=msg,
    )


class OculusCompressedAdapter(SensorAdapter):
    """Blueprint Subsea Oculus ping with a JPEG-compressed image payload."""

    msg_type = "sonar_oculus/msg/OculusPing"

    def __call__(self, msg):
        buf = np.frombuffer(msg.ping.data, np.uint8)
        img = np.array(cv2.imdecode(buf, cv2.IMREAD_COLOR)).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return _oculus_ping(msg, img)


class OculusUncompressedAdapter(SensorAdapter):
    """Blueprint Subsea Oculus ping with a raw image payload."""

    msg_type = "sonar_oculus/msg/OculusPingUncompressed"

    def __call__(self, msg):
        img = np.array(
            _bridge.imgmsg_to_cv2(msg.ping, desired_encoding="passthrough"),
            dtype=np.uint8,
        )
        return _oculus_ping(msg, img)


class GenericImageSonarAdapter(SensorAdapter):
    """A plain grayscale polar sonar image (sensor_msgs/Image).

    A bare image carries no acoustic geometry, so ``sonar/range_resolution`` (m
    per range bin) and ``sonar/horizontal_fov`` (radians, default 130 deg) are
    read from parameters. Range bins map to image rows and beams to columns; the
    ping id is an internal counter since Image has no sequence field.
    """

    msg_type = "sensor_msgs/msg/Image"

    def __init__(self, node):
        super().__init__(node)
        self.range_resolution = float(node.get_param("sonar/range_resolution", 0.0))
        if self.range_resolution <= 0.0:
            raise ValueError(
                "sonar/range_resolution (m per range bin) must be set for the "
                "generic 'image' sonar driver")
        self.horizontal_fov = float(
            node.get_param("sonar/horizontal_fov", float(np.deg2rad(130.0))))
        self._count = 0

    def __call__(self, msg):
        img = np.array(
            _bridge.imgmsg_to_cv2(msg, desired_encoding="mono8"), dtype=np.uint8)
        num_ranges, num_beams = img.shape[0], img.shape[1]
        bearings = np.linspace(
            -self.horizontal_fov / 2.0, self.horizontal_fov / 2.0, num_beams,
            dtype=np.float32)
        self._count += 1
        return SonarPing(
            header=msg.header,
            image=img,
            bearings=bearings,
            range_resolution=self.range_resolution,
            num_ranges=num_ranges,
            ping_id=self._count,
            fire_msg=FireMsg(),
            part_number=None,
            raw=msg,
        )


SONAR_ADAPTERS = {
    "oculus_compressed": OculusCompressedAdapter,
    "oculus_uncompressed": OculusUncompressedAdapter,
    "image": GenericImageSonarAdapter,
}


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def make_adapter(registry, driver, node):
    """Instantiate the adapter for ``driver`` and resolve its message class.

    Args:
        registry (dict): one of DVL_ADAPTERS / DEPTH_ADAPTERS / GYRO_ADAPTERS / SONAR_ADAPTERS
        driver (str): the driver name (key into the registry)
        node: the owning rclpy node (for parameter access)

    Returns:
        (adapter_instance, message_class): the callable adapter and its ROS 2 message type
    """
    if driver not in registry:
        raise ValueError(
            "Unknown sensor driver '{}'. Available: {}".format(
                driver, ", ".join(sorted(registry))))
    adapter = registry[driver](node)
    return adapter, registry[driver].message_class()
