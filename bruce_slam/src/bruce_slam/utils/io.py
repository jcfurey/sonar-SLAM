import timeit
from functools import wraps
import numpy as np
from tqdm.auto import tqdm
from threading import Event

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSHistoryPolicy, QoSReliabilityPolicy

from .conversions import to_sec

offline = False
callback_lock_event = Event()
callback_lock_event.set()

# module level logger, usable before/without a node being constructed
_LOGGER = rclpy.logging.get_logger("bruce_slam")


def latched_qos(depth: int = 1) -> QoSProfile:
    """QoS profile approximating a ROS 1 "latched" publisher.

    Late-joining subscribers receive the last published message.

    Args:
        depth (int): the history depth. Defaults to 1.

    Returns:
        QoSProfile: a transient-local, reliable, keep-last profile
    """

    return QoSProfile(
        depth=depth,
        history=QoSHistoryPolicy.KEEP_LAST,
        reliability=QoSReliabilityPolicy.RELIABLE,
        durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
    )


class BruceNode(Node):
    """Common base class for all bruce_slam ROS 2 nodes.

    Parameters supplied through launch files / YAML overrides are declared
    automatically, and :meth:`get_param` mimics the old ``rospy.get_param``
    behaviour (including translating ROS 1 style ``a/b`` names to ROS 2 ``a.b``).
    """

    def __init__(self, node_name: str, **kwargs):
        super().__init__(
            node_name,
            automatically_declare_parameters_from_overrides=True,
            **kwargs,
        )

    # sentinel distinguishing "no default supplied" from an explicit None
    _REQUIRED = object()

    def get_param(self, name: str, default=_REQUIRED):
        """Fetch a parameter value, declaring it if necessary.

        Args:
            name (str): the parameter name (``~foo`` and ``a/b`` are accepted)
            default: value to declare/return when the parameter is not provided.
                Omit it to make the parameter required.

        Raises:
            KeyError: if the parameter is required (no default) and not set —
                mirroring ``rospy.get_param``'s behaviour. (Declaring a missing
                parameter with a ``None`` default would make rclpy raise an
                opaque type-inference error instead.)

        Returns:
            the parameter value
        """

        name = name.lstrip("~").replace("/", ".")
        if not self.has_parameter(name):
            if default is BruceNode._REQUIRED or default is None:
                raise KeyError(
                    "Required parameter '{}' is not set for node '{}'. "
                    "Check the YAML/launch configuration.".format(name, self.get_name())
                )
            self.declare_parameter(name, default)
        return self.get_parameter(name).value


def add_lock(callback):
    """
    Lock decorator for callback functions, which is
    very helpful for running ROS offline with bag files.
    The lock forces callback functions sequentially,
    so we can show matplotlib plot, etc.

    """

    @wraps(callback)
    def lock_callback(*args, **kwargs):
        if not offline:
            callback(*args, **kwargs)
        else:
            callback_lock_event.wait()
            callback_lock_event.clear()
            callback(*args, **kwargs)
            callback_lock_event.set()

    return lock_callback


class LOGCOLORS:
    DK_GRAY = "\033[30m"
    DK_RED = "\033[31m"
    DK_GREEN = "\033[32m"
    DK_YELLOW = "\033[33m"
    DK_BLUE = "\033[34m"
    DK_PURPLE = "\033[35m"
    DK_CYAN = "\033[36m"
    DK_WHITE = "\033[37m"

    DK_BG_GRAY = "\033[40m"
    DK_BG_RED = "\033[41m"
    DK_BG_GREEN = "\033[42m"
    DK_BG_YELLOW = "\033[43m"
    DK_BG_BLUE = "\033[44m"
    DK_BG_PURPLE = "\033[45m"
    DK_BG_CYAN = "\033[46m"
    DK_BG_WHITE = "\033[47m"

    GRAY = "\033[90m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    PURPLE = "\033[95m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"

    BG_GRAY = "\033[100m"
    BG_RED = "\033[101m"
    BG_GREEN = "\033[102m"
    BG_YELLOW = "\033[103m"
    BG_BLUE = "\033[104m"
    BG_PURPLE = "\033[105m"
    BG_CYAN = "\033[106m"
    BG_WHITE = "\033[107m"

    END = "\033[0m"


def colorlog(color, str):
    return color + str + LOGCOLORS.END


def loginfo(msg):
    if offline:
        tqdm.write(msg)
    else:
        _LOGGER.info(str(msg))


def logdebug(msg):
    if offline:
        tqdm.write(colorlog(LOGCOLORS.BLUE, msg))
    else:
        _LOGGER.debug(str(msg))


def logwarn(msg):
    if offline:
        tqdm.write(colorlog(LOGCOLORS.YELLOW, msg))
    else:
        _LOGGER.warn(str(msg))


def logerror(msg):
    if offline:
        tqdm.write(colorlog(LOGCOLORS.RED, msg))
    else:
        _LOGGER.error(str(msg))


def common_parser(description="node"):
    import argparse

    parser = argparse.ArgumentParser(description=description)

    parser.add_argument("--file", type=str, default="", help="ROS 2 bag (directory)")
    parser.add_argument(
        "--start",
        type=float,
        default=None,
        help="start the video from START seconds (default: 0.0)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="duration of the video from START (default: -1)",
    )

    return parser


def read_bag(file, start=None, duration=None, progress=True, topics=None):
    """Iterate over (topic, deserialized message) pairs in a ROS 2 bag.

    Uses ``rosbag2_py``; the storage plugin is auto-detected from the bag
    metadata (works for sqlite3 and mcap on Humble and newer). ``file`` is the
    bag *directory* (or a single storage file), not a ROS 1 ``.bag``.

    Args:
        file (str): path to the rosbag2 directory / storage file
        start (float): seconds into the bag to start (default 0)
        duration (float): seconds to read from ``start`` (default: to the end)
        progress (bool): show a tqdm progress bar
        topics (list): optional list of topics to restrict to

    Yields:
        (str, Any): topic name and deserialized message
    """

    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message

    reader = rosbag2_py.SequentialReader()
    storage_options = rosbag2_py.StorageOptions(uri=file, storage_id="")
    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format="cdr", output_serialization_format="cdr"
    )
    reader.open(storage_options, converter_options)

    type_map = {t.name: t.type for t in reader.get_all_topics_and_types()}
    if topics is not None:
        reader.set_filter(rosbag2_py.StorageFilter(topics=list(topics)))

    start = start if start is not None else 0
    start_ns = None
    end_ns = None
    pbar = None

    while reader.has_next():
        topic, data, t_ns = reader.read_next()

        if start_ns is None:
            start_ns = t_ns + int(start * 1e9)
            if duration is not None and 0 <= duration != float("inf"):
                end_ns = start_ns + int(duration * 1e9)
            if progress:
                total = int(duration) if (duration and duration > 0) else None
                pbar = tqdm(total=total, unit="s")

        if t_ns < start_ns:
            continue
        if end_ns is not None and t_ns > end_ns:
            break

        msg = deserialize_message(data, get_message(type_map[topic]))
        if pbar is not None:
            pbar.update(int((t_ns - start_ns) * 1e-9) - pbar.n)
        yield topic, msg

    if pbar is not None:
        pbar.close()
    del reader


def load_nav_data(file, start=0, duration=None, progress=True):
    import gtsam
    from .topics import IMU_TOPIC, DVL_TOPIC, DEPTH_TOPIC

    dvl, depth, imu = [], [], []
    # filter at the storage layer so large unused topics (sonar images)
    # are never deserialized
    nav_topics = [IMU_TOPIC, DVL_TOPIC, DEPTH_TOPIC]
    for topic, msg in read_bag(file, start, duration, progress, topics=nav_topics):
        t = to_sec(msg.header.stamp)
        if topic == DVL_TOPIC:
            dvl.append(
                (t, msg.velocity.x, msg.velocity.y, msg.velocity.z, msg.altitude)
            )
        elif topic == DEPTH_TOPIC:
            depth.append((t, msg.depth))
        elif topic == IMU_TOPIC:
            ax = msg.linear_acceleration.x
            ay = msg.linear_acceleration.y
            az = msg.linear_acceleration.z
            wx = msg.angular_velocity.x
            wy = msg.angular_velocity.y
            wz = msg.angular_velocity.z
            qx = msg.orientation.x
            qy = msg.orientation.y
            qz = msg.orientation.z
            qw = msg.orientation.w
            # IMU is -roll90
            y, p, r = (
                gtsam.Rot3.Quaternion(qw, qx, qy, qz)
                .compose(gtsam.Rot3.Roll(np.pi / 2.0))
                .ypr()
            )
            tt = msg.linear_acceleration_covariance[0]
            imu.append((t, ax, ay, az, wx, wy, wz, r, p, y, tt))

    dvl = np.array(dvl)
    depth = np.array(depth)
    imu = np.array(imu)
    t0 = [a[0, 0] for a in (dvl, depth, imu) if len(a)]
    if not t0:
        return None, None, None
    else:
        t0 = min(t0)

    if len(dvl):
        dvl[:, 0] -= t0
    if len(imu):
        imu[:, 0] -= t0
        imu[:, -1] -= imu[0, -1]
    if len(depth):
        depth[:, 0] -= t0
    return dvl, depth, imu


class CodeTimer(object):
    """Timer class used with `with` statement

    - Disable output by setting CodeTimer.silent = False
    - Change log_func to print/tqdm.write/rospy.loginfo/etc

    with CodeTimer("Some function"):
        some_func()

    """

    silent = False

    def __init__(self, name="Code block"):
        self.name = name

    def __enter__(self):
        """Start measuring at the start of indent"""
        if not CodeTimer.silent:
            self.start = timeit.default_timer()

    def __exit__(self, exc_type, exc_value, traceback):
        """
            Stop measuring at the end of indent. This will run even
            if the indented lines raise an exception.
        """
        if not CodeTimer.silent:
            self.took = timeit.default_timer() - self.start

            if not CodeTimer.silent:
                msg = "{} : {:.5f} s".format(self.name, float(self.took))
                logdebug(msg)
