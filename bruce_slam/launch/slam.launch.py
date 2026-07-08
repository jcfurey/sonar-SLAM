"""ROS 2 launch file for the BlueROV sonar SLAM system.

Online mode (default): starts the localization (or Kalman), gyro, feature
extraction and SLAM nodes, a static map->world transform and RViz2.

Offline mode: pass ``file:=/path/to/rosbag2_dir`` to replay a recorded bag
through a single slam node (best-effort port of the ROS 1 offline mode).
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import (
    LaunchConfiguration,
    PathJoinSubstitution,
    PythonExpression,
)
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    pkg = FindPackageShare("bruce_slam")

    def cfg(name):
        return PathJoinSubstitution([pkg, "config", name])

    icp_config = cfg("icp.yaml")

    # ------------------------------------------------------------------ args
    rviz = LaunchConfiguration("rviz")
    enable_slam = ParameterValue(LaunchConfiguration("enable_slam"), value_type=bool)
    kalman_dead_reckoning = LaunchConfiguration("kalman_dead_reckoning")
    file = LaunchConfiguration("file")
    start = LaunchConfiguration("start")
    duration = LaunchConfiguration("duration")

    args = [
        DeclareLaunchArgument("rviz", default_value="true"),
        DeclareLaunchArgument("enable_slam", default_value="true"),
        DeclareLaunchArgument("kalman_dead_reckoning", default_value="false"),
        DeclareLaunchArgument("file", default_value=""),
        DeclareLaunchArgument("start", default_value="0.0"),
        DeclareLaunchArgument("duration", default_value="-1"),
    ]

    # online when 'file' is empty, offline otherwise
    online = IfCondition(PythonExpression(["'", file, "' == ''"]))
    offline = IfCondition(PythonExpression(["'", file, "' != ''"]))
    use_kalman = IfCondition(kalman_dead_reckoning)
    no_kalman = UnlessCondition(kalman_dead_reckoning)

    ns = "bruce/slam"

    # --------------------------------------------------------------- online
    online_group = GroupAction(
        condition=online,
        actions=[
            # start the gyro integration node (unless the Kalman filter is used)
            Node(
                condition=no_kalman,
                package="bruce_slam", executable="gyro_node.py",
                name="gyro_fusion", namespace=ns, output="screen",
                parameters=[cfg("gyro.yaml")],
            ),
            # start the dead reckoning node (unless the Kalman filter is used)
            Node(
                condition=no_kalman,
                package="bruce_slam", executable="dead_reckoning_node.py",
                name="localization", namespace=ns, output="screen",
                parameters=[cfg("dead_reckoning.yaml")],
            ),
            # start the Kalman filter if requested
            Node(
                condition=use_kalman,
                package="bruce_slam", executable="kalman_node.py",
                name="kalman", namespace=ns, output="screen",
                parameters=[cfg("kalman.yaml")],
            ),
            # feature extraction
            Node(
                package="bruce_slam", executable="feature_extraction_node.py",
                name="feature_extraction", namespace=ns, output="screen",
                parameters=[cfg("feature.yaml")],
            ),
            # SLAM back-end
            Node(
                package="bruce_slam", executable="slam_node.py",
                name="slam", namespace=ns, output="screen",
                parameters=[
                    cfg("slam.yaml"),
                    {"enable_slam": enable_slam, "icp_config": icp_config},
                ],
            ),
            # map -> world so everything is visualized in a z-down frame in rviz
            Node(
                package="tf2_ros", executable="static_transform_publisher",
                name="map_to_world_tf_publisher",
                arguments=[
                    "--x", "0", "--y", "0", "--z", "0",
                    "--yaw", "0", "--pitch", "0", "--roll", "3.14159",
                    "--frame-id", "world", "--child-frame-id", "map",
                ],
            ),
        ],
    )

    # -------------------------------------------------------------- offline
    # Only slam.yaml is passed here: the offline process constructs the
    # localization/feature/gyro sub-nodes itself, each loading its own config
    # file with use_global_arguments disabled (per-node parameter isolation —
    # passing all YAMLs here would merge their /**-rooted keys onto every node).
    slam_offline = Node(
        condition=offline,
        package="bruce_slam", executable="slam_node.py",
        name="slam", namespace=ns, output="screen",
        arguments=["--file", file, "--start", start, "--duration", duration],
        parameters=[
            cfg("slam.yaml"),
            {"enable_slam": enable_slam, "icp_config": icp_config},
        ],
    )

    # ----------------------------------------------------------------- rviz
    # In offline mode the bag pump publishes /clock, so rviz runs on sim time
    # (otherwise tf lookups against wall-clock 'now' would find nothing).
    use_sim_time = ParameterValue(
        PythonExpression(["'", file, "' != ''"]), value_type=bool
    )
    rviz_node = Node(
        condition=IfCondition(rviz),
        package="rviz2", executable="rviz2", name="rviz",
        arguments=["-d", PathJoinSubstitution([pkg, "rviz", "video.rviz"])],
        parameters=[{"use_sim_time": use_sim_time}],
    )

    return LaunchDescription(args + [online_group, slam_offline, rviz_node])
