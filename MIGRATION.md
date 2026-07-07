# ROS 1 → ROS 2 migration notes

This document summarizes the migration of `sonar-SLAM` from ROS 1 (Noetic / catkin)
to ROS 2 (Humble / Jazzy / Rolling), and the caveats to verify on a real ROS 2 machine.

> This conversion was authored without a ROS 2 environment available to build against,
> so it is a source-level port verified by review and Python syntax checks, not by
> `colcon build`. Please build and smoke-test on a ROS 2 machine and report issues.

## Build system

- **catkin → ament / colcon** for every package, using **`ament_cmake_auto`**:
  each `CMakeLists.txt` calls `ament_auto_find_build_dependencies()` (which
  `find_package`s all ROS deps declared in `package.xml`) and `ament_auto_package()`.
  Non-CMake system deps whose rosdep keys are not CMake package names
  (`python3-dev`, `libpcl-all-dev`) are quietly skipped by ament_auto and found
  explicitly (`Python3`, `PCL`).
- `bruce`: catkin metapackage → `ament_cmake` metapackage (`ament_auto_package()`).
- `bruce_msgs`: `message_generation` → `rosidl_generate_interfaces`; `Header` fields are
  now `std_msgs/Header`. Only `ISAM2Update` plus the three services are generated (the
  GTSAM-dependent C++ conversion helpers remain disabled, as upstream).
- `bruce_slam`: mixed C++/Python package. `ament_auto_find_build_dependencies()`
  handles the ROS deps; `ament_cmake_python` / `Python3` / `pybind11` / `PCL` /
  `libpointmatcher` are found explicitly. The `pcl` and `cfar` pybind11 modules are
  installed next to the Python package so `from bruce_slam import pcl` / `cfar`
  resolve, node scripts install to `lib/bruce_slam` (run with
  `ros2 run bruce_slam <node>.py`), and `ament_auto_package(INSTALL_TO_SHARE launch
  config rviz)` installs the resources. The catkin `setup.py` was removed.

## Vendored driver message stubs

`sonar_oculus`, `rti_dvl`, `bar30_depth`, and `kvh_gyro` message packages are provided
in-repo as `rosidl` packages so the workspace builds standalone. **Their fields were
reconstructed from how `bruce_slam` uses them**, not copied from the upstream Argonaut
definitions — if you record/replay real hardware bags, prefer the upstream driver
packages so the message types (and therefore the serialized bag data) match exactly.

## Python (`rclpy`)

- `rospy` → `rclpy`. Every node class now inherits from a small `BruceNode(rclpy.node.Node)`
  base (in `utils/io.py`) that auto-declares parameters from launch/YAML overrides and
  provides a `get_param()` helper (translating ROS 1 `~a/b` names to ROS 2 `a.b`).
- `rospy.get_param` → `self.get_param`; `rospy.Publisher/Subscriber` →
  `create_publisher/create_subscription`; latched publishers → transient-local QoS
  (`latched_qos()`).
- `tf`/`tf.TransformBroadcaster` → `tf2_ros.TransformBroadcaster` sending
  `TransformStamped` (helper `make_transform`). `tf.transformations.euler_from_quaternion`
  → `scipy` in `kalman.py`.
- `ros_numpy` → `cv_bridge` (images) and `sensor_msgs_py.point_cloud2` +
  a local `pointcloud2_to_xyz_array` (clouds).
- `sensor_msgs.point_cloud2` → `sensor_msgs_py.point_cloud2`; `PointField` now comes
  from `sensor_msgs.msg` and all ROS message constructors use keyword arguments
  (ROS 2 rejects positional construction).
- `rosbag` → `rosbag2_py` in `read_bag` (bag storage auto-detected).
- Message timestamps: `builtin_interfaces/Time` no longer supports subtraction /
  `.to_sec()`, so a `to_sec()` helper is used throughout (`slam.py`, `dead_reckoning.py`, …).
- ROS 2 message fields are strictly typed: integer literals assigned to float fields
  were changed to floats, and `OccupancyGrid.data` uses `.tolist()`.

## Parameters and config

ROS 2 parameter files use the `/**: ros__parameters:` form. Notable transforms:

- The ROS 1 `deg(...)` rosparam helper is unsupported → angles pre-converted to radians.
- `$(find bruce_slam)/config/icp.yaml` substitution is unsupported → the launch file
  passes an absolute path, and the SLAM node falls back to the installed package share
  copy if `icp_config` is empty/missing.
- **ROS 2 parameters cannot hold 2-D arrays** → all Kalman matrices in `config/kalman.yaml`
  are stored flattened (row-major) and reshaped in `kalman.py`.

## Launch / rviz

- `slam.launch` (XML) → `slam.launch.py`. Same args (`rviz`, `enable_slam`,
  `kalman_dead_reckoning`, `file`, `start`, `duration`). `enable_slam` is passed as a
  typed bool. The static `map`→`world` transform uses the ROS 2
  `static_transform_publisher` named-argument form.
- `rviz/video.rviz`: display/tool/view class names were remapped to
  `rviz_default_plugins/*` and panels to `rviz_common/*`. RViz 2's display schema
  differs from RViz 1, so some displays may need minor re-tuning (open, adjust, re-save).

## Known caveats / to verify on ROS 2

1. **Not built here** — verify `colcon build` (pybind11 module install path, PCL /
   libpointmatcher discovery).
2. **Driver message ABI** — vendored stubs are reconstructed; match the upstream Argonaut
   drivers for real data.
3. **Offline mode** is a best-effort port: it hosts all nodes in one process with a
   background executor and expects a `rosbag2` bag (convert legacy `.bag` first). Because
   the config files use the `/**` wildcard, a couple of parameters that differ between the
   SLAM and localization nodes (e.g. `keyframe_translation`) will take a single merged
   value in the single-process offline run. The online launch (separate processes) is
   unaffected.
4. **`sensor_msgs_py.point_cloud2.read_points`** return type has varied across ROS 2
   releases; `pointcloud2_to_xyz_array` handles both structured-array and iterable forms.
