cmake_minimum_required(VERSION 3.8)
project(bruce_msgs)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/ISAM2Update.msg"
  # "msg/keyframe_image.msg"   # needs sonar_oculus, enable if required
  # "msg/posehistory.msg"
  "srv/QueryExplorationPath.srv"
  "srv/PredictSLAMUpdate.srv"
  "srv/GetOccupancyMap.srv"
  DEPENDENCIES std_msgs geometry_msgs nav_msgs sensor_msgs
)

# NOTE: the C++ ISAM2 <-> ROS conversion helpers (include/bruce_msgs/conversions.h,
# src/bruce_msgs/conversions.cpp) depend on GTSAM and were disabled upstream. They
# are left in the tree for reference and are not built here.

ament_package()
