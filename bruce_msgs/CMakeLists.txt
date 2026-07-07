cmake_minimum_required(VERSION 3.8)
project(bruce_msgs)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()

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
# are left in the tree for reference and are not built here. USE_SCOPED_HEADER_INSTALL_DIR
# installs the existing include/ under include/${PROJECT_NAME} (avoids include pollution).
ament_auto_package(USE_SCOPED_HEADER_INSTALL_DIR)
