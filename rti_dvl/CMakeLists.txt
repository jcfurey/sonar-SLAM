cmake_minimum_required(VERSION 3.8)
project(rti_dvl)

find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/DVL.msg"
  DEPENDENCIES std_msgs geometry_msgs
)

ament_auto_package()
