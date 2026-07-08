cmake_minimum_required(VERSION 3.8...3.28)
project(kvh_gyro)

find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/Gyro.msg"
  DEPENDENCIES std_msgs
)

ament_auto_package()
