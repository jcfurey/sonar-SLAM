cmake_minimum_required(VERSION 3.8...3.28)
project(bruce_slam)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-O3)
endif()

find_package(ament_cmake_auto REQUIRED)
# auto-find all ament/ROS build dependencies declared in package.xml
ament_auto_find_build_dependencies()

# non-ament (system) dependencies: their rosdep keys (python3-dev, libpcl-all-dev)
# are not CMake package names, so ament_auto quietly skips them -> find explicitly.
find_package(ament_cmake_python REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED)
find_package(libpointmatcher REQUIRED)

# Install the pure-Python package located under src/bruce_slam.
ament_python_install_package(${PROJECT_NAME} PACKAGE_DIR src/${PROJECT_NAME})

# ---------------------------------------------------------------------------
# pybind11 C++ extension modules (pcl, cfar)
# ---------------------------------------------------------------------------
pybind11_add_module(pcl src/${PROJECT_NAME}/cpp/pcl.cpp)
target_include_directories(pcl PRIVATE
  ${PCL_INCLUDE_DIRS}
  ${libpointmatcher_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
)
target_link_libraries(pcl PRIVATE
  ${PCL_LIBRARIES}
  ${libpointmatcher_LIBRARIES}
)

# cfar.cpp only needs Eigen (pulled in globally in the old catkin build)
pybind11_add_module(cfar src/${PROJECT_NAME}/cpp/cfar.cpp)
target_include_directories(cfar PRIVATE
  ${EIGEN3_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
)

# Install the compiled extension modules next to the Python package so that
# `from bruce_slam import pcl` / `from bruce_slam import cfar` resolve.
install(TARGETS pcl cfar
  DESTINATION "${PYTHON_INSTALL_DIR}/${PROJECT_NAME}"
)

# ---------------------------------------------------------------------------
# Node executables (run with `ros2 run bruce_slam <node>`)
# ---------------------------------------------------------------------------
install(PROGRAMS
  scripts/slam_node.py
  scripts/feature_extraction_node.py
  scripts/dead_reckoning_node.py
  scripts/gyro_node.py
  scripts/kalman_node.py
  scripts/mapping_node.py
  DESTINATION lib/${PROJECT_NAME}
)

# Installs launch/config/rviz to share/${PROJECT_NAME} and calls ament_package().
ament_auto_package(INSTALL_TO_SHARE launch config rviz)
