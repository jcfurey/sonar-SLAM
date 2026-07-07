cmake_minimum_required(VERSION 3.8)
project(bruce_slam)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-O3)
endif()

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclpy REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
# pybind11_vendor makes a suitable pybind11 available on the ament prefix path
find_package(pybind11_vendor QUIET)
find_package(pybind11 REQUIRED)
find_package(PCL REQUIRED)
find_package(libpointmatcher REQUIRED)

# Install the pure-Python package located under src/bruce_slam so that
# `import bruce_slam` works after building.
ament_python_install_package(${PROJECT_NAME} PACKAGE_DIR src/${PROJECT_NAME})

# ---------------------------------------------------------------------------
# pybind11 C++ extension modules (pcl, cfar)
# ---------------------------------------------------------------------------
pybind11_add_module(pcl src/${PROJECT_NAME}/cpp/pcl.cpp)
target_include_directories(pcl PRIVATE
  ${PCL_INCLUDE_DIRS}
  ${libpointmatcher_INCLUDE_DIRS}
)
target_link_libraries(pcl PRIVATE
  ${PCL_LIBRARIES}
  ${libpointmatcher_LIBRARIES}
)

pybind11_add_module(cfar src/${PROJECT_NAME}/cpp/cfar.cpp)

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

# ---------------------------------------------------------------------------
# Resources
# ---------------------------------------------------------------------------
install(DIRECTORY launch config rviz
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
