cmake_minimum_required(VERSION 3.8)
project(example_robot_control)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(robot_state_publisher REQUIRED)
find_package(joint_state_publisher REQUIRED)
find_package(physical_ai_interfaces REQUIRED)

# Include directories
include_directories(include)

# Create executable
add_executable(example_robot_node src/example_robot_node.cpp)
ament_target_dependencies(example_robot_node 
  rclcpp 
  std_msgs 
  geometry_msgs 
  sensor_msgs 
  nav_msgs 
  tf2 
  tf2_ros
  physical_ai_interfaces
)

# Install executables
install(TARGETS
  example_robot_node
  DESTINATION lib/${PROJECT_NAME}
)

ament_export_dependencies(ament_cmake)

ament_package()