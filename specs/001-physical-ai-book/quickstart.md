# Quick Start Guide: Physical AI & Humanoid Robotics

## Overview
This guide provides a quick introduction to getting started with the Physical AI & Humanoid Robotics book project. It covers the essential setup steps to begin working with the curriculum.

## Prerequisites

### Hardware Requirements
- **Digital Twin Workstation**: 
  - RTX 4070 Ti+ (12–24GB VRAM)
  - Intel i7 / AMD Ryzen 9 processor
  - 32–64GB RAM
  - Ubuntu 22.04 LTS
- **Edge AI Kit (Optional for real robot work)**:
  - NVIDIA Jetson Orin Nano/NX
  - Intel RealSense D435i depth camera
  - USB microphone array
  - IMU sensor
- **Robot Platforms (Choose One)**:
  - Budget: Unitree Go2
  - Mid: OP3 / Unitree G1 Mini
  - Premium: Unitree G1 Humanoid

### Software Requirements
- ROS 2 Humble Hawksbill
- Gazebo Harmonic
- Unity 2023.2+ (Personal or Pro)
- NVIDIA Isaac Sim
- Docker and Docker Compose
- Python 3.10+

## Initial Setup

### 1. Install ROS 2 Humble Hawksbill
```bash
# Add ROS 2 repository
sudo apt update && sudo apt install curl gnupg lsb-release
curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key | sudo gpg --dearmor -o /usr/share/keyrings/ros-archive-keyring.gpg

echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(source /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

sudo apt update
sudo apt install ros-humble-desktop
sudo apt install python3-rosdep python3-rosinstall python3-rosinstall-generator python3-wstool build-essential
```

### 2. Set up ROS 2 Environment
```bash
source /opt/ros/humble/setup.bash
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
```

### 3. Install Gazebo
```bash
sudo apt install gz-harmonic
```

### 4. Install Python Dependencies
```bash
pip3 install numpy pyyaml transforms3d openai whisper speechrecognition
```

## Basic ROS 2 Workspace Setup

Create a workspace for working with the book examples:

```bash
mkdir -p ~/physical_ai_ws/src
cd ~/physical_ai_ws
colcon build
source install/setup.bash
```

## Running Your First Simulation

### 1. Launch a Basic Robot Simulation
```bash
# From your workspace directory
cd ~/physical_ai_ws
source install/setup.bash
ros2 launch example_robot_bringup example_robot.launch.py
```

### 2. View Robot in RViz
```bash
# In a new terminal
source ~/physical_ai_ws/install/setup.bash
rviz2
```

### 3. Send Basic Commands
```bash
# In a new terminal
source ~/physical_ai_ws/install/setup.bash

# Publish a velocity command to make the robot move forward
ros2 topic pub /cmd_vel geometry_msgs/Twist "{linear: {x: 0.5, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.5}}"
```

## Understanding the Book Structure

The book is organized into 6 main chapters covering a 13-week curriculum:

1. **Physical AI Foundations (Weeks 1-2)**: Core concepts of embodied intelligence
2. **ROS 2 Fundamentals (Weeks 3-5)**: Nodes, topics, services, URDF
3. **Simulation Environments (Weeks 6-7)**: Gazebo and Unity integration
4. **NVIDIA Isaac Platform (Weeks 8-10)**: AI perception and navigation
5. **Humanoid Control (Weeks 11-12)**: Kinematics, locomotion, manipulation
6. **Conversational Robotics (Week 13)**: Voice commands and GPT integration

## Key ROS 2 Concepts Covered

### Nodes, Topics, and Services
- **Nodes**: Individual processes that perform computation
- **Topics**: Message-based communication between nodes (publish/subscribe)
- **Services**: Request/response communication pattern

### Example Node Structure
```
example_node/
├── CMakeLists.txt
├── package.xml
├── src/
│   └── example_node.cpp
└── launch/
    └── example_launch.py
```

### Common Message Types
- `sensor_msgs/` - Sensor data (images, point clouds, etc.)
- `geometry_msgs/` - Position, orientation, and velocity data
- `nav_msgs/` - Navigation-specific messages (paths, maps)
- `trajectory_msgs/` - Trajectory and joint command messages

## Common Commands

### Check Active Nodes
```bash
ros2 node list
```

### Check Active Topics
```bash
ros2 topic list
```

### Echo Topic Data
```bash
ros2 topic echo /topic_name message_type
```

### Run a Node
```bash
ros2 run package_name executable_name
```

### Launch File
```bash
ros2 launch package_name launch_file.py
```

## Troubleshooting

### Common Issues and Solutions

1. **"Command 'ros2' not found"**
   - Ensure ROS 2 environment is sourced: `source /opt/ros/humble/setup.bash`

2. **Simulation runs slowly**
   - Check GPU drivers: `nvidia-smi`
   - Ensure proper GPU acceleration is enabled

3. **"Unable to load plugin" error**
   - Verify all dependencies are installed
   - Check that ROS 2 environment is sourced

### Useful Debugging Commands
```bash
# Check ROS 2 environment
printenv | grep ROS

# Check network connectivity for ROS 2
ros2 doctor

# Get detailed node information
ros2 node info node_name
```

## Next Steps

1. Complete the exercises in Chapter 2: ROS 2 Fundamentals
2. Set up the simulation environment as described in Chapter 3
3. Explore the NVIDIA Isaac platform covered in Chapter 4
4. Work through the humanoid control examples in Chapter 5
5. Implement the capstone conversational robotics project in Chapter 6

For further assistance, refer to the detailed documentation in the `documentation/` directory and the troubleshooting guides specific to each chapter.