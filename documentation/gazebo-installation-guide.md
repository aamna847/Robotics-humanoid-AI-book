# Gazebo Harmonic Installation Guide

This document outlines the installation steps for Gazebo Harmonic on Ubuntu 22.04. This is for reference purposes, as actual installation requires administrative privileges and a Linux environment.

## Prerequisites
- Ubuntu 22.04 LTS
- Administrative privileges (sudo access)
- Internet connection
- ROS 2 Humble Hawksbill (installed separately)

## Installation Steps

1. Set up the Gazebo repository:
```bash
sudo apt update
sudo apt install wget lsb-release gnupg
wget https://packages.osrfoundation.org/gazebo.gpg -O /tmp/gazebo.gpg
if [ ! -d /etc/apt/keyrings ]; then
  sudo mkdir /etc/apt/keyrings
fi
sudo cp /tmp/gazebo.gpg /etc/apt/keyrings/
rm /tmp/gazebo.gpg
echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/gazebo.gpg] http://packages.osrfoundation.org/gazebo/ubuntu-stable $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/gazebo-stable.list > /dev/null
```

2. Install Gazebo Harmonic:
```bash
sudo apt update
sudo apt install gazebo-harmonic
```

3. Install ROS 2 Gazebo bridge (for integration with ROS 2):
```bash
sudo apt install ros-humble-ros-gz
```

4. Verify installation:
```bash
gz sim --version
```

## Basic Functionality Verification

1. Launch Gazebo with an empty world:
```bash
gz sim -r -v 1 empty.sdf
```

2. Test basic functionality:
   - Verify the GUI opens correctly
   - Add a model from the Gazebo database
   - Verify physics simulation is working
   - Test camera controls and navigation

3. Create a simple test world to verify functionality:
```bash
# Create a basic SDF world file
echo '<?xml version="1.0" ?>
<sdf version="1.7">
  <world name="test_world">
    <include>
      <uri>model://sun</uri>
    </include>
    <include>
      <uri>model://ground_plane</uri>
    </include>
  </world>
</sdf>' > test_world.sdf

# Launch with the test world
gz sim -r test_world.sdf
```

## Integration with ROS 2

Verify Gazebo integration with ROS 2:
```bash
# Source ROS 2
source /opt/ros/humble/setup.bash

# Launch a ROS 2 enabled simulation
ros2 launch ros_gz_sim gazebo.launch.py
```

## Troubleshooting

- If you get graphics errors, check for proper GPU drivers (especially for RTX cards)
- If Gazebo fails to launch, check that all dependencies are installed
- If the GUI doesn't respond properly, check for X11 forwarding if running remotely
- For performance issues, verify adequate hardware resources (especially for RTX 4070 Ti+)

## Additional Resources

- [Gazebo Harmonic Documentation](https://gazebosim.org/docs/harmonic)
- [ROS 2 Gazebo Integration Guide](https://github.com/gazebosim/ros_gz)
- [Performance Tuning Guide](https://gazebosim.org/tutorials?tut=performance_tuning&cat=)