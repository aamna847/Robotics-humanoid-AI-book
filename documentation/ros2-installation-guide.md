# ROS 2 Humble Hawksbill Installation Script

This script outlines the installation steps for ROS 2 Humble Hawksbill on Ubuntu 22.04. This is for reference purposes, as actual installation requires administrative privileges and a Linux environment.

## Prerequisites
- Ubuntu 22.04 LTS
- Administrative privileges (sudo access)
- Internet connection

## Installation Steps

1. Set up locale:
```bash
locale  # check for UTF-8
sudo apt update && sudo apt install locales
sudo locale-gen en_US.UTF-8
sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
export LANG=en_US.UTF-8
```

2. Set up sources:
```bash
sudo apt update && sudo apt install curl gnupg lsb-release
curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key | sudo gpg --dearmor -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(source /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
```

3. Install ROS 2:
```bash
sudo apt update
sudo apt install ros-humble-desktop
sudo apt install ros-humble-cv-bridge ros-humble-tf2-tools ros-humble-tf2-eigen ros-humble-tf2-geometry-msgs ros-humble-tf2-sensor-msgs ros-humble-tf2-tools ros-humble-tf-transformations
```

4. Install colcon build tools:
```bash
sudo apt install python3-colcon-common-extensions
sudo apt install python3-rosdep python3-rosinstall python3-rosinstall-generator
```

5. Initialize rosdep:
```bash
sudo rosdep init
rosdep update
```

6. Add sourcing to bashrc:
```bash
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
```

## Verification

After installation, verify ROS 2 is working:
```bash
source /opt/ros/humble/setup.bash
ros2 topic list  # Should run without errors
```

## Troubleshooting

If you encounter issues:
- Make sure your system is fully updated: `sudo apt update && sudo apt upgrade`
- Check that your locale is properly set to UTF-8
- Verify that your Ubuntu version matches the ROS 2 distribution
- Check that the ROS 2 repository is properly added to your sources list

Note: This installation requires a Linux system with Ubuntu 22.04. For other platforms, see the official ROS 2 installation guide.