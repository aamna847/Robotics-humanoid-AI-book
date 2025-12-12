# Edge AI Kit (Physical AI) Setup Guide

This guide provides instructions for setting up the Edge AI Kit for the Physical AI & Humanoid Robotics curriculum, using the NVIDIA Jetson platform.

## Components List

### Required Components
- **Jetson Platform**: NVIDIA Jetson Orin Nano or Jetson Orin NX
- **Camera**: Intel RealSense D435i depth camera
- **Audio**: USB microphone array (e.g. ReSpeaker 4-Mic Array)
- **Inertial Sensor**: IMU (Integrated in some Jetson carriers or separate)
- **Power Supply**: Appropriate for your Jetson platform
- **Storage**: High-speed microSD card (64GB+ recommended)

### Optional Components
- **Enclosure**: To protect components
- **Additional Sensors**: Additional LiDAR, thermal, etc.
- **Communication**: WiFi/Bluetooth module if not built-in

## Prerequisites

- Host computer (x86_64) with Ubuntu 18.04 or later
- MicroSD card reader
- Internet connection for downloads
- USB-C cable for initial setup

## Installation Steps

### 1. Prepare the Jetson Platform

#### For Jetson Orin Nano/NX:
1. Download NVIDIA JetPack SDK from developer.nvidia.com
2. Use NVIDIA SDK Manager to flash the Jetson with:
   - Linux OS
   - CUDA
   - TensorRT
   - OpenCV
   - VPI
   - CUDA-X libraries

```bash
# Note: This is typically done using NVIDIA SDK Manager on a host computer
# The Jetson connects to the host via USB-C during flashing
```

### 2. Initial Setup

1. Insert microSD card into Jetson
2. Connect peripherals:
   - HDMI display
   - USB keyboard and mouse
   - Ethernet cable (recommended for initial setup)
3. Power on the Jetson
4. Complete the initial Ubuntu setup

### 3. Install Robot-Specific Software

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install ROS 2 Humble Hawksbill
sudo apt install software-properties-common
sudo add-apt-repository universe
sudo apt update && sudo apt install curl -y
curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
sudo apt update
sudo apt install ros-humble-ros-base
sudo apt install python3-rosdep python3-rosinstall python3-rosinstall-generator python3-wstool build-essential

# Initialize rosdep
sudo rosdep init
rosdep update
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
```

### 4. Install RealSense SDK

```bash
# Add Intel repository
sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-key F6E65AC044F831AC80A06380C8B3A54A55BA8C29
sudo add-apt-repository "deb https://librealsense.intel.com/Debian/apt-repo $(lsb_release -cs) main" -u
sudo apt-get install librealsense2-dkms
sudo apt-get install librealsense2-dev
```

### 5. Install Audio Processing Libraries

```bash
# Install audio libraries
sudo apt install pulseaudio alsa-utils portaudio19-dev python3-pyaudio

# Test audio input
arecord -l  # List recording devices
```

### 6. Install Python Dependencies

```bash
# Create a virtual environment for the project
python3 -m venv ~/ai_robot_env
source ~/ai_robot_env/bin/activate
pip install --upgrade pip

# Install required packages
pip install numpy pyyaml transforms3d openai openai-whisper SpeechRecognition
pip install opencv-python pyrealsense2
```

### 7. Configure RealSense Camera

```bash
# Test RealSense camera
rs-enumerate-devices  # Should list your RealSense device
rs-capture -d 0       # Test camera capture
```

## Performance Optimization

### 1. Power Mode Configuration
```bash
# Check available power modes
sudo jetson_clocks --show

# Set to maximum performance mode (requires external power)
sudo nvpmodel -m 0
sudo jetson_clocks
```

### 2. Memory Management
```bash
# Monitor memory usage
watch -n 1 free -h
```

### 3. Thermal Management
- Ensure adequate ventilation around the Jetson
- Consider adding heatsinks or fans for sustained high-performance operation
- Monitor temperature: `cat /sys/class/thermal/thermal_zone*/temp`

## Verification

### Test RealSense Camera
```bash
# Run a simple Python script to test camera
python3 -c "
import pyrealsense2 as rs
import numpy as np
import cv2

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)

try:
    for i in range(30):  # Capture 30 frames
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        # Stack both images horizontally
        images = np.hstack((color_image, depth_colormap))

        # Show images
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', images)
        cv2.waitKey(1)

finally:
    # Stop streaming
    pipeline.stop()
"
```

### Test Audio Input
```bash
# Record a short audio clip to test microphone
arecord -D hw:1,0 -f cd -d 5 test.wav
# Play back to verify recording was successful
aplay test.wav
```

## Troubleshooting

### Common Issues:

1. **RealSense camera not detected**:
   - Check USB connection
   - Verify UDEV rules are set up
   - Try different USB port if available

2. **Performance issues**:
   - Check power mode with `nvpmodel -q`
   - Monitor thermal throttling
   - Reduce sensor resolutions if needed

3. **Audio problems**:
   - Check list of audio devices: `arecord -l`
   - Configure default audio device in ~/.asoundrc

## Safety Considerations

- Always use appropriate power supply for your Jetson platform
- Ensure proper ventilation for sustained operation
- Handle the RealSense camera carefully to avoid damage
- Secure all connections to prevent disconnection during robotics operation

## Integration with Robotics Platform

For connecting the Edge AI Kit to a robot platform:

1. Mount the Jetson securely with shock absorption if needed
2. Connect to robot's power system with appropriate voltage regulation
3. Connect sensors to appropriate interfaces (USB, GPIO, I2C, etc.)
4. Ensure wireless connectivity to robot's main controller if needed
5. Test all connections before full operation

## Additional Resources

- [Jetson Downloads](https://developer.nvidia.com/embedded/downloads)
- [RealSense SDK Documentation](https://github.com/IntelRealSense/librealsense)
- [Jetson Performance Guide](https://docs.nvidia.com/jetson/l4t/index.html)
- [ROS 2 on Jetson](https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debians.html)