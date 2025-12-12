# Troubleshooting Guide for Physical AI & Humanoid Robotics

This guide provides solutions for common issues encountered during the Physical AI & Humanoid Robotics curriculum.

## ROS 2 Common Issues

### 1. Nodes Not Communicating
**Problem**: Nodes are not able to communicate via topics/services.

**Solutions**:
- Verify all nodes are on the same ROS_DOMAIN_ID:
  ```bash
  echo $ROS_DOMAIN_ID
  ```
- Check that nodes are on the same network if distributed
- Ensure all nodes source the same ROS 2 installation and workspace
- Verify ROS 2 agents are running: `ps aux | grep ros`
- Check firewall settings if nodes are on different machines

### 2. Missing Package Dependencies
**Problem**: Error message "package not found" when building.

**Solutions**:
- Update package lists: `sudo apt update`
- Install the missing package:
  ```bash
  sudo apt install ros-humble-<package-name>
  ```
- Search for the correct package name:
  ```bash
  apt search ros-humble
  ```
- Verify ROS 2 distribution (Humble Hawksbill) is correctly installed

### 3. Colcon Build Failures
**Problem**: Build fails with compilation errors.

**Solutions**:
- Clean the workspace: `rm -rf build install log`
- Verify all dependencies are installed
- Check CMakeLists.txt and package.xml syntax
- Ensure proper file permissions
- Check for missing header files

## Simulation Issues

### 1. Gazebo Not Starting
**Problem**: Gazebo fails to start or shows a black screen.

**Solutions**:
- Check graphics drivers: `nvidia-smi` (for NVIDIA) or `glxinfo | grep "OpenGL renderer"`
- Verify GPU compatibility with simulation
- Run with software rendering:
  ```bash
  export LIBGL_ALWAYS_SOFTWARE=1
  gazebo
  ```
- Check available VRAM and system RAM

### 2. Robot Falls Through Ground
**Problem**: Robot model falls through the ground plane in simulation.

**Solutions**:
- Check physics engine configuration in the world file
- Verify mass and inertia values in URDF
- Ensure proper collision geometry is defined
- Check for missing or incorrect joint limits

### 3. Simulation Runs Slowly
**Problem**: Simulation runs at less than real-time speed.

**Solutions**:
- Reduce physics update rate in world file
- Simplify collision meshes
- Lower sensor resolutions
- Check system resources (CPU, GPU, RAM)
- Disable unnecessary plugins

## Hardware Integration Issues

### 1. Jetson Not Connecting
**Problem**: Cannot connect to Jetson device.

**Solutions**:
- Verify network connection: `ping <jetson_ip>`
- Check SSH access: `ssh jetson@<jetson_ip>`
- Verify ROS 2 network setup
- Check power supply to Jetson
- Ensure proper cooling to prevent thermal throttling

### 2. Sensor Data Not Receiving
**Problem**: No data from sensors (camera, IMU, etc.).

**Solutions**:
- Check physical connections
- Verify sensor drivers are installed
- Check sensor permissions: `ls -l /dev/<sensor_device>`
- Test sensor independently of ROS 2
- Check if sensor node is running: `ros2 node list`

## Vision-Language-Action System Issues

### 1. Voice Commands Not Recognized
**Problem**: Voice commands are not processed correctly.

**Solutions**:
- Check microphone access and permissions
- Verify audio input levels
- Test with `arecord` and `aplay`
- Ensure Whisper model is properly installed
- Check audio format compatibility

### 2. Planning Service Failures
**Problem**: Service calls to planning services fail.

**Solutions**:
- Verify service server is running: `ros2 service list`
- Check service interface compatibility
- Verify robot and environment state parameters
- Check for timeout issues with complex planning requests

## Networking Issues

### 1. Multi-Robot Communication
**Problem**: Issues with communication between multiple robots/computers.

**Solutions**:
- Ensure all devices are on the same subnet
- Set consistent ROS_DOMAIN_ID across all devices
- Check firewall settings
- Verify ROS_LOCALHOST_ONLY=0 if using multiple machines
- Test network speed and latency

## Performance Issues

### 1. High CPU Usage
**Problem**: Nodes consume excessive CPU resources.

**Solutions**:
- Check loop rates and sleep calls in nodes
- Reduce sensor update rates if possible
- Optimize algorithms for better complexity
- Use threading where appropriate
- Monitor with `htop` to identify bottlenecks

### 2. Memory Leaks
**Problem**: Process memory usage increases over time.

**Solutions**:
- Use tools like `valgrind` for C++ or memory profilers for Python
- Ensure proper cleanup of resources
- Check for infinite loops creating objects
- Monitor with `watch -n 1 free -h`

## Development Environment Issues

### 1. IDE Not Recognizing ROS 2 Packages
**Problem**: IDE cannot find ROS 2 headers or packages.

**Solutions**:
- Ensure ROS 2 environment is properly sourced
- Configure IDE to use the correct compiler and include paths
- For VS Code, install ROS extension and reload window
- Check C++/Python path configurations in IDE

### 2. Virtual Environment Issues
**Problem**: Python packages not available after sourcing ROS 2.

**Solutions**:
- Source ROS 2 before activating virtual environment
- Or activate virtual environment before sourcing ROS 2 (may break some ROS tools)
- Use ROS 2's Python environment directly for ROS-specific packages

## Build and Installation Issues

### 1. Permission Errors
**Problem**: Cannot install packages or build workspace due to permissions.

**Solutions**:
- Do not use `sudo` with colcon build
- Check and fix file ownership in workspace
- Use `--symlink-install` with colcon to avoid permission issues
- Verify user has necessary permissions

### 2. Missing Dependencies
**Problem**: Build fails due to missing libraries.

**Solutions**:
- Run `rosdep install --from-paths src --ignore-src -r -y` in workspace
- Manually install missing system dependencies
- Check if dependencies are available for your ROS 2 distribution

## Testing Issues

### 1. Tests Failing
**Problem**: Unit or integration tests failing.

**Solutions**:
- Run specific test to get detailed error: `ros2 run <package> test_<name>`
- Check test dependencies are installed
- Ensure proper test environment setup
- Verify test data files are accessible

### 2. Simulation Tests Not Running
**Problem**: Tests requiring simulation environment fail.

**Solutions**:
- Ensure Gazebo is properly installed and accessible
- Check for display/graphics issues in headless environments
- Verify simulation parameters and world files
- Test simulation independently before running tests

## Debugging Strategies

### 1. Enable Logging
```bash
# Set logging level
export RCUTILS_LOGGING_SEVERITY_THRESHOLD=DEBUG
# Or for a specific node
export RCUTILS_LOGGING_SEVERITY_THRESHOLD=INFO
```

### 2. Use Debug Launch Files
Create launch files with debugging options enabled:
- Launch nodes in separate terminals
- Enable core dumps
- Set up remote debugging if needed

### 3. Monitor ROS 2 System
```bash
# View system status
ros2 doctor

# Monitor topics
ros2 topic hz <topic_name>
ros2 topic bw <topic_name>

# View system resources
ros2 run top top
```

## Getting Help

### 1. Check Documentation
- Review the relevant chapter documentation
- Check ROS 2 official documentation
- Look for examples in the curriculum code

### 2. Community Resources
- ROS Answers: https://answers.ros.org/
- Robotics Stack Exchange: https://robotics.stackexchange.com/
- Official ROS Discourse: https://discourse.ros.org/

### 3. Create Minimal Examples
When seeking help, create minimal reproducible examples:
- Isolate the specific issue
- Remove unnecessary code
- Include complete error messages

## When to Restart
Sometimes, the simplest solution is to restart services:

```bash
# Restart daemon if needed
sudo systemctl restart <service_name>
# Or restart ROS 2 system
# Close all terminals and reopen with fresh environment
```

Remember to always document the issue and solution for future reference.