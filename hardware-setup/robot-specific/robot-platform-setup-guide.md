# Robot Platform Setup Guide

This guide covers the setup and configuration for various robot platforms supported in the Physical AI & Humanoid Robotics curriculum.

## Supported Platforms

### 1. Unitree Go2 (Budget Option)
- **Type**: Quadruped robot
- **Price Range**: ~$16,000
- **Key Features**: 4 legs for dynamic movement, multiple sensors, programmable

### 2. OP3 (Mid-Range Option)
- **Type**: Humanoid robot
- **Price Range**: ~$100,000
- **Key Features**: 29 DOF, vision system, multiple sensors, ROS-compatible

### 3. Unitree G1 Mini (Mid-Range Option)
- **Type**: Humanoid robot
- **Price Range**: ~$16,000
- **Key Features**: Bipedal design, affordable humanoid platform

### 4. Unitree G1 (Premium Option)
- **Type**: Full-sized humanoid robot
- **Price Range**: ~$115,000
- **Key Features**: Full-sized humanoid, advanced control, high DOF

## Prerequisites

- Robot platform purchased and delivered
- Appropriate workspace for robot operation
- Safety equipment (safety glasses, etc.)
- Network infrastructure for communication
- Knowledge of robot-specific safety procedures

## Unitree Go2 Setup

### 1. Initial Unboxing and Inspection
- Inspect robot for any shipping damage
- Verify all components are present (charger, remote, documentation)
- Check that all legs are properly attached
- Verify battery is present and secured

### 2. Charging
- Connect provided charger to robot
- Allow 2-3 hours for full charge (LED indicator will show when complete)
- Do not operate robot with low battery

### 3. Network Setup
- Connect to robot's WiFi network: `UnitreeGo2_XXXX`
- Access robot's control interface through provided app or web interface
- Configure your network settings to connect robot to local WiFi if needed

### 4. Software Setup
```bash
# Install Unitree SDK
git clone https://github.com/unitreerobotics/unitree_ros.git
cd unitree_ros
git checkout humble  # For ROS 2 Humble compatibility
colcon build
source install/setup.bash
```

### 5. Basic Testing
- Power on the robot using the power button
- Use the mobile app or remote to control basic movements
- Verify all legs respond to commands
- Test basic movements: stand up, sit down, basic walking

### 6. ROS 2 Integration
```bash
# Launch the Go2 ROS interface
roslaunch go2_ros_interface go2_interface.launch.py
```

## OP3 Setup

### 1. Unboxing and Initial Inspection
- Verify all components are present (robot, battery, charger, cables)
- Check that all joints are secure
- Inspect external sensors and cameras

### 2. Battery Installation
- Remove battery from transport mode if applicable
- Install battery with proper polarity
- Connect main power cable

### 3. Network Configuration
- Connect robot to network via Ethernet or WiFi
- Find robot's IP address on network
- Test network connectivity

### 4. Software Setup
```bash
# Install OpenMANIPULATOR packages
sudo apt install ros-humble-open-manipulator-*
sudo apt install ros-humble-robotis-manipulator-h

# Set up robotis environment
source /opt/ros/humble/setup.bash
source /usr/local/robotis/setup.sh
```

### 5. Basic Controls
- Launch the basic control interface
- Test individual joint movements
- Test camera and sensor feeds
- Verify all DOFs function properly

## Unitree G1 Mini Setup

### 1. Unboxing and Safety Check
- Inspect robot for damage
- Ensure robot is in shipping lock mode if applicable
- Verify all protective elements are removed

### 2. Battery and Power
- Install main battery pack if not pre-installed
- Charge battery according to manufacturer specifications
- Connect to power supply for initial activation

### 3. Calibration
- Perform initial joint calibration following manufacturer instructions
- Run initial setup routines
- Check all joint ranges of motion

### 4. Network and Control
- Configure network settings
- Set up monitoring and control software
- Test basic movements and balance

## Unitree G1 Setup

### 1. Pre-Installation Requirements
- Dedicated space with appropriate flooring
- Safety measures for handling large robot
- Professional assistance for initial setup may be required

### 2. Physical Setup
- Secure robot to floor if required
- Connect to power supply
- Install any site-specific accessories

### 3. Software Configuration
```bash
# Set up G1 specific ROS packages
# (Exact steps vary by manufacturer)
git clone https://github.com/unitreerobotics/g1_ros.git
cd g1_ros
# Follow specific G1 ROS setup instructions
```

### 4. Safety Systems
- Configure emergency stops
- Set up safety perimeters
- Verify all safety systems function correctly

## Safety Procedures

### Before Operation
- Clear workspace of obstacles
- Verify safety systems are active
- Ensure proper ventilation
- Have safety equipment accessible

### During Operation
- Maintain safe distance during dynamic movements
- Monitor robot's behavior for anomalies
- Be ready to activate emergency stop
- Do not attempt to correct robot behavior manually

### After Operation
- Safely power down robot
- Secure robot if needed
- Log any issues or anomalies
- Charge batteries as needed

## Troubleshooting

### Common Issues:

1. **Communication Problems**:
   - Verify network connectivity
   - Check robot's IP address
   - Restart network interface if needed

2. **Joint Issues**:
   - Check for obstructions
   - Verify calibration
   - Check error codes on robot interface

3. **Battery Problems**:
   - Verify battery charge level
   - Check battery connections
   - Replace battery if needed based on cycle count

## Maintenance Schedule

### Daily
- Visual inspection for damage
- Check battery levels
- Verify all joints are clear of obstructions

### Weekly
- Clean external surfaces
- Check all connections
- Update software if available

### Monthly
- Deep battery calibration
- Detailed inspection of joints and actuators
- Backup robot configurations

## Integration with Curriculum

### For ROS 2 Development
- Connect to robot's ROS 2 network
- Use standard ROS 2 tools for control and monitoring
- Test custom nodes developed in simulation

### For AI Integration
- Connect Edge AI Kit to robot
- Test perception algorithms on real sensors
- Validate AI models with real-world data

### For Vision-Language-Action Systems
- Test voice command integration
- Validate planning algorithms with real robot
- Assess real-world performance vs. simulation

## Additional Resources

- [Unitree Robotics Documentation](https://www.unitree.com/support/)
- [ROBOTIS OpenMANIPULATOR Documentation](https://emanual.robotis.com/docs/en/platform/op3/)
- [ROS 2 Robot Integration Guide](http://wiki.ros.org/Robots)
- [Robot Safety Guidelines](https://www.iso.org/standard/45283.html) (ISO 10218)