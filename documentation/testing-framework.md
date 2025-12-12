# Testing Framework for Physical AI & Humanoid Robotics

This document outlines the testing framework for the Physical AI & Humanoid Robotics curriculum, designed to validate both simulation and hardware implementations according to the Physical AI constitution requirements.

## Testing Philosophy

The testing framework follows the Physical AI constitution's Test-First Robotics principle:
- TDD mandatory for all robotic behaviors
- Simulation tests written before hardware implementation
- Red-Green-Refactor cycle with Gazebo integration tests
- Safety-first approach with comprehensive validation

## Test Categories

### 1. Unit Tests
- Test individual robot components (sensors, actuators, controllers)
- Validate message and service interfaces
- Verify mathematical calculations (kinematics, transforms)
- Mock external dependencies

### 2. Integration Tests
- Test communication between robot subsystems
- Validate ROS 2 topic and service interactions
- Verify behavior in simulation environments
- Check sensor data processing pipelines

### 3. System Tests
- End-to-end functionality validation
- Sim-to-real transfer validation
- Performance benchmarking
- Safety system validation

### 4. Acceptance Tests
- Validate robot behavior against curriculum learning outcomes
- Test complete task execution (navigation, manipulation, perception)
- Verify compliance with Physical AI constitution
- Assess real-world performance metrics

## Testing Framework Components

### 1. Gazebo Integration Testing
- Model simulation in various environments
- Sensor data validation against ground truth
- Physics interaction verification
- Performance benchmarking in simulation

### 2. Hardware-in-the-Loop (HIL) Testing
- Connect real robot components to simulated environments
- Validate control algorithms with real sensors
- Test perception systems with real and simulated data
- Assess sim-to-real transfer effectiveness

### 3. Behavior Tree Testing
- Validate behavior tree execution logic
- Test state transitions under various conditions
- Verify safety condition handling
- Assess mission completion rates

## Test Structure

### Test File Organization
```
test/
├── unit/
│   ├── test_sensor_processing.cpp
│   ├── test_kinematics.cpp
│   ├── test_state_machine.cpp
│   └── test_navigation.cpp
├── integration/
│   ├── test_perception_pipeline.cpp
│   ├── test_manipulation_system.cpp
│   └── test_communication.cpp
├── system/
│   ├── test_navigation_simulation.cpp
│   ├── test_manipulation_real.cpp
│   └── test_vla_integration.cpp
├── fixtures/
│   ├── test_robot.urdf
│   ├── test_world.world
│   └── test_scenarios.yaml
├── launch/
│   ├── test_navigation.launch.py
│   └── test_manipulation.launch.py
└── results/
    ├── unit_results.xml
    ├── integration_results.xml
    └── system_results.xml
```

## Unit Testing Implementation

### Example Unit Test Structure (C++)
```cpp
#include <gtest/gtest.h>
#include "robot_state_machine/robot_state_machine.hpp"

class RobotStateMachineTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Set up test fixtures
        state_machine_ = std::make_unique<robot_state_machine::RobotStateMachine>("test_node");
    }

    void TearDown() override {
        // Clean up after tests
        state_machine_.reset();
    }

    std::unique_ptr<robot_state_machine::RobotStateMachine> state_machine_;
};

// Test state transitions
TEST_F(RobotStateMachineTest, TransitionFromIdleToNavigating) {
    EXPECT_EQ(state_machine_->get_current_state(), robot_state_machine::RobotState::IDLE);
    
    state_machine_->transition_to_navigating();
    EXPECT_EQ(state_machine_->get_current_state(), robot_state_machine::RobotState::NAVIGATING);
    
    state_machine_->transition_to_idle();
    EXPECT_EQ(state_machine_->get_current_state(), robot_state_machine::RobotState::IDLE);
}

// Test invalid transitions
TEST_F(RobotStateMachineTest, InvalidStateTransition) {
    // Implementation to verify invalid transitions are rejected
    // according to the constitution requirements
}
```

### Example Python Test for ROS 2 Components
```python
import unittest
import rclpy
from std_msgs.msg import String
from sensor_msgs.msg import LaserScan
import time

class TestRobotPerception(unittest.TestCase):
    
    def setUp(self):
        rclpy.init()
        self.node = rclpy.create_node('test_robot_perception')
        
        # Create subscribers and publishers for testing
        self.scan_sub = self.node.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10
        )
        
        self.scan_received = False
        self.scan_data = None
    
    def scan_callback(self, msg):
        self.scan_received = True
        self.scan_data = msg
    
    def test_scan_data_processing(self):
        '''Test that laser scan data is processed correctly'''
        # Wait for scan data
        timeout = time.time() + 60*2  # 2 minute timeout
        while not self.scan_received and time.time() < timeout:
            rclpy.spin_once(self.node, timeout_sec=0.1)
        
        self.assertTrue(self.scan_received, "Laser scan data not received")
        
        # Validate scan data properties
        self.assertGreater(len(self.scan_data.ranges), 0, "No range data in scan")
        self.assertLessEqual(max(self.scan_data.ranges), 10.0, "Invalid max range")
    
    def tearDown(self):
        self.node.destroy_node()
        rclpy.shutdown()
```

## Simulation Testing with Gazebo

### Gazebo Test Launch File
```python
import os
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
from launch_testing.actions import ReadyToTest
import launch_testing

def generate_launch_description():
    # Launch Gazebo with test environment
    gazebo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('gazebo_ros'),
                'launch',
                'gazebo.launch.py'
            ])
        ]),
        launch_arguments={
            'world': PathJoinSubstitution([
                FindPackageShare('robot_state_machine'),
                'test',
                'fixtures',
                'test_world.world'
            ])
        }.items()
    )
    
    return LaunchDescription([
        gazebo_launch,
        ReadyToTest()
    ])
```

## Hardware Testing Procedures

### Pre-Deployment Validation
1. Validate all behaviors in simulation first
2. Test with hardware-in-the-loop where possible
3. Verify safety systems before real hardware deployment
4. Conduct thorough risk assessment

### Safety Testing
```cpp
// Example safety test
TEST_F(RobotStateMachineTest, SafetyStopFunctionality) {
    // Initialize robot in navigating state
    state_machine_->transition_to_navigating();
    EXPECT_EQ(state_machine_->get_current_state(), robot_state_machine::RobotState::NAVIGATING);
    
    // Simulate obstacle detection (this would trigger safety callback in real implementation)
    // Verify transition to safety stop
    state_machine_->transition_to_safety_stop();
    EXPECT_EQ(state_machine_->get_current_state(), robot_state_machine::RobotState::SAFETY_STOP);
    
    // Verify robot movement stopped
    // (This would involve checking velocity commands in a real test)
}
```

## Continuous Integration Setup

### GitHub Actions Workflow
```yaml
name: Physical AI & Humanoid Robotics CI

on: [push, pull_request]

jobs:
  unit-tests:
    runs-on: ubuntu-22.04
    steps:
    - uses: actions/checkout@v3
    - name: Install ROS 2 Humble
      uses: ros-tooling/setup-ros@v0.7
      with:
        required-ros-distributions: humble
    - name: Run colcon tests
      uses: ros-tooling/action-ros-ci@v0.3
      with:
        package-name: robot_state_machine
        import-launch-script-dependencies: true
    - name: Run unit tests
      run: |
        source install/setup.bash
        colcon test --packages-select robot_state_machine
        colcon test-result --all

  simulation-tests:
    runs-on: ubuntu-22.04
    steps:
    - uses: actions/checkout@v3
    - name: Install ROS 2 Humble and Gazebo
      uses: ros-tooling/setup-ros@v0.7
      with:
        required-ros-distributions: humble
    - name: Run simulation tests
      run: |
        # Launch Gazebo tests
        source install/setup.bash
        # Run specific simulation test scenarios
```

## Test Metrics and Validation

### Performance Metrics
- Task completion rate
- Navigation accuracy (position and orientation error)
- Manipulation success rate
- Response time to voice commands
- System stability (uptime, crashes)
- Safety system effectiveness

### Validation Criteria
- 85% success rate for basic tasks (navigation, manipulation)
- <2 seconds response time for voice commands
- <5cm navigation accuracy in known environments
- 90% object detection accuracy in controlled scenarios
- 100% safety system activation when required

## Test Report Template

After each test run, generate reports containing:
- Test execution summary
- Performance metrics
- Failure analysis
- Recommendations for improvements
- Compliance with constitution requirements

## Running Tests

### Unit Tests
```bash
# Build with tests
colcon build --packages-select robot_state_machine --cmake-args -DTHIRDPARTY=ON

# Run unit tests
source install/setup.bash
colcon test --packages-select robot_state_machine
colcon test-result --all
```

### Integration Tests
```bash
# Run specific integration tests
source install/setup.bash
./install/robot_state_machine/lib/robot_state_machine/test_integrated_behavior
```

## Test Maintenance

- Regular review of test coverage
- Update tests as requirements evolve
- Verify test relevance to learning outcomes
- Maintain hardware test procedures
- Document test environment setup

This testing framework ensures that all implementations in the Physical AI & Humanoid Robotics curriculum meet both functional requirements and constitution guidelines for safety, simulation-first development, and proper integration between digital AI and physical robotics.