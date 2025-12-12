# Data Model: Physical AI & Humanoid Robotics Book

## Overview
This document defines the key entities and their relationships for the Physical AI & Humanoid Robotics book project. These data models represent the conceptual structure of the robotic systems covered in the curriculum.

## Core Entities

### Robot
- **Description**: Represents a physical or simulated robot platform
- **Fields**:
  - id: string (unique identifier for the robot)
  - name: string (human-readable name)
  - type: string (e.g., "Unitree Go2", "OP3", "Unitree G1")
  - urdf_path: string (path to URDF model file)
  - sensors: array<Sensor> (list of sensors on the robot)
  - actuators: array<Actuator> (list of actuators on the robot)
  - capabilities: array<string> (list of robot capabilities)
  - kinematics: KinematicModel (kinematic description)

### Sensor
- **Description**: A sensor component attached to a robot
- **Fields**:
  - id: string (unique identifier)
  - name: string (human-readable name)
  - type: string (e.g., "lidar", "camera", "imu", "depth", "microphone")
  - topic: string (ROS 2 topic name for sensor data)
  - frame_id: string (coordinate frame ID)
  - parameters: object (sensor-specific parameters)
  - specifications: SensorSpec (performance characteristics)

### SensorSpec
- **Description**: Technical specifications for a sensor
- **Fields**:
  - range_min: float (minimum detection range)
  - range_max: float (maximum detection range)
  - resolution: float (spatial or angular resolution)
  - frequency: float (update frequency in Hz)
  - accuracy: float (measurement accuracy)

### Actuator
- **Description**: An actuator component of a robot
- **Fields**:
  - id: string (unique identifier)
  - name: string (human-readable name)
  - type: string (e.g., "joint", "gripper", "wheel")
  - joint_names: array<string> (list of joint names controlled by this actuator)
  - topic: string (ROS 2 topic for control commands)
  - command_type: string (type of command message)

### KinematicModel
- **Description**: Kinematic description of a robot
- **Fields**:
  - urdf_path: string (path to URDF file)
  - joint_limits: object (limits for each joint)
  - base_frame: string (base coordinate frame)
  - end_effectors: array<string> (list of end effector frames)
  - kinematic_chain: array<string> (ordered list of joint names)

### Environment
- **Description**: A physical or simulated environment for the robot
- **Fields**:
  - id: string (unique identifier)
  - name: string (human-readable name)
  - type: string (e.g., "gazebo", "unity", "real_world")
  - description: string (brief description)
  - objects: array<EnvironmentObject> (objects in the environment)

### EnvironmentObject
- **Description**: An object in a robot's environment
- **Fields**:
  - id: string (unique identifier)
  - name: string (human-readable name)
  - type: string (e.g., "furniture", "obstacle", "target_object")
  - position: Position3D (world coordinates)
  - orientation: Orientation (3D orientation)
  - physical_properties: PhysicalProperties (mass, material, etc.)

### Position3D
- **Description**: 3D position in space
- **Fields**:
  - x: float (x coordinate in meters)
  - y: float (y coordinate in meters)
  - z: float (z coordinate in meters)

### Orientation
- **Description**: 3D orientation using quaternion
- **Fields**:
  - x: float (x component of quaternion)
  - y: float (y component of quaternion)
  - z: float (z component of quaternion)
  - w: float (w component of quaternion)

### PhysicalProperties
- **Description**: Physical properties of an object
- **Fields**:
  - mass: float (mass in kg)
  - material: string (material type)
  - friction: float (friction coefficient)
  - collision_enabled: boolean (whether collision detection is enabled)

### ControlCommand
- **Description**: Command sent to control a robot
- **Fields**:
  - id: string (unique identifier)
  - robot_id: string (robot to control)
  - command_type: string (e.g., "navigation", "manipulation", "motion")
  - parameters: object (command-specific parameters)
  - timestamp: string (ISO 8601 timestamp)
  - priority: integer (priority level, 0-10)

### PerceptionData
- **Description**: Data from robot perception systems
- **Fields**:
  - id: string (unique identifier)
  - robot_id: string (robot that generated data)
  - sensor_id: string (sensor that generated data)
  - data_type: string (e.g., "point_cloud", "image", "object_detection")
  - timestamp: string (ISO 8601 timestamp)
  - content: object (sensor-specific data)

### NavigationGoal
- **Description**: Goal for robot navigation system
- **Fields**:
  - id: string (unique identifier)
  - robot_id: string (robot to navigate)
  - position: Position3D (target position)
  - orientation: Orientation (target orientation)
  - frame_id: string (coordinate frame for goal)
  - tolerance: float (position tolerance in meters)
  - timestamp: string (ISO 8601 timestamp)

### ManipulationTask
- **Description**: Task for robot manipulation system
- **Fields**:
  - id: string (unique identifier)
  - robot_id: string (robot to execute task)
  - task_type: string (e.g., "grasp", "place", "move")
  - target_object: string (name of target object)
  - target_position: Position3D (target position for task)
  - grasp_type: string (type of grasp to use)
  - success_criteria: array<string> (criteria for task success)

### VoiceCommand
- **Description**: Voice command processed by the system
- **Fields**:
  - id: string (unique identifier)
  - text: string (recognized text)
  - confidence: float (confidence level of recognition)
  - intent: string (extracted intent)
  - entities: array<Entity> (extracted entities from command)
  - timestamp: string (ISO 8601 timestamp)

### Entity
- **Description**: Entity extracted from voice command
- **Fields**:
  - type: string (e.g., "object", "location", "action")
  - value: string (extracted value)
  - confidence: float (confidence of extraction)

### PlanningResult
- **Description**: Result of task planning process
- **Fields**:
  - id: string (unique identifier)
  - voice_command_id: string (original voice command)
  - robot_id: string (robot to execute plan)
  - plan_steps: array<PlanStep> (ordered steps of the plan)
  - success_probability: float (estimated success probability)
  - estimated_time: float (estimated time in seconds)

### PlanStep
- **Description**: A single step in a robot plan
- **Fields**:
  - id: string (unique identifier)
  - step_type: string (e.g., "navigate", "perceive", "manipulate", "wait")
  - parameters: object (step-specific parameters)
  - dependencies: array<string> (ids of dependent steps)
  - success_criteria: array<string> (criteria for step completion)

## Relationships

```
Robot 1 --- * Sensor
Robot 1 --- * Actuator
Robot 1 --- 1 KinematicModel

Sensor * --- * EnvironmentObject (via perception data)
Actuator * --- * ControlCommand (as targets)

Environment 1 --- * EnvironmentObject
Environment * --- * Robot (as deployment target)

ControlCommand 1 --- 1 Robot
ControlCommand 1 --- 1 NavigationGoal (if navigation command)
ControlCommand 1 --- 1 ManipulationTask (if manipulation command)

PerceptionData 1 --- 1 Robot
PerceptionData 1 --- 1 Sensor

VoiceCommand 1 --- 1 PlanningResult
PlanningResult 1 --- * PlanStep

PlanStep * --- * PlanStep (dependency relationship)
```

## State Transitions

### Robot State Machine
- IDLE → NAVIGATING (when navigation command received)
- IDLE → MANIPULATING (when manipulation command received)
- NAVIGATING → IDLE (when navigation completed successfully)
- NAVIGATING → IDLE (when navigation cancelled or failed)
- MANIPULATING → IDLE (when manipulation completed successfully)
- MANIPULATING → IDLE (when manipulation cancelled or failed)
- * → SAFETY_STOP (when safety conditions detected)

### Task Execution State Machine
- PENDING → EXECUTING (when task execution starts)
- EXECUTING → SUCCEEDED (when task completed successfully)
- EXECUTING → FAILED (when task execution failed)
- EXECUTING → CANCELLED (when task was cancelled)

## Validation Rules
1. All robot joints must have defined limits
2. Sensor frame IDs must match coordinate frames defined in URDF
3. All positions must be within environment bounds
4. All control commands must have valid target robots
5. All navigation goals must have reachable positions
6. Manipulation tasks must reference existing objects
7. Plan steps must have valid dependencies (no circular dependencies)
8. Timestamps must be in valid ISO 8601 format