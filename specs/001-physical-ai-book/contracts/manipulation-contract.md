# ROS 2 Service Contract: ManipulationPlanner

## Overview
This document defines the ROS 2 service interface for robot manipulation planning, implementing the grasping and manipulation component of the Physical AI system using MoveIt 2.

## Service Definition
- **Service Type**: `physical_ai_interfaces/srv/ManipulationPlanner`
- **Purpose**: Plans manipulation actions for robot arms/grippers to interact with objects
- **Pattern**: Request-Response using ROS 2 services

## Request Message

### Field Definitions
- `robot_id` (string)
  - Description: Identifier of the manipulating robot
  - Format: Alphanumeric string
  - Constraints: Must match an existing robot in the system
  - Required: Yes

- `manipulation_type` (string)
  - Description: Type of manipulation action
  - Values: "grasp", "place", "move", "retract", "custom"
  - Required: Yes

- `target_object` (physical_ai_interfaces/msg/ObjectInfo)
  - Description: Information about the target object
  - Format: ObjectInfo message
  - Required: For "grasp" and "place" actions

- `target_pose` (geometry_msgs/Pose)
  - Description: Target pose for end effector or object
  - Format: Standard ROS 2 Pose message with position and orientation
  - Required: For "place" and "move" actions

- `arm_name` (string)
  - Description: Name of the arm to use (for multi-arm robots)
  - Format: Alphanumeric string
  - Constraints: Must match an existing arm on the robot
  - Required: No (default: "main_arm")

- `grasp_strategy` (string)
  - Description: Strategy for grasping (for "grasp" actions)
  - Values: "top_grasp", "side_grasp", "pinch_grasp", "wrap_grasp"
  - Required: No (default: "top_grasp" for grasp actions)

- `end_effector` (string)
  - Description: Name of the end effector to use
  - Format: Alphanumeric string
  - Constraints: Must match an existing end effector on the robot
  - Required: No (default: "gripper")

- `approach_direction` (geometry_msgs/Vector3)
  - Description: Direction to approach the target from
  - Format: Standard ROS 2 Vector3 message (x, y, z)
  - Required: No (default: [0, 0, 1])

### Example Request
```yaml
robot_id: "unitree_g1_001"
manipulation_type: "grasp"
target_object:
  name: "red_cube"
  pose:
    position:
      x: 0.5
      y: 0.3
      z: 0.0
    orientation:
      x: 0.0
      y: 0.0
      z: 0.0
      w: 1.0
  dimensions:
    x: 0.05
    y: 0.05
    z: 0.05
  mass: 0.1
  material: "plastic"
arm_name: "right_arm"
grasp_strategy: "top_grasp"
end_effector: "right_gripper"
approach_direction:
  x: 0.0
  y: 0.0
  z: -1.0
```

## Response Message

### Field Definitions
- `success` (bool)
  - Description: Indicates if the manipulation planning was successful
  - Values: True if successful, False otherwise
  - Required: Yes

- `message` (string)
  - Description: Additional information about the result
  - Format: Human-readable string
  - Required: Yes

- `trajectory` (trajectory_msgs/JointTrajectory)
  - Description: Planned joint trajectory for the manipulation
  - Format: Standard ROS 2 JointTrajectory message
  - Required: If success is True

- `planner_info` (physical_ai_interfaces/msg/ManipulationInfo)
  - Description: Additional information about the manipulation planning process
  - Format: ManipulationInfo message
  - Required: If success is True

- `estimated_time` (float64)
  - Description: Estimated time to execute the trajectory (seconds)
  - Format: Positive floating point number
  - Required: If success is True

- `grasp_success_probability` (float64)
  - Description: Estimated probability of grasp success (for grasp actions)
  - Format: Floating point number between 0.0 and 1.0
  - Required: If manipulation_type is "grasp"

### Example Response
```yaml
success: true
message: "Grasp trajectory planned successfully"
trajectory:
  header:
    stamp:
      sec: 1678886400
      nanosec: 0
    frame_id: "base_link"
  joint_names:
    - "shoulder_pan_joint"
    - "shoulder_lift_joint"
    - "elbow_joint"
    - "wrist_1_joint"
    - "wrist_2_joint"
    - "wrist_3_joint"
  points:
    - positions: [0.0, -1.0, 0.0, -1.0, 0.0, 0.0]
      velocities: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
      time_from_start:
        sec: 1
        nanosec: 0
    - positions: [0.1, -0.9, -0.1, -1.1, 0.1, 0.05]
      velocities: [0.1, 0.1, -0.1, -0.1, 0.1, 0.05]
      time_from_start:
        sec: 2
        nanosec: 0
    # ... additional trajectory points
planner_info:
  planner_name: "OMPL"
  motion_type: "cartesian"
  grasp_approach: "top"
  grasp_quality: 0.85
estimated_time: 3.5
grasp_success_probability: 0.88
```

## ObjectInfo Message Definition

### Field Definitions
- `name` (string)
  - Description: Name of the object
  - Format: Alphanumeric string
  - Required: Yes

- `pose` (geometry_msgs/Pose)
  - Description: Pose of the object in space
  - Format: Standard ROS 2 Pose message
  - Required: Yes

- `dimensions` (geometry_msgs/Vector3)
  - Description: Dimensions of the object (for box approximation)
  - Format: Standard ROS 2 Vector3 with x, y, z as length, width, height
  - Required: Yes

- `mass` (float64)
  - Description: Mass of the object in kg
  - Format: Positive floating point number
  - Required: No (default: 0.5)

- `material` (string)
  - Description: Material of the object (for grasp planning)
  - Values: "plastic", "metal", "wood", "fabric", "glass", "other"
  - Required: No (default: "plastic")

## ManipulationInfo Message Definition

### Field Definitions
- `planner_name` (string)
  - Description: Name of the planner that generated the trajectory
  - Format: Human-readable string
  - Required: Yes

- `motion_type` (string)
  - Description: Type of motion planned
  - Values: "cartesian", "joint_space", "mixed"
  - Required: Yes

- `grasp_approach` (string)
  - Description: Approach direction for grasping
  - Values: "top", "side", "pinch", "wrap", "custom"
  - Required: For grasp actions

- `grasp_quality` (float64)
  - Description: Quality metric for the planned grasp (0.0 to 1.0)
  - Format: Floating point number between 0.0 and 1.0
  - Required: For grasp actions

## Error Handling
- Service returns false with appropriate message when:
  - Robot ID does not exist
  - Manipulation type is invalid
  - Target object information is missing for required actions
  - No valid grasp or placement pose can be found
  - Arm or end effector does not exist on the robot
  - Request times out

## Quality of Service (QoS)
- Reliability: Reliable
- Durability: Volatile
- History: Keep last 1
- Concurrency: Single-threaded

## Performance Expectations
- Average response time: < 2 seconds for simple grasps
- Maximum response time: < 10 seconds for complex manipulation planning
- Success rate: > 85% for objects in reachable workspace with known properties
- Grasp success probability accuracy: Correlates with actual grasp success > 80% of the time