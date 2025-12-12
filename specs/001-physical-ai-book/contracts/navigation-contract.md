# ROS 2 Service Contract: NavigationPlanner

## Overview
This document defines the ROS 2 service interface for robot navigation planning, implementing the path planning component of the Physical AI system using ROS 2 Navigation Stack (Nav2).

## Service Definition
- **Service Type**: `physical_ai_interfaces/srv/NavigationPlanner`
- **Purpose**: Plans navigation paths for robots in known or partially known environments
- **Pattern**: Request-Response using ROS 2 services

## Request Message

### Field Definitions
- `robot_id` (string)
  - Description: Identifier of the navigating robot
  - Format: Alphanumeric string
  - Constraints: Must match an existing robot in the system
  - Required: Yes

- `start_pose` (geometry_msgs/Pose)
  - Description: Starting pose of the robot
  - Format: Standard ROS 2 Pose message with position and orientation
  - Required: Yes

- `target_pose` (geometry_msgs/Pose)
  - Description: Target pose for navigation
  - Format: Standard ROS 2 Pose message with position and orientation
  - Required: Yes

- `map_name` (string)
  - Description: Name of the map to use for planning
  - Format: Alphanumeric string
  - Constraints: Must match an existing map
  - Required: No (default: "current_map")

- `planner_id` (string)
  - Description: ID of the specific planner to use
  - Format: Alphanumeric string
  - Constraints: Must match a configured planner
  - Required: No (default: "GridBased")

- `tolerance` (float64)
  - Description: Tolerance for reaching target position (meters)
  - Format: Positive floating point number
  - Constraints: Must be > 0.0 and < 10.0
  - Required: No (default: 0.5)

### Example Request
```yaml
robot_id: "unitree_g1_001"
start_pose:
  position:
    x: 0.0
    y: 0.0
    z: 0.0
  orientation:
    x: 0.0
    y: 0.0
    z: 0.0
    w: 1.0
target_pose:
  position:
    x: 5.0
    y: 3.0
    z: 0.0
  orientation:
    x: 0.0
    y: 0.0
    z: 0.707
    w: 0.707
map_name: "home_office"
planner_id: "GridBased"
tolerance: 0.25
```

## Response Message

### Field Definitions
- `success` (bool)
  - Description: Indicates if the path planning was successful
  - Values: True if successful, False otherwise
  - Required: Yes

- `message` (string)
  - Description: Additional information about the result
  - Format: Human-readable string
  - Required: Yes

- `path` (nav_msgs/Path)
  - Description: Planned navigation path
  - Format: Standard ROS 2 Path message with pose sequence
  - Required: If success is True

- `planner_info` (physical_ai_interfaces/msg/PlannerInfo)
  - Description: Additional information about the planning process
  - Format: PlannerInfo message
  - Required: If success is True

- `estimated_time` (float64)
  - Description: Estimated time to follow the path at default speed (seconds)
  - Format: Positive floating point number
  - Required: If success is True

- `path_length` (float64)
  - Description: Length of the planned path (meters)
  - Format: Positive floating point number
  - Required: If success is True

### Example Response
```yaml
success: true
message: "Path planned successfully"
path:
  header:
    stamp:
      sec: 1678886400
      nanosec: 0
    frame_id: "map"
  poses:
    - header:
        stamp:
          sec: 0
          nanosec: 0
        frame_id: "map"
      pose:
        position:
          x: 0.0
          y: 0.0
          z: 0.0
        orientation:
          x: 0.0
          y: 0.0
          z: 0.0
          w: 1.0
    - header:
        stamp:
          sec: 0
          nanosec: 0
        frame_id: "map"
      pose:
        position:
          x: 1.0
          y: 0.5
          z: 0.0
        orientation:
          x: 0.0
          y: 0.0
          z: 0.0
          w: 1.0
    # ... additional poses in the path
planner_info:
  planner_name: "NavFn"
  computation_time: 0.234
  path_cost: 8.76
estimated_time: 12.5
path_length: 7.8
```

## PlannerInfo Message Definition

### Field Definitions
- `planner_name` (string)
  - Description: Name of the planner that generated the path
  - Format: Human-readable string
  - Required: Yes

- `computation_time` (float64)
  - Description: Time taken to compute the path (seconds)
  - Format: Positive floating point number
  - Required: Yes

- `path_cost` (float64)
  - Description: Calculated cost of the path
  - Format: Positive floating point number
  - Required: Yes

## Error Handling
- Service returns false with appropriate message when:
  - Robot ID does not exist
  - Start or target pose is invalid
  - No path can be found between start and target
  - Map does not exist
  - Request times out

## Quality of Service (QoS)
- Reliability: Reliable
- Durability: Volatile
- History: Keep last 1
- Concurrency: Single-threaded

## Performance Expectations
- Average response time: < 1 second for simple paths in small areas
- Maximum response time: < 5 seconds for complex paths in large areas
- Success rate: > 95% for reachable targets in mapped environments
- Path optimality: Within 10% of optimal path length when using default planners