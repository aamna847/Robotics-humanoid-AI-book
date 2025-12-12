# ROS 2 Service Contract: VoiceCommandToAction

## Overview
This document defines the ROS 2 service interface for converting voice commands to robot actions, implementing the Vision-Language-Action component of the Physical AI system.

## Service Definition
- **Service Type**: `physical_ai_interfaces/srv/VoiceCommandToAction`
- **Purpose**: Accepts voice commands in text form and returns planned robot actions
- **Pattern**: Request-Response using ROS 2 services

## Request Message

### Field Definitions
- `voice_command` (string)
  - Description: The voice command in text form
  - Format: Natural language command (e.g., "Pick up the red block")
  - Constraints: Maximum 256 characters
  - Required: Yes

- `robot_id` (string)
  - Description: Identifier of the target robot
  - Format: Alphanumeric string
  - Constraints: Must match an existing robot in the system
  - Required: Yes

- `context` (string)
  - Description: Context information for the command
  - Format: JSON string containing environment and robot state
  - Constraints: Maximum 1024 characters
  - Required: No (default: empty string)

- `timeout` (float64)
  - Description: Maximum time to wait for plan completion (seconds)
  - Format: Positive floating point number
  - Constraints: Must be > 0.0 and < 3600.0
  - Required: No (default: 30.0 seconds)

### Example Request
```yaml
voice_command: "move to the kitchen and pick up the green bottle"
robot_id: "unitree_g1_001"
context: '{"environment": "home_office", "objects": [{"name": "green_bottle", "position": [1.2, 0.5, 0.0]}]}'
timeout: 60.0
```

## Response Message

### Field Definitions
- `success` (bool)
  - Description: Indicates if the planning was successful
  - Values: True if successful, False otherwise
  - Required: Yes

- `message` (string)
  - Description: Additional information about the result
  - Format: Human-readable string
  - Required: Yes

- `plan_id` (string)
  - Description: Identifier for the planned sequence of actions
  - Format: UUID string
  - Required: If success is True

- `action_sequence` (physical_ai_interfaces/msg/ActionStep[])
  - Description: Ordered list of actions to execute
  - Format: Array of ActionStep messages
  - Constraints: Minimum 1 step, maximum 20 steps
  - Required: If success is True

- `estimated_duration` (float64)
  - Description: Estimated time to complete the entire plan (seconds)
  - Format: Positive floating point number
  - Required: If success is True

- `confidence` (float64)
  - Description: Confidence in the plan's success (0.0 to 1.0)
  - Format: Floating point number
  - Constraints: Between 0.0 and 1.0
  - Required: Yes

### Example Response
```yaml
success: true
message: "Plan generated successfully"
plan_id: "a1b2c3d4-e5f6-7890-1234-567890abcdef"
action_sequence:
  - action_type: "navigate"
    parameters: '{"target_position": {"x": 1.5, "y": 2.3, "z": 0.0}, "target_orientation": {"x": 0.0, "y": 0.0, "z": 0.707, "w": 0.707}}'
  - action_type: "perceive"
    parameters: '{"target_object": "green_bottle", "sensor_config": {"fov": 60.0, "range": 2.0}}}'
  - action_type: "manipulate"
    parameters: '{"action": "grasp", "target_object": "green_bottle", "grasp_type": "top_grasp"}'
estimated_duration: 45.5
confidence: 0.87
```

## ActionStep Message Definition

### Field Definitions
- `action_type` (string)
  - Description: Type of action to perform
  - Values: "navigate", "perceive", "manipulate", "wait", "speak", "custom"
  - Required: Yes

- `parameters` (string)
  - Description: Action-specific parameters in JSON format
  - Format: JSON string
  - Constraints: Maximum 512 characters
  - Required: Yes

- `timeout` (float64)
  - Description: Maximum time to wait for this action (seconds)
  - Format: Positive floating point number
  - Required: No (default: 30.0 seconds)

## Error Handling
- Service returns false with appropriate message when:
  - Voice command is malformed or unclear
  - Robot ID does not exist
  - Planning fails due to environmental constraints
  - Request times out

## Quality of Service (QoS)
- Reliability: Reliable
- Durability: Volatile
- History: Keep last 1
- Concurrency: Single-threaded

## Performance Expectations
- Average response time: < 2 seconds for simple commands
- Maximum response time: < 5 seconds for complex commands
- Success rate: > 90% for well-formed commands in known environments