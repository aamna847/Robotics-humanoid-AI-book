---
slug: chapter-11-humanoid-kinematics-locomotion
title: Chapter 11 - Humanoid Kinematics & Bipedal Locomotion
description: Comprehensive guide to humanoid kinematics and bipedal locomotion for robotics
tags: [humanoid, kinematics, locomotion, bipedal, walking, robotics]
---

# 📚 Chapter 11: Humanoid Kinematics & Bipedal Locomotion 📚

## 🎯 Learning Objectives 🎯

By the end of this chapter, students will be able to:
- Understand the unique challenges of controlling humanoid robots
- Implement advanced control strategies for bipedal locomotion
- Design balance and stability control systems
- Integrate whole-body control approaches
- Apply machine learning to humanoid movement learning
- Evaluate and validate humanoid control systems

## 👋 11.1 Introduction to Humanoid Control Challenges 👋

Humanoid robots, with their human-like form factor, present unique control challenges that differ significantly from simpler robotic platforms. These challenges stem from the robot's complex kinematic structure, underactuation, balance requirements, and the need to operate in human-designed environments.

### ℹ️ 11.1.1 Kinematic Complexity ℹ️

Humanoid robots typically have 30+ degrees of freedom (DOF) distributed across legs, arms, and torso. This creates complex inverse kinematics problems where multiple joint configurations can achieve the same end-effector position.

```python
import numpy as np
import roboticstoolbox as rtb
from spatialmath import SE3
import matplotlib.pyplot as plt

class HumanoidKinematics:
    def __init__(self):
        # Simplified humanoid model with 12 DOF (6 per leg, 0 for upper body for simplicity)
        # In practice, humanoids have many more DOF
        self.dof = 12
        
        # Define joint limits
        self.joint_limits = {
            'hip_yaw': (-0.5, 0.5),
            'hip_roll': (-0.5, 0.5),
            'hip_pitch': (-1.0, 1.0),
            'knee': (0.0, 2.0),
            'ankle_pitch': (-0.5, 0.5),
            'ankle_roll': (-0.5, 0.5)
        }
        
        # Define link lengths (simplified)
        self.link_lengths = {
            'thigh': 0.4,  # meters
            'shin': 0.4,   # meters
        }
    
    def forward_kinematics(self, joint_angles):
        """
        Compute forward kinematics for simplified humanoid leg
        joint_angles: array of 6 joint angles for one leg
        Returns: end-effector position
        """
        # Simplified 3D forward kinematics for a leg with 6 DOF
        # In practice, this would be much more complex
        q = joint_angles
        
        # Compute positions of each joint in the chain
        # This is a greatly simplified version - real implementations use rotation matrices
        x = (self.link_lengths['thigh'] * np.sin(q[2]) + 
             self.link_lengths['shin'] * np.sin(q[2] + q[3]))
        
        y = 0  # Simplified, ignoring roll and yaw for this example
        z = -(self.link_lengths['thigh'] * np.cos(q[2]) + 
              self.link_lengths['shin'] * np.cos(q[2] + q[3]))
        
        return np.array([x, y, z])
    
    def inverse_kinematics(self, target_pos, leg_offset=np.array([0, 0, 0])):
        """
        Compute inverse kinematics for reaching a target position
        target_pos: desired end-effector position
        leg_offset: offset from robot center to leg
        Returns: joint angles to reach target position
        """
        # Simplified inverse kinematics for planar 2-link manipulator
        # In practice, this would handle the full 6-DOF leg
        target = target_pos - leg_offset
        
        # Calculate distance to target
        dist = np.linalg.norm(target)
        
        # Leg lengths
        l1 = self.link_lengths['thigh']
        l2 = self.link_lengths['shin']
        
        # Check if target is reachable
        if dist > (l1 + l2):
            # Target out of reach, extend fully toward target
            target = target * (l1 + l2) / dist
        
        if dist < abs(l1 - l2):
            # Target inside workspace, move as close as possible
            target = target * abs(l1 - l2) / dist
        
        # Inverse kinematics solution for 2-link planar manipulator
        x, y, z = target
        hip_pitch = np.arctan2(y, x)
        
        # Calculate leg angle in the xz plane
        xz_dist = np.sqrt(x**2 + z**2)
        
        # Apply cosine rule for 2-link manipulator
        cos_angle = (l1**2 + l2**2 - xz_dist**2) / (2 * l1 * l2)
        cos_angle = np.clip(cos_angle, -1, 1)  # Clamp to valid range
        knee_angle = np.pi - np.arccos(cos_angle)
        
        # Calculate hip angle
        angle2 = np.arctan2(z, x)  # Angle from x-axis to target
        angle1 = np.arccos((l1**2 + xz_dist**2 - l2**2) / (2 * l1 * xz_dist))
        hip_angle = angle2 - angle1
        
        # Fill in remaining DOF with defaults (real implementation would be more complex)
        joint_angles = np.zeros(6)
        joint_angles[0] = 0  # hip_yaw (simplified)
        joint_angles[1] = 0  # hip_roll (simplified) 
        joint_angles[2] = hip_angle  # hip_pitch
        joint_angles[3] = knee_angle  # knee
        joint_angles[4] = 0  # ankle_pitch (simplified)
        joint_angles[5] = 0  # ankle_roll (simplified)
        
        return joint_angles

class AdvancedHumanoidController:
    def __init__(self, robot_model):
        self.robot = robot_model
        self.kinematics = HumanoidKinematics()
        self.balance_controller = BalanceController()
        self.walk_engine = WalkEngine()
        
    def move_to_pose(self, target_pose):
        """
        Move the humanoid to a target pose while maintaining balance
        target_pose: SE3 transformation matrix
        """
        # Plan full-body motion to achieve target pose
        joint_angles = self.solve_full_body_IK(target_pose)
        
        # Execute motion while monitoring balance
        self.execute_balanced_motion(joint_angles)
    
    def solve_full_body_IK(self, target_pose):
        """Solve full-body inverse kinematics problem"""
        # This would use advanced methods like:
        # - Task-priority based IK
        # - Whole-body IK solvers (e.g., KDL, Pinocchio)
        # - Optimization-based approaches
        
        # For this example, we'll return a simplified solution
        return np.zeros(self.robot.nq)  # Placeholder
    
    def execute_balanced_motion(self, joint_angles):
        """Execute motion while maintaining balance"""
        # Use the balance controller to modulate movements
        balanced_angles = self.balance_controller.adjust_for_balance(joint_angles)
        
        # Send commands to robot
        self.robot.set_joint_targets(balanced_angles)
```

### 📋 11.1.2 Balance and Stability Requirements 📋

Unlike wheeled or tracked robots, humanoid robots must maintain balance on two points of contact or dynamically transition between balance states.

```python
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

class BalanceController:
    def __init__(self, robot_mass=75.0, com_height=0.8):
        self.robot_mass = robot_mass  # kg
        self.com_height = com_height  # Center of mass height in meters
        
        # Control gains
        self.k_p = 500  # Proportional gain for position control
        self.k_d = 100  # Derivative gain for velocity control
        
        # Support polygon boundaries
        self.foot_separation = 0.3  # Distance between feet in meters
        
        # Current state estimates
        self.com_position = np.zeros(3)  # Center of mass position [x, y, z]
        self.com_velocity = np.zeros(3)  # Center of mass velocity
        self.zmp = np.zeros(2)  # Zero Moment Point [x, y]
        
        # ZMP tracking controller
        self.zmp_controller = self._create_zmp_controller()
        
    def _create_zmp_controller(self):
        """Create ZMP (Zero Moment Point) tracking controller"""
        # Simple PID controller for ZMP tracking
        # In practice, this would be more sophisticated
        return {
            'kp': 100.0,  # Proportional gain
            'ki': 10.0,   # Integral gain  
            'kd': 50.0,   # Derivative gain
            'integral_error': 0,
            'prev_error': 0
        }
    
    def update_state(self, sensor_data):
        """Update controller with current sensor data"""
        # Extract center of mass information from sensor data
        # This would come from IMU, encoders, and possibly vision
        self.com_position = sensor_data.get('com_position', self.com_position)
        self.com_velocity = sensor_data.get('com_velocity', self.com_velocity)
        self.zmp = sensor_data.get('zmp', self.zmp)
    
    def compute_balance_correction(self, target_zmp, dt):
        """Compute balance corrections based on ZMP error"""
        # Calculate error between current and target ZMP
        zmp_error = target_zmp - self.zmp
        
        # ZMP controller (simplified PID)
        controller = self.zmp_controller
        
        # Proportional component
        p_term = controller['kp'] * zmp_error
        
        # Integral component
        controller['integral_error'] += zmp_error * dt
        i_term = controller['ki'] * controller['integral_error']
        
        # Derivative component
        derivative_error = (zmp_error - controller['prev_error']) / dt
        d_term = controller['kd'] * derivative_error
        
        controller['prev_error'] = zmp_error
        
        # Total correction
        correction = p_term + i_term + d_term
        
        return correction
    
    def adjust_for_balance(self, joint_commands):
        """Adjust joint commands to maintain balance"""
        # Calculate necessary adjustments based on current COM position
        # and planned movement
        adjustments = np.zeros_like(joint_commands)
        
        # This would involve more complex calculations in practice
        # to maintain the robot's center of mass within the support polygon
        
        return joint_commands + adjustments
    
    def compute_support_polygon(self, left_foot_pos, right_foot_pos):
        """Compute support polygon based on foot positions"""
        # For two feet, this is the convex hull of both foot contact points
        # Simplified as a rectangle for this example
        foot_length = 0.2  # meters
        foot_width = 0.1   # meters
        
        # Center points
        center = (left_foot_pos + right_foot_pos) / 2
        dx = right_foot_pos[0] - left_foot_pos[0]
        dy = right_foot_pos[1] - left_foot_pos[1]
        
        # Support polygon vertices (simplified)
        vertices = np.array([
            [center[0] - foot_length/2, center[1] - foot_width],
            [center[0] + foot_length/2, center[1] - foot_width],
            [center[0] + foot_length/2, center[1] + foot_width],
            [center[0] - foot_length/2, center[1] + foot_width]
        ])
        
        # Add points for each foot
        # Left foot
        vertices = np.vstack([
            vertices,
            [
                [left_foot_pos[0] - foot_length/2, left_foot_pos[1] - foot_width/2],
                [left_foot_pos[0] + foot_length/2, left_foot_pos[1] - foot_width/2],
                [left_foot_pos[0] + foot_length/2, left_foot_pos[1] + foot_width/2],
                [left_foot_pos[0] - foot_length/2, left_foot_pos[1] + foot_width/2]
            ]
        ])
        
        # Right foot
        vertices = np.vstack([
            vertices,
            [
                [right_foot_pos[0] - foot_length/2, right_foot_pos[1] - foot_width/2],
                [right_foot_pos[0] + foot_length/2, right_foot_pos[1] - foot_width/2],
                [right_foot_pos[0] + foot_length/2, right_foot_pos[1] + foot_width/2],
                [right_foot_pos[0] - foot_length/2, right_foot_pos[1] + foot_width/2]
            ]
        ])
        
        return vertices
    
    def is_balanced(self, com_pos, support_polygon):
        """Check if center of mass is within support polygon"""
        # Simplified 2D check (x, y plane)
        # In practice, this would use more sophisticated geometric methods
        com_xy = com_pos[:2]
        
        # Check if point is inside polygon (ray casting algorithm - simplified)
        x, y = com_xy[0], com_xy[1]
        
        # For this example, we'll use a bounding box check
        min_x = np.min(support_polygon[:, 0])
        max_x = np.max(support_polygon[:, 0])
        min_y = np.min(support_polygon[:, 1])
        max_y = np.max(support_polygon[:, 1])
        
        return (min_x <= x <= max_x) and (min_y <= y <= max_y)

class ZMPCalculator:
    def __init__(self, gravity=9.81):
        self.g = gravity
    
    def compute_zmp_simple(self, com_pos, com_acc):
        """Compute ZMP from center of mass position and acceleration"""
        # ZMP_x = CoM_x - (CoM_z - foot_z) / g * CoM_acc_x
        # ZMP_y = CoM_y - (CoM_z - foot_z) / g * CoM_acc_y
        
        foot_height = 0  # Simplified: foot at z=0
        zmp_x = com_pos[0] - (com_pos[2] - foot_height) / self.g * com_acc[0]
        zmp_y = com_pos[1] - (com_pos[2] - foot_height) / self.g * com_acc[1]
        
        return np.array([zmp_x, zmp_y])
```

## 🎛️ 11.2 Bipedal Locomotion Control 🎛️

### ℹ️ 11.2.1 Walking Pattern Generation ℹ️

Generating stable walking patterns for bipedal robots requires careful consideration of balance, momentum, and ground contact forces.

```python
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt

class WalkEngine:
    def __init__(self, step_length=0.3, step_height=0.05, step_time=0.8):
        self.step_length = step_length  # meters
        self.step_height = step_height  # meters
        self.step_time = step_time      # seconds
        
        # Walking parameters
        self.stride = step_length
        self.foot_lift = step_height
        self.cycle_time = step_time
        
        # Gait phase tracking
        self.phase = 0  # 0.0 to 1.0
        self.swing_leg = 'left'  # Which leg is swinging
        self.double_support_ratio = 0.2  # 20% of step cycle in double support
        
        # Store trajectory
        self.trajectory = []
        
    def generate_foot_trajectory(self, start_pos, goal_pos, current_time=0):
        """Generate smooth trajectory for foot movement"""
        # Define key frames for the foot trajectory
        # 1. Lift foot from ground
        # 2. Move foot forward in arc
        # 3. Lower foot to ground
        
        # Create time array for the step
        t_step = np.linspace(0, self.step_time, num=100)
        
        # X trajectory: simple linear movement
        x_traj = np.linspace(start_pos[0], goal_pos[0], num=100)
        
        # Y trajectory: adjust Y position if changing stance legs
        y_start, y_goal = start_pos[1], goal_pos[1]
        y_traj = np.linspace(y_start, y_goal, num=100)
        
        # Z trajectory: parabolic arc for foot lift
        z_lift = np.zeros_like(t_step)
        
        # Calculate when to lift and lower foot (single vs double support)
        lift_start = self.double_support_ratio * self.step_time / 2
        lift_end = self.step_time - lift_start
        
        for i, t in enumerate(t_step):
            if t < lift_start or t > lift_end:
                # Foot on ground
                z_lift[i] = 0.0
            else:
                # Foot in swing phase - lift in parabolic arc
                t_lift = (t - lift_start) / (lift_end - lift_start)  # Normalize to [0,1]
                
                # Parabolic trajectory: 4 * h * t * (1-t) for normalized t
                z_lift[i] = 4 * self.step_height * t_lift * (1 - t_lift)
        
        # Combine into trajectory
        trajectory = np.column_stack([x_traj, y_traj, z_lift])
        
        return trajectory, t_step
    
    def generate_com_trajectory(self, walking_speed):
        """Generate Center of Mass trajectory for stable walking"""
        # Use inverted pendulum model for CoM trajectory
        # This creates a stable walking pattern by moving CoM appropriately
        
        # Walking parameters
        step_time = self.step_time
        step_length = self.step_length
        
        # Time vector
        t = np.linspace(0, step_time, num=100)
        
        # Generate CoM trajectory in X direction (forward movement)
        com_x = np.linspace(0, walking_speed * step_time, num=100)
        
        # Generate CoM trajectory in Z direction (up and down movement)
        # Humans naturally move CoM up and down to save energy
        com_z = 0.8 + 0.02 * np.sin(2 * np.pi * t / step_time)  # Small oscillation
        
        # Generate CoM trajectory in Y direction (lateral movement)
        # Shift CoM toward stance leg to maintain balance
        com_y = 0.0  # For this example, simplified
        if self.swing_leg == 'right':
            # Shift CoM slightly to left (stance leg side)
            com_y = -0.05 * np.ones_like(t)
        else:
            # Shift CoM slightly to right (stance leg side)
            com_y = 0.05 * np.ones_like(t)
        
        return np.column_stack([com_x, com_y, com_z]), t
    
    def update_walking_phase(self, dt):
        """Update the walking phase based on time"""
        self.phase += dt / self.step_time
        
        # Keep phase in [0, 1]
        if self.phase >= 1.0:
            self.phase = 0.0
            # Switch swing leg
            self.swing_leg = 'right' if self.swing_leg == 'left' else 'left'
    
    def is_double_support_phase(self):
        """Check if currently in double support phase"""
        # Double support at beginning and end of step cycle
        return (self.phase < self.double_support_ratio / 2 or 
                self.phase > 1.0 - self.double_support_ratio / 2)

class WalkingController:
    def __init__(self, walk_engine, balance_controller):
        self.walk_engine = walk_engine
        self.balance_controller = balance_controller
        
        # Store previous step information
        self.left_foot_pos = np.array([0.0, 0.15, 0.0])   # Starting position
        self.right_foot_pos = np.array([0.0, -0.15, 0.0])  # Starting position
        
        # Walking state
        self.target_velocity = np.array([0.0, 0.0, 0.0])  # Target walking velocity
        self.is_walking = False
        
    def start_walking(self, velocity):
        """Start walking with specified velocity"""
        self.target_velocity = velocity
        self.is_walking = True
    
    def stop_walking(self):
        """Stop walking"""
        self.target_velocity = np.array([0.0, 0.0, 0.0])
        self.is_walking = False
    
    def compute_walking_step(self, dt):
        """Compute the next step in walking"""
        if not self.is_walking:
            return None
        
        # Update walking phase
        self.walk_engine.update_walking_phase(dt)
        
        # Determine which foot to move based on walking phase
        if self.walk_engine.swing_leg == 'left':
            stance_foot_pos = self.right_foot_pos
            swing_foot_pos = self.left_foot_pos
        else:
            stance_foot_pos = self.left_foot_pos
            swing_foot_pos = self.right_foot_pos
        
        # Calculate next swing foot position
        # Move forward by step length in the walking direction
        goal_offset = self.target_velocity * self.walk_engine.step_time
        goal_pos = stance_foot_pos + goal_offset
        
        # Generate foot trajectory
        trajectory, time_steps = self.walk_engine.generate_foot_trajectory(
            swing_foot_pos, goal_pos
        )
        
        # Calculate CoM trajectory for stability
        com_trajectory, com_time = self.walk_engine.generate_com_trajectory(
            np.linalg.norm(self.target_velocity)
        )
        
        # Return computed trajectories
        return {
            'swing_foot_trajectory': trajectory,
            'com_trajectory': com_trajectory,
            'step_timing': time_steps,
            'stance_foot_pos': stance_foot_pos
        }
    
    def adjust_step_for_balance(self, step_data, sensor_data):
        """Adjust step parameters based on balance information"""
        # This would modify the planned step based on current stability
        # For example, if the robot is leaning too far, adjust foot placement
        
        # Extract current state
        current_com = sensor_data.get('com_position', np.zeros(3))
        current_zmp = sensor_data.get('zmp', np.zeros(2))
        
        # Adjust goal position based on balance state
        adjusted_data = step_data.copy()
        
        # Example: if robot is leaning right, step to the right more
        if current_com[1] > 0.05:  # Leaning right
            adjustment = np.array([0.0, -0.02, 0.0])  # Step more left
            adjusted_data['swing_foot_trajectory'] += adjustment
        
        return adjusted_data
```

### 🔄 11.2.2 Advanced Locomotion Patterns 🔄

Beyond basic walking, humanoid robots need to handle various locomotion patterns:

```python
class LocomotionPatternGenerator:
    def __init__(self, robot_properties):
        self.properties = robot_properties
        self.current_gait = 'walk'
        
    def generate_walk_pattern(self, speed, direction):
        """Generate walking pattern at specified speed and direction"""
        # For this example, we'll use the WalkEngine
        # In practice, this would generate more sophisticated patterns
        walk_engine = WalkEngine()
        return {
            'pattern_type': 'walk',
            'speed': speed,
            'direction': direction,
            'engine': walk_engine
        }
    
    def generate_walk_pattern(self, speed, direction):
        """Generate walking pattern at specified speed and direction"""
        # Calculate step parameters based on desired speed
        step_length = np.clip(speed * 0.6, 0.1, 0.5)  # Step length proportional to speed
        step_time = max(0.5, 1.0 - speed * 0.2)       # Faster steps for higher speeds
        
        # Create walk engine with calculated parameters
        walk_engine = WalkEngine(
            step_length=step_length,
            step_height=0.05,
            step_time=step_time
        )
        
        return {
            'pattern_type': 'walk',
            'speed': speed,
            'direction': direction,
            'engine': walk_engine,
            'step_length': step_length,
            'step_time': step_time
        }
    
    def generate_run_pattern(self, speed, direction):
        """Generate running pattern at specified speed and direction"""
        # Running has different parameters than walking
        # - Higher step frequency
        # - Greater step length
        # - Aerial phase
        
        # For simplicity in this example
        step_length = speed * 1.0  # Longer steps for running
        step_time = max(0.2, 0.8 - speed * 0.3)  # Faster steps
        flight_time_ratio = 0.2  # 20% of step cycle in flight phase
        
        return {
            'pattern_type': 'run',
            'speed': speed,
            'direction': direction,
            'step_length': step_length,
            'step_time': step_time,
            'flight_time_ratio': flight_time_ratio
        }
    
    def generate_turn_pattern(self, angle, direction):
        """Generate turning pattern"""
        # Turning requires different foot placement and CoM movement
        # This is simplified for this example
        
        return {
            'pattern_type': 'turn',
            'angle': angle,
            'direction': direction,  # 'left' or 'right'
            'step_asymmetry': abs(angle) * 0.1  # More asymmetric for larger turns
        }
    
    def generate_climb_pattern(self, step_height):
        """Generate stair climbing pattern"""
        # Climbing requires extra vertical movement
        step_length = 0.2  # Shorter steps for stability
        step_height = max(0.15, step_height)  # Minimum step height
        
        return {
            'pattern_type': 'climb',
            'step_height': step_height,
            'step_length': step_length,
            'vertical_clearance': step_height + 0.05  # Extra clearance
        }

class GaitAdaptationController:
    def __init__(self, locomotion_generator, terrain_classifier):
        self.generator = locomotion_generator
        self.terrain_classifier = terrain_classifier
        self.current_pattern = None
        
    def adapt_gait_to_terrain(self, terrain_type):
        """Adapt gait pattern based on terrain type"""
        if terrain_type == 'flat':
            # Use normal walking gait
            pattern = self.generator.generate_walk_pattern(speed=0.5, direction='forward')
        elif terrain_type == 'rough':
            # Use more cautious gait with smaller steps
            pattern = self.generator.generate_walk_pattern(speed=0.3, direction='forward')
            pattern['step_length'] *= 0.7  # Shorter steps on rough terrain
            pattern['step_height'] += 0.02  # Higher foot clearance
        elif terrain_type == 'stairs':
            # Use climbing pattern
            pattern = self.generator.generate_climb_pattern(step_height=0.17)  # Standard stair height
        elif terrain_type == 'narrow':
            # Use gait with feet closer together
            pattern = self.generator.generate_walk_pattern(speed=0.4, direction='forward')
            # Adjust foot placement to be more centered
        else:
            # Default to normal walking
            pattern = self.generator.generate_walk_pattern(speed=0.5, direction='forward')
        
        self.current_pattern = pattern
        return pattern
    
    def adapt_gait_to_disturbance(self, disturbance_type, magnitude):
        """Adapt gait in response to external disturbances"""
        if not self.current_pattern:
            return
        
        if disturbance_type == 'push':
            # If pushed, widen stance and reduce speed
            self.current_pattern['step_length'] *= 0.8
            self.current_pattern['speed'] = max(0.1, self.current_pattern['speed'] * 0.7)
        elif disturbance_type == 'slip':
            # If slipping, increase ground clearance and slow down
            self.current_pattern['step_height'] += 0.02
            self.current_pattern['speed'] = max(0.1, self.current_pattern['speed'] * 0.5)
        
        return self.current_pattern
```

## 🎛️ 11.3 Whole-Body Control 🎛️

### 🎛️ 11.3.1 Task-Priority Based Control 🎛️

Whole-body control coordinates multiple tasks with different priorities to achieve complex behaviors:

```python
import numpy as np
from scipy.linalg import block_diag

class WholeBodyController:
    def __init__(self, robot_model, num_joints):
        self.model = robot_model
        self.n_joints = num_joints
        
        # Control hierarchy
        self.tasks = []
        self.priorities = []
        
    def add_task(self, task_jacobian, task_error, priority, weight=1.0):
        """
        Add a control task to the hierarchy
        task_jacobian: Jacobian matrix for the task (3xN or 6xN for spatial task)
        task_error: Error vector for the task (3x1 or 6x1)
        priority: Priority level (0 = highest priority, higher numbers = lower priority)
        weight: Weight for this task within its priority level
        """
        task = {
            'jacobian': task_jacobian,
            'error': task_error,
            'priority': priority,
            'weight': weight
        }
        
        self.tasks.append(task)
        self.priorities.append(priority)
        
        # Keep tasks sorted by priority
        sorted_indices = sorted(range(len(self.priorities)), 
                                key=lambda i: self.priorities[i])
        
        self.tasks = [self.tasks[i] for i in sorted_indices]
        self.priorities = [self.priorities[i] for i in sorted_indices]
    
    def compute_command(self):
        """
        Compute joint velocity commands using task-priority based control
        Based on the stack of tasks (Saab et al. approach)
        """
        # Initialize nullspace projection matrix as identity
        N_current = np.eye(self.n_joints)
        joint_velocity = np.zeros(self.n_joints)
        
        # Process tasks in priority order
        for task in self.tasks:
            # Project task jacobian onto current nullspace
            A_proj = task['jacobian'] @ N_current
            
            # Compute damped pseudo-inverse
            # J# = W^-1 * J^T * (J * W^-1 * J^T + λ^2 * I)^-1
            # For simplicity, using standard damped inverse
            damping = 1e-4
            J_damped_inv = np.linalg.pinv(
                A_proj @ A_proj.T + damping * np.eye(A_proj.shape[0])
            ) @ A_proj
            
            # Compute desired task velocity
            task_vel = task['error'] * task['weight']  # Proportional control
            
            # Compute joint velocity contribution
            delta_q = J_damped_inv @ task_vel
            
            # Add to total velocity (projected onto current nullspace)
            joint_velocity += N_current @ delta_q
            
            # Update nullspace projection matrix
            # N_new = N_current * (I - J_damped_inv * A_proj)
            N_current = N_current @ (np.eye(self.n_joints) - 
                                   np.linalg.pinv(A_proj, rcond=1e-4) @ A_proj)
        
        return joint_velocity
    
    def compute_com_balance_task(self, desired_com_pos, current_com_pos, 
                                 com_jacobian):
        """Create a center of mass balance task"""
        com_error = desired_com_pos - current_com_pos
        
        # Add this task with high priority
        self.add_task(com_jacobian, com_error, priority=0, weight=2.0)
    
    def compute_foot_placement_task(self, desired_foot_pos, current_foot_pos,
                                    foot_jacobian):
        """Create a foot placement task"""
        foot_error = desired_foot_pos - current_foot_pos
        
        # Add this task with medium priority
        self.add_task(foot_jacobian, foot_error, priority=1, weight=1.5)
    
    def compute_arm_posture_task(self, desired_joint_angles, current_joint_angles,
                                 arm_jacobian):
        """Create an arm posture task"""
        posture_error = desired_joint_angles - current_joint_angles
        
        # Add this task with low priority
        self.add_task(arm_jacobian, posture_error, priority=2, weight=0.5)

class OperationalSpaceController:
    def __init__(self, robot_model, num_joints):
        self.model = robot_model
        self.n_joints = num_joints
        
        # Mass matrix and other model parameters will be computed as needed
        self.M_inv = None  # Inverse of mass matrix
        self.C = None      # Coriolis and centrifugal forces
        self.G = None      # Gravity forces
        
    def compute_operational_force(self, task_jacobian, desired_accel, 
                                  current_pos, desired_pos, current_vel, 
                                  desired_vel, kp=100, kd=20):
        """
        Compute operational space force for a task
        """
        # Position error
        pos_error = desired_pos - current_pos
        
        # Velocity error  
        vel_error = desired_vel - current_vel
        
        # Desired acceleration in operational space
        op_acc_desired = (kp * pos_error + 
                         kd * vel_error + 
                         desired_accel)
        
        # Operational space mass matrix
        # Lambda = (J * M^-1 * J^T)^-1
        if self.M_inv is None:
            # Compute inverse mass matrix (would come from robot dynamics)
            self.M_inv = np.eye(self.n_joints)  # Placeholder
        
        J = task_jacobian
        lambda_op = np.linalg.inv(J @ self.M_inv @ J.T)
        
        # Operational space force
        F_op = lambda_op @ op_acc_desired
        
        # Convert to joint space force
        tau = J.T @ F_op
        
        return tau
    
    def compute_walking_controller(self, left_foot_pos, right_foot_pos, 
                                   com_pos, com_vel, dt):
        """
        Compute control forces for walking
        """
        # Define desired trajectories
        # This would come from gait planner
        
        # Compute ZMP-based balance control
        zmp_desired = self.compute_zmp_reference(left_foot_pos, right_foot_pos)
        zmp_current = self.compute_current_zmp(com_pos, com_vel)
        
        # Balance control
        zmp_error = zmp_desired - zmp_current[:2]  # Only X,Y components
        
        # Convert balance error to CoM acceleration command
        com_acc_cmd = self.balance_control(zmp_error)
        
        # Compute operational forces for each foot and CoM
        left_foot_jac = self.compute_jacobian('left_foot')
        right_foot_jac = self.compute_jacobian('right_foot')
        com_jac = self.compute_jacobian('com')
        
        # Left foot: try to maintain contact or move to next position
        left_foot_tau = self.compute_operational_force(
            left_foot_jac, 
            desired_accel=np.zeros(3),  # Keep at current position
            current_pos=left_foot_pos[:3], 
            desired_pos=left_foot_pos[:3],
            current_vel=np.zeros(3),  # Assume at rest for simplicity
            desired_vel=np.zeros(3),
            kp=500, kd=100
        )
        
        # CoM: track balance trajectory
        com_tau = self.compute_operational_force(
            com_jac,
            desired_accel=com_acc_cmd,
            current_pos=com_pos,
            desired_pos=com_pos + com_vel * dt + 0.5 * com_acc_cmd * dt**2,
            current_vel=com_vel,
            desired_vel=com_vel + com_acc_cmd * dt,
            kp=100, kd=20
        )
        
        return left_foot_tau + com_tau  # Combine torques
    
    def compute_jacobian(self, link_name):
        """Compute Jacobian matrix for a given link"""
        # This would interface with robot dynamics library
        # For simplicity, return a placeholder
        if link_name == 'left_foot':
            return np.random.rand(6, self.n_joints)  # Placeholder (6 for spatial, N for joints)
        elif link_name == 'right_foot':
            return np.random.rand(6, self.n_joints)
        elif link_name == 'com':
            return np.random.rand(3, self.n_joints)  # 3 for CoM position
        else:
            return np.zeros((6, self.n_joints))
    
    def compute_zmp_reference(self, left_foot_pos, right_foot_pos):
        """Compute reference ZMP position based on foot positions"""
        # For double support, ZMP is between feet
        # For single support, ZMP is under stance foot
        support_center = (left_foot_pos + right_foot_pos) / 2
        return support_center[:2]  # X,Y of support polygon center
    
    def compute_current_zmp(self, com_pos, com_vel):
        """Compute current ZMP from CoM state"""
        # ZMP_x = CoM_x - (CoM_z - h) / g * CoM_acc_x
        # We'll approximate acceleration from velocity
        g = 9.81  # Gravity constant
        h = com_pos[2]  # Approximate CoM height
        
        # For simplicity, using finite difference for acceleration
        # In practice, this would come from IMU or state estimation
        zmp_x = com_pos[0]  # Rough approximation
        zmp_y = com_pos[1]
        
        return np.array([zmp_x, zmp_y, 0.0])
    
    def balance_control(self, zmp_error):
        """Compute CoM acceleration command for balance"""
        # Simple PD controller for ZMP error
        kp = 50.0
        kd = 10.0
        
        # Convert ZMP error to CoM acceleration command
        com_acc_cmd = kp * zmp_error  # Proportional to error
        
        return np.array([com_acc_cmd[0], com_acc_cmd[1], 0.0])
```

### 🎛️ 11.3.2 Model Predictive Control for Humanoids 🎛️

Model predictive control (MPC) is particularly valuable for humanoid robots due to its ability to handle constraints:

```python
import numpy as np
from scipy.optimize import minimize
import cvxpy as cp

class MPCBalanceController:
    def __init__(self, prediction_horizon=20, dt=0.01, com_height=0.8):
        self.N = prediction_horizon  # Number of prediction steps
        self.dt = dt                 # Time step
        self.h = com_height          # CoM height
        
        # System matrices for inverted pendulum model
        # CoM dynamics: x(k+1) = A*x(k) + B*u(k)
        # where x = [px, py, vx, vy] (position and velocity of CoM)
        # and u = [ux, uy] (CoM acceleration)
        self.A = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        self.B = np.array([
            [0.5*dt**2, 0],
            [0, 0.5*dt**2],
            [dt, 0],
            [0, dt]
        ])
    
    def compute_optimal_control(self, current_state, reference_trajectory, 
                                support_feet_positions):
        """
        Compute optimal CoM trajectory using MPC
        current_state: [px, py, vx, vy] - current CoM state
        reference_trajectory: shape (N, 4) - desired CoM states over horizon
        support_feet_positions: list of [x, y] positions for each time step
        """
        N = self.N
        
        # Decision variables: CoM states and accelerations over the horizon
        X = cp.Variable((N+1, 4))  # State trajectory [px, py, vx, vy]
        U = cp.Variable((N, 2))    # Control trajectory [ux, uy] (acceleration)
        
        # Objective function: track reference trajectory with minimal control effort
        cost = 0
        for k in range(N):
            # State tracking cost
            cost += cp.sum_squares(X[k, :] - reference_trajectory[k, :])
            
            # Control effort cost
            cost += 0.1 * cp.sum_squares(U[k, :])
        
        # Add terminal cost
        cost += 10 * cp.sum_squares(X[N, :] - reference_trajectory[-1, :])
        
        # Constraints
        constraints = []
        
        # Initial state constraint
        constraints.append(X[0, :] == current_state)
        
        # System dynamics
        for k in range(N):
            constraints.append(X[k+1, :] == self.A @ X[k, :] + self.B @ U[k, :])
        
        # ZMP stability constraints
        # ZMP must be inside support polygon at each time step
        g = 9.81
        for k in range(N):
            # Compute ZMP from CoM state: zmp = [px, py] - h/g * [vx, vy]
            zmp_x = X[k, 0] - (self.h / g) * X[k, 2]
            zmp_y = X[k, 1] - (self.h / g) * X[k, 3]
            
            # For this example, assume rectangular support polygon around foot positions
            # In practice, this would check if ZMP is inside convex hull of contact points
            foot_pos = support_feet_positions[min(k, len(support_feet_positions)-1)]
            
            # Assume support polygon of 10cm x 20cm around foot center
            constraints.append(zmp_x >= foot_pos[0] - 0.05)
            constraints.append(zmp_x <= foot_pos[0] + 0.05)
            constraints.append(zmp_y >= foot_pos[1] - 0.10)
            constraints.append(zmp_y <= foot_pos[1] + 0.10)
        
        # Control limits (maximum CoM acceleration)
        for k in range(N):
            constraints.append(cp.norm(U[k, :], 'inf') <= 5.0)  # Max 5 m/s² acceleration
        
        # Solve optimization problem
        problem = cp.Problem(cp.Minimize(cost), constraints)
        
        try:
            problem.solve(solver=cp.ECOS, verbose=False)
            
            if problem.status not in ["infeasible", "unbounded"]:
                # Return first control command and predicted trajectory
                return U[0, :].value, X.value
            else:
                print(f"MPC problem status: {problem.status}")
                return np.zeros(2), np.tile(current_state, (N+1, 1))
        except Exception as e:
            print(f"MPC optimization error: {e}")
            return np.zeros(2), np.tile(current_state, (N+1, 1))

class PredictiveWalkingController:
    def __init__(self, mpc_controller, walk_engine):
        self.mpc = mpc_controller
        self.walk_engine = walk_engine
        self.footstep_planner = FootstepPlanner()
        
    def plan_footsteps(self, walk_command):
        """Plan future footsteps based on walking command"""
        # This would use path planning and gait analysis
        # For this example, we'll generate a simple sequence
        
        current_pos = np.array([0.0, 0.0])  # Robot's current position
        steps = []
        
        # Generate sequence of footsteps
        step_length = self.walk_engine.step_length
        n_steps = int(walk_command['distance'] / step_length) if walk_command.get('distance') else 5
        
        for i in range(n_steps):
            # Alternate feet in a walk pattern
            x = current_pos[0] + step_length * (i + 1)
            y = current_pos[1] + (-1)**i * 0.15  # Alternate left/right
            steps.append(np.array([x, y]))
        
        return steps
    
    def compute_walking_control(self, current_state, walk_command):
        """Compute walking control using MPC"""
        # Plan footsteps
        footsteps = self.plan_footsteps(walk_command)
        
        # Generate reference CoM trajectory to follow footsteps
        ref_trajectory = self.generate_com_reference(footsteps, current_state)
        
        # Get support foot positions for each time step
        support_positions = self.interpolate_support_polygon(footsteps)
        
        # Solve MPC problem
        optimal_control, predicted_trajectory = self.mpc.compute_optimal_control(
            current_state, 
            ref_trajectory,
            support_positions
        )
        
        return optimal_control, predicted_trajectory, footsteps
    
    def generate_com_reference(self, footsteps, current_state):
        """Generate CoM reference trajectory based on footsteps"""
        N = self.mpc.N
        dt = self.mpc.dt
        
        # For simplicity, generate a reference that moves between foot positions
        ref_trajectory = np.zeros((N, 4))  # [px, py, vx, vy]
        
        # Initialize with current state
        ref_trajectory[0, :2] = current_state[:2]  # Position
        ref_trajectory[0, 2:] = current_state[2:]  # Velocity
        
        # Generate smooth transition between footstep locations
        if len(footsteps) > 0:
            for k in range(1, N):
                # Move toward the next relevant footstep
                # This is a simplified approach
                target_idx = min(int(k * len(footsteps) / N), len(footsteps) - 1)
                target_pos = footsteps[target_idx][:2]
                
                # Simple first-order tracking
                alpha = 0.1  # Smoothing factor
                ref_trajectory[k, :2] = (1 - alpha) * ref_trajectory[k-1, :2] + alpha * target_pos
                
                # Compute approximate velocity
                if k > 0:
                    ref_trajectory[k, 2:] = (ref_trajectory[k, :2] - ref_trajectory[k-1, :2]) / dt
        
        return ref_trajectory
    
    def interpolate_support_polygon(self, footsteps):
        """Interpolate support polygon positions over the prediction horizon"""
        N = self.mpc.N
        support_positions = []
        
        # For this example, alternate support between feet
        for k in range(N):
            if k < len(footsteps):
                # Use the appropriate foot position based on gait phase
                if k % 2 == 0:
                    # Right foot is stance foot
                    if k + 1 < len(footsteps):
                        support_pos = footsteps[k + 1][:2]
                    else:
                        support_pos = footsteps[k][:2]
                else:
                    # Left foot is stance foot
                    support_pos = footsteps[k][:2]
            else:
                # Use last foot position
                support_pos = footsteps[-1][:2] if footsteps else np.array([0.0, 0.0])
            
            support_positions.append(support_pos)
        
        return support_positions

class FootstepPlanner:
    def __init__(self):
        self.step_width = 0.3  # Default distance between feet
        self.max_step_length = 0.5  # Maximum step length
        
    def plan_to_target(self, current_pos, target_pos, terrain_map=None):
        """Plan footsteps from current position to target position"""
        # Calculate required steps
        displacement = target_pos - current_pos
        distance = np.linalg.norm(displacement)
        
        # Calculate number of steps needed
        n_steps = max(1, int(np.ceil(distance / self.max_step_length)))
        
        # Generate evenly spaced steps
        footsteps = []
        for i in range(1, n_steps + 1):
            ratio = i / n_steps
            step_pos = current_pos + ratio * displacement
            
            # Alternate foot placement (left-right-left...)
            if i % 2 == 1:
                # Offset in Y for first foot
                step_pos[1] += self.step_width / 2
            else:
                # Offset in Y for second foot
                step_pos[1] -= self.step_width / 2
                
            footsteps.append(step_pos)
        
        return footsteps
```

## 🎯 11.4 Learning-Based Control 🎯

### 🎯 11.4.1 Imitation Learning for Humanoid Movements 🎯

Humanoid robots can learn complex movements through imitation of human demonstrations:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class HumanoidImitationLearner(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(HumanoidImitationLearner, self).__init__()
        
        # State encoder
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Action decoder
        self.action_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()  # Actions are normalized to [-1, 1]
        )

    def forward(self, state):
        encoded_state = self.state_encoder(state)
        action = self.action_decoder(encoded_state)
        return action

class HumanoidImitationController:
    def __init__(self, state_dim, action_dim, learning_rate=1e-4):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.learner = HumanoidImitationLearner(state_dim, action_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.learner.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        
        # Storage for demonstration data
        self.demonstration_states = []
        self.demonstration_actions = []
        
    def add_demonstration(self, states, actions):
        """Add a demonstration trajectory to the dataset"""
        self.demonstration_states.extend(states)
        self.demonstration_actions.extend(actions)
    
    def train(self, epochs=100, batch_size=64):
        """Train the imitation learning model"""
        if len(self.demonstration_states) == 0:
            print("No demonstrations available for training")
            return
        
        # Convert to tensors
        states_tensor = torch.FloatTensor(self.demonstration_states).to(self.device)
        actions_tensor = torch.FloatTensor(self.demonstration_actions).to(self.device)
        
        dataset = torch.utils.data.TensorDataset(states_tensor, actions_tensor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        self.learner.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch_states, batch_actions in dataloader:
                self.optimizer.zero_grad()
                
                predicted_actions = self.learner(batch_states)
                loss = self.criterion(predicted_actions, batch_actions)
                
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
            
            if epoch % 20 == 0:
                print(f"Epoch {epoch}, Loss: {total_loss/len(dataloader):.4f}")
    
    def get_action(self, state):
        """Get action for a given state"""
        self.learner.eval()
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action = self.learner(state_tensor)
            return action.cpu().numpy().squeeze()

# ℹ️ Demonstration collection system ℹ️
class DemonstrationCollector:
    def __init__(self, robot_interface, human_motion_capture):
        self.robot = robot_interface
        self.motion_capture = human_motion_capture
        self.current_demonstration = {'states': [], 'actions': []}
        
    def start_demonstration(self):
        """Start collecting a new demonstration"""
        self.current_demonstration = {'states': [], 'actions': []}
        print("Started collecting demonstration")
    
    def record_step(self, robot_state, human_motion_state):
        """Record a single step of the demonstration"""
        # Map human motion to robot actions
        robot_action = self.map_human_to_robot(human_motion_state)
        
        self.current_demonstration['states'].append(robot_state)
        self.current_demonstration['actions'].append(robot_action)
    
    def map_human_to_robot(self, human_pose):
        """Map human motion capture data to robot joint commands"""
        # This would involve complex kinematic mapping
        # taking into account differences in kinematic structure
        # For this example, we'll use a simplified mapping
        
        # Assume human pose contains joint angles for a human skeleton
        # and we need to map them to robot joint angles
        n_robot_joints = 30  # Example number of humanoid joints
        
        # Simplified mapping (in practice this would be more sophisticated)
        robot_action = np.zeros(n_robot_joints)
        
        # Map common joints: hips, knees, ankles, shoulders, elbows, wrists
        # This is highly simplified - real implementations would use IK to match poses
        for i in range(min(len(human_pose), n_robot_joints)):
            robot_action[i] = human_pose[i] if i < len(human_pose) else 0.0
        
        return robot_action
    
    def end_demonstration(self):
        """End the current demonstration and return it"""
        demo = self.current_demonstration.copy()
        self.current_demonstration = {'states': [], 'actions': []}
        print(f"Completed demonstration with {len(demo['states'])} steps")
        return demo
```

### 🎯 11.4.2 Reinforcement Learning for Humanoid Control 🎯

Reinforcement learning can be used to learn complex humanoid behaviors:

```python
import torch
import torch.nn as nn
import numpy as np
import random
from collections import deque

class HumanoidActor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(HumanoidActor, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )
        
        # Learnable standard deviation for exploration
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, state):
        action_mean = self.network(state)
        action_std = torch.exp(self.log_std)
        return action_mean, action_std

class HumanoidCritic(nn.Module):
    def __init__(self, state_dim, hidden_dim=256):
        super(HumanoidCritic, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state):
        return self.network(state)

class HumanoidPPOAgent:
    def __init__(self, state_dim, action_dim, lr_actor=3e-4, lr_critic=1e-3, 
                 gamma=0.99, clip_epsilon=0.2, epochs=10):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Networks
        self.actor = HumanoidActor(state_dim, action_dim).to(self.device)
        self.critic = HumanoidCritic(state_dim).to(self.device)
        
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)
        
        # Hyperparameters
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.epochs = epochs
        
        # For updating old policy
        self.old_actor = HumanoidActor(state_dim, action_dim).to(self.device)
        self.update_old_policy()
        
    def update_old_policy(self):
        """Update old policy network with current policy parameters"""
        self.old_actor.load_state_dict(self.actor.state_dict())
    
    def select_action(self, state):
        """Select action using current policy"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action_mean, action_std = self.old_actor(state_tensor)
            
            # Sample action from normal distribution
            action_distribution = torch.distributions.Normal(action_mean, action_std)
            action = action_distribution.sample()
            log_prob = action_distribution.log_prob(action).sum(dim=-1)
        
        return action.cpu().numpy()[0], log_prob.cpu().numpy()[0]
    
    def evaluate(self, state, action):
        """Evaluate state-action pairs"""
        state_tensor = torch.FloatTensor(state).to(self.device)
        action_tensor = torch.FloatTensor(action).to(self.device)
        
        action_mean, action_std = self.actor(state_tensor)
        action_distribution = torch.distributions.Normal(action_mean, action_std)
        log_prob = action_distribution.log_prob(action_tensor).sum(dim=-1, keepdim=True)
        
        entropy = action_distribution.entropy().sum(dim=-1, keepdim=True)
        state_value = self.critic(state_tensor)
        
        return log_prob, entropy, state_value

    def update(self, states, actions, rewards, dones, log_probs, values):
        """Update policy using PPO"""
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)
        old_log_probs = torch.FloatTensor(log_probs).to(self.device)
        
        # Compute discounted rewards (returns)
        returns = []
        R = 0
        for reward, done in zip(reversed(rewards), reversed(dones)):
            if done:
                R = 0
            R = reward + self.gamma * R
            returns.insert(0, R)
        returns = torch.FloatTensor(returns).to(self.device).unsqueeze(1)
        
        # Advantages
        advantages = returns - values
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Update policy for several epochs
        for _ in range(self.epochs):
            # Get new probabilities and values
            new_log_probs, entropy, new_values = self.evaluate(states, actions)
            
            # Compute ratios
            ratios = torch.exp(new_log_probs - old_log_probs)
            
            # Compute PPO surrogates
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            
            # Critic loss
            critic_loss = F.mse_loss(new_values, returns)
            
            # Total loss
            total_loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy.mean()
            
            # Update networks
            self.actor_optimizer.zero_grad()
            total_loss.backward()
            self.actor_optimizer.step()
            
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()
        
        # Update old policy
        self.update_old_policy()

# ℹ️ Humanoid Environment for RL ℹ️
class HumanoidEnvironment:
    def __init__(self, robot_model):
        self.robot = robot_model
        self.action_space_dim = robot_model.n_joints
        self.observation_space_dim = 100  # Simplified state space
        
        # Episode tracking
        self.step_count = 0
        self.max_steps = 1000
        
        # Reward components
        self.reward_weights = {
            'forward_progress': 1.0,
            'balance': 2.0,
            'energy_efficiency': 0.1,
            'upright': 1.5
        }
    
    def reset(self):
        """Reset the environment"""
        self.step_count = 0
        
        # Reset robot to standing position
        initial_state = self.robot.get_initial_state()
        
        return initial_state
    
    def step(self, action):
        """Take a step in the environment"""
        self.step_count += 1
        
        # Apply action to robot
        self.robot.apply_action(action)
        
        # Get new state
        new_state = self.robot.get_state()
        
        # Calculate reward
        reward = self.calculate_reward(new_state, action)
        
        # Check if episode is done
        done = (self.step_count >= self.max_steps or 
                self.check_fall_condition(new_state))
        
        # Additional info (for debugging)
        info = {
            'step': self.step_count,
            'com_height': new_state[2],  # Simplified
            'is_fallen': self.check_fall_condition(new_state)
        }
        
        return new_state, reward, done, info
    
    def calculate_reward(self, state, action):
        """Calculate reward based on state and action"""
        reward = 0.0
        
        # Forward progress reward (simplified)
        com_velocity_x = state[3]  # Simplified - assume 4th element is CoM velocity X
        reward += self.reward_weights['forward_progress'] * max(0, com_velocity_x)
        
        # Balance reward - penalize deviation from upright position
        robot_orientation = state[6:9]  # Simplified - assume orientation is elements 6-8
        upright_penalty = np.sum(robot_orientation[0:2]**2)  # Penalize roll and pitch
        reward -= self.reward_weights['balance'] * upright_penalty
        
        # Energy efficiency - penalize large actions
        action_penalty = np.sum(action**2)
        reward -= self.reward_weights['energy_efficiency'] * action_penalty
        
        # Upright bonus - reward for maintaining upright position
        if abs(state[2] - 0.8) < 0.1:  # If CoM height is close to nominal
            reward += self.reward_weights['upright']
        
        return reward
    
    def check_fall_condition(self, state):
        """Check if robot has fallen"""
        # Simplified fall detection - check if CoM is too low or tilted too much
        com_height = state[2]  # Simplified assumption
        orientation = state[6:9]  # Simplified assumption
        
        # Fall if CoM is too low or tilted beyond threshold
        fallen = (com_height < 0.3 or 
                 abs(orientation[0]) > 0.5 or  # Roll
                 abs(orientation[1]) > 0.5)    # Pitch
        
        return fallen

# 🤖 Training loop for humanoid RL 🤖
def train_humanoid_rl_agent():
    """Training loop for humanoid reinforcement learning"""
    # Initialize environment and agent
    robot_model = SimpleHumanoidModel()  # Placeholder
    env = HumanoidEnvironment(robot_model)
    
    agent = HumanoidPPOAgent(
        state_dim=env.observation_space_dim,
        action_dim=env.action_space_dim,
        lr_actor=3e-4,
        lr_critic=1e-3
    )
    
    # Training parameters
    max_episodes = 1000
    update_timestep = 2000  # Update policy every N timesteps
    
    # Storage for trajectories
    states = []
    actions = []
    log_probs = []
    rewards = []
    is_terminals = []
    
    # Track performance
    running_reward = 0
    avg_length = 0
    
    timestep = 0
    
    for episode in range(max_episodes):
        state = env.reset()
        episode_reward = 0
        time_steps = 0
        
        for _ in range(update_timestep):
            timestep += 1
            time_steps += 1
            
            # Select action
            action, log_prob = agent.select_action(state)
            
            # Take action in environment
            new_state, reward, done, info = env.step(action)
            
            # Store data
            states.append(state)
            actions.append(action)
            log_probs.append(log_prob)
            rewards.append(reward)
            is_terminals.append(done)
            
            state = new_state
            episode_reward += reward
            
            if done:
                break
        
        # Update running reward
        running_reward += episode_reward
        avg_length += time_steps
        
        # Update policy if enough samples collected
        if timestep % update_timestep == 0:
            # Convert to numpy arrays for PPO update
            np_states = torch.FloatTensor(states).detach().numpy()
            np_actions = torch.FloatTensor(actions).detach().numpy()
            np_rewards = torch.FloatTensor(rewards).detach().numpy()
            np_is_terminals = torch.BoolTensor(is_terminals).detach().numpy()
            
            # Get old values for PPO update
            with torch.no_grad():
                old_values = agent.critic(torch.FloatTensor(states)).detach().numpy()
            
            # Update agent
            agent.update(np_states, np_actions, np_rewards, np_is_terminals, log_probs, old_values)
            
            # Clear storage
            states = []
            actions = []
            log_probs = []
            rewards = []
            is_terminals = []
        
        # Print average reward every 10 episodes
        if episode % 10 == 0:
            avg_reward = running_reward / 10
            avg_length = int(avg_length / 10)
            print(f'Episode {episode}, avg length: {avg_length}, reward: {avg_reward:.2f}')
            running_reward = 0
            avg_length = 0

# 🤖 Placeholder for simple humanoid model (would be implemented with real robot/physics) 🤖
class SimpleHumanoidModel:
    def __init__(self):
        self.n_joints = 30  # Example humanoid joint count
        self.mass = 75  # kg
        self.height = 1.8  # meters
    
    def get_initial_state(self):
        """Get initial state of the robot"""
        # Return a 100-dimensional state vector (simplified)
        return np.zeros(100)
    
    def get_state(self):
        """Get current state of the robot"""
        # Return current state (simplified)
        return np.random.randn(100)
    
    def apply_action(self, action):
        """Apply action to the robot"""
        # In real implementation, send commands to robot
        pass
```

## 🎛️ 11.5 Control Validation and Testing 🎛️

### 🎮 11.5.1 Simulation-Based Validation 🎮

Validating humanoid control systems in simulation before real-world deployment:

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

class HumanoidSimulator:
    def __init__(self, physics_engine='pybullet'):
        self.engine = physics_engine
        self.timestep = 0.001  # 1kHz simulation
        self.robot_id = None
        
        # Robot properties
        self.mass = 75.0  # kg
        self.height = 1.7  # meters
        self.base_position = [0, 0, 0.8]  # Standing position
        
        # Simulation state
        self.sim_step = 0
        self.contact_points = []
        
    def reset(self):
        """Reset simulation to initial state"""
        self.sim_step = 0
        self.contact_points = []
        
        # Initialize robot at standing position
        # Implementation would depend on physics engine
        pass
    
    def step(self, joint_commands):
        """Step simulation with joint commands"""
        # Apply joint commands to robot
        # Update physics simulation
        # Check for contacts, collisions, etc.
        
        self.sim_step += 1
        
        # Return updated robot state
        # This would include joint positions, velocities, IMU data, etc.
        return self.get_robot_state()
    
    def get_robot_state(self):
        """Get current robot state"""
        # Return comprehensive robot state for controller
        state = {
            'joint_positions': np.random.randn(30),  # Placeholder
            'joint_velocities': np.random.randn(30),  # Placeholder
            'imu_data': {
                'orientation': np.random.randn(4),  # Quaternion
                'angular_velocity': np.random.randn(3),
                'linear_acceleration': np.random.randn(3)
            },
            'ft_sensors': np.random.randn(6),  # 6-axis force/torque (placeholder)
            'com_state': np.random.randn(6),  # Position and velocity of CoM
            'contact_states': self.get_contact_states()
        }
        return state
    
    def get_contact_states(self):
        """Get current contact states"""
        # Return information about which parts of the robot are in contact
        return {
            'left_foot': np.random.random() > 0.5,  # Whether in contact
            'right_foot': np.random.random() > 0.5,
            'contact_forces': np.random.randn(2, 6)  # Forces at each foot
        }
    
    def add_disturbance(self, force, position, duration=100):
        """Add an external disturbance to the robot"""
        # Apply external force for specified duration
        # This is useful for testing robustness
        pass

class ControlValidator:
    def __init__(self, simulator):
        self.simulator = simulator
        self.test_results = {}
        
    def run_stability_test(self, controller, test_duration=10.0):
        """Test controller stability over time"""
        self.simulator.reset()
        
        initial_state = self.simulator.get_robot_state()
        states = []
        commands = []
        
        n_steps = int(test_duration / self.simulator.timestep)
        
        for i in range(n_steps):
            current_state = self.simulator.get_robot_state()
            states.append(current_state)
            
            # Get control command
            command = controller.compute_command(current_state)
            commands.append(command)
            
            # Step simulation
            next_state = self.simulator.step(command)
            
            # Check for failure conditions
            if self.check_failure_conditions(next_state):
                print(f"Stability test failed at step {i}")
                break
        
        # Analyze results
        stability_metrics = self.analyze_stability(states, commands)
        
        self.test_results['stability'] = stability_metrics
        return stability_metrics
    
    def run_robustness_test(self, controller, disturbances=None):
        """Test controller robustness to disturbances"""
        if disturbances is None:
            disturbances = [
                {'force': [50, 0, 0], 'position': [0, 0, 0.5], 'duration': 100},
                {'force': [0, -30, 0], 'position': [0.2, 0, 0.5], 'duration': 100},
                {'force': [0, 0, -100], 'position': [0, 0, 0.7], 'duration': 50}
            ]
        
        self.simulator.reset()
        initial_state = self.simulator.get_robot_state()
        
        for i, dist in enumerate(disturbances):
            print(f"Applying disturbance {i+1}: {dist['force']}")
            
            # Apply disturbance
            self.simulator.add_disturbance(
                dist['force'], 
                dist['position'], 
                dist['duration']
            )
            
            # Run controller for a while after disturbance
            for j in range(500):  # 0.5 seconds at 1kHz
                current_state = self.simulator.get_robot_state()
                command = controller.compute_command(current_state)
                self.simulator.step(command)
                
                # Check recovery
                if self.check_recovery(current_state, initial_state):
                    print(f"Recovered from disturbance {i+1} at step {j}")
                    break
        
        # Analyze robustness metrics
        robustness_metrics = self.analyze_robustness()
        self.test_results['robustness'] = robustness_metrics
        return robustness_metrics
    
    def check_failure_conditions(self, state):
        """Check if robot has failed (fallen, etc.)"""
        com_height = state['com_state'][2]  # Z component of CoM
        robot_orientation = state['imu_data']['orientation']
        
        # Convert quaternion to roll/pitch angles
        r = R.from_quat(robot_orientation)
        euler = r.as_euler('xyz')
        
        # Fail if fallen (CoM too low or too tilted)
        fallen = (com_height < 0.3 or 
                 abs(euler[0]) > 0.5 or  # Roll
                 abs(euler[1]) > 0.5)    # Pitch
        
        return fallen
    
    def check_recovery(self, current_state, initial_state, tolerance=0.1):
        """Check if robot has recovered to stable state"""
        # Check if CoM position is close to initial
        com_diff = np.linalg.norm(
            current_state['com_state'][:2] - initial_state['com_state'][:2]
        )
        
        return com_diff < tolerance
    
    def analyze_stability(self, states, commands):
        """Analyze stability metrics"""
        # Calculate CoM stability metrics
        com_positions = np.array([s['com_state'][:2] for s in states])
        
        # Mean deviation from center
        mean_deviation = np.mean(np.linalg.norm(com_positions, axis=1))
        
        # Variance of CoM position
        com_variance = np.var(np.linalg.norm(com_positions, axis=1))
        
        # Calculate joint command smoothness
        commands = np.array(commands)
        command_derivatives = np.diff(commands, axis=0)
        command_smoothness = np.mean(np.abs(command_derivatives))
        
        return {
            'mean_com_deviation': mean_deviation,
            'com_variance': com_variance,
            'command_smoothness': command_smoothness,
            'max_com_drift': np.max(np.linalg.norm(com_positions, axis=1))
        }
    
    def analyze_robustness(self):
        """Analyze robustness metrics"""
        # This would analyze how well the controller handled disturbances
        # and recovered to stable behavior
        return {
            'recovery_time_avg': 0.5,  # Placeholder
            'max_disturbance_response': 0.2,  # Placeholder
            'stability_after_disturbance': True  # Placeholder
        }
    
    def visualize_test_results(self):
        """Visualize control validation results"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        if 'stability' in self.test_results:
            stability = self.test_results['stability']
            
            # Plot CoM stability over time (simplified)
            # In practice, you'd have time series data
            time = np.linspace(0, 10, 1000)
            com_x = np.random.randn(1000) * stability['mean_com_deviation'] / 2
            com_y = np.random.randn(1000) * stability['mean_com_deviation'] / 2
            
            axes[0, 0].plot(com_x, com_y)
            axes[0, 0].set_title('Center of Mass Trajectory')
            axes[0, 0].set_xlabel('X Position (m)')
            axes[0, 0].set_ylabel('Y Position (m)')
            axes[0, 0].grid(True)
            
            # Plot command smoothness
            axes[0, 1].plot(time, np.convolve(np.abs(com_x), np.ones(50)/50, mode='same'))
            axes[0, 1].set_title('Smoothed CoM Position X')
            axes[0, 1].set_xlabel('Time (s)')
            axes[0, 1].set_ylabel('Position (m)')
            
        if 'robustness' in self.test_results:
            # Robustness metrics plot
            metrics = list(self.test_results['robustness'].keys())
            values = list(self.test_results['robustness'].values())
            
            axes[1, 0].bar(metrics, values)
            axes[1, 0].set_title('Robustness Metrics')
            axes[1, 0].set_ylabel('Value')
            plt.setp(axes[1, 0].get_xticklabels(), rotation=45)
        
        # Balance margin visualization
        # Create a support polygon and show CoM position
        foot_positions = np.array([
            [0.1, 0.15],   # Left foot
            [0.1, -0.15],  # Right foot
            [-0.1, -0.15], # Right foot (back)
            [-0.1, 0.15]   # Left foot (back)
        ])
        
        axes[1, 1].fill(foot_positions[:, 0], foot_positions[:, 1], alpha=0.3, label='Support Polygon')
        axes[1, 1].plot(0, 0, 'ro', label='CoM (stable)')
        axes[1, 1].plot(0.15, 0.2, 'rx', markersize=10, label='CoM (unstable)')
        axes[1, 1].set_title('Balance Margin Visualization')
        axes[1, 1].set_xlabel('X Position (m)')
        axes[1, 1].set_ylabel('Y Position (m)')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.show()

# ℹ️ Example usage of validation framework ℹ️
def run_control_validation():
    """Run comprehensive control system validation"""
    print("Starting humanoid control validation...")
    
    # Initialize simulator
    simulator = HumanoidSimulator()
    
    # Create validator
    validator = ControlValidator(simulator)
    
    # For this example, we'll create a simple controller
    # In practice, this would be your actual controller
    class DummyController:
        def compute_command(self, state):
            # Return random commands for this example
            return np.random.randn(30)
    
    controller = DummyController()
    
    # Run stability test
    print("Running stability test...")
    stability_results = validator.run_stability_test(controller, test_duration=5.0)
    print(f"Stability metrics: {stability_results}")
    
    # Run robustness test
    print("Running robustness test...")
    robustness_results = validator.run_robustness_test(controller)
    print(f"Robustness metrics: {robustness_results}")
    
    # Visualize results
    print("Visualizing results...")
    validator.visualize_test_results()
    
    print("Control validation completed.")
```

## 🤖 11.6 Safety and Failsafe Systems 🤖

### ⚙️ 11.6.1 Emergency Response Systems ⚙️

Critical safety systems for humanoid robots:

```python
import time
import threading
from enum import Enum

class SafetyState(Enum):
    NORMAL = "normal"
    WARNING = "warning"
    EMERGENCY = "emergency"
    SHUTDOWN = "shutdown"

class EmergencyResponseSystem:
    def __init__(self, robot_controller):
        self.controller = robot_controller
        self.state = SafetyState.NORMAL
        self.emergency_threads = []
        
        # Safety thresholds
        self.safety_thresholds = {
            'joint_temp': 80.0,      # Celsius
            'current': 20.0,         # Amps
            'torque': 100.0,         # Nm
            'velocity': 5.0,         # rad/s
            'acceleration': 50.0,    # rad/s²
            'imu_angle': 0.5,        # Radians (about 28 degrees)
            'com_drift': 1.0,        # Meters from nominal
            'contact_force': 500.0    # Newtons
        }
        
        # Emergency responses
        self.emergency_responses = {
            'overheat': self._handle_overheat,
            'excessive_current': self._handle_excessive_current,
            'fall_detected': self._handle_fall,
            'high_torque': self._handle_high_torque,
            'imu_violation': self._handle_imu_violation
        }
        
    def start_monitoring(self):
        """Start safety monitoring in background threads"""
        # Start joint monitoring thread
        joint_monitor = threading.Thread(target=self._joint_monitor_loop)
        joint_monitor.daemon = True
        joint_monitor.start()
        self.emergency_threads.append(joint_monitor)
        
        # Start IMU monitoring thread
        imu_monitor = threading.Thread(target=self._imu_monitor_loop)
        imu_monitor.daemon = True
        imu_monitor.start()
        self.emergency_threads.append(imu_monitor)
        
        # Start contact force monitoring thread
        force_monitor = threading.Thread(target=self._force_monitor_loop)
        force_monitor.daemon = True
        force_monitor.start()
        self.emergency_threads.append(force_monitor)
        
        print("Emergency response system activated")
    
    def _joint_monitor_loop(self):
        """Monitor joint safety parameters"""
        while True:
            if self.state == SafetyState.SHUTDOWN:
                break
                
            # Get current joint states
            joint_states = self.controller.get_joint_states()
            
            for i, joint_state in enumerate(joint_states):
                # Check temperature
                if joint_state['temperature'] > self.safety_thresholds['joint_temp']:
                    self.trigger_emergency('overheat', f'Joint {i} temperature too high')
                
                # Check current
                if abs(joint_state['current']) > self.safety_thresholds['current']:
                    self.trigger_emergency('excessive_current', f'Joint {i} current too high')
                
                # Check torque
                if abs(joint_state['torque']) > self.safety_thresholds['torque']:
                    self.trigger_emergency('high_torque', f'Joint {i} torque too high')
                
                # Check velocity
                if abs(joint_state['velocity']) > self.safety_thresholds['velocity']:
                    self.trigger_emergency('high_velocity', f'Joint {i} velocity too high')
            
            time.sleep(0.01)  # Check every 10ms
    
    def _imu_monitor_loop(self):
        """Monitor IMU data for safety violations"""
        while True:
            if self.state == SafetyState.SHUTDOWN:
                break
                
            # Get IMU data
            imu_data = self.controller.get_imu_data()
            
            # Check orientation
            orientation = imu_data['orientation']
            if max(abs(orientation[:3])) > self.safety_thresholds['imu_angle']:
                self.trigger_emergency('imu_violation', f'Excessive tilt detected: {orientation[:3]}')
            
            # Check acceleration
            linear_acc = imu_data['linear_acceleration']
            if np.linalg.norm(linear_acc) > 50.0:  # Very high acceleration indicates impact
                self.trigger_emergency('imu_violation', f'High acceleration detected: {linear_acc}')
            
            time.sleep(0.01)
    
    def _force_monitor_loop(self):
        """Monitor contact forces"""
        while True:
            if self.state == SafetyState.SHUTDOWN:
                break
                
            # Get force/torque sensor data
            ft_data = self.controller.get_force_torque_data()
            
            # Check for excessive forces
            for i, force_read in enumerate(ft_data):
                if np.linalg.norm(force_read[:3]) > self.safety_thresholds['contact_force']:
                    self.trigger_emergency('high_force', f'High contact force on sensor {i}')
            
            time.sleep(0.01)
    
    def trigger_emergency(self, emergency_type, description):
        """Trigger emergency response"""
        print(f"EMERGENCY TRIGGERED: {emergency_type} - {description}")
        
        # Update safety state
        self.state = SafetyState.EMERGENCY
        
        # Execute emergency response
        if emergency_type in self.emergency_responses:
            self.emergency_responses[emergency_type]()
        
        # Log the emergency
        self._log_emergency(emergency_type, description)
    
    def _handle_overheat(self):
        """Handle overheat emergency"""
        print("Handling overheat emergency...")
        # Reduce power to joints
        # Activate cooling systems if available
        # Slow down movements
        self.controller.reduce_power(0.5)
    
    def _handle_excessive_current(self):
        """Handle excessive current emergency"""
        print("Handling excessive current emergency...")
        # Reduce torque commands
        # Check for mechanical obstructions
        self.controller.safety_stop()
    
    def _handle_fall(self):
        """Handle fall detection"""
        print("Handling fall emergency...")
        # Execute fall protection sequence
        # Minimize impact
        # Prepare for recovery
        self.controller.execute_fall_protection()
    
    def _handle_high_torque(self):
        """Handle high torque emergency"""
        print("Handling high torque emergency...")
        # Reduce torque commands
        # Check for collisions
        self.controller.safety_stop()
    
    def _handle_imu_violation(self):
        """Handle IMU safety violation"""
        print("Handling IMU violation emergency...")
        # Stop dynamic movements
        # Return to stable pose if possible
        self.controller.return_to_safe_pose()
    
    def _log_emergency(self, emergency_type, description):
        """Log emergency to safety system"""
        timestamp = time.time()
        log_entry = {
            'timestamp': timestamp,
            'type': emergency_type,
            'description': description,
            'state': self.state.value,
            'controller_state': self.controller.get_state()
        }
        
        # In practice, this would write to a safety log file
        print(f"Logged emergency: {log_entry}")
    
    def shutdown_safely(self):
        """Shut down the robot safely"""
        print("Initiating safe shutdown...")
        
        # Update state
        self.state = SafetyState.SHUTDOWN
        
        # Stop all movements
        self.controller.emergency_stop()
        
        # Power down systems safely
        self.controller.power_down()
        
        print("Robot safely shut down")

# ℹ️ Fall Protection System ℹ️
class FallProtectionSystem:
    def __init__(self, robot_controller):
        self.controller = robot_controller
        self.fall_threshold = 0.3  # Threshold for fall detection (CoM height)
        self.arm_positions = np.zeros(10)  # Positions for protective arm movements
        
    def detect_fall(self, sensor_data):
        """Detect if a fall is occurring"""
        # Check if CoM height is dropping rapidly
        com_height = sensor_data.get('com_height', 0.8)
        com_velocity_z = sensor_data.get('com_velocity', [0, 0, 0])[2]
        
        # Check if falling (negative z velocity and low height)
        is_falling = (com_velocity_z < -0.5 and com_height < self.fall_threshold)
        
        # Also check orientation
        orientation = sensor_data.get('orientation', [0, 0, 0, 1])
        roll_pitch = np.abs(orientation[:2])
        high_tilt = np.any(roll_pitch > 0.6)  # About 34 degrees
        
        return is_falling or high_tilt
    
    def execute_protection_sequence(self):
        """Execute fall protection sequence"""
        print("Executing fall protection...")
        
        # Move arms to protective positions
        self.controller.move_arms_to_protection()
        
        # If possible, try to break fall with legs
        self.controller.prepare_legs_for_impact()
        
        # Reduce stiffness to minimize damage
        self.controller.reduce_impedance()
        
        # Activate shock absorption if available
        self.controller.activate_shock_absorption()

# ℹ️ Safe Recovery System ℹ️
class SafeRecoverySystem:
    def __init__(self, robot_controller):
        self.controller = robot_controller
        self.recovery_poses = self._define_recovery_poses()
        
    def _define_recovery_poses(self):
        """Define safe recovery poses"""
        return {
            'kneel': np.array([0, 0, -0.5, 0, 0, 0] * 2 + [0] * 18),  # Kneeling position
            'crawl': np.array([0.2, 0, -0.3, -0.2, 0, 0.3] * 2 + [0] * 18),  # Crawling position
            'crawl_forward': np.array([0.3, 0.1, -0.2, -0.3, -0.1, 0.2] * 2 + [0] * 18)  # Moving forward on knees
        }
    
    def attempt_recovery(self, from_pose='fallen'):
        """Attempt to recover from fallen state"""
        if from_pose == 'fallen':
            # First, move to a stable intermediate pose
            success = self.controller.move_to_pose(self.recovery_poses['kneel'], duration=3.0)
            
            if success:
                # Then attempt to stand up
                return self._attempt_standup()
            else:
                print("Could not move to kneeling position, requesting assistance")
                return False
        
        return False
    
    def _attempt_standup(self):
        """Attempt to stand up from kneeling position"""
        print("Attempting to stand up...")
        
        # Gradual standup motion
        standup_trajectory = self._generate_standup_trajectory()
        
        # Execute standup
        success = self.controller.execute_trajectory(standup_trajectory, duration=5.0)
        
        if success:
            print("Successfully stood up")
            # Verify balance
            if self._verify_balance():
                print("Balance verified after standup")
                return True
            else:
                print("Failed to achieve stable balance after standup")
                return False
        else:
            print("Standup motion failed")
            return False
    
    def _generate_standup_trajectory(self):
        """Generate standup motion trajectory"""
        # This would generate a smooth trajectory from kneeling to standing
        # Simplified for this example
        n_points = 50
        trajectory = []
        
        # Interpolate from kneeling to standing
        kneeling_pos = self.recovery_poses['kneel']
        standing_pos = np.zeros(30)  # Default standing position
        
        for i in range(n_points):
            ratio = i / (n_points - 1)
            pos = kneeling_pos * (1 - ratio) + standing_pos * ratio
            trajectory.append(pos)
        
        return trajectory
    
    def _verify_balance(self):
        """Verify that robot is in stable balance after recovery"""
        # Check that CoM is within support polygon
        # Check that robot is upright
        # Check that no joints are in unsafe configurations
        
        state = self.controller.get_state()
        com_height = state.get('com_height', 0.8)
        orientation = state.get('orientation', [0, 0, 0, 1])
        
        # Check if upright
        is_upright = abs(orientation[0]) < 0.1 and abs(orientation[1]) < 0.1  # Small roll/pitch
        is_at_correct_height = 0.7 < com_height < 0.9  # Within reasonable range for standing
        
        return is_upright and is_at_correct_height
```

### 📏 11.6.2 Compliance and Safety Standards 📏

Understanding and implementing safety standards for humanoid robots:

```python
class SafetyComplianceSystem:
    def __init__(self):
        self.safety_standards = {
            'ISO 13482': {
                'description': 'Safety requirements for personal care robots',
                'requirements': [
                    'Risk assessment and mitigation',
                    'Emergency stop functionality',
                    'Safe interaction with humans',
                    'System reliability'
                ]
            },
            'ISO 12100': {
                'description': 'Safety of machinery - General principles',
                'requirements': [
                    'Risk analysis',
                    'Safety functions',
                    'Verification and validation'
                ]
            }
        }
        
        self.compliance_status = {}
        
    def verify_compliance(self, robot_system):
        """Verify compliance with safety standards"""
        results = {}
        
        for standard, requirements in self.safety_standards.items():
            standard_result = {
                'compliant': True,
                'violations': [],
                'recommendations': []
            }
            
            # Check each requirement
            for req in requirements['requirements']:
                if not self._check_requirement(robot_system, req):
                    standard_result['compliant'] = False
                    standard_result['violations'].append(req)
                    standard_result['recommendations'].append(
                        self._get_recommendation_for_requirement(req)
                    )
            
            results[standard] = standard_result
        
        self.compliance_status = results
        return results
    
    def _check_requirement(self, robot_system, requirement):
        """Check if a specific requirement is met"""
        # Implementation would check specific requirements
        # For this example, we'll implement a few common checks
        
        if 'emergency stop' in requirement.lower():
            # Check if emergency stop is properly implemented
            return robot_system.has_emergency_stop()
        
        elif 'risk assessment' in requirement.lower():
            # Check if risk assessment has been documented
            return robot_system.has_risk_assessment()
        
        elif 'safe interaction' in requirement.lower():
            # Check if safe interaction protocols are implemented
            return robot_system.has_safe_interaction_protocols()
        
        else:
            # Default to compliant for this example
            return True
    
    def _get_recommendation_for_requirement(self, requirement):
        """Get recommendation for addressing a requirement"""
        recommendations = {
            'emergency stop functionality': 'Implement emergency stop button accessible to operator',
            'risk assessment and mitigation': 'Conduct formal risk assessment and document mitigation strategies',
            'safe interaction with humans': 'Implement collision detection and force limiting in human interaction zones',
            'system reliability': 'Implement redundant safety systems and regular health checks'
        }
        
        return recommendations.get(requirement, 'Consult relevant safety standard for specific requirements')

# ℹ️ Safety documentation and certification ℹ️
class SafetyDocumentation:
    def __init__(self, robot_name, serial_number):
        self.robot_name = robot_name
        self.serial_number = serial_number
        self.safety_case = {}
        
    def generate_safety_case(self):
        """Generate complete safety case for the robot"""
        safety_case = {
            'robot_identification': {
                'name': self.robot_name,
                'serial': self.serial_number,
                'model': 'Generic Humanoid v1.0'
            },
            'hazard_analysis': self._perform_hazard_analysis(),
            'safety_requirements': self._define_safety_requirements(),
            'validation_results': self._get_validation_results(),
            'residual_risks': self._identify_residual_risks(),
            'safe_operating_procedures': self._define_operating_procedures()
        }
        
        self.safety_case = safety_case
        return safety_case
    
    def _perform_hazard_analysis(self):
        """Perform systematic hazard analysis"""
        hazards = [
            {
                'hazard': 'Uncontrolled motion',
                'hazardous_situation': 'Robot moves unexpectedly during maintenance',
                'harm': 'Physical injury to personnel',
                'severity': 'High',
                'probability': 'Medium',
                'risk_level': 'High',
                'mitigation': 'Implement lockout/tagout procedures and motion interlocks'
            },
            {
                'hazard': 'Falling',
                'hazardous_situation': 'Robot loses balance during operation',
                'harm': 'Robot falls onto nearby personnel or objects',
                'severity': 'High',
                'probability': 'Low',
                'risk_level': 'Medium',
                'mitigation': 'Implement balance control and fall protection systems'
            },
            {
                'hazard': 'High contact force',
                'hazardous_situation': 'Robot applies excessive force during interaction',
                'harm': 'Injury during human-robot interaction',
                'severity': 'Medium',
                'probability': 'Low',
                'risk_level': 'Low',
                'mitigation': 'Implement force limiting and compliant control'
            }
        ]
        
        return hazards
    
    def _define_safety_requirements(self):
        """Define safety requirements based on hazard analysis"""
        requirements = [
            {
                'id': 'SR-001',
                'requirement': 'The robot shall stop all motion within 100ms of emergency stop signal',
                'verification_method': 'Test with emergency stop button'
            },
            {
                'id': 'SR-002', 
                'requirement': 'The robot shall maintain balance with a 5cm CoM displacement during normal operation',
                'verification_method': 'Stability tests with external disturbances'
            },
            {
                'id': 'SR-003',
                'requirement': 'All joint torques shall be limited to 50 Nm during human interaction mode',
                'verification_method': 'Force measurement during interaction tests'
            }
        ]
        
        return requirements
    
    def _get_validation_results(self):
        """Get results from safety validation and testing"""
        return {
            'functional_safety_tests': {
                'status': 'Passed',
                'date': '2024-01-15',
                'test_procedure': 'SFT-001'
            },
            'drop_tests': {
                'status': 'Passed',
                'date': '2024-01-18',
                'test_procedure': 'DT-002'
            },
            'emergy_stop_tests': {
                'status': 'Passed', 
                'date': '2024-01-20',
                'test_procedure': 'EST-003'
            }
        }
    
    def _identify_residual_risks(self):
        """Identify risks that remain after mitigation"""
        residual_risks = [
            {
                'risk': 'Very high impact disturbance',
                'residual_probability': 'Very Low',
                'residual_severity': 'High',
                'mitigation_status': 'Mitigated but not eliminated'
            },
            {
                'risk': 'Software failure in safety system',
                'residual_probability': 'Low',
                'residual_severity': 'High',
                'mitigation_status': 'Mitigated with redundant systems'
            }
        ]
        
        return residual_risks
    
    def _define_operating_procedures(self):
        """Define safe operating procedures"""
        procedures = {
            'pre_operation_checklist': [
                'Verify emergency stop functionality',
                'Check all joint limits and ranges of motion',
                'Confirm communication with safety systems',
                'Verify adequate space for operation'
            ],
            'normal_operation': [
                'Maintain safe distance during autonomous operation',
                'Monitor robot state continuously',
                'Be prepared to activate emergency stop'
            ],
            'maintenance_mode': [
                'Use lockout/tagout procedures',
                'Follow proper isolation procedures',
                'Verify robot is in safe pose before maintenance'
            ]
        }
        
        return procedures
    
    def export_certification_package(self):
        """Export safety documentation for certification"""
        # This would generate proper documentation files
        # suitable for safety certification
        print(f"Generating safety certification package for {self.robot_name}")
        
        # In practice, this would create structured documents,
        # safety reports, test records, etc. for certification bodies
        return {
            'safety_case_report': 'safety_case.pdf',
            'test_reports': ['functional_tests.pdf', 'safety_tests.pdf'],
            'risk_assessment': 'risk_assessment.pdf',
            'safety_requirements': 'safety_requirements.pdf'
        }
```

## 📝 11.7 Summary 📝

Advanced humanoid control is a complex field that requires balancing multiple competing requirements: stability, mobility, safety, and efficiency. This chapter covered key aspects including:

1. **Humanoid Kinematics & Control**: Understanding the unique challenges of controlling robots with human-like form factors, including complex kinematic chains and balance requirements.

2. **Balance & Stability Control**: Implementing sophisticated control systems to maintain balance using ZMP control, whole-body control, and predictive approaches.

3. **Bipedal Locomotion**: Generating stable walking patterns with proper foot placement, CoM control, and adaptation to different terrains.

4. **Whole-Body Control**: Coordinating multiple tasks with different priorities using operational space control and task-prioritized frameworks.

5. **Learning-Based Control**: Using imitation learning and reinforcement learning to acquire complex behaviors.

6. **Validation & Testing**: Ensuring control systems are safe and reliable through simulation and real-world testing.

7. **Safety Systems**: Implementing comprehensive safety and emergency response systems.

The field of humanoid robotics continues to evolve rapidly, with machine learning and advanced control techniques enabling ever more sophisticated behaviors.

### ℹ️ Key Takeaways: ℹ️
- Humanoid control requires specialized approaches due to the bipedal nature and high degrees of freedom
- Balance control is fundamental and often achieved through ZMP-based methods
- Whole-body control frameworks coordinate multiple tasks simultaneously
- Learning approaches can complement traditional control methods
- Safety and validation are paramount in physical AI systems

## 🤔 Knowledge Check 🤔

1. Explain the challenges of controlling humanoid robots compared to simpler robotic platforms.
2. Describe how ZMP (Zero Moment Point) control contributes to bipedal stability.
3. What are the key components of a whole-body control system for humanoid robots?
4. How can reinforcement learning be applied to learn complex humanoid behaviors?
5. What are the critical safety considerations for humanoid robots?

---
*Continue to [Chapter 12: Capstone - Autonomous Humanoid](./chapter-12-capstone-autonomous-humanoid.md)*