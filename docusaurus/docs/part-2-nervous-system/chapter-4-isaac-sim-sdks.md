---
slug: chapter-7-nvidia-isaac-sim-sdk
title: Chapter 7 - NVIDIA Isaac Sim & SDK
description: Comprehensive guide to NVIDIA Isaac Sim and SDK for robotics and AI
tags: [nvidia, isaac, robotics, ai, simulation, perception]
---

# 📚 Chapter 7: NVIDIA Isaac Sim & SDK 📚

## 🎯 Learning Objectives 🎯

- Understand NVIDIA Isaac Sim architecture and capabilities for robotics simulation
- Install and configure Isaac Sim with ROS 2 integration
- Create and import robot models for Isaac Sim
- Implement perception systems using Isaac's AI capabilities
- Develop synthetic data generation pipelines
- Integrate Isaac Sim with NVIDIA Isaac ROS packages
- Build AI-powered navigation and manipulation systems
- Optimize performance for sim-to-real transfer

## 📋 Table of Contents 📋

- [Introduction to NVIDIA Isaac Platform](#introduction-to-nvidia-isaac-platform)
- [Isaac Sim Architecture](#isaac-sim-architecture)
- [Installation & Setup](#installation--setup)
- [Isaac Sim Basics](#isaac-sim-basics)
- [Robot Model Integration](#robot-model-integration)
- [Perception Systems](#perception-systems)
- [Synthetic Data Generation](#synthetic-data-generation)
- [Isaac ROS Integration](#isaac-ros-integration)
- [AI Navigation & Control](#ai-navigation--control)
- [Performance Optimization](#performance-optimization)
- [Sim-to-Real Transfer](#sim-to-real-transfer)
- [Chapter Summary](#chapter-summary)
- [Knowledge Check](#knowledge-check)

## 👋 Introduction to NVIDIA Isaac Platform 👋

The NVIDIA Isaac Platform represents a comprehensive solution for developing AI-powered robotics systems. It combines high-fidelity simulation capabilities with advanced AI tools, enabling researchers and developers to accelerate the development and deployment of sophisticated robotic applications.

### 🧩 Key Components of the Isaac Platform 🧩

The Isaac Platform consists of several key components:

1. **Isaac Sim**: Physically accurate 3D simulation environment based on NVIDIA Omniverse
2. **Isaac ROS**: Collection of ROS 2 packages for perception, navigation, and manipulation
3. **Isaac Lab**: Framework for reinforcement learning and robotics research
4. **Omniverse**: Collaborative simulation platform with USD format support
5. **Triton Inference Server**: AI model serving for robotics applications

### ℹ️ Benefits of Isaac Platform ℹ️

The Isaac Platform offers significant advantages for Physical AI development:

1. **Photorealistic Rendering**: High-fidelity visual simulation for computer vision training
2. **Physically Accurate Physics**: Realistic physics engine for accurate robot simulation
3. **Synthetic Data Generation**: Tools to generate large datasets for training AI models
4. **AI Integration**: Native support for reinforcement learning, perception, and navigation
5. **Real-time Performance**: Optimized for fast simulation on NVIDIA GPUs
6. **ROS 2 Integration**: Seamless integration with the ROS 2 ecosystem
7. **Cloud Scalability**: Ability to scale to large distributed simulation environments
8. **Sim-to-Real Transfer**: Tools and techniques to minimize the reality gap

### ⚖️ Isaac Sim vs. Traditional Simulators ⚖️

Compared to traditional simulators like Gazebo, Isaac Sim offers:

- **Enhanced Rendering**: NVIDIA RTX-accelerated ray tracing and global illumination
- **USD-Based**: Universal Scene Description format for industry-standard asset exchange
- **Multi-GPU Scaling**: Support for complex multi-GPU simulation environments
- **DL Integration**: Direct integration with NVIDIA's deep learning frameworks
- **NVidia Hardware Optimization**: Optimized for CUDA, TensorRT, and other NVIDIA technologies
- **Professional Pipeline**: Tools for professional content creation and validation

## 🏗️ Isaac Sim Architecture 🏗️

Isaac Sim is built on NVIDIA's Omniverse platform, which is designed for collaborative 3D simulation and design. Understanding its architecture is crucial for effective utilization.

### 🏗️ Core Architecture Components 🏗️

```
Isaac Sim Architecture

[Application Layer]
  ├── Isaac Sim Application
  ├── User Extensions
  └── Simulation Scenes

[Framework Layer]
  ├── OmniGraphNode System
  ├── Physics Engine (PhysX/Bullet)
  ├── Rendering Engine (RTX)
  └── USD Scene Management

[Extension Layer]
  ├── Isaac Sim Extensions
  ├── User Extensions
  └── External Extensions

[Interface Layer]
  ├── ROS 2 Bridge
  ├── Python API
  ├── Omniverse Kit
  └── Graphics API (DX11/DX12/Vulkan)
```

### ℹ️ USD (Universal Scene Description) ℹ️

USD is the core technology underlying Isaac Sim. It provides:

- **Scene Representation**: Hierarchical, value-based data model
- **Asset Definition**: Rich asset relationships and schemas
- **Composition**: Assembly of scene elements with composition arcs
- **Animation**: Time-sampled animation data
- **Render Delegate**: Render representation for visualization
- **Schema System**: Extensible type system for custom objects

### 🔗 Physics Engine Integration 🔗

Isaac Sim integrates with multiple physics engines:

- **NVIDIA PhysX**: High-performance physics engine optimized for GPUs
- **Bullet Physics**: Open-source physics engine with good robotics support
- **Omniverse PhysX**: Isaac Sim-specific physics extensions

### ℹ️ Rendering System ℹ️

Built on NVIDIA RTX technology, Isaac Sim provides:

- **Path Tracing**: Physically accurate lighting simulation
- **Global Illumination**: Realistic indirect lighting effects
- **Real-time Ray Tracing**: Dynamic reflections and shadows
- **Multi-resolution Shading**: Optimized rendering for different sensor types

## ℹ️ Installation & Setup ℹ️

### 📋 System Requirements 📋

To get optimal performance with Isaac Sim:

**Minimum Requirements:**
- NVIDIA GPU: RTX 3070 (8GB VRAM)
- CPU: Intel i7 / AMD Ryzen 7
- RAM: 16GB
- OS: Ubuntu 20.04/22.04 or Windows 10/11

**Recommended Requirements:**
- NVIDIA GPU: RTX 4070 Ti+ (12+GB VRAM)
- CPU: Intel i9 / AMD Ryzen 9
- RAM: 32GB+
- OS: Ubuntu 22.04 LTS
- Storage: SSD with 100GB+ free space

### ℹ️ Installing Isaac Sim ℹ️

#### ℹ️ Option 1: Docker (Recommended) ℹ️

```bash
# 🤖 Pull the Isaac Sim container 🤖
docker pull nvcr.io/nvidia/isaac-sim:4.2.0

# ℹ️ Run Isaac Sim with GUI support (Linux) ℹ️
xhost +local:docker
docker run --gpus all -it --rm \
  --name isaac-sim \
  -e "ACCEPT_EULA=Y" \
  -e "PRIVACY_CONSENT=Y" \
  --net=host \
  --mount "type=bind,src=/tmp/.X11-unix,dst=/tmp/.X11-unix" \
  --mount "type=bind,src=/home/$USER,dst=/home/user" \
  --mount "type=bind,src=/dev/shm,dst=/dev/shm" \
  --device=/dev/dri \
  --privileged \
  -v $HOME/isaac-sim-cache:/isaac-sim/cache \
  nvcr.io/nvidia/isaac-sim:4.2.0
```

#### ℹ️ Option 2: Standalone Installation ℹ️

For a standalone installation:

1. Download Isaac Sim from NVIDIA Developer Zone
2. Install Omniverse Launcher
3. Install Isaac Sim through the launcher
4. Configure environment variables

```bash
# ℹ️ Add to your .bashrc or .zshrc ℹ️
export ISAACSIM_PATH="/path/to/isaac-sim"
export PYTHONPATH="$ISAACSIM_PATH/python:$PYTHONPATH"
export PATH="$ISAACSIM_PATH:$PATH"
```

### ℹ️ Setting Up Isaac ROS Bridge ℹ️

```bash
# ℹ️ Create a new ROS 2 workspace ℹ️
mkdir -p ~/isaac_ros_ws/src
cd ~/isaac_ros_ws

# ℹ️ Clone Isaac ROS packages ℹ️
git clone -b ros2 https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_common.git src/isaac_ros_common
git clone -b ros2 https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_visual_slam.git src/isaac_ros_visual_slam
git clone -b ros2 https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_manipulator.git src/isaac_ros_manipulator
git clone -b ros2 https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_point_cloud_processing.git src/isaac_ros_point_cloud_processing

# 🔗 Install dependencies 🔗
rosdep install --from-paths src --ignore-src -r -y

# ℹ️ Build the workspace ℹ️
colcon build --symlink-install --packages-select \
  isaac_ros_common \
  isaac_ros_visual_slam \
  isaac_ros_point_cloud_processing
```

## ℹ️ Isaac Sim Basics ℹ️

### ℹ️ Getting Started with Isaac Sim ℹ️

Once Isaac Sim is installed, start by familiarizing yourself with the interface:

1. Launch Isaac Sim
2. Create a new stage
3. Set up basic lighting and camera
4. Import or create objects

### ℹ️ Omniverse Extensions ℹ️

Isaac Sim uses a powerful extension system that allows for feature enhancement:

```python
# ℹ️ Example extension structure ℹ️
from omni.kit.property.usd.property_widget_builder import PropertyWidgetBuilder
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.objects import DynamicCuboid
import carb
import omni.ext

class IsaacSimTutorialExtension(omni.ext.IExt):
    """Isaac Sim Tutorial Extension"""

    def on_startup(self, ext_id):
        self._logger = carb.logger.acquire_logger(ext_id)
        self._logger.info("IsaacSimTutorialExtension startup")
        
        # Create a cube in the simulation
        cube = DynamicCuboid(
            prim_path="/World/cube",
            name="my_cube",
            position=(0, 0, 1),
            size=0.1
        )
        
        self._logger.info("Added cube to stage")

    def on_shutdown(self):
        self._logger.info("IsaacSimTutorialExtension shutdown")
        self._logger = None
```

### ℹ️ USD Format and Scene Structure ℹ️

Understanding USD format is essential for working with Isaac Sim:

```python
# ℹ️ Python API for USD operations ℹ️
from pxr import Usd, UsdGeom, Gf, Sdf, UsdPhysics, PhysxSchema
import omni.usd

def create_simple_robot_stage(stage_path):
    """Create a simple robot stage using USD API"""
    stage = Usd.Stage.CreateNew(stage_path)
    
    # Root world prim
    world_prim = stage.DefinePrim("/World", "Xform")
    
    # Add some basic lighting
    light_prim = stage.DefinePrim("/World/light", "DistantLight")
    light_prim.GetAttribute("inputs:intensity").Set(3000)
    
    # Define robot prim
    robot_prim = stage.DefinePrim("/World/Robot", "Xform")
    robot_prim.GetAttribute("xformOp:translate").Set(Gf.Vec3d(0, 0, 0.5))
    
    # Add a simple base link
    base_link_prim = stage.DefinePrim("/World/Robot/base_link", "Cylinder")
    base_link_prim.GetAttribute("radius").Set(0.1)
    base_link_prim.GetAttribute("height").Set(0.2)
    
    # Create physics properties
    body_api = UsdPhysics.RigidBodyAPI.Apply(base_link_prim, "")
    body_api.CreateMassThresholdAttr(0.1)
    
    stage.Save()
    return stage

def add_sensor_to_robot(robot_path, sensor_type="Camera"):
    """Add a sensor to the robot using USD API"""
    stage = omni.usd.get_context().get_stage()
    
    if sensor_type == "Camera":
        camera_path = f"{robot_path}/sensor_camera"
        camera_prim = stage.DefinePrim(camera_path, "Camera")
        
        # Set camera properties
        camera_prim.GetAttribute("focalLength").Set(24.0)
        camera_prim.GetAttribute("horizontalAperture").Set(36.0)
        camera_prim.GetAttribute("verticalAperture").Set(20.25)
    
    stage.Save()
```

### ℹ️ Basic Physics Setup ℹ️

Setting up realistic physics in Isaac Sim:

```python
from omni.isaac.core.physics_context import PhysicsContext
from omni.isaac.core.prims import RigidPrim
from pxr import PhysxSchema, UsdPhysics

def setup_physics_environment():
    """Configure physics environment for robot simulation"""
    # Create physics context
    physics_ctx = PhysicsContext()
    physics_ctx.set_gravity(9.81)
    
    # Configure physics settings
    physics_ctx.set_fixed_timestep(1.0/60.0)  # 60 Hz physics
    physics_ctx.set_max_substeps(4)  # Max substeps for stability
    
    # Enable GPU acceleration if available
    physx_scene_api = PhysxSchema.PhysxSceneAPI.Apply(physics_ctx.scene)
    physx_scene_api.CreateGpuDynamicParticlesEnabledAttr(True)
    physx_scene_api.CreateGpuCollisionFrameCountAttr(24)
    
    return physics_ctx

def setup_material_properties(material_path, static_friction=0.5, dynamic_friction=0.5, restitution=0.1):
    """Create material properties for realistic contact simulation"""
    stage = omni.usd.get_context().get_stage()
    
    # Create material prim
    material_prim = stage.DefinePrim(material_path, "Material")
    
    # Add surface outputs
    surface_output = material_prim.CreateOutput("outputs:surface", Sdf.ValueTypeNames.Token)
    
    # Add physics material properties
    friction_attr = material_prim.CreateAttribute("physics:staticFriction", Sdf.ValueTypeNames.Float)
    friction_attr.Set(static_friction)
    
    dynamic_friction_attr = material_prim.CreateAttribute("physics:dynamicFriction", Sdf.ValueTypeNames.Float)
    dynamic_friction_attr.Set(dynamic_friction)
    
    restitution_attr = material_prim.CreateAttribute("physics:restitution", Sdf.ValueTypeNames.Float)
    restitution_attr.Set(restitution)
    
    return material_prim
```

## 🤖 Robot Model Integration 🤖

### 🏗️ Importing URDF Models 🏗️

Isaac Sim can import existing URDF models, though they may require some adaptation:

```python
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.viewports import set_camera_view
from omni.isaac.core import World
from omni.isaac.core.robots import Robot

class IsaacSimRobot:
    def __init__(self, usd_path, name, position, orientation):
        self.name = name
        self.position = position
        self.orientation = orientation
        self.usd_path = usd_path
        self.robot = None
        
        # Import robot into Isaac Sim
        self.import_robot()
    
    def import_robot(self):
        """Add robot to Isaac Sim stage"""
        # Add robot to stage with specified transform
        add_reference_to_stage(
            usd_path=self.usd_path,
            prim_path=f"/World/{self.name}",
            position=self.position,
            orientation=self.orientation
        )
        
        # Create a Robot object for high-level control
        self.robot = Robot(
            prim_path=f"/World/{self.name}",
            name=self.name,
            position=self.position,
            orientation=self.orientation
        )
    
    def setup_robot_controllers(self):
        """Set up controllers for robot joint control"""
        from omni.isaac.core.articulations import Articulation
        from omni.isaac.core.controllers import DifferentialController
        
        # Get articulation (robot model with joints)
        self.articulation = Articulation(prim_path=f"/World/{self.name}")
        
        # Example: Set up differential controller for wheeled robot
        self.diff_controller = DifferentialController(
            name="diff_controller",
            wheel_radius=0.1,
            wheel_base=0.4
        )
        
        # For humanoid robots, controllers would be set up differently
        # with individual joint controllers for each articulation
```

### 🤖 Creating Isaac-Compatible Robot Models 🤖

While importing URDF works, creating Isaac-native robot models often yields better results:

```python
def create_isaac_robot_model(robot_name, joints_config):
    """Create a robot model specifically designed for Isaac Sim"""
    from pxr import Gf, Usd, UsdGeom, UsdPhysics, PhysxSchema
    
    stage = omni.usd.get_context().get_stage()
    
    # Create robot root
    robot_prim = stage.DefinePrim(f"/World/{robot_name}", "Xform")
    
    # Create base link
    base_link_path = f"/World/{robot_name}/base_link"
    base_link_prim = stage.DefinePrim(base_link_path, "Capsule")
    base_link_prim.GetAttribute("radius").Set(0.15)
    base_link_prim.GetAttribute("height").Set(0.4)
    
    # Create collision and physics
    collision_api = UsdPhysics.CollisionAPI.Apply(base_link_prim)
    rigid_body_api = UsdPhysics.RigidBodyAPI.Apply(base_link_prim)
    
    # Create joints based on configuration
    for joint_idx, joint_config in enumerate(joints_config):
        joint_path = f"/World/{robot_name}/joint_{joint_idx}"
        
        if joint_config['type'] == 'revolute':
            joint_prim = stage.DefinePrim(joint_path, "PhysicsRevoluteJoint")
        elif joint_config['type'] == 'prismatic':
            joint_prim = stage.DefinePrim(joint_path, "PhysicsPrismaticJoint")
        
        # Set joint properties
        joint_prim.GetAttribute("physics:body0").Set(base_link_path)
        
        # Add child link
        child_link_path = f"/World/{robot_name}/link_{joint_idx+1}"
        child_link_prim = stage.DefinePrim(child_link_path, joint_config['geometry'])
        child_link_api = UsdPhysics.CollisionAPI.Apply(child_link_prim)
        
        # Configure joint limits
        if 'limit' in joint_config:
            lower_limit = joint_config['limit']['lower']
            upper_limit = joint_config['limit']['upper']
            
            # Apply limits based on joint type
            if joint_config['type'] == 'revolute':
                joint_prim.GetAttribute("physics:limitLower").Set(lower_limit)
                joint_prim.GetAttribute("physics:limitUpper").Set(upper_limit)
    
    stage.Save()

# ℹ️ Example joint configuration for a simple arm ℹ️
simple_arm_joints = [
    {
        'type': 'revolute',
        'name': 'shoulder_yaw',
        'geometry': 'Capsule',
        'limit': {'lower': -1.57, 'upper': 1.57},
        'drive': {'type': 'angular', 'damping': 10.0, 'stiffness': 1000.0}
    },
    {
        'type': 'revolute',
        'name': 'shoulder_pitch',
        'geometry': 'Capsule',
        'limit': {'lower': -1.57, 'upper': 1.57},
        'drive': {'type': 'angular', 'damping': 10.0, 'stiffness': 1000.0}
    },
    {
        'type': 'revolute',
        'name': 'elbow_pitch',
        'geometry': 'Capsule',
        'limit': {'lower': -2.0, 'upper': 0.5},
        'drive': {'type': 'angular', 'damping': 10.0, 'stiffness': 1000.0}
    }
]
```

### 🤖 Robot Actuator Integration 🤖

Setting up realistic actuator models for humanoid robots:

```python
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.stage import get_current_stage
from pxr import PhysxSchema, UsdPhysics
import numpy as np

class RobotActuatorManager:
    def __init__(self, robot_prim_path):
        self.robot_path = robot_prim_path
        self.joint_paths = self.get_joint_paths()
        self.setup_actuators()
    
    def get_joint_paths(self):
        """Get all joint paths in the robot"""
        stage = get_current_stage()
        robot_prim = stage.GetPrimAtPath(self.robot_path)
        
        joint_paths = []
        for child in robot_prim.GetAllChildren():
            # Look for physics joints
            if child.GetTypeName() in ["PhysicsRevoluteJoint", "PhysicsPrismaticJoint"]:
                joint_paths.append(child.GetPath())
        
        return joint_paths
    
    def setup_actuators(self):
        """Configure actuator properties for joints"""
        for joint_path in self.joint_paths:
            joint_prim = get_prim_at_path(str(joint_path))
            
            # Set up actuator properties
            # Create drive targets - these will be controlled in simulation
            
            # For revolute joints, add angular drive
            joint_type = joint_prim.GetTypeName()
            if joint_type == "PhysicsRevoluteJoint":
                # Add angular drive for torque control
                drive_api = PhysxSchema.DriveAPI.Apply(joint_prim, "angular")
                drive_api.CreateStiffnessAttr(1000.0)  # Spring constant
                drive_api.CreateDampingAttr(100.0)     # Damping coefficient
                drive_api.CreateMaxForceAttr(100.0)     # Maximum actuation force
                
                # Add target for position control
                joint_prim.CreateAttribute("angularDrive:targetPosition", 
                                          Sdf.ValueTypeNames.Float, False).Set(0.0)
                joint_prim.CreateAttribute("angularDrive:targetVelocity", 
                                          Sdf.ValueTypeNames.Float, False).Set(0.0)
    
    def command_joint_positions(self, joint_commands):
        """Send position commands to robot joints"""
        # This would interface with Isaac's control system
        # Implementation depends on specific controller setup
        pass
    
    def command_joint_velocities(self, joint_velocities):
        """Send velocity commands to robot joints"""
        # Velocity control implementation
        pass
    
    def command_joint_efforts(self, joint_efforts):
        """Send effort/torque commands to robot joints"""
        # Force/torque control implementation
        pass

# 🎛️ Example of integrating with ROS 2 for control 🎛️
class IsaacROSRobotInterface:
    def __init__(self, robot_name):
        self.robot_name = robot_name
        self.actuator_manager = RobotActuatorManager(f"/World/{robot_name}")
        
        # ROS 2 interface components
        self.node = None  # Will be set by main node
        self.joint_command_sub = None
        self.joint_state_pub = None
        
    def setup_ros_interface(self, node):
        """Setup ROS 2 interfaces for the robot"""
        self.node = node
        
        from sensor_msgs.msg import JointState
        from trajectory_msgs.msg import JointTrajectory
        from control_msgs.msg import JointTrajectoryControllerState
        
        # Joint state publisher
        self.joint_state_pub = self.node.create_publisher(JointState, 
                                                          f'/{self.robot_name}/joint_states', 
                                                          10)
        
        # Joint trajectory subscriber for control
        self.joint_command_sub = self.node.create_subscription(
            JointTrajectory,
            f'/{self.robot_name}/joint_trajectory',
            self.joint_command_callback,
            10
        )
        
        # Controller state publisher
        self.controller_state_pub = self.node.create_publisher(
            JointTrajectoryControllerState,
            f'/{self.robot_name}/controller_state',
            10
        )
    
    def joint_command_callback(self, msg):
        """Handle incoming joint trajectory commands"""
        # Extract command positions, velocities, or efforts
        if len(msg.points) > 0:
            current_point = msg.points[0]
            
            # For position control
            if len(current_point.positions) > 0:
                self.actuator_manager.command_joint_positions(current_point.positions)
            
            # For velocity control
            if len(current_point.velocities) > 0:
                self.actuator_manager.command_joint_velocities(current_point.velocities)
    
    def publish_joint_states(self):
        """Publish current joint states"""
        from sensor_msgs.msg import JointState
        import time
        
        msg = JointState()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.name = self.get_joint_names()  # Get from robot model
        msg.position = self.get_joint_positions()  # Get current positions
        msg.velocity = self.get_joint_velocities()  # Get current velocities
        msg.effort = self.get_joint_efforts()  # Get current efforts
        
        self.joint_state_pub.publish(msg)
    
    def get_joint_names(self):
        """Get names of all joints in the robot"""
        # Implementation to extract joint names from USD stage
        joint_names = []
        for i, joint_path in enumerate(self.actuator_manager.joint_paths):
            joint_names.append(f"joint_{i}")
        return joint_names
    
    def get_joint_positions(self):
        """Get current joint positions"""
        # Implementation to read current joint positions from simulation
        # This requires interfacing with Isaac Sim physics
        return [0.0] * len(self.actuator_manager.joint_paths)
    
    def get_joint_velocities(self):
        """Get current joint velocities"""
        # Implementation to read current joint velocities
        return [0.0] * len(self.actuator_manager.joint_paths)
    
    def get_joint_efforts(self):
        """Get current joint efforts/torques"""
        # Implementation to read current joint forces/torques
        return [0.0] * len(self.actuator_manager.joint_paths)
```

## 👁️ Perception Systems 👁️

### 🔗 Advanced Sensor Integration 🔗

Isaac Sim includes sophisticated perception systems for various types of sensors:

```python
from omni.isaac.sensor import Camera, LidarRtx
from omni.isaac.core.utils.prims import set_targets
from omni.isaac.core.objects import DynamicCuboid
from pxr import Gf, Sdf, UsdGeom
import numpy as np

class IsaacPerceptionSensors:
    def __init__(self, robot_prim_path):
        self.robot_path = robot_prim_path
        self.cameras = {}
        self.lidars = {}
        self.setup_sensors()
    
    def setup_sensors(self):
        """Set up various perception sensors on the robot"""
        # 1. RGB Camera
        self.add_rgb_camera(
            name="front_camera",
            position=[0.2, 0.0, 0.8],  # 20cm forward, centered, 80cm high
            orientation=[0, 0, 0, 1],  # Looking forward
            resolution=(640, 480),
            fov=1.047  # 60 degrees
        )
        
        # 2. Depth Camera
        self.add_depth_camera(
            name="depth_camera",
            position=[0.2, 0.0, 0.8],
            orientation=[0, 0, 0, 1],
            resolution=(640, 480),
            fov=1.047
        )
        
        # 3. 3D LiDAR
        self.add_lidar(
            name="3d_lidar",
            position=[0.15, 0.0, 0.9],
            orientation=[0, 0, 0, 1],
            config="40m-10hz"
        )
        
        # 4. IMU
        self.add_imu(
            name="imu_sensor",
            position=[0.0, 0.0, 0.5]  # At robot center of mass
        )
    
    def add_rgb_camera(self, name, position, orientation, resolution, fov):
        """Add an RGB camera to the robot"""
        camera_path = f"{self.robot_path}/{name}"
        
        # Create camera prim
        camera_prim = UsdGeom.Camera.Define(
            get_current_stage(), 
            camera_path
        )
        
        # Set camera properties
        camera_prim.GetFocalLengthAttr().Set(24.0)
        camera_prim.GetHorizontalApertureAttr().Set(36.0)
        camera_prim.GetVerticalApertureAttr().Set(20.25)
        camera_prim.GetProjectionAttr().Set(UsdGeom.Tokens.perspective)
        
        # Apply transforms
        xform = UsdGeom.Xformable(camera_prim)
        xform.MakeMatrixXform().SetOpValue(
            Gf.Matrix4d().SetTranslateOnly(Gf.Vec3d(*position))
        )
        
        # Create Isaac Camera object
        camera = Camera(
            prim_path=camera_path,
            frequency=30,  # 30 Hz
            resolution=resolution,
            position=position,
            orientation=orientation
        )
        
        self.cameras[name] = camera
    
    def add_depth_camera(self, name, position, orientation, resolution, fov):
        """Add a depth camera to the robot"""
        depth_camera_path = f"{self.robot_path}/{name}_depth"
        
        # Create depth camera prim
        depth_camera_prim = UsdGeom.Camera.Define(
            get_current_stage(), 
            depth_camera_path
        )
        
        # Similar setup as RGB camera
        depth_camera_prim.GetFocalLengthAttr().Set(24.0)
        depth_camera_prim.GetHorizontalApertureAttr().Set(36.0)
        depth_camera_prim.GetVerticalApertureAttr().Set(20.25)
        
        # Create Isaac depth camera object
        camera = Camera(
            prim_path=depth_camera_path,
            frequency=30,
            resolution=resolution,
            position=position,
            orientation=orientation
        )
        
        # Enable depth capture
        camera.add_observed_properties({"distance_to_image_plane"})
        
        self.cameras[f"{name}_depth"] = camera
    
    def add_lidar(self, name, position, orientation, config="40m-10hz"):
        """Add a 3D LiDAR to the robot"""
        lidar_path = f"{self.robot_path}/{name}"
        
        # Different LiDAR model based on configuration
        if config == "40m-10hz":
            lidar = LidarRtx(
                prim_path=lidar_path,
                translation=position,
                orientation=orientation,
                config="ShortRange",
                rotation_frequency=10,
                # Set up LiDAR parameters
                horizontal_samples=1080,
                vertical_samples=64,
                horizontal_field_of_view=360,
                vertical_field_of_view=30,
                range=40
            )
        elif config == "100m-5hz":
            lidar = LidarRtx(
                prim_path=lidar_path,
                translation=position,
                orientation=orientation,
                config="LongRange",
                rotation_frequency=5,
                horizontal_samples=2160,
                vertical_samples=128,
                horizontal_field_of_view=360,
                vertical_field_of_view=45,
                range=100
            )
        
        self.lidars[name] = lidar
    
    def add_imu(self, name, position):
        """Add IMU sensor to the robot"""
        # IMUs are typically attached to the main body (base link)
        # In Isaac Sim, IMU sensing is often implemented through rigid body properties
        # and stage transforms
        
        from omni.isaac.core.sensors import Imu
        from omni.isaac.core.prims import RigidPrim
        
        imu_path = f"{self.robot_path}/{name}"
        
        # Create a small visual prim for the IMU (just for reference)
        cube_geom = UsdGeom.Cube.Define(get_current_stage(), imu_path)
        cube_geom.GetSizeAttr().Set(0.01)  # 1cm cube
        
        # Apply transforms
        xform = UsdGeom.Xformable(cube_geom)
        xform.MakeMatrixXform().SetOpValue(
            Gf.Matrix4d().SetTranslateOnly(Gf.Vec3d(*position))
        )
        
        # Create Isaac IMU sensor
        imu_sensor = Imu(
            prim_path=imu_path,
            frequency=100,  # 100 Hz
            position=position
        )
        
        self.imus[name] = imu_sensor
    
    def get_sensor_data(self):
        """Get data from all sensors"""
        sensor_data = {}
        
        # Get camera data
        for name, camera in self.cameras.items():
            sensor_data[f"{name}_rgb"] = camera.get_rgb()
            if "depth" in name:
                sensor_data[f"{name}_depth"] = camera.get_depth()
        
        # Get LiDAR data
        for name, lidar in self.lidars.items():
            sensor_data[f"{name}_points"] = lidar.get_point_cloud()
            sensor_data[f"{name}_ranges"] = lidar.get_ranges()
        
        # Get IMU data
        for name, imu in self.imus.items():
            sensor_data[f"{name}_accelerometer"] = imu.get_linear_acceleration()
            sensor_data[f"{name}_gyroscope"] = imu.get_angular_velocity()
        
        return sensor_data

# ℹ️ Example usage ℹ️
def setup_robot_with_sensors(robot_path, sensor_config):
    """Helper function to set up robot with perception sensors"""
    perception_manager = IsaacPerceptionSensors(robot_path)
    
    # Configure sensors based on robot type
    if sensor_config['robot_type'] == 'humanoid':
        # Humanoid-specific sensors
        perception_manager.add_rgb_camera(
            name="head_camera",
            position=[0.0, 0.0, 0.8],  # Head position
            orientation=[0, 0, 0, 1],
            resolution=(1280, 720),
            fov=1.047
        )
        
        perception_manager.add_lidar(
            name="chest_lidar",
            position=[0.05, 0.0, 0.6],  # Chest position
            orientation=[0, 0, 0, 1],
            config="40m-10hz"
        )
    
    return perception_manager
```

### ℹ️ Synthetic Image Generation ℹ️

Isaac Sim excels at generating synthetic images for training computer vision models:

```python
from omni.isaac.synthetic_utils import SyntheticDataHelper
from omni.isaac.core.utils.stage import add_reference_to_stage
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os

class SyntheticDataGenerator:
    def __init__(self, output_dir, scene_configs):
        self.output_dir = output_dir
        self.scene_configs = scene_configs
        self.synthetic_helper = SyntheticDataHelper()
        self.setup_output_directories()
    
    def setup_output_directories(self):
        """Create directories for synthetic data storage"""
        dirs = [
            os.path.join(self.output_dir, "images"),
            os.path.join(self.output_dir, "depth_maps"),
            os.path.join(self.output_dir, "segmentation_masks"),
            os.path.join(self.output_dir, "annotations"),
            os.path.join(self.output_dir, "metadata")
        ]
        
        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)
    
    def setup_scenes(self):
        """Setup multiple scenes for synthetic data generation"""
        for i, scene_config in enumerate(self.scene_configs):
            # Load environment
            env_path = scene_config['environment_path']
            add_reference_to_stage(
                usd_path=env_path,
                prim_path=f"/World/Environment_{i}"
            )
            
            # Add lighting variations
            self.add_lighting_variations(i, scene_config.get('lighting', []))
            
            # Add objects for detection
            self.add_objects_for_training(i, scene_config.get('objects', []))
    
    def add_lighting_variations(self, scene_id, lighting_configs):
        """Add different lighting conditions"""
        for j, light_config in enumerate(lighting_configs):
            light_path = f"/World/Environment_{scene_id}/Light_{j}"
            
            # Create different light types based on config
            if light_config['type'] == 'distant':
                # Sun-like lighting
                light_prim = add_reference_to_stage(
                    usd_path="omniverse://localhost/NVIDIA/Assets/SkinnedMeshes/Known/Lights/DistantLight.usdz",
                    prim_path=light_path
                )
                
                # Configure light properties
                stage = get_current_stage()
                light_prim = stage.GetPrimAtPath(light_path)
                
                # Set intensity and direction
                intensity_attr = light_prim.GetAttribute("inputs:intensity")
                if intensity_attr:
                    intensity_attr.Set(light_config.get('intensity', 3000))
                
                # Set direction
                direction_attr = light_prim.GetAttribute("inputs:direction")
                if direction_attr:
                    direction_attr.Set(Gf.Vec3f(*light_config.get('direction', [0, 0, -1])))
            
            elif light_config['type'] == 'rect':
                # Area light
                light_prim = add_reference_to_stage(
                    usd_path="omniverse://localhost/NVIDIA/Assets/SkinnedMeshes/Known/Lights/RectLight.usdz",
                    prim_path=light_path
                )
    
    def add_objects_for_training(self, scene_id, object_configs):
        """Add objects that will be used for training"""
        for i, obj_config in enumerate(object_configs):
            obj_path = f"/World/Environment_{scene_id}/Object_{i}"
            
            # Add object to stage
            add_reference_to_stage(
                usd_path=obj_config['usd_path'],
                prim_path=obj_path
            )
            
            # Apply random transforms if requested
            if obj_config.get('randomize_position', True):
                self.randomize_object_position(obj_path, obj_config.get('position_bounds', {}))
            
            if obj_config.get('randomize_rotation', True):
                self.randomize_object_rotation(obj_path, obj_config.get('rotation_bounds', {}))
            
            if obj_config.get('randomize_appearance', True):
                self.randomize_object_appearance(obj_path, obj_config.get('appearance_configs', []))
    
    def randomize_object_position(self, obj_path, bounds):
        """Randomly position object within bounds"""
        import random
        
        x_min = bounds.get('x_min', -10.0)
        x_max = bounds.get('x_max', 10.0)
        y_min = bounds.get('y_min', -10.0)
        y_max = bounds.get('y_max', 10.0)
        z_min = bounds.get('z_min', 0.1)
        z_max = bounds.get('z_max', 2.0)
        
        # Generate random position
        pos = [
            random.uniform(x_min, x_max),
            random.uniform(y_min, y_max),
            random.uniform(z_min, z_max)
        ]
        
        # Set position
        stage = get_current_stage()
        prim = stage.GetPrimAtPath(obj_path)
        xform = UsdGeom.Xformable(prim)
        xform.MakeMatrixXform().SetOpValue(
            Gf.Matrix4d().SetTranslateOnly(Gf.Vec3d(*pos))
        )
    
    def randomize_object_appearance(self, obj_path, appearance_options):
        """Randomize object appearance for domain randomization"""
        import random
        
        if not appearance_options:
            return
        
        # Select a random appearance option
        appearance = random.choice(appearance_options)
        
        # Apply material changes
        stage = get_current_stage()
        prim = stage.GetPrimAtPath(obj_path)
        
        # This would involve changing material properties
        # Implementation depends on object's material setup
        pass
    
    def generate_synthetic_dataset(self, num_samples=1000, output_format="coco"):
        """Generate synthetic dataset for training"""
        
        # Set up synthetic data helper
        self.synthetic_helper.initialize()
        
        # Configure annotations needed
        self.synthetic_helper.set_annotators([
            "bbox",          # Bounding boxes
            "mask",          # Segmentation masks  
            "depth",         # Depth maps
            "instance_id",   # Instance segmentation
            "normal",        # Surface normals
            "pose",          # Object poses
            "semseg"         # Semantic segmentation
        ])
        
        # Generate samples
        for i in range(num_samples):
            # Randomize scene
            self.randomize_scene()
            
            # Capture data
            frame_data = self.capture_frame_data()
            
            # Save data
            self.save_frame_data(frame_data, i, output_format)
            
            # Progress update
            if i % 100 == 0:
                print(f"Generated {i}/{num_samples} synthetic samples")
    
    def randomize_scene(self):
        """Randomize the scene for domain randomization"""
        # Randomize lighting
        self.randomize_lighting_conditions()
        
        # Randomize object positions
        self.randomize_object_positions()
        
        # Randomize camera positions (for multiple views)
        self.randomize_camera_positions()
        
        # Randomize environmental conditions
        self.randomize_weather_conditions()  # Fog, rain, etc.
    
    def capture_frame_data(self):
        """Capture all synthetic data for a single frame"""
        frame_data = {}
        
        # Get RGB image
        rgb_data = self.synthetic_helper.get_rgb()
        frame_data['rgb'] = rgb_data
        
        # Get depth data
        depth_data = self.synthetic_helper.get_depth()
        frame_data['depth'] = depth_data
        
        # Get segmentation masks
        seg_data = self.synthetic_helper.get_semantic_segmentation()
        frame_data['semantic_mask'] = seg_data
        
        # Get instance segmentation
        inst_data = self.synthetic_helper.get_instance_segmentation()
        frame_data['instance_mask'] = inst_data
        
        # Get bounding boxes
        bbox_data = self.synthetic_helper.get_bounding_boxes()
        frame_data['bounding_boxes'] = bbox_data
        
        # Get object poses
        pose_data = self.synthetic_helper.get_object_poses()
        frame_data['poses'] = pose_data
        
        return frame_data
    
    def save_frame_data(self, frame_data, frame_id, format="coco"):
        """Save frame data in specified format"""
        
        # Create image filename
        img_filename = f"image_{frame_id:06d}.png"
        img_path = os.path.join(self.output_dir, "images", img_filename)
        
        # Save RGB image
        img_array = frame_data['rgb']
        img_pil = Image.fromarray(img_array)
        img_pil.save(img_path)
        
        # Save depth map
        depth_filename = f"depth_{frame_id:06d}.png"
        depth_path = os.path.join(self.output_dir, "depth_maps", depth_filename)
        
        # Normalize depth data to 0-255 range for PNG
        depth_normalized = ((frame_data['depth'] - np.min(frame_data['depth'])) / 
                           (np.max(frame_data['depth']) - np.min(frame_data['depth'])) * 255).astype(np.uint8)
        depth_img = Image.fromarray(depth_normalized)
        depth_img.save(depth_path)
        
        # Save segmentation mask
        seg_filename = f"seg_{frame_id:06d}.png"
        seg_path = os.path.join(self.output_dir, "segmentation_masks", seg_filename)
        
        seg_img = Image.fromarray(frame_data['semantic_mask'].astype(np.uint8))
        seg_img.save(seg_path)
        
        # Create annotation based on format
        if format == "coco":
            annotation = self.create_coco_annotation(frame_data, frame_id, img_filename)
        elif format == "yolo":
            annotation = self.create_yolo_annotation(frame_data, frame_id, img_filename)
        
        # Save annotation
        annot_filename = f"annotation_{frame_id:06d}.json"
        annot_path = os.path.join(self.output_dir, "annotations", annot_filename)
        
        import json
        with open(annot_path, 'w') as f:
            json.dump(annotation, f, indent=2)
        
        # Save metadata
        metadata = {
            'frame_id': frame_id,
            'timestamp': time.time(),
            'scene_config': self.get_current_scene_config(),
            'camera_config': self.get_current_camera_config()
        }
        
        meta_filename = f"meta_{frame_id:06d}.json"
        meta_path = os.path.join(self.output_dir, "metadata", meta_filename)
        
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def create_coco_annotation(self, frame_data, frame_id, img_filename):
        """Create COCO format annotation for frame"""
        annotation = {
            "info": {
                "year": 2025,
                "version": "1.0",
                "description": "Synthetic dataset generated with Isaac Sim",
                "contributor": "Physical AI & Humanoid Robotics",
                "date_created": time.strftime("%Y-%m-%d"),
                "image_dir": os.path.join(self.output_dir, "images")
            },
            "licenses": [{
                "id": 1,
                "name": "Synthetic Data License",
                "url": "http://example.com/license"
            }],
            "categories": [],  # This would be populated with object categories
            "images": [{
                "id": frame_id,
                "file_name": img_filename,
                "width": frame_data['rgb'].shape[1],
                "height": frame_data['rgb'].shape[0],
                "date_captured": time.strftime("%Y-%m-%d %H:%M:%S"),
                "license": 1,
                "coco_url": "",
                "flickr_url": ""
            }],
            "annotations": []  # This would contain bounding box/segmentation annotations
        }
        
        return annotation
    
    def get_current_scene_config(self):
        """Get current scene configuration for metadata"""
        # Implementation would return current lighting, objects, etc.
        return {"lighting": "random", "objects": "random"}
    
    def get_current_camera_config(self):
        """Get current camera configuration for metadata"""
        # Implementation would return camera position, etc.
        return {"position": [0, 0, 1], "rotation": [0, 0, 0]}
```

## 🔗 Isaac ROS Integration 🔗

### ℹ️ Isaac ROS Packages ℹ️

The Isaac ROS collection provides hardware-accelerated perception and navigation nodes tightly integrated with Isaac Sim:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import numpy as np
import cv2

class IsaacROSPerceptionPipeline(Node):
    def __init__(self):
        super().__init__('isaac_ros_perception_pipeline')
        
        # Create publisher for processed images
        self.image_pub = self.create_publisher(Image, 'processed_image', 10)
        self.bridge = CvBridge()
        
        # Example: subscribing to Isaac Sim camera feed
        self.camera_sub = self.create_subscription(
            Image,
            '/camera/color/image_raw',
            self.camera_callback,
            10
        )
        
        # Example: camera info for rectification
        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            '/camera/color/camera_info',
            self.camera_info_callback,
            10
        )
        
        self.camera_matrix = None
        self.distortion_coeffs = None
        
        self.get_logger().info('Isaac ROS Perception Pipeline initialized')
    
    def camera_callback(self, msg):
        """Process incoming camera images"""
        # Convert ROS image to OpenCV
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        
        # Perform Isaac ROS-like processing
        # This is a simplified example - real Isaac ROS includes hardware-accelerated processing
        processed_image = self.process_image(cv_image)
        
        # Publish processed image
        processed_msg = self.bridge.cv2_to_imgmsg(processed_image, encoding='rgb8')
        processed_msg.header = msg.header
        self.image_pub.publish(processed_msg)
    
    def camera_info_callback(self, msg):
        """Process camera calibration information"""
        self.camera_matrix = np.array(msg.k).reshape(3, 3)
        self.distortion_coeffs = np.array(msg.d)
    
    def process_image(self, image):
        """Example image processing pipeline"""
        # This would typically include:
        # - Rectification using camera calibration
        # - Feature detection
        # - Object detection using accelerated AI
        # - Preprocessing for downstream tasks
        
        # For this example, just perform basic processing
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Edge detection
        edges = cv2.Canny(blurred, 50, 150)
        
        # Convert back to 3-channel for visualization
        result = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        
        return result

class IsaacROSVSLAM(Node):
    def __init__(self):
        super().__init__('isaac_ros_vs_slam')
        
        # Isaac ROS Visual Slam uses stereo cameras or RGB-D
        self.left_image_sub = self.create_subscription(
            Image, 
            '/stereo_camera/left/image_rect_color', 
            self.left_image_callback, 
            10
        )
        
        self.right_image_sub = self.create_subscription(
            Image, 
            '/stereo_camera/right/image_rect_color', 
            self.right_image_callback, 
            10
        )
        
        # Publish pose estimates
        self.pose_pub = self.create_publisher(PoseStamped, 'camera_pose', 10)
        
        # Initialize VSLAM algorithm
        self.setup_vs_slam()
        
        self.get_logger().info('Isaac ROS VSLAM Node initialized')
    
    def setup_vs_slam(self):
        """Setup visual SLAM algorithm with Isaac optimizations"""
        # Isaac ROS VSLAM typically uses hardware-accelerated features
        # like ORB-SLAM with GPU acceleration
        pass
    
    def left_image_callback(self, msg):
        """Process left camera image for stereo SLAM"""
        # Store left image for stereo processing
        pass
    
    def right_image_callback(self, msg):
        """Process right camera image for stereo SLAM"""
        # Perform stereo matching with stored left image
        pass
    
    def estimate_pose(self, left_image, right_image):
        """Estimate camera pose using stereo vision"""
        # Implementation would use Isaac's optimized VSLAM algorithms
        pass
```

### ℹ️ Isaac ROS Manipulation ℹ️

```python
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, WrenchStamped
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from std_srvs.srv import SetBool
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal

class IsaacROSManipulationController(Node):
    def __init__(self):
        super().__init__('isaac_ros_manipulation_controller')
        
        # Publishers for joint control
        self.joint_cmd_pub = self.create_publisher(JointTrajectory, 
                                                  'arm_controller/joint_trajectory', 10)
        
        # Subscribers for feedback
        self.joint_state_sub = self.create_subscription(
            JointState, 'joint_states', self.joint_state_callback, 10
        )
        
        # Service for grasping
        self.grasp_service = self.create_service(
            SetBool, 'execute_grasp', self.execute_grasp_callback
        )
        
        # Action client for trajectory execution
        self.trajectory_client = ActionClient(
            self, FollowJointTrajectory, 'arm_controller/follow_joint_trajectory'
        )
        
        # Current joint state
        self.current_joint_positions = {}
        self.current_joint_velocities = {}
        self.current_joint_efforts = {}
        
        self.get_logger().info('Isaac ROS Manipulation Controller initialized')
    
    def joint_state_callback(self, msg):
        """Update current joint state"""
        for i, name in enumerate(msg.name):
            if i < len(msg.position):
                self.current_joint_positions[name] = msg.position[i]
            if i < len(msg.velocity):
                self.current_joint_velocities[name] = msg.velocity[i]
            if i < len(msg.effort):
                self.current_joint_efforts[name] = msg.effort[i]
    
    def move_to_pose(self, target_pose, cartesian_path=True):
        """Move manipulator to target Cartesian pose"""
        if cartesian_path:
            # Plan Cartesian path to avoid obstacles
            joint_trajectory = self.cartesian_to_joint_trajectory(target_pose)
        else:
            # Use inverse kinematics for direct pose
            joint_positions = self.inverse_kinematics(target_pose)
            joint_trajectory = self.create_single_point_trajectory(joint_positions)
        
        self.execute_trajectory(joint_trajectory)
    
    def cartesian_to_joint_trajectory(self, target_pose):
        """Plan a Cartesian path and convert to joint-space trajectory"""
        # This would typically use MoveIt2 or other path planning libraries
        # integrated with Isaac's perception system
        
        # For this example, we'll just do a simple IK solution
        return self.simple_cartesian_path_planner(target_pose)
    
    def inverse_kinematics(self, pose):
        """Compute inverse kinematics for target pose"""
        # Isaac ROS includes optimized IK solvers
        # This is a simplified example
        pass
    
    def create_single_point_trajectory(self, joint_positions):
        """Create trajectory with single point"""
        traj = JointTrajectory()
        traj.joint_names = list(joint_positions.keys())
        
        point = JointTrajectoryPoint()
        point.positions = list(joint_positions.values())
        point.time_from_start.sec = 5  # 5 seconds to reach position
        traj.points = [point]
        
        return traj
    
    def execute_trajectory(self, trajectory):
        """Execute joint trajectory"""
        # Wait for action server
        self.trajectory_client.wait_for_server()
        
        # Create goal
        goal = FollowJointTrajectoryGoal()
        goal.trajectory = trajectory
        goal.goal_time_tolerance.sec = 5
        
        # Send goal
        future = self.trajectory_client.send_goal_async(goal)
        
        # Wait for result
        rclpy.spin_until_future_complete(self, future)
        result = future.result()
        
        return result
    
    def execute_grasp_callback(self, request, response):
        """Execute grasp based on request"""
        if request.data:  # Execute grasp
            # Move to pre-grasp position
            pre_grasp_pose = self.get_pre_grasp_pose()
            self.move_to_pose(pre_grasp_pose)
            
            # Close gripper
            self.close_gripper()
            
            # Lift object
            lift_pose = self.get_lift_pose()
            self.move_to_pose(lift_pose)
            
            response.success = True
            response.message = "Grasp completed successfully"
        else:  # Release object
            # Open gripper
            self.open_gripper()
            
            response.success = True
            response.message = "Object released"
        
        return response
    
    def close_gripper(self):
        """Close the robot gripper"""
        # Implementation to close gripper
        pass
    
    def open_gripper(self):
        """Open the robot gripper"""
        # Implementation to open gripper
        pass
    
    def get_pre_grasp_pose(self):
        """Calculate pre-grasp pose based on object position"""
        # Would use Isaac perception system to find object
        pass
    
    def get_lift_pose(self):
        """Calculate lift pose after grasp"""
        # Would lift object by small amount
        pass
```

## 🤖 AI Navigation & Control 🤖

### 🧭 Isaac Navigation Stack 🧭

Isaac Sim includes advanced navigation capabilities that leverage GPU acceleration and AI:

```python
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import OccupancyGrid, Path
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool
import numpy as np
import cv2
from pathlib import Path

class IsaacNavigationController(Node):
    def __init__(self):
        super().__init__('isaac_navigation_controller')
        
        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.goal_pub = self.create_publisher(PoseStamped, 'goal_pose', 10)
        self.local_map_pub = self.create_publisher(OccupancyGrid, 'local_costmap/costmap', 10)
        
        # Subscribers
        self.scan_sub = self.create_subscription(
            LaserScan, 'scan', self.scan_callback, 10
        )
        self.odom_sub = self.create_subscription(
            Odometry, 'odom', self.odom_callback, 10
        )
        
        # Services
        self.nav_to_pose_srv = self.create_service(
            Pose, 'nav_to_pose', self.navigate_to_pose
        )
        
        # Navigation state
        self.current_pose = None
        self.current_scan = None
        self.global_plan = None
        self.local_plan = None
        
        # Navigation parameters
        self.linear_speed = 0.5
        self.angular_speed = 0.5
        self.safety_distance = 0.5
        self.arrival_threshold = 0.2
        
        # Setup navigation components
        self.setup_costmap()
        self.setup_path_planner()
        self.setup_local_controller()
        
        self.get_logger().info('Isaac Navigation Controller initialized')
    
    def setup_costmap(self):
        """Setup local and global costmaps with Isaac optimizations"""
        # Isaac provides GPU-accelerated costmap computation
        # This is a simplified example of the concept
        self.local_costmap = LocalCostmap(
            resolution=0.05,
            width=200,  # 10m x 10m at 5cm resolution
            height=200,
            robot_radius=0.3
        )
        
        self.global_costmap = GlobalCostmap(
            resolution=0.1,
            width=1000,  # 100m x 100m at 10cm resolution
            height=1000
        )
    
    def setup_path_planner(self):
        """Setup path planner with Isaac optimizations"""
        # Isaac includes GPU-accelerated path planners
        # such as multi-criteria planners that consider multiple objectives
        self.planner = IsaacPathPlanner(
            costmap_resolution=0.1,
            planner_type='multicriteria',  # Optimizes for multiple objectives
            gpu_accelerated=True
        )
    
    def setup_local_controller(self):
        """Setup local controller for path following"""
        # Isaac provides multiple local controllers optimized for different scenarios
        self.local_controller = IsaacLocalPlanner(
            controller_type='teb',  # Timed Elastic Band
            max_vel_x=0.5,
            max_vel_theta=1.0,
            min_turn_radius=0.2
        )
    
    def scan_callback(self, msg):
        """Process laser scan data for obstacle detection"""
        self.current_scan = msg
        
        # Update local costmap with scan data
        self.update_local_costmap(msg)
    
    def odom_callback(self, msg):
        """Process odometry data for localization"""
        self.current_pose = msg.pose.pose
        
        # Update local planner with current pose
        self.local_controller.update_pose(self.current_pose)
    
    def update_local_costmap(self, scan_msg):
        """Update local costmap with laser scan data"""
        # Convert scan to obstacle points
        angles = np.linspace(
            scan_msg.angle_min, 
            scan_msg.angle_max, 
            len(scan_msg.ranges)
        )
        
        # Get obstacle positions in robot frame
        ranges = np.array(scan_msg.ranges)
        valid_indices = (ranges > scan_msg.range_min) & (ranges < scan_msg.range_max)
        
        if np.any(valid_indices):
            x_points = ranges[valid_indices] * np.cos(angles[valid_indices])
            y_points = ranges[valid_indices] * np.sin(angles[valid_indices])
            
            # Transform to map frame if needed
            # Update costmap with obstacle information
            obstacle_points = np.column_stack((x_points, y_points))
            self.local_costmap.add_obstacles(obstacle_points, self.current_pose)
        
        # Publish updated local costmap
        costmap_msg = self.local_costmap.to_ros_msg()
        costmap_msg.header.stamp = self.get_clock().now().to_msg()
        costmap_msg.header.frame_id = 'map'
        self.local_map_pub.publish(costmap_msg)
    
    def navigate_to_pose(self, request, response):
        """Navigate to goal pose"""
        goal_pose = request.pose
        
        # Plan global path
        if not self.plan_global_path(goal_pose):
            response.success = False
            response.message = "Failed to find global path"
            return response
        
        # Execute navigation
        navigation_success = self.follow_path()
        
        response.success = navigation_success
        if navigation_success:
            response.message = "Navigation completed successfully"
        else:
            response.message = "Navigation failed to reach goal"
        
        return response
    
    def plan_global_path(self, goal_pose):
        """Plan global path to goal"""
        if self.current_pose is None:
            self.get_logger().warn('Current pose not available')
            return False
        
        # Use Isaac's GPU-accelerated A* or other planners
        path = self.planner.plan(
            start_pose=self.current_pose,
            goal_pose=goal_pose,
            use_gpu=True
        )
        
        if path is not None:
            self.global_plan = path
            self.get_logger().info(f'Global path planned with {len(path.poses)} waypoints')
            return True
        else:
            self.get_logger().error('Failed to plan global path')
            return False
    
    def follow_path(self):
        """Follow the planned path using local controller"""
        if self.global_plan is None:
            self.get_logger().warn('No global plan to follow')
            return False
        
        # Break down global plan into local segments
        for i, waypoint in enumerate(self.global_plan.poses):
            # Check if we've reached this waypoint
            distance = self.calculate_distance(
                self.current_pose.position,
                waypoint.pose.position
            )
            
            if distance < self.arrival_threshold:
                continue
            
            # Generate local plan to this waypoint
            local_goal = waypoint.pose
            local_plan = self.local_controller.plan_to_waypoint(
                current_pose=self.current_pose,
                goal_pose=local_goal
            )
            
            # Execute local plan
            if not self.execute_local_plan(local_plan):
                self.get_logger().warn(f'Failed to execute local plan to waypoint {i}')
                return False
        
        # If we've reached all waypoints, we've arrived
        final_distance = self.calculate_distance(
            self.current_pose.position,
            self.global_plan.poses[-1].pose.position
        )
        
        if final_distance < self.arrival_threshold:
            self.get_logger().info('Successfully reached goal')
            return True
        else:
            self.get_logger().warn('Did not reach final goal')
            return False
    
    def execute_local_plan(self, local_plan):
        """Execute local plan by sending velocity commands"""
        if local_plan is None:
            return False
        
        # Convert path to velocity commands using Isaac's local controller
        velocity_cmd = self.local_controller.compute_velocity_command(
            current_pose=self.current_pose,
            local_plan=local_plan
        )
        
        # Check for obstacles
        if self.detect_immediate_obstacles():
            # Emergency stop
            stop_cmd = Twist()
            self.cmd_vel_pub.publish(stop_cmd)
            return False
        
        # Publish velocity command
        self.cmd_vel_pub.publish(velocity_cmd)
        
        return True
    
    def detect_immediate_obstacles(self):
        """Detect obstacles immediately ahead of robot"""
        if self.current_scan is None:
            return False
        
        # Check laser scan for obstacles in front of robot
        # Front sector: -30 to 30 degrees
        front_sector = (self.current_scan.angle_min + 30*np.pi/180, 
                       self.current_scan.angle_min - 30*np.pi/180)
        
        # Get ranges in front sector
        angles = np.linspace(self.current_scan.angle_min, 
                            self.current_scan.angle_max, 
                            len(self.current_scan.ranges))
        front_indices = np.where((angles >= -30*np.pi/180) & 
                                (angles <= 30*np.pi/180))[0]
        
        if len(front_indices) > 0:
            front_ranges = np.array(self.current_scan.ranges)[front_indices]
            min_front_range = np.min(front_ranges[np.isfinite(front_ranges)])
            
            if min_front_range < self.safety_distance:
                return True
        
        return False
    
    def calculate_distance(self, pos1, pos2):
        """Calculate distance between two positions"""
        dx = pos1.x - pos2.x
        dy = pos1.y - pos2.y
        dz = pos1.z - pos2.z
        return np.sqrt(dx*dx + dy*dy + dz*dz)

class LocalCostmap:
    """Simplified local costmap implementation"""
    def __init__(self, resolution, width, height, robot_radius):
        self.resolution = resolution
        self.width = width
        self.height = height
        self.robot_radius = robot_radius
        
        # Center of costmap is robot position
        self.center_x = width // 2
        self.center_y = height // 2
        
        self.costmap = np.zeros((height, width), dtype=np.uint8)
    
    def add_obstacles(self, obstacle_points, robot_pose):
        """Add obstacle points to costmap"""
        # Convert obstacle points from robot frame to costmap coordinates
        for point in obstacle_points:
            # Calculate map coordinates
            map_x = int(self.center_x + point[0] / self.resolution)
            map_y = int(self.center_y + point[1] / self.resolution)
            
            # Check bounds
            if 0 <= map_x < self.width and 0 <= map_y < self.height:
                # Add obstacle with inflation
                self.inflate_obstacle(map_x, map_y, self.robot_radius / self.resolution)
    
    def inflate_obstacle(self, x, y, radius):
        """Inflate obstacle with robot radius"""
        radius_int = int(radius)
        y, x = np.ogrid[-y:self.height-y, -x:self.width-x]
        mask = x*x + y*y <= radius_int*radius_int
        self.costmap[mask] = 254  # Mark as occupied
    
    def to_ros_msg(self):
        """Convert to ROS OccupancyGrid message"""
        from nav_msgs.msg import OccupancyGrid
        from std_msgs.msg import Header
        
        msg = OccupancyGrid()
        msg.header = Header()
        msg.info.resolution = self.resolution
        msg.info.width = self.width
        msg.info.height = self.height
        # Set origin appropriately
        msg.data = self.costmap.flatten().tolist()
        
        return msg

class IsaacPathPlanner:
    """Placeholder for Isaac's GPU-accelerated path planner"""
    def __init__(self, costmap_resolution, planner_type='multicriteria', gpu_accelerated=True):
        self.costmap_resolution = costmap_resolution
        self.planner_type = planner_type
        self.gpu_accelerated = gpu_accelerated
    
    def plan(self, start_pose, goal_pose, use_gpu=True):
        """Plan path from start to goal"""
        # This would use Isaac's optimized path planning algorithms
        # In a real implementation, this would interface with Isaac's planners
        path_msg = Path()
        path_msg.header.frame_id = 'map'
        
        # Create a simple straight-line path as example
        # In reality, this would be computed using sophisticated planners
        path_msg.poses = self.generate_straight_path(start_pose, goal_pose)
        
        return path_msg
    
    def generate_straight_path(self, start_pose, goal_pose):
        """Generate a straight-line path (for demonstration)"""
        from geometry_msgs.msg import PoseStamped
        import math
        
        num_waypoints = 10
        path_poses = []
        
        start_x = start_pose.position.x
        start_y = start_pose.position.y
        goal_x = goal_pose.position.x
        goal_y = goal_pose.position.y
        
        for i in range(num_waypoints + 1):
            t = i / num_waypoints
            x = start_x + t * (goal_x - start_x)
            y = start_y + t * (goal_y - start_y)
            
            pose_stamped = PoseStamped()
            pose_stamped.pose.position.x = x
            pose_stamped.pose.position.y = y
            pose_stamped.pose.position.z = 0.0
            
            # Simple orientation toward goal (simplified)
            angle = math.atan2(goal_y - start_y, goal_x - start_x)
            from tf_transformations import quaternion_from_euler
            quat = quaternion_from_euler(0, 0, angle)
            pose_stamped.pose.orientation.x = quat[0]
            pose_stamped.pose.orientation.y = quat[1]
            pose_stamped.pose.orientation.z = quat[2]
            pose_stamped.pose.orientation.w = quat[3]
            
            path_poses.append(pose_stamped)
        
        return path_poses
```

## 📈 Performance Optimization 📈

### 📈 Isaac Sim Performance Tuning 📈

Isaac Sim can be optimized for various use cases:

```python
from omni.isaac.core.utils.settings import set_carb_setting
import carb

class IsaacSimOptimizer:
    def __init__(self):
        """Initialize Isaac Sim performance optimizer"""
        self.settings = carb.settings.get_settings()
    
    def optimize_for_training(self):
        """Optimize settings for reinforcement learning training"""
        # Disable rendering for faster physics simulation
        self.settings.set("/app/renderer/enabled", False)
        self.settings.set("/app/isaacsim/render_frequency", 10)  # Only render occasionally
        
        # Increase physics substeps for stability
        self.settings.set("/physics/solverType", 0)  # TGS solver for stability
        self.settings.set("/physics/solverPositionIterationCount", 8)
        self.settings.set("/physics/solverVelocityIterationCount", 4)
        
        # Optimize physics for training
        self.settings.set("/physics/worker_thread_count", 2)
        self.settings.set("/physics/frictionModel", "lcp")
        
        # Disable unnecessary systems
        self.settings.set("/app/show_developer_menu", False)
        self.settings.set("/app/show_timeline", False)
        
        # Optimize for high-frequency simulation
        self.settings.set("/app/play_update_frequency", 60)  # Match physics frequency
    
    def optimize_for_visualization(self):
        """Optimize settings for high-quality visualization"""
        # Enable rendering
        self.settings.set("/app/renderer/enabled", True)
        self.settings.set("/app/isaacsim/render_frequency", 60)  # Match display frequency
        
        # Enable advanced rendering features
        self.settings.set("/rtx/enable_super_sampling", True)
        self.settings.set("/rtx/enable_denoise", True)
        self.settings.set("/rtx/enable_ray_tracing", True)
        
        # Quality settings
        self.settings.set("/renderer/antiAliasing", 2)  # MSAA 2x
        self.settings.set("/renderer/resolution/width", 1280)
        self.settings.set("/renderer/resolution/height", 720)
        
        # Enable visual debugging aids
        self.settings.set("/app/show_physics_visualization", True)
        self.settings.set("/app/show_collision_meshes", False)
    
    def optimize_for_sensor_simulation(self):
        """Optimize settings for sensor simulation"""
        # Balance quality and performance for sensors
        self.settings.set("/app/isaacsim/render_frequency", 30)  # Balance performance and quality
        
        # Optimize specific sensors
        self.settings.set("/app/sensor/async_mode", True)  # Async sensor updates
        self.settings.set("/app/sensor/max_update_frequency", 30)  # Match sensor rates
        
        # Physics settings for stable sensor readings
        self.settings.set("/physics/stepSize", 1.0/60.0)  # 60Hz physics
        
        # Sensor-specific optimizations
        self.settings.set("/app/sim/sensor_update_rate", 30)
        self.settings.set("/app/sim/lidar_update_rate", 10)
    
    def enable_multi_gpu(self):
        """Enable multi-GPU if available (requires compatible hardware)"""
        self.settings.set("/app/renderer/multi_gpu/enabled", True)
        self.settings.set("/app/renderer/multi_gpu/primary_gpu", 0)
        self.settings.set("/app/renderer/multi_gpu/physx_gpu", 0)  # GPU 0 for physics
    
    def memory_management(self):
        """Optimize memory usage"""
        # USD stage optimization
        self.settings.set("/app/usd/cache_size", 512 * 1024 * 1024)  # 512MB cache
        
        # Physics memory
        self.settings.set("/physics/convex_mesh_cache_size", 128)
        self.settings.set("/physics/triangle_mesh_cache_size", 128)
        
        # Renderer memory
        self.settings.set("/renderer/max_gpu_memory_allocation", 0.9)  # Use 90% of GPU memory

def setup_simulation_optimization(simulation_type="training"):
    """Configure Isaac Sim for specific use case"""
    optimizer = IsaacSimOptimizer()
    
    if simulation_type == "training":
        optimizer.optimize_for_training()
        print("Configured Isaac Sim for reinforcement learning training")
    elif simulation_type == "visualization":
        optimizer.optimize_for_visualization()
        print("Configured Isaac Sim for high-quality visualization")
    elif simulation_type == "sensor_simulation":
        optimizer.optimize_for_sensor_simulation()
        print("Configured Isaac Sim for sensor simulation")
    elif simulation_type == "benchmarking":
        # Optimize for performance benchmarking
        optimizer.optimize_for_training()  # Same as training
        print("Configured Isaac Sim for performance benchmarking")

# ℹ️ Usage example ℹ️
setup_simulation_optimization(simulation_type="training")
```

## ℹ️ Sim-to-Real Transfer ℹ️

###  🔧 Techniques for Improving Reality Gap  🔧

```python
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
import random

class DomainRandomizationEngine:
    def __init__(self):
        """Engine for applying domain randomization techniques"""
        self.randomization_params = {
            # Lighting parameters for photorealistic rendering
            'light_intensity_range': (1000, 5000),
            'light_color_temperature': (3000, 8000),  # Kelvin
            'light_direction_variance': (0.1, 0.5),
            
            # Texture parameters
            'texture_brightness_range': (0.5, 1.5),
            'texture_contrast_range': (0.8, 1.2),
            'texture_saturation_range': (0.8, 1.2),
            
            # Geometric parameters
            'object_scale_range': (0.8, 1.2),
            'object_position_jitter': (0.02, 0.02, 0.02),
            
            # Sensor parameters
            'camera_noise_std': (0.001, 0.01),
            'laser_noise_std': (0.001, 0.005),
            
            # Physics parameters
            'friction_range': (0.1, 1.0),
            'restitution_range': (0.0, 0.5),
            'mass_variance': 0.1  # 10% variance
        }
    
    def randomize_lighting(self, light_prim):
        """Apply random lighting parameters"""
        # Randomize light intensity
        intensity = random.uniform(
            self.randomization_params['light_intensity_range'][0],
            self.randomization_params['light_intensity_range'][1]
        )
        intensity_attr = light_prim.GetAttribute("inputs:intensity")
        if intensity_attr:
            intensity_attr.Set(intensity)
        
        # Randomize light color (temperature-based)
        temp = random.uniform(
            self.randomization_params['light_color_temperature'][0],
            self.randomization_params['light_color_temperature'][1]
        )
        # Convert temperature to RGB approx.
        rgb = self.color_temperature_to_rgb(temp)
        
        # Apply to light
        color_attr = light_prim.GetAttribute("inputs:color")
        if color_attr:
            color_attr.Set(carb.Float3(rgb))
    
    def color_temperature_to_rgb(self, temp_kelvin):
        """Approximate conversion from color temperature to RGB"""
        temp_kelvin /= 100
        
        if temp_kelvin <= 66:
            red = 255
        else:
            red = temp_kelvin - 60
            red = 329.698727446 * (red ** -0.1332047592)
        
        if temp_kelvin <= 66:
            green = temp_kelvin
            green = 99.4708025861 * np.log(green) - 161.1195681661
        else:
            green = temp_kelvin - 60
            green = 288.1221695283 * (green ** -0.0755148492)
        
        if temp_kelvin >= 66:
            blue = 255
        elif temp_kelvin <= 19:
            blue = 0
        else:
            blue = temp_kelvin - 10
            blue = 138.5177312231 * np.log(blue) - 305.0447927307
        
        return [max(0, min(255, c))/255.0 for c in [red, green, blue]]
    
    def randomize_material_properties(self, material_prim):
        """Apply randomization to material properties"""
        # Randomize surface properties
        brightness = random.uniform(
            self.randomization_params['texture_brightness_range'][0],
            self.randomization_params['texture_brightness_range'][1]
        )
        
        contrast = random.uniform(
            self.randomization_params['texture_contrast_range'][0],
            self.randomization_params['texture_contrast_range'][1]
        )
        
        saturation = random.uniform(
            self.randomization_params['texture_saturation_range'][0],
            self.randomization_params['texture_saturation_range'][1]
        )
        
        # Apply these randomizations through shader parameters
        # Implementation would depend on specific material shader
        pass
    
    def randomize_object_physical_properties(self, rigid_body_prim):
        """Randomize physical properties of rigid bodies"""
        # Get current mass
        mass_api = UsdPhysics.MassAPI(rigid_body_prim)
        current_mass_attr = mass_api.GetMassAttr()
        current_mass = current_mass_attr.Get()
        
        # Apply mass variance
        mass_variance = self.randomization_params['mass_variance']
        new_mass = current_mass * random.uniform(1 - mass_variance, 1 + mass_variance)
        current_mass_attr.Set(new_mass)
        
        # Randomize friction
        friction_attr = rigid_body_prim.GetAttribute("physics:staticFriction")
        if friction_attr:
            new_friction = random.uniform(
                self.randomization_params['friction_range'][0],
                self.randomization_params['friction_range'][1]
            )
            friction_attr.Set(new_friction)
        
        # Randomize restitution
        restitution_attr = rigid_body_prim.GetAttribute("physics:restitution")
        if restitution_attr:
            new_restitution = random.uniform(
                self.randomization_params['restitution_range'][0],
                self.randomization_params['restitution_range'][1]
            )
            restitution_attr.Set(new_restitution)

class SyntheticToRealAdapter:
    def __init__(self):
        """Adapt synthetic data to be more realistic"""
        self.noise_models = {
            'camera': self.camera_noise_model,
            'lidar': self.lidar_noise_model,
            'imu': self.imu_noise_model
        }
    
    def add_camera_noise(self, image_tensor):
        """Add realistic noise to camera images"""
        # Add different types of noise that real cameras have
        img_np = image_tensor.numpy()
        
        # Add Gaussian noise
        gaussian_noise = np.random.normal(
            0, 
            random.uniform(0.001, 0.01),  # std dev based on config
            img_np.shape
        )
        img_noisy = img_np + gaussian_noise
        
        # Add salt and pepper noise
        prob_salt = random.uniform(0, 0.001)
        prob_pepper = random.uniform(0, 0.001)
        
        random_matrix = np.random.random(img_np.shape[:2])
        img_noisy[random_matrix < prob_salt] = 1.0
        img_noisy[random_matrix > 1 - prob_pepper] = 0.0
        
        # Add blur to simulate lens imperfections
        kernel_size = random.choice([3, 5])
        sigma = random.uniform(0.1, 0.5)
        
        # Apply Gaussian blur
        img_blur = cv2.GaussianBlur(img_noisy, (kernel_size, kernel_size), sigma)
        
        return torch.tensor(img_blur)
    
    def camera_noise_model(self, image):
        """Complete camera noise model"""
        # Apply multiple noise transformations
        image = self.add_camera_noise(image)
        
        # Add other real-world effects
        image = self.simulate_motion_blur(image)
        image = self.add_vignetting(image)
        
        return image
    
    def simulate_motion_blur(self, image):
        """Simulate motion blur from camera/scene motion"""
        kernel_size = random.randint(3, 7)
        
        # Create motion blur kernel
        kernel = np.zeros((kernel_size, kernel_size))
        kernel[int((kernel_size-1)/2), :] = np.ones(kernel_size)
        kernel = kernel / kernel_size
        
        # Apply convolution
        image_np = image.numpy()
        if len(image_np.shape) == 3:  # 3-channel image
            for i in range(3):  # Apply to each channel
                image_np[:, :, i] = cv2.filter2D(image_np[:, :, i], -1, kernel)
        else:  # Single channel
            image_np = cv2.filter2D(image_np, -1, kernel)
        
        return torch.tensor(image_np)
    
    def add_vignetting(self, image):
        """Add vignetting effect (darkening at corners)"""
        h, w = image.shape[:2]
        
        # Create coordinate grids
        x = np.arange(w)
        y = np.arange(h)
        x_grid, y_grid = np.meshgrid(x, y)
        
        # Calculate distances from center
        x_center = w / 2
        y_center = h / 2
        distances = np.sqrt((x_grid - x_center)**2 + (y_grid - y_center)**2)
        
        # Maximum possible distance (corner to center)
        max_dist = np.sqrt(x_center**2 + y_center**2)
        
        # Create vignette mask (1 at center, 0 at corners)
        intensity = random.uniform(0.1, 0.4)  # How much to darken corners
        vignette = 1 - (distances / max_dist) * intensity
        
        # Apply to image
        image_np = image.numpy()
        if len(image_np.shape) == 3:  # Color image
            for i in range(3):
                image_np[:, :, i] *= vignette
        else:  # Grayscale
            image_np *= vignette
        
        return torch.tensor(image_np)
    
    def lidar_noise_model(self, pointcloud):
        """Add realistic noise to LiDAR point clouds"""
        # In simulation, we have perfect depth information
        # In reality, LiDAR has various noise sources:
        # - Range measurement noise
        # - Angular resolution limitations
        # - Multi-path effects
        # - Weather effects
        
        points = pointcloud.copy()
        
        # Add range-dependent noise (noise increases with distance)
        distances = np.linalg.norm(points[:, :3], axis=1)
        noise_std = 0.001 + 0.005 * (distances / 20.0)  # 1mm base, up to 6mm at 20m
        
        for i, std in enumerate(noise_std):
            points[i, :3] += np.random.normal(0, std, 3)
        
        # Add spurious points (like those caused by multi-path effects)
        if random.random() < 0.05:  # 5% chance of spurious points
            num_spurious = random.randint(1, 10)
            spurious_points = np.random.random((num_spurious, points.shape[1])) * 20  # Random points within 20m
            points = np.vstack([points, spurious_points])
        
        return points
    
    def adapt_for_real_world(self, synthetic_data, sensor_type):
        """Adapt synthetic data to be more realistic for the specified sensor type"""
        if sensor_type in self.noise_models:
            return self.noise_models[sensor_type](synthetic_data)
        else:
            return synthetic_data  # Return unchanged if no adapter exists

class RealityGapMinimizer:
    def __init__(self):
        """Class to minimize the sim-to-real gap"""
        self.domain_rand_engine = DomainRandomizationEngine()
        self.adapter = SyntheticToRealAdapter()
        
        # Store statistics from sim and real environments
        self.sim_statistics = {}
        self.real_statistics = {}
    
    def collect_statistical_data(self):
        """Collect statistical characteristics of sim vs real"""
        # This would run analysis comparing synthetic and real data
        # to identify key differences to focus on
        pass
    
    def generate_randomized_scenes(self, num_scenes=1000):
        """Generate varied scenes with domain randomization"""
        for i in range(num_scenes):
            # Apply randomizations to scene
            self.domain_rand_engine.randomize_lighting(self.get_random_light())
            self.domain_rand_engine.randomize_material_properties(self.get_random_material())
            self.domain_rand_engine.randomize_object_physical_properties(self.get_random_object())
            
            # Collect data from this randomized scene
            yield self.collect_current_scene_data()
    
    def validate_transfer(self, policy, sim_env, real_env):
        """Validate that a policy trained in simulation works in reality"""
        # Test policy in simulation environment
        sim_performance = self.evaluate_policy(policy, sim_env, episodes=100)
        
        # Test policy in real environment
        real_performance = self.evaluate_policy(policy, real_env, episodes=100) 
        
        # Calculate sim-to-real gap
        gap = abs(sim_performance - real_performance) / max(abs(sim_performance), 1e-6)
        
        print(f"Sim-to-real gap: {gap:.3f}")
        
        # Return True if gap is within acceptable threshold
        return gap < 0.1  # Less than 10% gap is acceptable
    
    def evaluate_policy(self, policy, env, episodes=100):
        """Evaluate policy performance"""
        total_reward = 0
        success_count = 0
        
        for episode in range(episodes):
            obs = env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                action = policy(obs)
                obs, reward, done, info = env.step(action)
                episode_reward += reward
            
            total_reward += episode_reward
            if info.get('success', False):
                success_count += 1
        
        avg_reward = total_reward / episodes
        success_rate = success_count / episodes
        
        return avg_reward
```

## 📝 Chapter Summary 📝

This chapter provided a comprehensive overview of building ROS 2 nodes with Python for Physical AI applications, focusing on the NVIDIA Isaac ecosystem. Key topics covered include:

1. **Python in ROS 2**: Understanding rclpy client library and its advantages for rapid development
2. **Node Architecture**: Building nodes with proper publishers, subscribers, services, and actions
3. **Isaac Sim Integration**: Creating sensor-rich simulation environments with realistic physics
4. **Perception Systems**: Implementing cameras, LiDAR, and IMU sensors with synthetic data generation
5. **Isaac ROS Integration**: Using Isaac's optimized perception and navigation packages
6. **AI Navigation**: Building navigation systems with GPU-accelerated path planning
7. **Performance Optimization**: Configuring Isaac Sim for different use cases (training, visualization, sensor sim)
8. **Sim-to-Real Transfer**: Techniques to minimize the reality gap using domain randomization

The implementation of Physical AI systems using Isaac Sim and ROS 2 requires understanding both the distributed communication architecture and the physics simulation capabilities. Proper use of Isaac's GPU-accelerated components and perception pipelines enables efficient training of real-world capable robotic systems.

## 🤔 Knowledge Check 🤔

1. Explain the differences between ROS 1 and ROS 2 architectures, particularly in relation to distributed computing.
2. Describe the three communication patterns in ROS 2 (topics, services, actions) and when each should be used.
3. What are the advantages of using Isaac Sim over traditional simulators like Gazebo?
4. How does domain randomization help bridge the sim-to-real gap in robotics?
5. What are the key components of the Isaac ROS ecosystem and how do they improve robotic perception?
6. Explain how Quality of Service (QoS) settings impact communication in ROS 2.
7. What are the benefits and challenges of using Python for ROS 2 node development?

### ℹ️ Practical Exercise ℹ️

Create a complete ROS 2 Python node that:
1. Subscribes to sensor data (LiDAR and IMU)
2. Integrates this data to estimate robot pose
3. Publishes velocity commands to navigate to a goal
4. Uses Isaac Sim for simulation with realistic physics
5. Implements basic obstacle avoidance
6. Includes appropriate logging and error handling

### 💬 Discussion Questions 💬

1. How might you design a ROS 2 system to be resilient to sensor failures in a Physical AI application?
2. What are the trade-offs between simulation fidelity and computational performance in Isaac Sim?
3. How could the techniques learned in this chapter apply to other robotic platforms beyond humanoid robots?
4. What challenges arise when scaling from single-robot to multi-robot Isaac Sim environments?