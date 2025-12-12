---
slug: chapter-6-simulation-gazebo-urdf-sdf
title: Chapter 6 - Simulation with Gazebo (URDF/SDF)
description: Comprehensive guide to simulation with Gazebo using URDF and SDF for robotics
tags: [simulation, gazebo, urdf, sdf, robotics, physics, 3d]
---

# 📚 Chapter 6: Simulation with Gazebo (URDF/SDF) 📚

## 🎯 Learning Objectives 🎯

- Understand the role of simulation in Physical AI development
- Master Gazebo simulation for physics, collisions, and sensors
- Learn Unity for high-fidelity visual rendering and VR/AR integration
- Create and configure robot models with URDF/SDF for simulation
- Implement sensor simulation with realistic parameters
- Design simulation environments for physical AI training
- Connect simulation to real ROS 2 systems (sim-to-real transfer)
- Optimize simulation performance for training efficiency
- Implement domain randomization techniques to bridge sim-to-real gap

## 📋 Table of Contents 📋

- [Introduction to Robotics Simulation](#introduction-to-robotics-simulation)
- [Gazebo Simulation Environment](#gazebo-simulation-environment)
- [Unity Simulation Environment](#unity-simulation-environment)
- [URDF & SDF Models](#urdf--sdf-models)
- [Sensor Simulation](#sensor-simulation)
- [Physics Simulation](#physics-simulation)
- [Digital Twin & Environment Design](#digital-twin--environment-design)
- [ROS 2 Integration](#ros-2-integration)
- [Simulation Testing & Validation](#simulation-testing--validation)
- [Sim-to-Real Transfer](#sim-to-real-transfer)
- [Performance Optimization](#performance-optimization)
- [Chapter Summary](#chapter-summary)
- [Knowledge Check](#knowledge-check)

## 👋 Introduction to Robotics Simulation 👋

Simulation plays a critical role in Physical AI development by providing safe, cost-effective, and controllable environments for testing and training robotic systems. In the context of humanoid robotics, simulation is especially valuable as it allows for extensive testing of complex physical interactions without risking expensive hardware or human safety.

### 🎮 Primary Functions of Simulation 🎮

1. **Algorithm Development**: Test new algorithms without hardware constraints
2. **Safety Validation**: Ensure robot behaviors are safe before real-world deployment
3. **Training Data Generation**: Create large datasets for machine learning applications
4. **Performance Evaluation**: Benchmark robot capabilities in controlled scenarios
5. **Multi-Robot Coordination**: Test multi-robot systems with complex interactions

### 🤖 Simulation Challenges in Physical AI 🤖

Simulation for Physical AI presents specific challenges:

- **Reality Gap**: Differences between simulated and real physics, sensors, and dynamics
- **Computational Requirements**: Complex physics simulation demands significant computation
- **Model Fidelity**: Balance between accuracy and performance requirements
- **Transfer Learning**: Ensuring learned behaviors work in real-world scenarios

### 🎮 Simulation Fidelity Levels 🎮

Simulation fidelity can be categorized into:

1. **Kinematic Simulation**: Basic motion without physics forces
2. **Dynamic Simulation**: Includes forces, torques, and physical interactions
3. **Sensor Simulation**: Realistic modeling of real sensors
4. **Environmental Simulation**: Detailed representations of real-world conditions
5. **System-Level Simulation**: Full hardware-in-the-loop simulation

## 🎮 Gazebo Simulation Environment 🎮

Gazebo is the standard simulation environment for ROS 2, providing high-fidelity physics simulation and sensor modeling specifically designed for robotics research and development.

### 🏗️ Gazebo Architecture 🏗️

Gazebo follows a client-server architecture:

```
Gazebo Server (gzserver)
├── Physics Engine (ODE, Bullet, Simbody)
├── Sensor System
├── Model Database
└── Plugin System

Gazebo Client (gzclient)
├── Visualization
├── GUI Controls
└── Rendering (OGRE)

ROS 2 Bridge (ros_gz)
├── Topic Bridges
├── Service Bridges
└── Parameter Bridges
```

### ℹ️ Installing Gazebo Harmonic ℹ️

With ROS 2 Humble, Gazebo Harmonic is included as part of the standard installation:

```bash
# ℹ️ Install Gazebo Harmonic ℹ️
sudo apt update
sudo apt install gz-harmonic

# ℹ️ Install ROS 2 Gazebo bridge packages ℹ️
sudo apt install ros-humble-ros-gz
```

### ℹ️ Basic Gazebo Concepts ℹ️

#### ℹ️ Worlds ℹ️

Worlds define the simulation environment:

```xml
<?xml version="1.0"?>
<sdf version="1.7">
  <world name="humanoid_lab">
    <!-- Include standard models -->
    <include>
      <uri>model://sun</uri>
    </include>
    
    <include>
      <uri>model://ground_plane</uri>
    </include>
    
    <!-- Physics engine configuration -->
    <physics type="ode">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1.0</real_time_factor>
      <real_time_update_rate>1000.0</real_time_update_rate>
      <gravity>0 0 -9.8</gravity>
    </physics>
    
    <!-- Lighting -->
    <light name="main_light" type="directional">
      <pose>0 0 10 0 0 0</pose>
      <diffuse>0.8 0.8 0.8 1</diffuse>
      <specular>0.2 0.2 0.2 1</specular>
      <attenuation>
        <range>1000</range>
      </attenuation>
      <direction>-0.4 0.2 -1.0</direction>
    </light>
    
    <!-- Environment Models -->
    <model name="table">
      <pose>2 0 0 0 0 0</pose>
      <link name="link">
        <collision name="collision">
          <geometry>
            <box>
              <size>1.0 0.6 0.8</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>1.0 0.6 0.8</size>
            </box>
          </geometry>
          <material>
            <ambient>0.8 0.6 0.4 1</ambient>
            <diffuse>0.8 0.6 0.4 1</diffuse>
          </material>
        </visual>
        <inertial>
          <mass>10.0</mass>
          <inertia>
            <ixx>1.0</ixx>
            <ixy>0.0</ixy>
            <ixz>0.0</ixz>
            <iyy>1.0</iyy>
            <iyz>0.0</iyz>
            <izz>1.0</izz>
          </inertia>
        </inertial>
      </link>
    </model>
    
    <!-- Robot spawn point -->
    <state world_name="humanoid_lab">
      <model name="my_humanoid_robot">
        <pose>0 0 1 0 0 0</pose>
      </model>
    </state>
  </world>
</sdf>
```

#### 🏗️ Models 🏗️

Models represent objects in the environment with:

- **Visual**: How the model appears in the simulation
- **Collision**: Physical collision representation
- **Inertial**: Mass properties for physics calculations
- **Links**: Rigid bodies connected by joints
- **Joints**: Connections between links
- **Sensors**: Simulated sensor components

Example model definition:

```xml
<?xml version="1.0"?>
<sdf version="1.7">
  <model name="humanoid_robot">
    <!-- Base link -->
    <link name="base_link">
      <pose>0 0 1.0 0 0 0</pose>
      
      <inertial>
        <mass>10.0</mass>
        <inertia>
          <ixx>0.1</ixx>
          <ixy>0.0</ixy>
          <ixz>0.0</ixz>
          <iyy>0.1</iyy>
          <iyz>0.0</iyz>
          <izz>0.2</izz>
        </inertia>
      </inertial>
      
      <visual name="base_visual">
        <geometry>
          <cylinder>
            <radius>0.15</radius>
            <length>0.3</length>
          </cylinder>
        </geometry>
        <material>
          <ambient>0.1 0.1 0.8 1</ambient>
          <diffuse>0.1 0.1 0.8 1</diffuse>
        </material>
      </visual>
      
      <collision name="base_collision">
        <geometry>
          <cylinder>
            <radius>0.15</radius>
            <length>0.3</length>
          </cylinder>
        </geometry>
      </collision>
    </link>
    
    <!-- Hip joint -->
    <joint name="left_hip_joint" type="revolute">
      <parent>base_link</parent>
      <child>left_thigh</child>
      <axis>
        <xyz>0 1 0</xyz>
        <limit>
          <lower>-1.57</lower>
          <upper>1.57</upper>
          <effort>100.0</effort>
          <velocity>1.0</velocity>
        </limit>
      </axis>
      <pose>-0.1 -0.1 0 0 0 0</pose>
    </joint>
    
    <!-- Thigh link -->
    <link name="left_thigh">
      <inertial>
        <mass>2.0</mass>
        <inertia>
          <ixx>0.01</ixx>
          <ixy>0.0</ixy>
          <ixz>0.0</ixz>
          <iyy>0.01</iyy>
          <iyz>0.0</iyz>
          <izz>0.005</izz>
        </inertia>
      </inertial>
      
      <visual name="thigh_visual">
        <geometry>
          <cylinder>
            <radius>0.05</radius>
            <length>0.4</length>
          </cylinder>
        </geometry>
        <pose>0 0 -0.2 0 1.57 0</pose>
      </visual>
      
      <collision name="thigh_collision">
        <geometry>
          <cylinder>
            <radius>0.05</radius>
            <length>0.4</length>
          </cylinder>
        </geometry>
        <pose>0 0 -0.2 0 1.57 0</pose>
      </collision>
    </link>
    
    <!-- Sensors -->
    <sensor name="rgbd_camera" type="rgbd_camera">
      <pose>0.15 0 0.1 0 0 0</pose>
      <camera>
        <horizontal_fov>1.047</horizontal_fov>
        <image>
          <width>640</width>
          <height>480</height>
        </image>
        <clip>
          <near>0.1</near>
          <far>10.0</far>
        </clip>
      </camera>
    </sensor>
  </model>
</sdf>
```

### 📡 Sensors in Gazebo 📡

Gazebo supports various sensor types:

- **Camera**: RGB and depth cameras
- **LiDAR**: 2D and 3D laser range finders
- **IMU**: Inertial measurement units
- **GPS**: Global positioning systems
- **Contact**: Touch sensors
- **Force/Torque**: Force and torque sensors
- **Sonar**: Ultrasonic range finders
- **Ray**: Generic ray-based sensors

Example LiDAR sensor configuration:

```xml
<sensor name="front_lidar" type="gpu_lidar">
  <pose>0.2 0 0.3 0 0 0</pose>
  <topic>scan</topic>
  <update_rate>10</update_rate>
  <ray>
    <scan>
      <horizontal>
        <samples>360</samples>
        <resolution>1.0</resolution>
        <min_angle>-3.14159</min_angle>  <!-- -π -->
        <max_angle>3.14159</max_angle>   <!-- π -->
      </horizontal>
    </scan>
    <range>
      <min>0.1</min>
      <max>30.0</max>
      <resolution>0.01</resolution>
    </range>
  </ray>
  <always_on>true</always_on>
  <visualize>true</visualize>
  <noise>
    <type>gaussian</type>
    <mean>0.0</mean>
    <stddev>0.01</stddev>
  </noise>
</sensor>
```

### ℹ️ Plugins in Gazebo ℹ️

Gazebo plugins extend functionality:

- **Physics plugins**: Custom physics behaviors
- **Model plugins**: Model-specific behaviors
- **Sensor plugins**: Custom sensor processing
- **GUI plugins**: Interface enhancements

Example physics plugin:

```xml
<plugin name="gravity_compensator" filename="libGravityCompensator.so">
  <gravity_vector>0 0 0</gravity_vector>
</plugin>
```

## 🎮 Unity Simulation Environment 🎮

Unity offers high-fidelity visual rendering and game engine features that complement Gazebo's physics simulation. Unity is increasingly used in robotics for visual realism and VR/AR integration.

### 🤖 Unity for Robotics 🤖

Unity is valuable for:

- **High-fidelity rendering**: Photo-realistic visuals for computer vision training
- **VR/AR integration**: Immersive teleoperation interfaces
- **Synthetic data generation**: Training datasets with perfect ground truth
- **Human-robot interaction**: Natural interaction environments
- **Collaborative simulation**: Multiple users in shared virtual spaces

### 🤖 Unity Robotics Simulation Setup 🤖

Unity Robotics provides the Unity Robotics Simulation Package (URP) for robotics applications:

1. **Install Unity Hub** and a compatible Unity Editor version
2. **Import the Unity Robotics Simulation Package** via Package Manager
3. **Setup ROS TCP Connector** for ROS 2 communication
4. **Configure physics settings** for accurate simulation

### ℹ️ Unity Scene Structure ℹ️

A typical Unity robotics simulation includes:

- **Robot Prefabs**: Reusable robot models with components
- **Environment Assets**: Terrain, buildings, objects
- **Sensor Components**: Camera, LiDAR, IMU simulators
- **Physics Materials**: Friction and bounciness settings
- **Lighting Settings**: Realistic lighting for computer vision

Unity C# script example for robot control:

```csharp
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Geometry;
using RosMessageTypes.Std;

public class UnityRobotController : MonoBehaviour
{
    [SerializeField] private float linearSpeed = 1.0f;
    [SerializeField] private float angularSpeed = 1.0f;
    
    private float linearVelocity = 0.0f;
    private float angularVelocity = 0.0f;
    
    private Rigidbody rb;
    
    void Start()
    {
        rb = GetComponent<Rigidbody>();
        
        // Subscribe to ROS topic for velocity commands
        ROSConnection.instance.Subscribe<TwistMsg>("/cmd_vel", CmdVelCallback);
    }
    
    public void SetVelocity(float linear, float angular)
    {
        linearVelocity = linear;
        angularVelocity = angular;
    }
    
    void FixedUpdate()
    {
        // Apply movement based on velocities
        Vector3 forwardMove = transform.forward * linearVelocity * linearSpeed * Time.fixedDeltaTime;
        transform.position += forwardMove;
        
        // Apply rotation
        float rotation = angularVelocity * angularSpeed * Time.fixedDeltaTime;
        transform.Rotate(Vector3.up, rotation);
    }
    
    void CmdVelCallback(TwistMsg msg)
    {
        SetVelocity((float)msg.linear.x, (float)msg.angular.z);
    }
}
```

### ℹ️ Physics Configuration in Unity ℹ️

Unity's physics system can be configured for robotics:

```csharp
using UnityEngine;

public class RobotPhysicsSettings : MonoBehaviour
{
    [Header("Friction Settings")]
    public PhysicMaterial wheelMaterial;
    
    [Header("Joints Configuration")]
    public ConfigurableJoint[] joints;
    
    [Header("Simulation Parameters")]
    public float fixedTimestep = 0.02f;  // 50 Hz
    public float maxAngularVelocity = 50f;
    
    void Start()
    {
        ConfigurePhysics();
    }
    
    private void ConfigurePhysics()
    {
        // Set global physics settings
        Time.fixedDeltaTime = fixedTimestep;
        Physics.defaultSolverIterations = 10;
        Physics.defaultSolverVelocityIterations = 20;
        Physics.maxAngularVelocity = maxAngularVelocity;
        
        // Configure joint properties
        foreach (var joint in joints)
        {
            joint.configuredInWorldSpace = false;
            joint.projectionMode = JointProjectionMode.PositionAndRotation;
            joint.projectionDistance = 0.1f;
            joint.projectionAngle = 10f;
            
            // Configure joint limits
            SoftJointLimit lowLimit = joint.lowAngularXLimit;
            lowLimit.limit = -Mathf.PI / 2;  // -90 degrees
            joint.lowAngularXLimit = lowLimit;
            
            SoftJointLimit highLimit = joint.highAngularXLimit;
            highLimit.limit = Mathf.PI / 2;  // 90 degrees
            joint.highAngularXLimit = highLimit;
        }
        
        // Configure wheel material properties
        if (wheelMaterial != null)
        {
            wheelMaterial.staticFriction = 1.0f;
            wheelMaterial.dynamicFriction = 0.8f;
            wheelMaterial.frictionCombine = PhysicMaterialCombine.MaximumFriction;
        }
    }
}
```

### 🎮 Sensor Simulation in Unity 🎮

Unity can simulate various sensors:

#### 🎮 Camera Simulation 🎮

```csharp
using UnityEngine;
using UnityEngine.Rendering.HighDefinition;

public class UnityCameraSimulation : MonoBehaviour
{
    [Header("Camera Settings")]
    public int imageWidth = 640;
    public int imageHeight = 480;
    public float fov = 60.0f;
    
    [Header("Noise Parameters")]
    [Range(0.0f, 0.1f)] public float noiseIntensity = 0.02f;
    
    private Camera cam;
    private RenderTexture renderTexture;
    private Texture2D texture2D;
    
    void Start()
    {
        InitializeCamera();
    }
    
    private void InitializeCamera()
    {
        cam = GetComponent<Camera>();
        cam.fieldOfView = fov;
        
        // Create render texture for camera output
        renderTexture = new RenderTexture(imageWidth, imageHeight, 24);
        cam.targetTexture = renderTexture;
        
        texture2D = new Texture2D(imageWidth, imageHeight, TextureFormat.RGB24, false);
    }
    
    // Method to capture and process camera image
    public Texture2D CaptureImage()
    {
        // Set the camera to render to the texture
        RenderTexture.active = renderTexture;
        
        // Render the camera's view
        cam.Render();
        
        // Read from the render texture and save to texture2D
        texture2D.ReadPixels(new Rect(0, 0, imageWidth, imageHeight), 0, 0);
        texture2D.Apply();
        
        // Generate noisy version if needed
        if (noiseIntensity > 0.0f)
        {
            AddNoiseToImage(texture2D);
        }
        
        // Reset active render texture
        RenderTexture.active = null;
        
        return texture2D;
    }
    
    private void AddNoiseToImage(Texture2D tex)
    {
        Color[] pixels = tex.GetPixels();
        
        for (int i = 0; i < pixels.Length; i++)
        {
            float noise = Random.Range(-noiseIntensity, noiseIntensity);
            pixels[i] = new Color(
                Mathf.Clamp01(pixels[i].r + noise),
                Mathf.Clamp01(pixels[i].g + noise),
                Mathf.Clamp01(pixels[i].b + noise),
                pixels[i].a
            );
        }
        
        tex.SetPixels(pixels);
        tex.Apply();
    }
}
```

#### 🎮 LiDAR Simulation 🎮

```csharp
using System.Collections.Generic;
using UnityEngine;

public class UnityLidarSimulation : MonoBehaviour
{
    [Header("LiDAR Configuration")]
    public int numberOfBeams = 360;
    public float minAngle = -Mathf.PI;
    public float maxAngle = Mathf.PI;
    public float maxDistance = 30.0f;
    public LayerMask detectionLayers;
    
    [Header("Noise Parameters")]
    public float rangeNoiseStdDev = 0.01f;
    public float angleNoiseStdDev = 0.001f;
    
    // Store the latest scan data
    private float[] ranges;
    
    void Start()
    {
        ranges = new float[numberOfBeams];
    }
    
    public float[] GetLaserScan()
    {
        for (int i = 0; i < numberOfBeams; i++)
        {
            float angle = Mathf.Lerp(minAngle, maxAngle, (float)i / (numberOfBeams - 1));
            
            // Add some noise to the angle for realistic simulation
            angle += RandomGaussian(angleNoiseStdDev);
            
            Vector3 direction = new Vector3(Mathf.Cos(angle), 0, Mathf.Sin(angle));
            
            RaycastHit hit;
            if (Physics.Raycast(transform.position, direction, out hit, maxDistance, detectionLayers))
            {
                float distance = hit.distance;
                // Add noise to the range measurement
                distance += RandomGaussian(rangeNoiseStdDev);
                ranges[i] = Mathf.Min(distance, maxDistance);
            }
            else
            {
                ranges[i] = maxDistance;
            }
        }
        
        return ranges;
    }
    
    private float RandomGaussian(float stdDev)
    {
        // Box-Muller transform for Gaussian noise
        float u1 = Random.value;
        float u2 = Random.value;
        float gaussian = Mathf.Sqrt(-2.0f * Mathf.Log(u1)) * Mathf.Cos(2.0f * Mathf.PI * u2);
        return gaussian * stdDev;
    }
}
```

## 🏗️ URDF & SDF Models 🏗️

URDF (Unified Robot Description Format) and SDF (Simulation Description Format) are essential for defining robot models in ROS and Gazebo respectively.

### ℹ️ URDF Structure ℹ️

URDF is XML-based and defines robot kinematics and dynamics:

```xml
<?xml version="1.0"?>
<robot name="humanoid_robot">
  <!-- Materials -->
  <material name="blue">
    <color rgba="0.0 0.0 0.8 1.0"/>
  </material>
  <material name="black">
    <color rgba="0.0 0.0 0.0 1.0"/>
  </material>
  
  <!-- Base link -->
  <link name="base_link">
    <visual>
      <geometry>
        <cylinder radius="0.15" length="0.3"/>
      </geometry>
      <material name="blue"/>
    </visual>
    
    <collision>
      <geometry>
        <cylinder radius="0.15" length="0.3"/>
      </geometry>
    </collision>
    
    <inertial>
      <mass value="10.0"/>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <inertia 
        ixx="0.1" ixy="0.0" ixz="0.0"
        iyy="0.1" iyz="0.0"
        izz="0.2"/>
    </inertial>
  </link>
  
  <!-- Spine joint and link -->
  <joint name="torso_joint" type="revolute">
    <parent link="base_link"/>
    <child link="torso_link"/>
    <origin xyz="0 0 0.25" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1.57" upper="1.57" effort="100" velocity="1"/>
  </joint>
  
  <link name="torso_link">
    <visual>
      <geometry>
        <box size="0.3 0.2 0.5"/>
      </geometry>
      <material name="black"/>
    </visual>
    
    <collision>
      <geometry>
        <box size="0.3 0.2 0.5"/>
      </geometry>
    </collision>
    
    <inertial>
      <mass value="5.0"/>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <inertia 
        ixx="0.2" ixy="0.0" ixz="0.0"
        iyy="0.3" iyz="0.0"
        izz="0.1"/>
    </inertial>
  </link>
  
  <!-- Sensors -->
  <gazebo reference="base_link">
    <sensor type="camera" name="rgbd_camera">
      <pose>0.15 0 0.1 0 0 0</pose>
      <camera>
        <horizontal_fov>1.047</horizontal_fov>
        <image>
          <width>640</width>
          <height>480</height>
        </image>
        <clip>
          <near>0.1</near>
          <far>10</far>
        </clip>
      </camera>
      <always_on>true</always_on>
      <update_rate>30.0</update_rate>
      <visualize>true</visualize>
    </sensor>
  </gazebo>
</robot>
```

### ℹ️ SDF Structure ℹ️

SDF is more comprehensive and includes both robot and environment descriptions:

```xml
<?xml version="1.0"?>
<sdf version="1.7">
  <model name="humanoid_with_urdf">
    <include>
      <uri>model://humanoid_robot/model.urdf</uri>
      <pose>0 0 1.0 0 0 0</pose>
    </include>
    
    <!-- Add more links and joints if needed -->
    <link name="head_link">
      <pose>0 0 0.15 0 0 0</pose>
      <inertial>
        <mass>2.0</mass>
        <inertia>
          <ixx>0.01</ixx>
          <ixy>0.0</ixy>
          <ixz>0.0</ixz>
          <iyy>0.01</iyy>
          <iyz>0.0</iyz>
          <izz>0.01</izz>
        </inertia>
      </inertial>
      
      <visual name="head_visual">
        <geometry>
          <sphere>
            <radius>0.1</radius>
          </sphere>
        </geometry>
        <material>
          <ambient>0.8 0.8 0.8 1</ambient>
          <diffuse>0.8 0.8 0.8 1</diffuse>
        </material>
      </visual>
      
      <collision name="head_collision">
        <geometry>
          <sphere>
            <radius>0.1</radius>
          </sphere>
        </geometry>
      </collision>
    </link>
    
    <joint name="neck_joint" type="ball">
      <parent>torso_link</parent>
      <child>head_link</child>
      <pose>0 0 0.25 0 0 0</pose>
    </joint>
  </model>
</sdf>
```

### ℹ️ Xacro for URDF ℹ️

Xacro is a macro language for URDF that allows for reusable robot definitions:

```xml
<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="humanoid_robot">
  
  <!-- Define properties -->
  <xacro:property name="M_PI" value="3.1415926535897931" />
  <xacro:property name="wheel_radius" value="0.05" />
  <xacro:property name="wheel_width" value="0.02" />
  <xacro:property name="base_mass" value="10.0" />
  
  <!-- Macro for wheel -->
  <xacro:macro name="wheel" params="prefix parent xyz joint_limit_position joint_limit_velocity effort velocity">
    <joint name="${prefix}_wheel_joint" type="continuous">
      <parent link="${parent}"/>
      <child link="${prefix}_wheel_link"/>
      <origin xyz="${xyz}" rpy="0 0 0"/>
      <axis xyz="0 1 0"/>
      <limit effort="${effort}" velocity="${velocity}"/>
    </joint>
    
    <link name="${prefix}_wheel_link">
      <visual>
        <geometry>
          <cylinder radius="${wheel_radius}" length="${wheel_width}"/>
        </geometry>
        <material name="black"/>
      </visual>
      <collision>
        <geometry>
          <cylinder radius="${wheel_radius}" length="${wheel_width}"/>
        </geometry>
      </collision>
      <inertial>
        <mass value="${wheel_radius * 2}"/>
        <inertia 
          ixx="${wheel_radius * 0.5}" ixy="0" ixz="0"
          iyy="${wheel_radius * 0.25}" iyz="0"
          izz="${wheel_radius * 0.5}"/>
      </inertial>
    </link>
  </xacro:macro>
  
  <!-- Include the wheel macro twice -->
  <xacro:wheel prefix="left" parent="base_link" 
                xyz="0 0.1 0" 
                effort="100" velocity="1"/>
  <xacro:wheel prefix="right" parent="base_link" 
                xyz="0 -0.1 0" 
                effort="100" velocity="1"/>
  
</robot>
```

### ℹ️ URDF to SDF Conversion ℹ️

When using URDF models in Gazebo, they get converted to SDF internally:

```bash
# ℹ️ Convert URDF to SDF ℹ️
gz sdf -p robot.urdf > robot.sdf

# ℹ️ Or use the ROS 2 utility ℹ️
ros2 run xacro xacro robot.urdf.xacro > robot.urdf
```

## 🎮 Sensor Simulation 🎮

Accurate sensor simulation is crucial for Physical AI development:

### 🎮 Camera Simulation 🎮

Camera sensors in simulation should match real-world properties:

```xml
<sensor name="camera" type="camera">
  <pose>0.2 0 0.3 0 0 0</pose>
  <topic>camera/image_raw</topic>
  <update_rate>30</update_rate>
  <camera name="head_camera">
    <horizontal_fov>1.047</horizontal_fov> <!-- 60 degrees -->
    <image>
      <width>1280</width>
      <height>720</height>
      <format>R8G8B8</format>
    </image>
    <clip>
      <near>0.1</near>
      <far>10.0</far>
    </clip>
    <noise>
      <type>gaussian</type>
      <mean>0.0</mean>
      <stddev>0.007</stddev>
    </noise>
  </camera>
  <always_on>true</always_on>
  <visualize>true</visualize>
</sensor>
```

### 🎮 LiDAR Simulation 🎮

LiDAR simulation requires careful consideration of physical properties:

```xml
<sensor name="3d_lidar" type="gpu_lidar">
  <pose>0.15 0 0.5 0 0 0</pose>
  <topic>scan_3d</topic>
  <update_rate>10</update_rate>
  <ray>
    <scan>
      <horizontal>
        <samples>1080</samples>
        <resolution>1</resolution>
        <min_angle>-3.14159</min_angle> <!-- -π -->
        <max_angle>3.14159</max_angle>  <!-- π -->
      </horizontal>
      <vertical>
        <samples>64</samples>
        <resolution>1</resolution>
        <min_angle>-0.261799</min_angle> <!-- -15 degrees -->
        <max_angle>0.261799</max_angle>  <!-- 15 degrees -->
      </vertical>
    </scan>
    <range>
      <min>0.1</min>
      <max>25.0</max>
      <resolution>0.01</resolution>
    </range>
  </ray>
  <always_on>true</always_on>
  <visualize>true</visualize>
  <noise>
    <type>gaussian</type>
    <mean>0.0</mean>
    <stddev>0.01</stddev>
  </noise>
</sensor>
```

### 🎮 IMU Simulation 🎮

```xml
<sensor name="imu_sensor" type="imu">
  <pose>0 0 0.5 0 0 0</pose>
  <topic>imu</topic>
  <update_rate>100</update_rate>
  <always_on>true</always_on>
  <imu>
    <angular_velocity>
      <x>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>2e-4</stddev>
        </noise>
      </x>
      <y>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>2e-4</stddev>
        </noise>
      </y>
      <z>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>2e-4</stddev>
        </noise>
      </z>
    </angular_velocity>
    <linear_acceleration>
      <x>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>1.7e-2</stddev>
        </noise>
      </x>
      <y>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>1.7e-2</stddev>
        </noise>
      </y>
      <z>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>1.7e-2</stddev>
        </noise>
      </z>
    </linear_acceleration>
  </imu>
</sensor>
```

## 🎮 Physics Simulation 🎮

### ℹ️ Physics Engine Selection ℹ️

Different physics engines offer different trade-offs:

#### ℹ️ ODE (Open Dynamics Engine) ℹ️
- Used by default in Gazebo
- Good for general-purpose simulation
- Conservative but stable
- Good for ground vehicles and basic manipulation

#### ℹ️ Bullet Physics ℹ️
- Faster than ODE
- Better for complex contact scenarios
- Good for humanoid robots and bipedal locomotion
- More complex contact modeling

### ℹ️ Physics Configuration ℹ️

```xml
<physics type="bullet">
  <max_step_size>0.001</max_step_size>
  <real_time_factor>1.0</real_time_factor>
  <real_time_update_rate>1000.0</real_time_update_rate>
  <gravity>0 0 -9.8</gravity>
  
  <bullet>
    <solver>
      <type>sequential_impulse</type>
      <iters>50</iters>
      <sor>1.3</sor>
    </solver>
    <constraints>
      <contact_surface_layer>0.001</contact_surface_layer>
      <contact_max_correcting_vel>100.0</contact_max_correcting_vel>
      <cfm>0.0</cfm>
      <erp>0.2</erp>
      <contact_erp>0.2</contact_erp>
      <contact_cfm>0.0</contact_cfm>
    </constraints>
  </bullet>
</physics>
```

### ℹ️ Material Properties ℹ️

Accurate material properties are essential for realistic simulation:

```xml
<material name="rubber_wheel">
  <pbr>
    <metal>
      <albedo_map>materials/textures/rubber_color.png</albedo_map>
      <roughness>0.8</roughness>
      <metalness>0.0</metalness>
    </metal>
  </pbr>
</material>

<!-- Physically-based friction model -->
<gazebo reference="wheel_link">
  <mu1>1.0</mu1>  <!-- Primary friction coefficient -->
  <mu2>1.0</mu2>  <!-- Secondary friction coefficient -->
  <kp>1000000.0</kp>  <!-- Contact stiffness -->
  <kd>100.0</kd>    <!-- Damping coefficient -->
  <max_vel>100.0</max_vel>
  <min_depth>0.001</min_depth>
</gazebo>
```

## 🎮 Digital Twin & Environment Design 🎮

### 🌍 Creating Realistic Environments 🌍

Digital twins require careful environment design to maximize training effectiveness:

#### 🌍 Indoor Environments 🌍

```xml
<sdf version="1.7">
  <world name="humanoid_laboratory">
    <!-- Include standard models -->
    <include>
      <uri>model://sun</uri>
    </include>
    <include>
      <uri>model://ground_plane</uri>
    </include>
    
    <!-- Laboratory setup -->
    <model name="workbench">
      <pose>2 0 0 0 0 0</pose>
      <link name="workbench_link">
        <collision name="collision">
          <geometry>
            <box>
              <size>1.2 0.6 0.8</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>1.2 0.6 0.8</size>
            </box>
          </geometry>
          <material>
            <ambient>0.6 0.4 0.2 1</ambient>
            <diffuse>0.8 0.6 0.4 1</diffuse>
          </material>
        </visual>
      </link>
    </model>
    
    <!-- Objects for manipulation -->
    <model name="red_box">
      <pose>2.2 0.1 0.9 0 0 0</pose>
      <link name="box_link">
        <collision name="collision">
          <geometry>
            <box>
              <size>0.1 0.1 0.1</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>0.1 0.1 0.1</size>
            </box>
          </geometry>
          <material>
            <ambient>0.8 0.2 0.2 1</ambient>
            <diffuse>1.0 0.3 0.3 1</diffuse>
          </material>
        </visual>
        <inertial>
          <mass>0.1</mass>
          <inertia>
            <ixx>0.00017</ixx>
            <ixy>0.0</ixy>
            <ixz>0.0</ixz>
            <iyy>0.00017</iyy>
            <iyz>0.0</iyz>
            <izz>0.00017</izz>
          </inertia>
        </inertial>
      </link>
    </model>
    
    <!-- Room with obstacles -->
    <model name="obstacle_course">
      <pose>0 3 0 0 0 0</pose>
      <!-- Add various obstacles to challenge navigation -->
    </model>
  </world>
</sdf>
```

#### 🌍 Outdoor Environments 🌍

```xml
<sdf version="1.7">
  <world name="outdoor_park">
    <include>
      <uri>model://sun</uri>
    </include>
    
    <!-- Terraced terrain -->
    <model name="Terrain">
      <static>true</static>
      <link name="terrain_link">
        <collision name="collision">
          <geometry>
            <heightmap>
              <uri>model://outdoor_park/materials/textures/terrain.png</uri>
              <size>100 100 10</size>
              <pos>0 0 0</pos>
            </heightmap>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <heightmap>
              <uri>model://outdoor_park/materials/textures/terrain.png</uri>
              <size>100 100 10</size>
              <pos>0 0 0</pos>
            </heightmap>
          </geometry>
        </visual>
      </link>
    </model>
    
    <!-- Walking path -->
    <model name="walking_path">
      <static>true</static>
      <link name="path_link">
        <collision name="collision">
          <geometry>
            <mesh>
              <uri>model://outdoor_park/meshes/path.dae</uri>
            </mesh>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <mesh>
              <uri>model://outdoor_park/meshes/path.dae</uri>
            </mesh>
          </geometry>
          <material>
            <ambient>0.6 0.6 0.6 1</ambient>
            <diffuse>0.7 0.7 0.7 1</diffuse>
          </material>
        </visual>
      </link>
    </model>
  </world>
```

### ℹ️ Environment Parameterization ℹ️

For Physical AI training, environments should be parameterizable:

```xml
<sdf version="1.7">
  <world name="parametrized_environment">
    
    <!-- Include with custom parameters -->
    <model name="configurable_room">
      <sdf version="1.7">
        <model name="room_with_params" name="{room_name}">
          <link name="floor">
            <collision name="collision">
              <geometry>
                <box>
                  <size>{{room_length}} {{room_width}} 0.1</size>
                </box>
              </geometry>
            </collision>
            <visual name="visual">
              <geometry>
                <box>
                  <size>{{room_length}} {{room_width}} 0.1</size>
                </box>
              </geometry>
              <material>
                <ambient>0.7 0.7 0.7 1</ambient>
                <diffuse>0.8 0.8 0.8 1</diffuse>
              </material>
            </visual>
          </link>
          
          <!-- Parametrized objects -->
          <model name="table_1">
            <pose>{{table_1_x}} {{table_1_y}} 0 0 0 0</pose>
            <link name="table_link">
              <collision name="collision">
                <geometry>
                  <box>
                    <size>{{table_length}} {{table_width}} {{table_height}}</size>
                  </box>
                </geometry>
              </collision>
            </link>
          </model>
        </model>
      </sdf>
    </model>
  </world>
</sdf>
```

## 🔗 ROS 2 Integration 🔗

### ℹ️ ROS 2 Gazebo Bridge ℹ️

The `ros_gz` bridge enables communication between ROS 2 and Gazebo:

```python
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan, Image, Imu
from nav_msgs.msg import Odometry
from tf2_ros import TransformBroadcaster
import tf2_geometry_msgs

class RobotSimulator(Node):
    def __init__(self):
        super().__init__('robot_simulator')
        
        # Subscribe to velocity commands
        self.cmd_vel_sub = self.create_subscription(
            Twist, 'cmd_vel', self.cmd_vel_callback, 10
        )
        
        # Publishers for sensor data
        self.scan_pub = self.create_publisher(LaserScan, 'scan', 10)
        self.imu_pub = self.create_publisher(Imu, 'imu', 10)
        self.odom_pub = self.create_publisher(Odometry, 'odom', 10)
        
        # TF broadcaster
        self.tf_broadcaster = TransformBroadcaster(self)
        
        # Robot state
        self.position = [0.0, 0.0, 0.0]
        self.velocity = [0.0, 0.0, 0.0]  # x, y, theta
        self.linear_cmd = 0.0
        self.angular_cmd = 0.0
        
        # Timer for physics update
        self.timer = self.create_timer(0.01, self.update_physics)  # 100Hz
        
        self.get_logger().info('Robot Simulator Node Initialized')
    
    def cmd_vel_callback(self, msg):
        """Process velocity commands from ROS 2"""
        self.linear_cmd = msg.linear.x
        self.angular_cmd = msg.angular.z
    
    def update_physics(self):
        """Simple differential drive physics model"""
        dt = 0.01  # 100Hz
        
        # Update velocity based on commands (with some smoothing)
        self.velocity[0] += (self.linear_cmd - self.velocity[0]) * 0.1
        self.velocity[2] += (self.angular_cmd - self.velocity[2]) * 0.1
        
        # Update position
        self.position[0] += self.velocity[0] * math.cos(self.position[2]) * dt
        self.position[1] += self.velocity[0] * math.sin(self.position[2]) * dt
        self.position[2] += self.velocity[2] * dt
        
        # Publish odometry
        self.publish_odometry()
        
        # Simulate sensors
        self.simulate_lidar()
        self.simulate_imu()
    
    def publish_odometry(self):
        """Publish odometry information"""
        msg = Odometry()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'odom'
        msg.child_frame_id = 'base_link'
        
        # Position
        msg.pose.pose.position.x = self.position[0]
        msg.pose.pose.position.y = self.position[1]
        msg.pose.pose.position.z = 0.0
        
        # Convert Euler to Quaternion
        quat = tf_transformations.quaternion_from_euler(0, 0, self.position[2])
        msg.pose.pose.orientation.x = quat[0]
        msg.pose.pose.orientation.y = quat[1]
        msg.pose.pose.orientation.z = quat[2]
        msg.pose.pose.orientation.w = quat[3]
        
        # Velocities
        msg.twist.twist.linear.x = self.velocity[0]
        msg.twist.twist.angular.z = self.velocity[2]
        
        self.odom_pub.publish(msg)
        
        # Broadcast transform
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'odom'
        t.child_frame_id = 'base_link'
        t.transform.translation.x = self.position[0]
        t.transform.translation.y = self.position[1]
        t.transform.translation.z = 0.0
        t.transform.rotation.x = quat[0]
        t.transform.rotation.y = quat[1]
        t.transform.rotation.z = quat[2]
        t.transform.rotation.w = quat[3]
        
        self.tf_broadcaster.sendTransform(t)
    
    def simulate_lidar(self):
        """Publish simulated LiDAR data"""
        msg = LaserScan()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'laser_link'
        
        # Set parameters
        msg.angle_min = -math.pi
        msg.angle_max = math.pi
        msg.angle_increment = 2 * math.pi / 360
        msg.range_min = 0.05
        msg.range_max = 30.0
        
        # Generate simulated ranges based on position in environment
        ranges = []
        for i in range(360):
            angle = msg.angle_min + i * msg.angle_increment
            simulated_range = self.compute_lidar_range(angle)
            ranges.append(simulated_range)
        
        msg.ranges = ranges
        self.scan_pub.publish(msg)
    
    def compute_lidar_range(self, angle_in_robot_frame):
        """Compute what the LiDAR would see at a given angle"""
        # This would need to interface with the Gazebo environment
        # For this example, we'll simulate a simple environment
        world_angle = self.position[2] + angle_in_robot_frame
        cos_world = math.cos(world_angle)
        sin_world = math.sin(world_angle)
        
        # Simulate objects in environment
        # In a real implementation, this would use Gazebo's scene information
        simulated_distance = 10.0  # Default max range
        
        # Example: simulate a wall at x = 5.0
        if abs(cos_world) > 0.1:
            dist_to_wall = (5.0 - self.position[0]) / cos_world
            if 0.1 < dist_to_wall < simulated_distance:
                simulated_distance = dist_to_wall
        
        # Add noise
        noise = random.uniform(-0.05, 0.05)
        return max(0.1, simulated_distance + noise)
    
    def simulate_imu(self):
        """Publish simulated IMU data"""
        msg = Imu()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'imu_link'
        
        # Orientation (from robot position)
        quat = tf_transformations.quaternion_from_euler(0, 0, self.position[2])
        msg.orientation.x = quat[0]
        msg.orientation.y = quat[1]
        msg.orientation.z = quat[2]
        msg.orientation.w = quat[3]
        
        # Angular velocity (from robot motion)
        msg.angular_velocity.z = self.velocity[2]  # Turning rate
        
        # Linear acceleration (simplified)
        msg.linear_acceleration.x = self.linear_cmd  # Commanded acceleration
        
        self.imu_pub.publish(msg)
```

### ℹ️ Gazebo Services ℹ️

Gazebo provides services for simulation control:

```python
from rclpy.node import Node
from rclpy.action import ActionClient
from std_srvs.srv import Empty
from gazebo_msgs.srv import SpawnEntity, DeleteEntity, GetEntityState, SetEntityState
from geometry_msgs.msg import Pose, Twist

class SimulationManager(Node):
    def __init__(self):
        super().__init__('simulation_manager')
        
        # Connect to Gazebo services
        self.spawn_client = self.create_client(SpawnEntity, '/spawn_entity')
        self.delete_client = self.create_client(DeleteEntity, '/delete_entity')
        self.get_state_client = self.create_client(GetEntityState, '/get_entity_state')
        self.set_state_client = self.create_client(SetEntityState, '/set_entity_state')
        
        # Wait for services
        while not self.spawn_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Spawn entity service not available, waiting again...')
        
        while not self.delete_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Delete entity service not available, waiting again...')
    
    def spawn_robot(self, robot_name, robot_xml, initial_pose):
        """Spawn a robot in the simulation"""
        req = SpawnEntity.Request()
        req.name = robot_name
        req.xml = robot_xml
        req.initial_pose = initial_pose
        
        future = self.spawn_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        
        result = future.result()
        if result.success:
            self.get_logger().info(f'Successfully spawned {robot_name}')
        else:
            self.get_logger().error(f'Failed to spawn {robot_name}: {result.status_message}')
        
        return result.success
    
    def reset_simulation(self):
        """Reset the simulation to initial state"""
        reset_req = Empty.Request()
        reset_client = self.create_client(Empty, '/reset_simulation')
        
        while not reset_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Reset service not available, waiting again...')
        
        future = reset_client.call_async(reset_req)
        rclpy.spin_until_future_complete(self, future)
        
        self.get_logger().info('Simulation reset complete')
```

## 🎮 Simulation Testing & Validation 🎮

### 🎮 Testing Simulation Accuracy 🎮

It's crucial to validate that simulations accurately represent real-world behavior:

```python
import unittest
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import JointState
import time

class SimulationAccuracyTest(unittest.TestCase):
    def setUp(self):
        rclpy.init()
        self.node = TestAccuracyNode()
        self.executor = rclpy.executors.SingleThreadedExecutor()
        self.executor.add_node(self.node)
    
    def tearDown(self):
        self.node.destroy_node()
        rclpy.shutdown()
    
    def test_kinematic_accuracy(self):
        """Test that simulated kinematics match expected values"""
        # Send a known command to the robot
        cmd_msg = Twist()
        cmd_msg.linear.x = 1.0  # Move forward at 1 m/s
        self.node.cmd_pub.publish(cmd_msg)
        
        # Wait for physics to update
        time.sleep(2.0)  # Move for 2 seconds
        
        # Check that position changed by expected amount (with tolerance)
        expected_x = 2.0  # 1 m/s * 2 s
        actual_x = self.node.current_pos_x
        tolerance = 0.1  # 10 cm tolerance
        
        self.assertAlmostEqual(expected_x, actual_x, delta=tolerance,
                              msg=f"Expected x={expected_x}, got x={actual_x}")
    
    def test_dynamic_response(self):
        """Test that simulated dynamics respond appropriately"""
        # Send impulse command
        cmd_msg = Twist()
        cmd_msg.linear.x = 2.0  # Strong forward command
        self.node.cmd_pub.publish(cmd_msg)
        
        # Wait and check if velocity increases appropriately
        time.sleep(0.5)
        initial_vel = self.node.current_vel_x
        
        # Remove command and check for gradual decrease (due to damping)
        zero_cmd = Twist()
        self.node.cmd_pub.publish(zero_cmd)
        
        time.sleep(1.0)
        final_vel = self.node.current_vel_x
        
        # Velocity should be lower than initial after removing command
        self.assertLess(final_vel, initial_vel,
                       msg="Velocity did not decrease after removing command")
    
    def test_sensor_consistency(self):
        """Test that sensor readings are consistent with robot state"""
        # Publish known robot state
        # Check that sensors return expected values
        
        # Implementation would depend on specific sensor testing needs
        pass

class TestAccuracyNode(Node):
    def __init__(self):
        super().__init__('test_accuracy_node')
        
        # Publishers and subcribers for testing
        self.cmd_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.odom_sub = self.create_subscription(Odometry, 'odom', self.odom_callback, 10)
        
        # State tracking
        self.current_pos_x = 0.0
        self.current_vel_x = 0.0
    
    def odom_callback(self, msg):
        self.current_pos_x = msg.pose.pose.position.x
        self.current_vel_x = msg.twist.twist.linear.x
```

## ℹ️ Sim-to-Real Transfer ℹ️

### 🤖 Domain Randomization 🤖

Domain randomization helps bridge the sim-to-real gap by training models on varied simulated environments:

```python
import random
import numpy as np

class DomainRandomizer:
    def __init__(self):
        # Define parameter ranges for randomization
        self.params = {
            # Physics parameters
            'gravity_range': [9.7, 9.9],
            'friction_range': [0.1, 1.5],
            'mass_variance': 0.1,  # ±10% mass variation
            
            # Sensor parameters
            'camera_noise_range': [0.001, 0.05],
            'lidar_noise_range': [0.005, 0.02],
            
            # Environment parameters
            'lighting_min_intensity': 0.1,
            'lighting_max_intensity': 1.0,
            'texture_variation': True
        }
    
    def randomize_environment(self, sdf_template):
        """Apply randomizations to environment parameters"""
        gravity = random.uniform(*self.params['gravity_range'])
        friction = random.uniform(*self.params['friction_range'])
        
        # Modify SDF template with randomized parameters
        sdf_modified = sdf_template.replace('{GRAVITY_CONSTANT}', str(gravity))
        sdf_modified = sdf_modified.replace('{FRICTION_COEFF}', str(friction))
        
        # Add more randomizations as needed
        return sdf_modified
    
    def randomize_robot_properties(self, urdf_model):
        """Apply randomizations to robot model properties"""
        # Randomize mass within variance
        mass_multiplier = 1.0 + random.uniform(-self.params['mass_variance'], self.params['mass_variance'])
        
        # In a real implementation, this would modify URDF mass values
        # This is a simplified example
        modified_urdf = urdf_model.replace('mass value="1.0"', f'mass value="{1.0 * mass_multiplier}"')
        
        # Add noise to inertial parameters
        # Modify sensor parameters
        # Adjust joint limits slightly
        
        return modified_urdf
    
    def apply_texture_randomization(self, model_path):
        """Apply texture and color randomization to improve visual domain transfer"""
        # This would modify material properties in the model
        # Change colors within plausible ranges
        # Apply different textures from a library
        pass
```

### ℹ️ System Identification for Parameter Tuning ℹ️

```python
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

class SystemIdentifier:
    def __init__(self, robot_model):
        self.robot_model = robot_model
        self.sim_params = {}
        self.real_params = {}
    
    def collect_trajectory_data(self, sim_robot, real_robot, trajectories):
        """Collect trajectory data from both simulation and real robot"""
        sim_data = []
        real_data = []
        
        for trajectory in trajectories:
            # Execute same trajectory in both sim and real
            sim_result = self.execute_trajectory(sim_robot, trajectory)
            real_result = self.execute_trajectory(real_robot, trajectory)
            
            sim_data.append(sim_result)
            real_data.append(real_result)
        
        return sim_data, real_data
    
    def objective_function(self, params):
        """Objective function to minimize sim-to-real discrepancy"""
        # Set simulation parameters
        self.set_simulation_params(params)
        
        # Collect data with current parameters
        sim_data, real_data = self.collect_trajectory_data(self.sim_robot, self.real_robot, self.trajectories)
        
        # Calculate discrepancy
        error = 0
        for sim_traj, real_traj in zip(sim_data, real_data):
            error += np.sum((np.array(sim_traj) - np.array(real_traj))**2)
        
        return error
    
    def tune_parameters(self, initial_params, bounds):
        """Tune simulation parameters to match real robot behavior"""
        result = minimize(
            self.objective_function,
            initial_params,
            bounds=bounds,
            method='L-BFGS-B'
        )
        
        self.best_params = result.x
        self.get_logger().info(f'Optimized parameters: {self.best_params}')
        
        return result.x
    
    def validate_transfer(self, policy):
        """Validate that a policy trained in simulation works on the real robot"""
        # Test performance in simulation
        sim_score = self.evaluate_policy(policy, self.sim_robot)
        
        # Test performance on real robot
        real_score = self.evaluate_policy(policy, self.real_robot)
        
        # Calculate sim-to-real gap
        gap = abs(sim_score - real_score) / max(abs(sim_score), abs(real_score), 1e-6)
        
        self.get_logger().info(f'Sim-to-real gap: {gap:.3f}')
        return gap < 0.1  # Return True if gap is below threshold
```

## 📈 Performance Optimization 📈

### 🎮 Simulation Performance Strategies 🎮

```python
class OptimizedSimulation(Node):
    def __init__(self):
        super().__init__('optimized_simulation')
        
        # Use efficient data structures and algorithms
        self.collision_grid = {}  # Spatial hash for collision detection
        
        # Optimize rendering
        self.rendering_enabled = True
        self.display_framerate = 60  # Target display framerate
        
        # Optimize physics
        self.physics_framerate = 1000  # Physics update rate
        self.max_substeps = 10  # Max substeps for stability
        
        # Use efficient communication patterns
        self.compression_enabled = False  # Consider compressing large sensor data
        self.throttling_active = True  # Throttle non-critical updates
        
        # Memory management
        self.message_pool = []  # Reuse message objects to reduce allocation
        self.buffer_size = 1000  # Size of message pool
        
        self.setup_optimized_publishers()
    
    def setup_optimized_publishers(self):
        """Set up publishers with optimized QoS profiles"""
        
        # High-frequency sensor data - use best-effort, keep last
        sensor_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST
        )
        
        self.lidar_pub = self.create_publisher(LaserScan, 'scan', sensor_qos)
        
        # Critical control data - use reliable, keep last
        control_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST
        )
        
        self.cmd_pub = self.create_publisher(Twist, 'cmd_vel', control_qos)
    
    def optimize_physics_step(self):
        """Optimize physics calculation"""
        # Use fixed timestep for consistency
        dt = 1.0 / self.physics_framerate
        
        # Limit substeps to prevent performance degradation
        if dt > self.max_substeps * 0.001:
            dt = self.max_substeps * 0.001
        
        # Perform physics update
        self.update_physics(dt)
        
        # Update rendering less frequently than physics
        if self.should_render():
            self.render_scene()
    
    def should_render(self):
        """Determine if rendering should occur this frame"""
        current_time = time.time()
        if hasattr(self, 'last_render_time'):
            elapsed = current_time - self.last_render_time
            target_interval = 1.0 / self.display_framerate
            if elapsed >= target_interval:
                self.last_render_time = current_time
                return True
        else:
            self.last_render_time = current_time
            return True
        return False
    
    def reuse_message_objects(self):
        """Reuse message objects to reduce garbage collection pressure"""
        if len(self.message_pool) > 0:
            msg = self.message_pool.pop()
        else:
            msg = LaserScan()  # Create new if pool is empty
        
        # Use the message...
        # When done, return to pool instead of deleting
        self.return_to_pool(msg)
    
    def return_to_pool(self, msg):
        """Return message object to pool for reuse"""
        if len(self.message_pool) < self.buffer_size:
            # Reset message to default state before returning to pool
            msg.ranges = []
            msg.intensities = []
            self.message_pool.append(msg)
```

## 📝 Chapter Summary 📝

Simulation environments are a cornerstone of Physical AI development, providing safe, cost-effective, and controllable platforms for testing and training robotic systems. Key points from this chapter:

1. **Gazebo**: Advanced physics simulation with realistic collision and sensor modeling
2. **Unity**: High-fidelity rendering and VR/AR capabilities for visual realism
3. **URDF/SDF**: Critical for robot and environment modeling
4. **Sensor Simulation**: Accurate modeling of real-world sensors with noise and characteristics
5. **Physics Simulation**: Proper parameterization for realistic interactions
6. **Digital Twin Creation**: Environment design for effective training
7. **ROS 2 Integration**: Bridging simulation and real-world control
8. **Sim-to-Real Transfer**: Techniques to minimize the reality gap
9. **Performance Optimization**: Efficient simulation execution
10. **Domain Randomization**: Techniques to improve sim-to-real transfer

The combination of accurate physics modeling, realistic sensor simulation, and proper domain randomization enables robots to learn in simulation and transfer their knowledge to real-world applications with minimal adaptation required.

## 🤔 Knowledge Check 🤔

1. Explain the differences between ROS 1 and ROS 2 communication architectures and their impact on simulation.
2. Compare Gazebo and Unity for robotics simulation. When would you choose each?
3. What are the key components of a Gazebo world file? Create a simple example.
4. How do you implement sensor fusion simulation with multiple sensor types?
5. What techniques can be used to reduce the sim-to-real gap?
6. Explain Quality of Service (QoS) profiles in ROS 2 and their importance for simulation.
7. What are the trade-offs between simulation fidelity and computational performance?

### ℹ️ Practical Exercise ℹ️

Create a complete simulation environment with:
1. A robot model with LiDAR, camera, and IMU sensors
2. A structured environment with obstacles
3. ROS 2 nodes for controlling the robot and processing sensor data
4. Performance monitoring tools
5. Domain randomization capabilities

### 💬 Discussion Questions 💬

1. How might you design a simulation environment specifically for training humanoid robots to walk on uneven terrain?
2. What are the challenges of simulating soft-body interactions for robotic manipulation?
3. How can you validate that your simulation accurately represents real-world physics?
4. What role does cloud computing play in large-scale robotics simulation?