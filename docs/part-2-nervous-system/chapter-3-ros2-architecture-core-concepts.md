---
slug: chapter-3-ros2-architecture-core-concepts
title: Chapter 3 - ROS 2 Architecture & Core Concepts
description: Introduction to ROS 2 architecture, nodes, topics, services, and core concepts
tags: [ros2, architecture, nodes, topics, services, robotics]
---

# 📚 Chapter 3: ROS 2 Architecture & Core Concepts 📚

## 🎯 Learning Objectives 🎯

- Understand the fundamental architecture of ROS 2 and how it differs from ROS 1
- Explain the concepts of nodes, topics, services, and actions in ROS 2
- Describe the communication patterns and message types used in ROS 2
- Implement basic ROS 2 nodes with publishers and subscribers
- Understand the role of DDS in ROS 2 communication
- Analyze the benefits of ROS 2's distributed architecture for Physical AI applications

## 📋 Table of Contents 📋

- [Introduction to ROS 2](#introduction-to-ros-2)
- [Architecture Overview](#architecture-overview)
- [Nodes in ROS 2](#nodes-in-ros-2)
- [Topics and Publishers/Subscribers](#topics-and-publisherssubscribers)
- [Services and Clients](#services-and-clients)
- [Actions](#actions)
- [Messages and Services](#messages-and-services)
- [Data Distribution Service (DDS)](#data-distribution-service-dds)
- [Launch Systems](#launch-systems)
- [Quality of Service (QoS)](#quality-of-service-qos)
- [ROS 2 Ecosystem](#ros-2-ecosystem)
- [Chapter Summary](#chapter-summary)
- [Knowledge Check](#knowledge-check)

## 👋 Introduction to ROS 2 👋

The Robot Operating System 2 (ROS 2) is a flexible framework for composing robotic applications using a distributed computing architecture. Unlike its predecessor ROS 1, ROS 2 is built from the ground up to support modern robotics requirements including real-time performance, multi-robot systems, and professional applications that require formal software life-cycle management.

ROS 2 represents the "nervous system" of robotic applications, providing a communication infrastructure that allows different components (nodes) to exchange information. This distributed architecture is particularly well-suited for Physical AI systems where different subsystems (perception, planning, control, etc.) run on different computing units and must coordinate over a network.

### ℹ️ Evolution from ROS 1 ℹ️

ROS 2 addresses several limitations of ROS 1:
- **Real-time support**: Critical for physical AI systems requiring precise control
- **Multi-robot coordination**: Native support for multi-robot and multi-platform applications
- **Formal release management**: Supports professional development practices
- **Improved security**: Authentication and encryption capabilities for sensitive deployments
- **Cross-platform support**: Unified experience across Linux, Windows, and macOS
- **Middleware abstraction**: Pluggable communication layers

### ℹ️ Core Philosophy ℹ️

ROS 2 follows the philosophy that:
- Each component should be as independent as possible
- Communication should be explicit and observable
- Components should be reusable and composable
- The system should support both simulation and real hardware

## 📊 Architecture Overview 📊

### ℹ️ Distributed Nature ℹ️

ROS 2 uses a peer-to-peer network architecture where each node is a process that can publish and/or subscribe to messages. Unlike ROS 1's centralized Master design, ROS 2 nodes can discover each other dynamically over the network without requiring a central coordinator.

```
ROS 2 Architecture

[Node A]     [Node B]     [Node C]
   |             |            |
   |-------------|------------|
          DDS Infrastructure
   |-------------|------------|
[Node D]     [Node E]     [Node F]

Each arrow represents potential topic/service connections
```

### ⚡ Middleware Abstraction Layer ⚡

ROS 2 introduces a middleware abstraction layer that separates the application code from the underlying communication mechanism. The default middleware is DDS (Data Distribution Service), but other middlewares can be swapped in if needed.

### 🤖 Domain IDs 🤖

ROS 2 uses Domain IDs to separate logical networks. Nodes with different Domain IDs cannot communicate, allowing for isolation between different applications or robot teams running in the same physical location.

## ℹ️ Nodes in ROS 2 ℹ️

Nodes are the fundamental execution units of ROS 2. Each node runs in its own process and encapsulates a specific functionality. Nodes can contain multiple threads and can perform multiple tasks, but typically they focus on a single concern.

### 🔨 Node Implementation 🔨

In Python:
```python
import rclpy
from rclpy.node import Node

class MyRobotNode(Node):
    def __init__(self):
        super().__init__('robot_controller')
        self.get_logger().info('Robot controller node initialized')

def main(args=None):
    rclpy.init(args=args)
    node = MyRobotNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

In C++:
```cpp
#include <rclcpp/rclcpp.hpp>

class MyRobotNode : public rclcpp::Node
{
public:
    MyRobotNode() : Node("robot_controller")
    {
        RCLCPP_INFO(this->get_logger(), "Robot controller node initialized");
    }
};

int main(int argc, char * argv[])
{
    rclpy::init(argc, argv);
    auto node = std::make_shared<MyRobotNode>();
    rclpy::spin(node);
    rclpy::shutdown();
    return 0;
}
```

### ℹ️ Node Concepts ℹ️

#### ℹ️ Node Names ℹ️
Every node must have a unique name within the domain. Names should be descriptive and consistent across deployments.

#### ℹ️ Parameters ℹ️
Nodes can be parameterized to support different configurations without code changes:
```python
self.declare_parameter('loop_frequency', 10)
frequency = self.get_parameter('loop_frequency').value
```

#### ℹ️ Timers ℹ️
Nodes can execute callbacks at regular intervals:
```python
timer_period = 0.1  # seconds
self.timer = self.create_timer(timer_period, self.timer_callback)
```

#### ℹ️ Lifecycles ℹ️
Advanced nodes can implement lifecycle management for complex initialization and finalization:
- Unconfigured → Inactive → Active → Finalized
- Supports safety-critical applications requiring controlled states

## ℹ️ Topics and Publishers/Subscribers ℹ️

Topics form the backbone of ROS 2's publish-subscribe communication model. This pattern enables loose coupling between nodes, where publishers don't need to know who their subscribers are and vice versa.

### 🔨 Publisher Implementation 🔨

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist

class SensorPublisher(Node):
    def __init__(self):
        super().__init__('sensor_publisher')
        self.publisher = self.create_publisher(LaserScan, 'scan', 10)
        self.timer = self.create_timer(0.1, self.publish_scan)
        
    def publish_scan(self):
        msg = LaserScan()
        # Populate the message with sensor data
        self.publisher.publish(msg)
```

### 🔨 Subscriber Implementation 🔨

```python
class VelocityController(Node):
    def __init__(self):
        super().__init__('velocity_controller')
        self.subscription = self.create_subscription(
            Twist,
            'cmd_vel',
            self.velocity_callback,
            10)
        self.subscription  # Prevent unused variable warning
        
    def velocity_callback(self, msg):
        # Process velocity command
        linear_x = msg.linear.x
        angular_z = msg.angular.z
        self.get_logger().info(f'Received velocity command: {linear_x}, {angular_z}')
```

### 🔄 Topic Patterns 🔄

#### 📊 Sensor Data Flow 📊
- Sensors publish data at high rates (e.g., camera frames at 30 Hz)
- Multiple subscribers may need sensor data (localization, mapping, control)

#### 🎛️ Control Command Flow 🎛️
- High-level planners publish commands
- Low-level controllers subscribe to execute them

#### ℹ️ State Broadcasting ℹ️
- Robot state published for monitoring
- Multiple visualization tools subscribe to state

### ℹ️ Message Types and Standard Messages ℹ️

Standard message types ensure interoperability:
- **sensor_msgs**: Sensor data (images, lasers, IMUs)
- **geometry_msgs**: Spatial relationships (poses, vectors, quaternions)
- **nav_msgs**: Navigation-specific messages (paths, grids)
- **std_msgs**: Basic data types (headers, colors, values)
- **trajectory_msgs**: Motion trajectories for manipulation/locomotion

## ℹ️ Services and Clients ℹ️

Services provide request-response communication, ideal for operations that have a definite outcome. Unlike topics, services are synchronous and guarantee that a request gets processed.

### 🔨 Service Server Implementation 🔨

```python
import rclpy
from rclpy.node import Node
from std_srvs.srv import SetBool

class SafetyServer(Node):
    def __init__(self):
        super().__init__('safety_server')
        self.srv = self.create_service(SetBool, 'emergency_stop', self.emergency_stop_callback)
        
    def emergency_stop_callback(self, request, response):
        if request.data:  # Emergency stop requested
            self.get_logger().warn('EMERGENCY STOP ACTIVATED')
            # Execute emergency stop procedures
            response.success = True
            response.message = 'Emergency stop activated'
        else:
            self.get_logger().info('System resumed from emergency stop')
            response.success = True
            response.message = 'System resumed'
        return response
```

### 🔨 Client Implementation 🔨

```python
import rclpy
from rclpy.node import Node
from std_srvs.srv import SetBool

class SafetyClient(Node):
    def __init__(self):
        super().__init__('safety_client')
        self.client = self.create_client(SetBool, 'emergency_stop')
        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Emergency stop service not available, waiting again...')
        
    def trigger_emergency_stop(self):
        request = SetBool.Request()
        request.data = True
        future = self.client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        return future.result()
```

### 🚀 Service Applications 🚀

#### ℹ️ Configuration Services ℹ️
- Changing parameters at runtime
- Calibrating sensors
- Activating/deactivating subsystems

#### ⚡ Action Triggering ⚡
- Starting complex behaviors
- Requesting task execution
- Diagnostic operations

#### ℹ️ Coordination ℹ️
- Multi-robot choreography
- Resource allocation
- State queries

## ⚡ Actions ⚡

Actions are a hybrid communication pattern combining features of topics (feedback) and services (result reporting). They're ideal for long-running operations like navigation or manipulation tasks.

### ⚡ Action Structure ⚡

An action has three parts:
1. **Goal**: What to do
2. **Feedback**: Progress updates
3. **Result**: Final outcome

### ⚡ Action Server Implementation ⚡

```python
import rclpy
from rclpy.action import ActionServer
from rclpy.node import Node
from example_interfaces.action import Fibonacci

class FibonacciActionServer(Node):
    def __init__(self):
        super().__init__('fibonacci_action_server')
        self._action_server = ActionServer(
            self,
            Fibonacci,
            'fibonacci',
            self.execute_callback)

    def execute_callback(self, goal_handle):
        self.get_logger().info('Executing goal...')
        
        feedback_msg = Fibonacci.Feedback()
        feedback_msg.sequence = [0, 1]
        
        for i in range(1, goal_handle.request.order):
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                self.get_logger().info('Goal canceled')
                return Fibonacci.Result()

            feedback_msg.sequence.append(
                feedback_msg.sequence[i] + feedback_msg.sequence[i-1])
            
            goal_handle.publish_feedback(feedback_msg)
            self.get_logger().info(f'Publishing feedback: {feedback_msg.sequence}')

        goal_handle.succeed()
        result = Fibonacci.Result()
        result.sequence = feedback_msg.sequence
        self.get_logger().info(f'Returning result: {result.sequence}')

        return result
```

### ⚡ Action Client Implementation ⚡

```python
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from example_interfaces.action import Fibonacci

class FibonacciActionClient(Node):
    def __init__(self):
        super().__init__('fibonacci_action_client')
        self._action_client = ActionClient(
            self,
            Fibonacci,
            'fibonacci')

    def send_goal(self, order):
        goal_msg = Fibonacci.Goal()
        goal_msg.order = order

        self._action_client.wait_for_server()
        self._send_goal_future = self._action_client.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_callback)

        self._send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Goal rejected :(')
            return

        self.get_logger().info('Goal accepted :)')

        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)

    def feedback_callback(self, feedback_msg):
        self.get_logger().info(f'Received feedback: {feedback_msg.feedback.sequence}')

    def get_result_callback(self, future):
        result = future.result().result
        self.get_logger().info(f'Result: {result.sequence}')
```

### ⚡ Action Applications ⚡

#### 🧭 Navigation 🧭
- Sending navigation goals
- Monitoring progress toward destinations
- Receiving success/failure results

#### ℹ️ Manipulation ℹ️
- Requesting complex manipulation tasks
- Monitoring grip strength, position
- Getting task completion status

#### ℹ️ Complex Behaviors ℹ️
- High-level task execution
- Multi-step operations
- Failure recovery procedures

## ℹ️ Messages and Services ℹ️

### ℹ️ Message Definition ℹ️

Messages are defined in `.msg` files using a simple syntax:
```
# ℹ️ In geometry_msgs/msg/Twist.msg ℹ️
geometry_msgs/Vector3 linear
geometry_msgs/Vector3 angular
```

Nested message definition:
```
# ℹ️ In geometry_msgs/msg/Vector3.msg ℹ️
float64 x
float64 y
float64 z
```

### ℹ️ Service Definition ℹ️

Services are defined in `.srv` files:
```
# ℹ️ In custom_interfaces/srv/GetObjectLocation.srv ℹ️
string object_name
---
geometry_msgs/Pose pose
bool found
```

### ⚡ Action Definition ⚡

Actions are defined in `.action` files:
```
# ⚡ In custom_interfaces/action/MoveGripper.action ⚡
float64 goal_position
---
float64 final_position
bool success
---
float64 current_position
float64 effort
```

## 📊 Data Distribution Service (DDS) 📊

DDS (Data Distribution Service) is the middleware standard that ROS 2 uses for communication. It provides a publisher-subscriber networking model that's designed for real-time systems.

### 🏗️ DDS Architecture 🏗️

DDS consists of:
- **Domain**: Logical network partition
- **Participants**: Applications participating in DDS
- **Topics**: Named data channels
- **Data Writers**: Publishers of data
- **Data Readers**: Subscribers of data
- **QoS Policies**: Configurable parameters for behavior

### ℹ️ QoS (Quality of Service) Profiles ℹ️

DDS provides rich configuration options:

#### ✅ Reliability ✅
- **Reliable**: All messages guaranteed to be delivered
- **Best Effort**: Messages sent without guarantee (faster)

#### ℹ️ Durability ℹ️
- **Transient Local**: Late-joining subscribers get historical data
- **Volatile**: Only current data available to new subscribers

#### ℹ️ History ℹ️
- **Keep Last**: Only the most recent N samples
- **Keep All**: All samples are kept (memory intensive)

#### ℹ️ Deadline and Lifespan ℹ️
- **Deadline**: How often data should be received
- **Lifespan**: How long data remains valid

### 🔨 DDS Implementations 🔨

Popular DDS implementations include:
- **FastDDS**: Open-source implementation from eProsima
- **CycloneDDS**: Eclipse Foundation open-source
- **RTI Connext**: Commercial implementation
- **OpenDDS**: Open-source from Object Computing

## ⚙️ Launch Systems ⚙️

Launch systems allow for starting multiple nodes simultaneously with appropriate configuration. This is crucial for complex robotic systems that require coordinated startup.

### ℹ️ XML Launch Files ℹ️

```xml
<launch>
  <node pkg="navigation" exec="localization_node" name="amcl" output="screen">
    <param name="use_map_topic" value="true" />
    <param name="odom_frame" value="odom" />
  </node>
  
  <node pkg="navigation" exec="planner_node" name="planner" output="screen">
    <param name="planner_frequency" value="5.0" />
  </node>
  
  <node pkg="rviz2" exec="rviz2" name="rviz" args="-d /path/to/config.rviz" />
</launch>
```

### ℹ️ Python Launch Files ℹ️

```python
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='navigation',
            executable='localization_node',
            name='amcl',
            parameters=[
                {'use_map_topic': True},
                {'odom_frame': 'odom'}
            ],
            output='screen'
        ),
        Node(
            package='navigation',
            executable='planner_node',
            name='planner',
            parameters=[
                {'planner_frequency': 5.0}
            ],
            output='screen'
        ),
        Node(
            package='rviz2',
            executable='rviz2',
            arguments=['-d', '/path/to/config.rviz']
        )
    ])
```

### ℹ️ Launch Composition ℹ️

Launch files support:
- **Conditional Launch**: Start nodes based on parameters
- **Remapping**: Redirect topic/service names
- **Environment Setup**: Set environment variables
- **Shutdown Handling**: Graceful node termination

## ℹ️ Quality of Service (QoS) ℹ️

QoS settings allow fine-tuning of communication behavior for different use cases.

### ℹ️ Common QoS Presets ℹ️

```python
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

# 📊 For real-time sensor data 📊
sensor_qos = QoSProfile(
    depth=10,
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST
)

# ℹ️ For critical state information ℹ️
critical_qos = QoSProfile(
    depth=1,
    reliability=ReliabilityPolicy.RELIABLE,
    history=HistoryPolicy.KEEP_LAST
)

# 📊 For configuration data that should persist 📊
config_qos = QoSProfile(
    depth=1,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=rclpy.qos.DurabilityPolicy.TRANSIENT_LOCAL,
    history=HistoryPolicy.KEEP_LAST
)
```

### ℹ️ QoS Matching ℹ️

When publishers and subscribers have incompatible QoS settings, they won't be able to communicate. ROS 2 provides warnings when this occurs to help debug communication issues.

## ℹ️ ROS 2 Ecosystem ℹ️

### ℹ️ Core Packages ℹ️

- **rclpy/rclcpp**: Client libraries for Python/C++
- **rcl**: Common client library implementation
- **rmw**: ROS Middleware Interface
- **builtin_interfaces**: Standard message definitions
- **std_msgs**: Basic datatypes
- **geometry_msgs**: Spatial data types
- **sensor_msgs**: Sensor data types
- **nav_msgs**: Navigation types
- **action_msgs**: Action types

### ℹ️ Tooling ℹ️

- **rviz2**: 3D visualization tool
- **rqt**: Graphical tools framework
- **rosbag2**: Data recording and playback
- **ros2cli**: Command-line interface tools
- **launch**: Node launching system

### 🎮 Simulation Integration 🎮

- **Gazebo**: Physics simulation
- **Ignition**: Next-generation simulation
- **Webots**: Alternative simulator
- **Unity**: Game engine simulation

### 🧭 Navigation and Manipulation 🧭

- **Navigation2**: Modern navigation stack
- **MoveIt2**: Motion planning framework
- **OpenVDB**: 3D mapping

## 📝 Chapter Summary 📝

ROS 2 provides a robust, distributed communication framework essential for Physical AI systems. Key concepts include:

- **Nodes**: Individual processes that encapsulate functionality
- **Topics**: Publish-subscribe communication for streaming data
- **Services**: Request-response for operations with clear outcomes
- **Actions**: Hybrid pattern for long-running operations with feedback
- **DDS Middleware**: Provides robust communication infrastructure
- **QoS**: Fine-grained control over communication behavior
- **Launch Systems**: Coordinated startup of complex systems

The distributed architecture of ROS 2 makes it ideal for Physical AI applications where different subsystems need to run on different hardware platforms while maintaining coordinated behavior. The combination of real-time capabilities, multi-robot support, and professional-grade tools makes ROS 2 the foundation for modern robotic systems.

## 🤔 Knowledge Check 🤔

1. Explain the difference between ROS 1 and ROS 2 architecture.
2. Compare the three communication patterns: topics, services, and actions. When would you use each?
3. What is DDS and why is it important for ROS 2?
4. Describe three QoS policies and their impact on communication.
5. How do launch files facilitate complex robotic system deployment?
6. Why might you choose a specific QoS profile for sensor data vs. command data?
7. Explain the concept of node lifecycles and their benefits for safety-critical applications.

### ℹ️ Practical Exercise ℹ️

Create a simple ROS 2 package with two nodes: a publisher that generates random sensor data (simulating LiDAR or IMU data) and a subscriber that processes this data and prints statistics. Use appropriate message types and QoS settings for your sensor simulation.

### 💬 Discussion Questions 💬

1. How does the distributed nature of ROS 2 support Physical AI applications running on multiple hardware platforms?
2. What are the benefits and challenges of ROS 2's middleware abstraction layer?
3. How might you implement fault tolerance in a ROS 2 system for critical Physical AI applications?