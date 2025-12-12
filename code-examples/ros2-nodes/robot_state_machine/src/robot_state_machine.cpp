#include "robot_state_machine/robot_state_machine.hpp"
#include <tf2_ros/transform_listener.h>
#include <iostream>

namespace robot_state_machine
{

RobotStateMachine::RobotStateMachine(const std::string & node_name)
: Node(node_name),
  current_state_(RobotState::IDLE),
  previous_state_(RobotState::IDLE),
  safety_distance_threshold_(0.5),  // 50cm safety distance
  emergency_stop_triggered_(false)
{
    RCLCPP_INFO(this->get_logger(), "Initializing Robot State Machine");

    // Create publishers and subscribers
    cmd_vel_publisher_ = this->create_publisher<geometry_msgs::msg::Twist>("cmd_vel", 10);
    laser_scan_subscriber_ = this->create_subscription<sensor_msgs::msg::LaserScan>(
        "scan", 10, 
        std::bind(&RobotStateMachine::laser_scan_callback, this, std::placeholders::_1));
    odometry_subscriber_ = this->create_subscription<nav_msgs::msg::Odometry>(
        "odom", 10,
        std::bind(&RobotStateMachine::odometry_callback, this, std::placeholders::_1));

    // Initialize TF buffer and listener
    tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

    // Create control timer (10 Hz)
    control_timer_ = this->create_wall_timer(
        std::chrono::milliseconds(100),
        std::bind(&RobotStateMachine::control_loop, this));

    RCLCPP_INFO(this->get_logger(), "Robot State Machine initialized in IDLE state");
}

RobotStateMachine::~RobotStateMachine()
{
    // Ensure robot stops before shutdown
    auto stop_msg = geometry_msgs::msg::Twist();
    cmd_vel_publisher_->publish(stop_msg);
    RCLCPP_INFO(this->get_logger(), "Robot State Machine destroyed");
}

void RobotStateMachine::laser_scan_callback(const sensor_msgs::msg::LaserScan::SharedPtr msg)
{
    // Check for obstacles within safety threshold
    for (auto range : msg->ranges) {
        if (range < safety_distance_threshold_ && std::isfinite(range)) {
            if (current_state_ != RobotState::SAFETY_STOP) {
                RCLCPP_WARN(this->get_logger(), "Obstacle detected! Initiating safety stop.");
                transition_to_safety_stop();
            }
            return;
        }
    }

    // If we were in safety stop and obstacle is cleared, transition back to previous state
    if (current_state_ == RobotState::SAFETY_STOP) {
        RCLCPP_INFO(this->get_logger(), "Obstacle cleared. Returning to previous state.");
        auto previous = previous_state_;
        transition_to_idle(); // temporarily go to idle to reset
        previous_state_ = previous; // restore previous state
        // We'll transition back to previous state in the next control loop iteration
    }
}

void RobotStateMachine::odometry_callback(const nav_msgs::msg::Odometry::SharedPtr msg)
{
    // Process odometry data if needed for state machine logic
    // This could include checking if navigation target is reached
    if (current_state_ == RobotState::NAVIGATING) {
        // Navigation-specific processing would go here
        // For now, just log the odometry
        RCLCPP_DEBUG(this->get_logger(), "Received odometry data: position=(%.2f, %.2f, %.2f)",
                    msg->pose.pose.position.x,
                    msg->pose.pose.position.y,
                    msg->pose.pose.position.z);
    }
}

void RobotStateMachine::control_loop()
{
    // This method is called at 10Hz to process state transitions
    switch (current_state_) {
        case RobotState::IDLE:
            // In IDLE state, robot is waiting for commands
            // Check if we should return to previous state after safety stop
            if (previous_state_ != RobotState::SAFETY_STOP && previous_state_ != RobotState::IDLE) {
                // For curriculum purposes, we might transition to a different state
                // based on external commands (not implemented here for simplicity)
            }
            break;

        case RobotState::NAVIGATING:
            // Navigation-specific processing
            break;

        case RobotState::MANIPULATING:
            // Manipulation-specific processing
            break;

        case RobotState::PERCEIVING:
            // Perception-specific processing
            break;

        case RobotState::SAFETY_STOP:
            // In safety stop, only allow transition back to previous state if obstacle cleared
            break;

        case RobotState::COMMUNICATING:
            // Communication-specific processing
            break;
    }
}

void RobotStateMachine::transition_to_idle()
{
    if (current_state_ != RobotState::IDLE) {
        RCLCPP_INFO(this->get_logger(), "Transitioning to IDLE state");
        previous_state_ = current_state_;
        current_state_ = RobotState::IDLE;
        
        // Stop robot movement
        auto stop_msg = geometry_msgs::msg::Twist();
        cmd_vel_publisher_->publish(stop_msg);
        
        on_state_change(previous_state_, current_state_);
    }
}

void RobotStateMachine::transition_to_navigating()
{
    if (is_valid_transition(current_state_, RobotState::NAVIGATING)) {
        RCLCPP_INFO(this->get_logger(), "Transitioning to NAVIGATING state");
        previous_state_ = current_state_;
        current_state_ = RobotState::NAVIGATING;
        on_state_change(previous_state_, current_state_);
    } else {
        RCLCPP_WARN(this->get_logger(), "Invalid transition from %d to NAVIGATING", 
                   static_cast<int>(current_state_));
    }
}

void RobotStateMachine::transition_to_manipulating()
{
    if (is_valid_transition(current_state_, RobotState::MANIPULATING)) {
        RCLCPP_INFO(this->get_logger(), "Transitioning to MANIPULATING state");
        previous_state_ = current_state_;
        current_state_ = RobotState::MANIPULATING;
        on_state_change(previous_state_, current_state_);
    } else {
        RCLCPP_WARN(this->get_logger(), "Invalid transition from %d to MANIPULATING", 
                   static_cast<int>(current_state_));
    }
}

void RobotStateMachine::transition_to_perceiving()
{
    if (is_valid_transition(current_state_, RobotState::PERCEIVING)) {
        RCLCPP_INFO(this->get_logger(), "Transitioning to PERCEIVING state");
        previous_state_ = current_state_;
        current_state_ = RobotState::PERCEIVING;
        on_state_change(previous_state_, current_state_);
    } else {
        RCLCPP_WARN(this->get_logger(), "Invalid transition from %d to PERCEIVING", 
                   static_cast<int>(current_state_));
    }
}

void RobotStateMachine::transition_to_safety_stop()
{
    RCLCPP_WARN(this->get_logger(), "Transitioning to SAFETY_STOP state");
    previous_state_ = current_state_;
    current_state_ = RobotState::SAFETY_STOP;
    
    // Immediately stop robot movement
    auto stop_msg = geometry_msgs::msg::Twist();
    cmd_vel_publisher_->publish(stop_msg);
    
    on_state_change(previous_state_, current_state_);
}

void RobotStateMachine::transition_to_communicating()
{
    if (is_valid_transition(current_state_, RobotState::COMMUNICATING)) {
        RCLCPP_INFO(this->get_logger(), "Transitioning to COMMUNICATING state");
        previous_state_ = current_state_;
        current_state_ = RobotState::COMMUNICATING;
        on_state_change(previous_state_, current_state_);
    } else {
        RCLCPP_WARN(this->get_logger(), "Invalid transition from %d to COMMUNICATING", 
                   static_cast<int>(current_state_));
    }
}

bool RobotStateMachine::is_valid_transition(RobotState from_state, RobotState to_state) const
{
    // Define valid state transitions following Physical AI constitution requirements
    // Safety stop can be entered from any state
    if (to_state == RobotState::SAFETY_STOP) {
        return true;
    }
    
    // Define valid transitions between states
    switch (from_state) {
        case RobotState::IDLE:
            return (to_state == RobotState::NAVIGATING || 
                   to_state == RobotState::MANIPULATING || 
                   to_state == RobotState::PERCEIVING ||
                   to_state == RobotState::COMMUNICATING);
        
        case RobotState::NAVIGATING:
            return (to_state == RobotState::IDLE || 
                   to_state == RobotState::MANIPULATING || 
                   to_state == RobotState::PERCEIVING ||
                   to_state == RobotState::COMMUNICATING);
        
        case RobotState::MANIPULATING:
            return (to_state == RobotState::IDLE || 
                   to_state == RobotState::NAVIGATING || 
                   to_state == RobotState::PERCEIVING ||
                   to_state == RobotState::COMMUNICATING);
        
        case RobotState::PERCEIVING:
            return (to_state == RobotState::IDLE || 
                   to_state == RobotState::NAVIGATING || 
                   to_state == RobotState::MANIPULATING ||
                   to_state == RobotState::COMMUNICATING);
        
        case RobotState::COMMUNICATING:
            return (to_state == RobotState::IDLE || 
                   to_state == RobotState::NAVIGATING || 
                   to_state == RobotState::MANIPULATING ||
                   to_state == RobotState::PERCEIVING);
        
        case RobotState::SAFETY_STOP:
            // Can only transition from safety stop to idle (or back to safety if still needed)
            return (to_state == RobotState::IDLE);
    }
    
    return false;
}

void RobotStateMachine::on_state_change(RobotState previous, RobotState current)
{
    RCLCPP_DEBUG(this->get_logger(), "State changed from %d to %d", 
                static_cast<int>(previous), static_cast<int>(current));
    
    // Add any additional logic that should happen on state change
    // For example, publishing state change messages, updating UI, etc.
}

} // namespace robot_state_machine