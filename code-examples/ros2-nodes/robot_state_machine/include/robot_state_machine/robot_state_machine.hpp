#ifndef ROBOT_STATE_MACHINE_HPP
#define ROBOT_STATE_MACHINE_HPP

#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <tf2_ros/transform_listener.h>
#include <memory>
#include <string>

namespace robot_state_machine
{

// Robot states following Physical AI constitution requirements
enum class RobotState {
    IDLE,
    NAVIGATING,
    MANIPULATING,
    PERCEIVING,
    SAFETY_STOP,
    COMMUNICATING
};

// Robot state machine class implementing the constitution requirements
class RobotStateMachine : public rclcpp::Node
{
public:
    explicit RobotStateMachine(const std::string & node_name);
    virtual ~RobotStateMachine();

    // Main control loop
    void control_loop();

    // State transition methods
    void transition_to_idle();
    void transition_to_navigating();
    void transition_to_manipulating();
    void transition_to_perceiving();
    void transition_to_safety_stop();
    void transition_to_communicating();

    // Get current state
    RobotState get_current_state() const { return current_state_; }

    // Check if state transition is valid
    bool is_valid_transition(RobotState from_state, RobotState to_state) const;

private:
    // Callbacks for sensor data
    void laser_scan_callback(const sensor_msgs::msg::LaserScan::SharedPtr msg);
    void odometry_callback(const nav_msgs::msg::Odometry::SharedPtr msg);

    // Publishers and subscribers
    rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr cmd_vel_publisher_;
    rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr laser_scan_subscriber_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odometry_subscriber_;
    
    // Timer for control loop
    rclcpp::TimerBase::SharedPtr control_timer_;

    // Current state and previous state
    RobotState current_state_;
    RobotState previous_state_;

    // TF listener
    std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

    // Safety parameters
    double safety_distance_threshold_;
    bool emergency_stop_triggered_;

    // State change callback
    void on_state_change(RobotState previous, RobotState current);
};

} // namespace robot_state_machine

#endif // ROBOT_STATE_MACHINE_HPP