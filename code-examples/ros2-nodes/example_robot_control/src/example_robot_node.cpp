// Basic example of a robot control node
// This would be expanded with actual functionality in the curriculum

#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include "sensor_msgs/msg/laser_scan.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "tf2_ros/transform_broadcaster.h"

class ExampleRobotNode : public rclcpp::Node
{
public:
    ExampleRobotNode() : Node("example_robot_node")
    {
        RCLCPP_INFO(this->get_logger(), "Example Robot Node initialized");
        
        // Create publisher for robot velocity commands
        cmd_vel_publisher_ = this->create_publisher<geometry_msgs::msg::Twist>(
            "cmd_vel", 10);
            
        // Create subscriber for laser scan data
        laser_subscriber_ = this->create_subscription<sensor_msgs::msg::LaserScan>(
            "scan", 10,
            std::bind(&ExampleRobotNode::laser_callback, this, std::placeholders::_1));
            
        // Timer for robot control loop
        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(100), // 10Hz
            std::bind(&ExampleRobotNode::control_loop, this));
    }

private:
    void laser_callback(const sensor_msgs::msg::LaserScan::SharedPtr msg)
    {
        RCLCPP_INFO(this->get_logger(), "Received laser scan data");
        // Process sensor data here
    }
    
    void control_loop()
    {
        // Example: publish a simple movement command
        auto msg = geometry_msgs::msg::Twist();
        msg.linear.x = 0.2;  // Move forward at 0.2 m/s
        msg.angular.z = 0.1; // Turn slightly
        cmd_vel_publisher_->publish(msg);
    }
    
    rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr cmd_vel_publisher_;
    rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr laser_subscriber_;
    rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<ExampleRobotNode>());
    rclcpp::shutdown();
    return 0;
}