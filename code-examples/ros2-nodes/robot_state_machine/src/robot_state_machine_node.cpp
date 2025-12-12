#include "robot_state_machine/robot_state_machine.hpp"
#include <memory>

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    
    // Create the robot state machine node
    auto node = std::make_shared<robot_state_machine::RobotStateMachine>("robot_state_machine_node");
    
    RCLCPP_INFO(node->get_logger(), "Robot State Machine node started");
    
    // Spin the node
    rclcpp::spin(node);
    
    rclcpp::shutdown();
    return 0;
}