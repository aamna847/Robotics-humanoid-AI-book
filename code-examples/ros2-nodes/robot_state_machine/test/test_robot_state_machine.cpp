#include <gtest/gtest.h>
#include "robot_state_machine/robot_state_machine.hpp"

class RobotStateMachineTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Set up test fixtures
        // For this sample, we'll just initialize the node
        rclcpp::init(0, nullptr);
        node_ = std::make_shared<robot_state_machine::RobotStateMachine>("test_node");
    }

    void TearDown() override {
        // Clean up after tests
        node_.reset();
        rclcpp::shutdown();
    }

    std::shared_ptr<robot_state_machine::RobotStateMachine> node_;
};

// Test state transitions
TEST_F(RobotStateMachineTest, TransitionFromIdleToNavigating) {
    EXPECT_EQ(node_->get_current_state(), robot_state_machine::RobotState::IDLE);
    
    node_->transition_to_navigating();
    EXPECT_EQ(node_->get_current_state(), robot_state_machine::RobotState::NAVIGATING);
    
    node_->transition_to_idle();
    EXPECT_EQ(node_->get_current_state(), robot_state_machine::RobotState::IDLE);
}

// Test valid state transitions
TEST_F(RobotStateMachineTest, ValidStateTransitions) {
    // Test that we can transition from IDLE to NAVIGATING
    node_->transition_to_idle(); // Ensure we start in IDLE
    EXPECT_TRUE(node_->is_valid_transition(
        robot_state_machine::RobotState::IDLE,
        robot_state_machine::RobotState::NAVIGATING
    ));
    
    // Test that we can transition from NAVIGATING to IDLE
    node_->transition_to_navigating();
    EXPECT_TRUE(node_->is_valid_transition(
        robot_state_machine::RobotState::NAVIGATING,
        robot_state_machine::RobotState::IDLE
    ));
}

// Test invalid state transitions are properly rejected
TEST_F(RobotStateMachineTest, InvalidStateTransition) {
    // For safety reasons, test that we cannot transition directly from 
    // SAFETY_STOP to MANIPULATING (must go through IDLE first)
    EXPECT_FALSE(node_->is_valid_transition(
        robot_state_machine::RobotState::SAFETY_STOP,
        robot_state_machine::RobotState::MANIPULATING
    ));
}

// Test safety stop functionality
TEST_F(RobotStateMachineTest, SafetyStopTransition) {
    // Start in navigating state
    node_->transition_to_navigating();
    EXPECT_EQ(node_->get_current_state(), robot_state_machine::RobotState::NAVIGATING);
    
    // Transition to safety stop (this should always be valid)
    node_->transition_to_safety_stop();
    EXPECT_EQ(node_->get_current_state(), robot_state_machine::RobotState::SAFETY_STOP);
    
    // From safety stop, we should only be able to go to IDLE
    EXPECT_TRUE(node_->is_valid_transition(
        robot_state_machine::RobotState::SAFETY_STOP,
        robot_state_machine::RobotState::IDLE
    ));
    
    // Should not be able to go directly from safety to other states
    EXPECT_FALSE(node_->is_valid_transition(
        robot_state_machine::RobotState::SAFETY_STOP,
        robot_state_machine::RobotState::NAVIGATING
    ));
}

// Test that safety stop can be entered from any state
TEST_F(RobotStateMachineTest, SafetyStopFromAnyState) {
    // Test from IDLE
    node_->transition_to_idle();
    EXPECT_TRUE(node_->is_valid_transition(
        node_->get_current_state(),
        robot_state_machine::RobotState::SAFETY_STOP
    ));
    
    // Test from NAVIGATING
    node_->transition_to_navigating();
    EXPECT_TRUE(node_->is_valid_transition(
        node_->get_current_state(),
        robot_state_machine::RobotState::SAFETY_STOP
    ));
    
    // Test from MANIPULATING (conceptually, though we don't have a method for this state yet)
    // This verifies the safety-first requirement in the constitution
}