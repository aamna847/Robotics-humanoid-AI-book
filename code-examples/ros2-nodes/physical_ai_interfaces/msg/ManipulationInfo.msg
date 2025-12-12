# ManipulationInfo.msg
# Additional information about the manipulation planning process

string planner_name           # Name of the planner that generated the trajectory
string motion_type            # Type of motion planned
                        # Values: "cartesian", "joint_space", "mixed"
string grasp_approach         # Approach direction for grasping (for grasp actions)
                        # Values: "top", "side", "pinch", "wrap", "custom"
float64 grasp_quality         # Quality metric for the planned grasp (0.0 to 1.0)