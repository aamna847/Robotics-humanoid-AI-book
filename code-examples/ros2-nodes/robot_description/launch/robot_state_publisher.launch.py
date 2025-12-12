import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node, SetParameter
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # Launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    robot_description_path = LaunchConfiguration('robot_description_path', 
                                                 default=[PathJoinSubstitution(
                                                     [FindPackageShare('robot_description'), 
                                                      'urdf', 
                                                      'simple_humanoid.urdf'])])
    
    # Read the URDF file
    robot_description_content = open(robot_description_path.perform({})).read()
    
    # Robot State Publisher node
    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        parameters=[
            {'use_sim_time': use_sim_time},
            {'robot_description': robot_description_content}
        ],
        output='screen'
    )
    
    # Joint State Publisher node (for simulation)
    joint_state_publisher_node = Node(
        package='joint_state_publisher',
        executable='joint_state_publisher',
        name='joint_state_publisher',
        parameters=[
            {'use_sim_time': use_sim_time}
        ],
        output='screen'
    )
    
    # Joint State Publisher GUI (for manual control during debugging)
    joint_state_publisher_gui_node = Node(
        package='joint_state_publisher_gui',
        executable='joint_state_publisher_gui',
        name='joint_state_publisher_gui',
        parameters=[
            {'use_sim_time': use_sim_time}
        ],
        output='screen',
        condition=launch.conditions.IfCondition(LaunchConfiguration('gui', default='false'))
    )
    
    # RViz2 node for visualization
    rviz_config_path = PathJoinSubstitution(
        [FindPackageShare('robot_description'), 'rviz', 'robot_view.rviz']
    )
    
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config_path],
        parameters=[
            {'use_sim_time': use_sim_time}
        ],
        output='screen'
    )
    
    # Launch description
    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='true',
            description='Use simulation clock if true'
        ),
        DeclareLaunchArgument(
            'robot_description_path',
            default_value=[PathJoinSubstitution(
                [FindPackageShare('robot_description'), 'urdf', 'simple_humanoid.urdf']
            )],
            description='Path to robot description file'
        ),
        DeclareLaunchArgument(
            'gui',
            default_value='false',
            description='Use GUI for joint state publisher'
        ),
        
        # Set parameters
        SetParameter(name='use_sim_time', value=use_sim_time),
        
        # Launch nodes
        robot_state_publisher_node,
        joint_state_publisher_node,
        joint_state_publisher_gui_node,
        rviz_node
    ])