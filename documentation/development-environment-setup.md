# Development Environment Setup Guide

This guide covers setting up a complete development environment for the Physical AI & Humanoid Robotics curriculum, including all necessary tools and configurations for effective learning and development.

## Prerequisites

Before setting up the development environment, ensure you have:

- A workstation meeting the minimum requirements (see workstation setup guide)
- Administrative access to install software
- Stable internet connection
- Basic familiarity with terminal/command line operations

## Development Tools

### 1. Version Control (Git)
Git is essential for tracking changes in your robotics projects.

```bash
# Install Git
sudo apt install git

# Configure Git
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

### 2. Integrated Development Environment (IDE)

#### Option A: VS Code (Recommended)
VS Code is well-suited for robotics development with ROS 2 extensions:

```bash
# Install VS Code (Ubuntu)
sudo snap install --classic code

# Key extensions for robotics development:
# - ROS (for ROS 2 development)
# - C/C++ (for native code)
# - Python (for Python scripting)
# - GitLens (for version control)
# - Pylance (for Python development)
```

#### Option B: CLion or Qt Creator
For C++ development with advanced debugging capabilities.

#### Option C: PyCharm
For Python-focused development.

### 3. Terminal Tools
```bash
# Install useful terminal tools
sudo apt install htop tmux tree ncdu
```

### 4. Build Tools
```bash
# Essential build tools
sudo apt install build-essential cmake pkg-config
```

## ROS 2 Development Setup

### 1. Workspace Setup
```bash
# Create a workspace for the Physical AI curriculum
mkdir -p ~/physical_ai_ws/src
cd ~/physical_ai_ws

# Source ROS 2 before building
source /opt/ros/humble/setup.bash

# Build the workspace
colcon build

# Source the workspace
source install/setup.bash
```

### 2. Environment Variables
Add these to your `~/.bashrc` file:

```bash
# ROS 2 Humble Setup
source /opt/ros/humble/setup.bash

# Physical AI Workspace
source ~/physical_ai_ws/install/setup.bash

# Optional: Set ROS_DOMAIN_ID for multiple robots
export ROS_DOMAIN_ID=0

# Use simulation time (important for simulation consistency)
export USE_SIM_TIME=true
```

### 3. ROS 2 Tools
```bash
# Install useful ROS 2 development tools
sudo apt install ros-humble-rqt ros-humble-rqt-common-plugins
sudo apt install ros-humble-turtlebot3-*
sudo apt install ros-humble-nav2-bringup
```

## Simulation Environment

### 1. Gazebo Setup
```bash
# Already installed as part of ROS 2, but additional plugins may be needed
sudo apt install ros-humble-gazebo-ros-pkgs ros-humble-gazebo-ros2-control
```

### 2. Unity Setup (for advanced simulation)
Follow the Unity installation guide in the documentation section.

### 3. NVIDIA Isaac Sim Setup
Follow the Isaac Sim installation guide in the documentation section.

## Python Development Environment

### 1. Virtual Environment Setup
```bash
# Create virtual environment for the curriculum
python3 -m venv ~/physical_ai_env

# Activate the environment
source ~/physical_ai_env/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install common dependencies
pip install numpy scipy matplotlib jupyter pandas
```

### 2. Jupyter Notebooks for Prototyping
```bash
# Install Jupyter in the virtual environment
pip install jupyter notebook

# Create a Jupyter configuration
jupyter notebook --generate-config
```

## Development Workflow

### 1. Project Structure
Organize your robotics projects following this structure:

```
robotics_project/
├── CMakeLists.txt          # Build configuration for C++
├── package.xml             # Package metadata
├── config/                 # Configuration files
├── launch/                 # Launch files
├── src/                    # Source code
│   ├── cpp/                # C++ code
│   └── python/             # Python code
├── include/                # C++ headers
├── scripts/                # Standalone scripts
├── test/                   # Unit tests
├── urdf/                   # Robot description files
├── meshes/                 # 3D models
└── worlds/                 # Simulation environments
```

### 2. Best Practices
- Use meaningful variable and function names
- Document all public functions and classes
- Follow ROS 2 naming conventions
- Use constants for magic numbers
- Separate concerns (keep nodes focused on a single task)

### 3. Code Quality Tools
```bash
# Install Python linting tools
pip install flake8 pylint black

# Install C++ linting tools
sudo apt install cppcheck clang-tidy
```

## Debugging Tools

### 1. ROS 2 Tools
```bash
# View topics
ros2 topic list
ros2 topic echo <topic_name>

# View services
ros2 service list
ros2 service call <service_name> <service_type>

# View nodes
ros2 node list
ros2 run rqt_graph rqt_graph  # Visualize node connections
```

### 2. IDE Debugging
Configure your IDE for debugging ROS 2 nodes with both C++ and Python code.

## Performance Monitoring

### 1. System Monitoring
```bash
# Monitor system resources
htop
# Monitor ROS 2 performance
ros2 run tf2_tools view_frames  # Check TF tree performance
```

### 2. Profiling Tools
```bash
# Python profiling
pip install cProfile snakeviz
python -m cProfile -o output.prof your_script.py
snakeviz output.prof

# C++ profiling
sudo apt install valgrind
valgrind --tool=callgrind your_program
```

## Documentation and Note-Taking

### 1. Sphinx for Documentation
```bash
pip install sphinx sphinx-rtd-theme
```

### 2. Confluence or Notion for Project Documentation
Consider using collaborative platforms for team projects.

## Troubleshooting Common Issues

### 1. Environment Not Sourcing Properly
- Check that ROS 2 and workspace are properly sourced in `.bashrc`
- Verify the order of source commands
- Ensure paths are correct

### 2. Package Installation Issues
- Update package lists: `sudo apt update`
- Check ROS 2 distribution compatibility
- Verify internet connection for package downloads

### 3. Build Failures
- Verify all dependencies are installed
- Check CMakeLists.txt and package.xml configurations
- Ensure proper file permissions

## Continuous Integration Setup (Optional)

For advanced users, consider setting up CI/CD for your robotics projects:

```bash
# Example GitHub Actions workflow for ROS 2
# .github/workflows/ros2-ci.yml
name: ROS 2 CI
on: [push, pull_request]
jobs:
  ci:
    runs-on: ubuntu-22.04
    steps:
    - uses: actions/checkout@v3
    - name: Install ROS 2 Humble
      uses: ros-tooling/setup-ros@v0.7
      with:
        required-ros-distributions: humble
    - name: Run colcon build
      uses: ros-tooling/action-ros-ci@v0.3
      with:
        package-name: your_package_name
```

## Next Steps

After completing the development environment setup:

1. Run the basic robot simulation to verify the setup
2. Create a simple ROS 2 node following the curriculum examples
3. Practice using the debugging tools
4. Explore the provided code examples
5. Start with the ROS 2 fundamentals chapter