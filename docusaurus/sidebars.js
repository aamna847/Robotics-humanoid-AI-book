// sidebars.js
// Sidebar configuration for the Physical AI & Humanoid Robotics book

/**
 * Creating a sidebar enables you to:
 - create an ordered group of docs
 - render a sidebar for each doc of that group
 - provide next/previous navigation

 The sidebars can be generated from the filesystem, or explicitly defined here.

 Create as many sidebars as you want.
 */

/** @type {import('@docusaurus/plugin-content-docs').SidebarsConfig} */
const sidebars = {
  tutorialSidebar: [
    {
      type: 'category',
      label: 'Introduction',
      items: [
        'introduction',
      ],
      collapsed: false,
    },
    {
      type: 'category',
      label: 'Part 1: Foundations',
      items: [
        'part-1-foundations/chapter-1-introduction-to-physical-ai',
        'part-1-foundations/chapter-2-sensors-perception',
      ],
      collapsed: false,
    },
    {
      type: 'category',
      label: 'Part 2: The Nervous System (ROS 2)',
      items: [
        'part-2-nervous-system/chapter-3-ros2-architecture-core-concepts',
        'part-2-nervous-system/chapter-4-building-ros2-nodes-python',
        'part-2-nervous-system/chapter-5-launch-systems-parameter-management',
      ],
      collapsed: false,
    },
    {
      type: 'category',
      label: 'Part 3: The Digital Twin',
      items: [
        'part-3-digital-twin/chapter-6-simulation-gazebo-urdf-sdf',  // Renamed from chapter-5-simulation-gazebo-unity-integration.md
        'part-3-digital-twin/chapter-7-physics-simulation-unity-integration',  // Renamed from chapter-7-digital-twin-environment-design.md
      ],
      collapsed: false,
    },
    {
      type: 'category',
      label: 'Part 4: The AI Brain',
      items: [
        'part-4-ai-brain/chapter-8-nvidia-isaac-sim-sdk',  // Moved from part-4-interaction/chapter-8-human-in-the-loop-teleoperation.md
        'part-4-ai-brain/chapter-9-robot-learning-ai-integration',  // Currently exists but content doesn't match title - needs update
        'part-4-ai-brain/chapter-10-vision-language-action',  // Currently exists but content doesn't match title - needs update
      ],
      collapsed: false,
    },
    {
      type: 'category',
      label: 'Part 5: Advanced Humanoids',
      items: [
        'part-5-advanced-humanoids/chapter-11-humanoid-kinematics-locomotion',  // Moved from part-4-ai-brain/chapter-11-advanced-humanoid-control.md
        'part-5-advanced-humanoids/chapter-12-vla-conversational-robotics',  // Moved from part-4-ai-brain/chapter-12-capstone-autonomous-humanoid.md
        'part-5-advanced-humanoids/chapter-13-hardware-requirements-robot-platforms',  // New chapter for hardware requirements
      ],
      collapsed: false,
    },
  ],
};

module.exports = sidebars;