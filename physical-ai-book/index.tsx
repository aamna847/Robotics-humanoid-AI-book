import React from 'react';
import Layout from '@theme/Layout';
import Navbar from '../components/Navbar'; // This is the custom navbar with Modules and Curriculum
import StatCard from '../components/StatCard';
import FeatureCard from '../components/FeatureCard';
import Link from '@docusaurus/Link';

const App = () => {
  const statsData = [
    { value: '120+', label: 'Hours of Content', icon: '⏱️' },
    { value: '4', label: 'Learning Modules', icon: '🧩' },
    { value: '13', label: 'Practical Sessions', icon: '⚡' },
    { value: '5+', label: 'Capstone Projects', icon: '🧩' }
  ];

  const featuresData = [
    {
      title: '🤖 Robotic Nervous System',
      description: 'Learn how to build sophisticated sensorimotor control systems that enable robots to perceive and interact with the physical world.',
      icon: '🧠'
    },
    {
      title: '🎮 Digital Twin & Simulation',
      description: 'Master physics-based simulation environments for training and validating robotic systems in virtual worlds before deployment.',
      icon: '⚡'
    },
    {
      title: '🤖 AI-Robot Brain Interface',
      description: 'Develop cognitive architectures that enable robots to learn, reason, and adapt to complex physical environments.',
      icon: '🧠'
    }
  ];

  return (
    <Layout title="Physical AI" description="Mastering humanoid robotics and embodied artificial intelligence">
      <div className="min-h-screen bg-gray-900 text-gray-100">
        <Navbar />
      <main className="container mx-auto px-4 py-8">
        {/* Hero Section */}
        <div className="text-center py-16 main-hero-section">
          <h1 className="text-6xl md:text-8xl font-bold mb-6 bg-clip-text text-transparent bg-gradient-to-r from-electric-cyan to-purple-accent">
            <span>🤖</span> Physical AI & Humanoid Robotics <span>🧠</span>
          </h1>
          <p className="text-xl text-gray-400 max-w-3xl mx-auto">
            Bridging the gap between digital AI (LLMs, Computer Vision, Reinforcement Learning) and physical robotics, focusing on embodied intelligence
          </p>
        </div>

        {/* Stats Section */}
        <section className="py-12">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            {statsData.map((stat, index) => (
              <div key={index} className="bg-gray-900/50 backdrop-blur-xs border border-gray-800 rounded-xl p-6 transition-all duration-300 hover:bg-gray-900/70 hover:shadow-[0_0_20px_rgba(0,255,255,0.3)] hover:scale-[1.02]">
                <div className="flex items-center justify-between">
                  <div>
                    <div className="text-3xl font-bold text-electric-cyan mb-1">{stat.value}</div>
                    <div className="text-gray-400">{stat.label}</div>
                  </div>
                  <div className="p-3 rounded-lg bg-gray-800/50 text-electric-cyan">
                    {stat.icon}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </section>

        {/* Features Section */}
        <section className="py-12">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {featuresData.map((feature, index) => (
              <div key={index} className="bg-gray-900/50 backdrop-blur-xs border border-gray-800 rounded-xl p-6 transition-all duration-300 hover:bg-gray-900/70 hover:shadow-[0_0_20px_rgba(127,79,255,0.3)] hover:scale-[1.02]">
                <div className="flex items-start space-x-4">
                  <div className="p-3 rounded-lg bg-purple-900/30 text-purple-accent">
                    {feature.icon}
                  </div>
                  <div>
                    <h3 className="text-xl font-bold mb-2 bg-clip-text text-transparent bg-gradient-to-r from-electric-cyan to-purple-accent">{feature.title}</h3>
                    <p className="text-gray-400">{feature.description}</p>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </section>

        {/* Curriculum Modules Section */}
        <section id="curriculum-modules" className="py-16">
          <h2 className="text-8xl font-bold text-center mb-12 bg-clip-text text-transparent bg-gradient-to-r from-electric-cyan to-purple-accent flex items-center justify-center gap-4">
            <span>🧩</span> <span>Curriculum Modules</span> <span>🧠</span>
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="bg-gray-900/50 backdrop-blur-xs border border-gray-800 rounded-xl p-6 transition-all duration-300 hover:bg-gray-900/70 hover:shadow-[0_0_20px_rgba(127,79,255,0.3)] hover:scale-[1.02]">
              <div className="flex items-center gap-4">
                <div className="p-3 rounded-lg bg-gray-800/50 text-electric-cyan">
                  🧠
                </div>
                <div>
                  <h3 className="text-xl font-bold mb-2 bg-clip-text text-transparent bg-gradient-to-r from-electric-cyan to-purple-accent">
                    <span>Weeks 1-2: </span> Physical AI Foundations
                  </h3>
                  <p className="text-gray-400">Core concepts of embodied intelligence, sensors, and humanoid robotics overview ⚡</p>
                </div>
              </div>
            </div>
            <div className="bg-gray-900/50 backdrop-blur-xs border border-gray-800 rounded-xl p-6 transition-all duration-300 hover:bg-gray-900/70 hover:shadow-[0_0_20px_rgba(127,79,255,0.3)] hover:scale-[1.02]">
              <div className="flex items-center gap-4">
                <div className="p-3 rounded-lg bg-gray-800/50 text-electric-cyan">
                  🤖
                </div>
                <div>
                  <h3 className="text-xl font-bold mb-2 bg-clip-text text-transparent bg-gradient-to-r from-electric-cyan to-purple-accent">
                    <span>Weeks 3-5: </span> ROS 2 Fundamentals
                  </h3>
                  <p className="text-gray-400">Nodes, topics, services, URDF, and Python integration ⚙️</p>
                </div>
              </div>
            </div>
            <div className="bg-gray-900/50 backdrop-blur-xs border border-gray-800 rounded-xl p-6 transition-all duration-300 hover:bg-gray-900/70 hover:shadow-[0_0_20px_rgba(127,79,255,0.3)] hover:scale-[1.02]">
              <div className="flex items-center gap-4">
                <div className="p-3 rounded-lg bg-gray-800/50 text-electric-cyan">
                  🎮
                </div>
                <div>
                  <h3 className="text-xl font-bold mb-2 bg-clip-text text-transparent bg-gradient-to-r from-electric-cyan to-purple-accent">
                    <span>Weeks 6-7: </span> Simulation Environments
                  </h3>
                  <p className="text-gray-400">Gazebo and Unity for physics, collisions, and sensor simulation ⚡</p>
                </div>
              </div>
            </div>
            <div className="bg-gray-900/50 backdrop-blur-xs border border-gray-800 rounded-xl p-6 transition-all duration-300 hover:bg-gray-900/70 hover:shadow-[0_0_20px_rgba(127,79,255,0.3)] hover:scale-[1.02]">
              <div className="flex items-center gap-4">
                <div className="p-3 rounded-lg bg-gray-800/50 text-electric-cyan">
                  🚀
                </div>
                <div>
                  <h3 className="text-xl font-bold mb-2 bg-clip-text text-transparent bg-gradient-to-r from-electric-cyan to-purple-accent">
                    <span>Weeks 8-10: </span> NVIDIA Isaac Platform
                  </h3>
                  <p className="text-gray-400">Isaac Sim, Reinforcement Learning, synthetic data, VSLAM, Nav2, sim-to-real ⚡</p>
                </div>
              </div>
            </div>
            <div className="bg-gray-900/50 backdrop-blur-xs border border-gray-800 rounded-xl p-6 transition-all duration-300 hover:bg-gray-900/70 hover:shadow-[0_0_20px_rgba(127,79,255,0.3)] hover:scale-[1.02]">
              <div className="flex items-center gap-4">
                <div className="p-3 rounded-lg bg-gray-800/50 text-electric-cyan">
                  🤖
                </div>
                <div>
                  <h3 className="text-xl font-bold mb-2 bg-clip-text text-transparent bg-gradient-to-r from-electric-cyan to-purple-accent">
                    <span>Weeks 11-12: </span> Humanoid Control
                  </h3>
                  <p className="text-gray-400">Kinematics, locomotion, and manipulation techniques ⚙️</p>
                </div>
              </div>
            </div>
            <div className="bg-gray-900/50 backdrop-blur-xs border border-gray-800 rounded-xl p-6 transition-all duration-300 hover:bg-gray-900/70 hover:shadow-[0_0_20px_rgba(127,79,255,0.3)] hover:scale-[1.02]">
              <div className="flex items-center gap-4">
                <div className="p-3 rounded-lg bg-gray-800/50 text-electric-cyan">
                  💬
                </div>
                <div>
                  <h3 className="text-xl font-bold mb-2 bg-clip-text text-transparent bg-gradient-to-r from-electric-cyan to-purple-accent">
                    <span>Week 13: </span> Conversational Robotics
                  </h3>
                  <p className="text-gray-400">Voice commands to ROS 2 execution using LLMs 🧠</p>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* Hardware Requirements Section */}
        <section className="py-12 hardware-section">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            <div className="bg-gray-900/50 backdrop-blur-xs border border-gray-800 rounded-xl p-6 transition-all duration-300 hover:bg-gray-900/70 hover:shadow-[0_0_20px_rgba(127,79,255,0.3)] hover:scale-[1.02]">
              <h2 className="text-5xl font-bold mb-4 bg-clip-text text-transparent bg-gradient-to-r from-electric-cyan to-purple-accent hardware-requirements">
                <span>🖥️</span> Hardware Requirements <span>🤖</span>
              </h2>
              <h3 className="text-xl font-semibold mb-3 bg-clip-text text-transparent bg-gradient-to-r from-electric-cyan to-purple-accent">
                <span>🎮</span> Digital Twin Workstation <span>⚡</span>
              </h3>
              <ul className="mb-4 text-gray-400">
                <li>🎮 RTX 4070 Ti+ (12–24GB VRAM) 🚀</li>
                <li>⚙️ Intel i7 / AMD Ryzen 9 processor 🧠</li>
                <li>💾 32–64GB RAM ⚙️</li>
                <li>🐧 Ubuntu 22.04 LTS 🧩</li>
              </ul>
              <h3 className="text-xl font-semibold mb-3 bg-clip-text text-transparent bg-gradient-to-r from-electric-cyan to-purple-accent">
                <span>🤖</span> Edge Kit (Physical AI) <span>⚡</span>
              </h3>
              <ul className="text-gray-400">
                <li>🚀 NVIDIA Jetson Orin Nano/NX 🤖</li>
                <li>👁️ Intel RealSense D435i depth camera 🎮</li>
                <li>🎤 USB microphone array 💬</li>
                <li>🧭 IMU sensor ⚡</li>
              </ul>
            </div>
            <div className="bg-gray-900/50 backdrop-blur-xs border border-gray-800 rounded-xl p-6 platforms-section transition-all duration-300 hover:bg-gray-900/70 hover:shadow-[0_0_20px_rgba(127,79,255,0.3)] hover:scale-[1.02]">
              <h3 className="text-5xl font-bold mb-4 bg-clip-text text-transparent bg-gradient-to-r from-electric-cyan to-purple-accent robot-platforms">
                <span>🤖</span> Robot Platforms <span>🚀</span>
              </h3>
              <ul className="text-gray-400">
                <li><strong>Budget:</strong> Unitree Go2 🤖</li>
                <li><strong>Mid:</strong> OP3 / Unitree G1 Mini 🧠</li>
                <li><strong>Premium:</strong> Unitree G1 Humanoid 🚀</li>
              </ul>
              <div className="text-center mt-8 pt-8">
                <Link
                  className="button button--primary button--lg"
                  to="/docs/introduction">
                  Start Learning
                </Link>
              </div>
            </div>
          </div>
        </section>
      </main>
    </div>
    </Layout>
  );
};

export default App;