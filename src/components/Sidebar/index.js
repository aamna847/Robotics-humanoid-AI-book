import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import Link from '@docusaurus/Link';
import { useLocation } from '@docusaurus/router';

const Sidebar = ({ isOpen, toggleSidebar }) => {
  const location = useLocation();

  useEffect(() => {
    // Close sidebar when route changes
    if (isOpen) {
      toggleSidebar();
    }
  }, [location.pathname]);

  const sidebarVariants = {
    hidden: { 
      x: '-100%', 
      opacity: 0 
    },
    visible: { 
      x: 0, 
      opacity: 1,
      transition: { 
        type: "spring", 
        damping: 25, 
        stiffness: 300,
        duration: 0.3 
      } 
    },
    exit: { 
      x: '-100%', 
      opacity: 0,
      transition: { 
        duration: 0.2 
      } 
    }
  };

  const itemVariants = {
    hidden: { opacity: 0, y: 20 },
    visible: { 
      opacity: 1, 
      y: 0,
      transition: { 
        duration: 0.3, 
        ease: "easeOut" 
      } 
    }
  };

  return (
    <AnimatePresence>
      {isOpen && (
        <>
          {/* Backdrop overlay */}
          <motion.div
            className="fixed inset-0 bg-black bg-opacity-50 z-[998] lg:hidden"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={toggleSidebar}
          />
          
          {/* Sidebar */}
          <motion.aside
            className="fixed top-0 left-0 h-full w-64 bg-gray-900 text-gray-100 z-[999] p-5 shadow-2xl lg:hidden"
            variants={sidebarVariants}
            initial="hidden"
            animate="visible"
            exit="exit"
          >
            <div className="flex justify-between items-center mb-8">
              <h2 className="text-xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-electric-cyan to-purple-accent">
                Menu
              </h2>
              <button 
                onClick={toggleSidebar}
                className="text-gray-400 hover:text-white focus:outline-none"
                aria-label="Close menu"
              >
                <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>

            <nav>
              <ul className="space-y-4">
                <motion.li variants={itemVariants}>
                  <Link
                    to="/"
                    className={`block py-3 px-4 rounded-lg transition-all ${
                      location.pathname === '/' 
                        ? 'bg-gradient-to-r from-electric-cyan/20 to-purple-accent/20 text-electric-cyan border-l-4 border-electric-cyan' 
                        : 'hover:bg-gray-800/50 hover:text-electric-cyan'
                    }`}
                    onClick={toggleSidebar}
                  >
                    <span className="flex items-center">
                      <span className="mr-3">🏠</span> Home
                    </span>
                  </Link>
                </motion.li>

                <motion.li variants={itemVariants} transition={{ delay: 0.1 }}>
                  <Link
                    to="/docs/introduction"
                    className={`block py-3 px-4 rounded-lg transition-all ${
                      location.pathname.startsWith('/docs/') && location.pathname !== '/docs/'
                        ? 'bg-gradient-to-r from-electric-cyan/20 to-purple-accent/20 text-electric-cyan border-l-4 border-electric-cyan' 
                        : 'hover:bg-gray-800/50 hover:text-electric-cyan'
                    }`}
                    onClick={toggleSidebar}
                  >
                    <span className="flex items-center">
                      <span className="mr-3">📖</span> Curriculum
                    </span>
                  </Link>
                </motion.li>

                <motion.li variants={itemVariants} transition={{ delay: 0.2 }}>
                  <Link
                    to="#"
                    className="block py-3 px-4 rounded-lg hover:bg-gray-800/50 hover:text-electric-cyan transition-all cursor-pointer"
                    onClick={(e) => {
                      e.preventDefault();
                      document.getElementById('curriculum-modules')?.scrollIntoView({ behavior: 'smooth' });
                      toggleSidebar();
                    }}
                  >
                    <span className="flex items-center">
                      <span className="mr-3">🧩</span> Modules
                    </span>
                  </Link>
                </motion.li>

                <motion.li variants={itemVariants} transition={{ delay: 0.3 }}>
                  <Link
                    to="/docs/part-1-foundations/chapter-1-introduction-to-physical-ai"
                    className="block py-3 px-4 rounded-lg hover:bg-gray-800/50 hover:text-electric-cyan transition-all"
                    onClick={toggleSidebar}
                  >
                    <span className="flex items-center">
                      <span className="mr-3">🧠</span> Foundations
                    </span>
                  </Link>
                </motion.li>

                <motion.li variants={itemVariants} transition={{ delay: 0.4 }}>
                  <Link
                    to="/docs/part-2-nervous-system/chapter-3-ros2-architecture-core-concepts"
                    className="block py-3 px-4 rounded-lg hover:bg-gray-800/50 hover:text-electric-cyan transition-all"
                    onClick={toggleSidebar}
                  >
                    <span className="flex items-center">
                      <span className="mr-3">⚡</span> Nervous System
                    </span>
                  </Link>
                </motion.li>

                <motion.li variants={itemVariants} transition={{ delay: 0.5 }}>
                  <Link
                    to="/docs/part-3-digital-twin/chapter-6-simulation-gazebo-urdf-sdf"
                    className="block py-3 px-4 rounded-lg hover:bg-gray-800/50 hover:text-electric-cyan transition-all"
                    onClick={toggleSidebar}
                  >
                    <span className="flex items-center">
                      <span className="mr-3">🎮</span> Digital Twin
                    </span>
                  </Link>
                </motion.li>

                <motion.li variants={itemVariants} transition={{ delay: 0.6 }}>
                  <Link
                    to="/docs/part-4-ai-brain/chapter-8-nvidia-isaac-sim-sdk"
                    className="block py-3 px-4 rounded-lg hover:bg-gray-800/50 hover:text-electric-cyan transition-all"
                    onClick={toggleSidebar}
                  >
                    <span className="flex items-center">
                      <span className="mr-3">🚀</span> AI Brain
                    </span>
                  </Link>
                </motion.li>

                <motion.li variants={itemVariants} transition={{ delay: 0.7 }}>
                  <Link
                    to="/docs/part-5-advanced-humanoids/chapter-11-humanoid-kinematics-locomotion"
                    className="block py-3 px-4 rounded-lg hover:bg-gray-800/50 hover:text-electric-cyan transition-all"
                    onClick={toggleSidebar}
                  >
                    <span className="flex items-center">
                      <span className="mr-3">🤖</span> Advanced Humanoids
                    </span>
                  </Link>
                </motion.li>
                
                <motion.li variants={itemVariants} transition={{ delay: 0.8 }}>
                  <Link
                    href="https://github.com/humanoid-robotics/physical-ai-book"
                    className="block py-3 px-4 rounded-lg hover:bg-gray-800/50 hover:text-electric-cyan transition-all flex items-center"
                    onClick={toggleSidebar}
                  >
                    <span className="mr-3">🐙</span> GitHub
                  </Link>
                </motion.li>
              </ul>
            </nav>

            <div className="absolute bottom-5 left-5 right-5">
              <div className="bg-gradient-to-r from-electric-cyan/10 to-purple-accent/10 p-4 rounded-xl border border-gray-700">
                <p className="text-sm text-gray-300">
                  Physical AI & Humanoid Robotics
                </p>
                <p className="text-xs text-gray-500 mt-1">
                  Bridging digital AI and physical robotics
                </p>
              </div>
            </div>
          </motion.aside>
        </>
      )}
    </AnimatePresence>
  );
};

export default Sidebar;