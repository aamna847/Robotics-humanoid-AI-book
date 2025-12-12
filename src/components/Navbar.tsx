import React, { useState } from 'react';
import Sidebar from '../components/Sidebar';

const Navbar = () => {
  const [sidebarOpen, setSidebarOpen] = useState(false);
  
  const toggleSidebar = () => {
    setSidebarOpen(!sidebarOpen);
  };

  const scrollToSection = (sectionId) => {
    const element = document.getElementById(sectionId);
    if (element) {
      window.scrollTo({
        top: element.offsetTop - 80, // Subtract navbar height for proper positioning
        behavior: 'smooth'
      });
    }
  };

  return (
    <>
      <nav className="w-full py-6 px-12 flex justify-between items-center bg-[#0a0f1c] sticky top-0 z-50">
        <div className="flex items-center space-x-6">
          <div className="w-16 h-16 rounded-xl bg-gradient-to-r from-electric-cyan to-purple-accent flex items-center justify-center shadow-lg shadow-electric-cyan/30">
            <span className="font-bold text-navy-black text-2xl">AI</span>
          </div>
        </div>

        <div className="flex flex-col items-center mx-auto">
          <h1 className="text-5xl font-bold text-white bg-clip-text text-transparent bg-gradient-to-r from-electric-cyan to-purple-accent">
            Physical AI
          </h1>
          <p className="text-lg text-gray-400 text-center mt-1">
            Humanoid Robotics
          </p>
        </div>

        <div className="flex items-center space-x-8">
          {/* Mobile Menu Button - Visible only on small devices */}
          <button
            className="p-2 rounded-lg bg-gray-800 hover:bg-gray-700 transition-colors hover:shadow-md hover:shadow-purple-accent/10 md:hidden"
            onClick={toggleSidebar}
            aria-label="Toggle menu"
            aria-expanded={sidebarOpen}
          >
            <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
              <path d="M4 6h16v2H4zm0 5h16v2H4zm0 5h16v2H4z"/>
            </svg>
          </button>
          
          {/* Desktop Navigation - Hidden on mobile */}
          <div className="hidden md:flex items-center space-x-8">
            <a
              href="#"
              onClick={(e) => {
                e.preventDefault();
                scrollToSection('curriculum-modules');
              }}
              className="hover:text-electric-cyan transition-colors flex items-center gap-1 cursor-pointer text-white"
            >
              <span>📚</span> <span>Modules</span>
            </a>
            <a
              href="/docs/introduction"
              className="hover:text-electric-cyan transition-colors flex items-center gap-1 text-white"
            >
              <span>📖</span> <span>Curriculum</span>
            </a>
            <a
              href="https://github.com/humanoid-robotics/physical-ai-book"
              target="_blank"
              rel="noopener noreferrer"
              className="p-2 rounded-lg bg-gray-800 hover:bg-gray-700 transition-colors hover:shadow-md hover:shadow-purple-accent/10"
            >
              <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
                <path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z"/>
              </svg>
            </a>
          </div>
        </div>
      </nav>
      
      {/* Sidebar Component */}
      <Sidebar isOpen={sidebarOpen} toggleSidebar={toggleSidebar} />
    </>
  );
};

export default Navbar;