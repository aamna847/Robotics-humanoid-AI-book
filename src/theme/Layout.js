import React from 'react';
import { motion } from 'framer-motion';
import OriginalLayout from '@theme-original/Layout';

const Layout = (props) => {
  // Tech-style emojis list
  const techEmojis = ['💻', '🖥️', '🛠️', '📡', '⚙️', '📱', '🔋', '🔌', '🔍', '🚀', '💡', '🌐', '💾', '💿', '⚛️', '⚡', '🔬', '🔭', '📡', '🧩', '🔄', '📦', '📋', '📌', '🔔', '🔒', '🔓', '🔑', '🔨', '🔧', '🧪', '🎨', '⚙️'];

  // Function to generate animated emojis for the background
  const renderFloatingEmojis = () => {
    // Adjust emoji count based on screen size for responsiveness
    const emojiCount = 12; // Fixed count but size will be responsive

    const emojis = [];
    for (let i = 0; i < emojiCount; i++) {
      const randomEmoji = techEmojis[Math.floor(Math.random() * techEmojis.length)];
      const leftPos = Math.random() * 100; // 0-100% horizontal position
      const topPos = Math.random() * 100;  // 0-100% vertical position
      const size = 20; // Base size, CSS will handle responsiveness
      const animationDuration = Math.random() * 10 + 15; // 15-25s random duration
      const delay = Math.random() * 5; // 0-5s random delay

      emojis.push(
        <motion.div
          key={i}
          className="floating-emoji fixed pointer-events-none opacity-5 z-0"
          initial={{ opacity: 0, y: 50, x: 0, rotate: 0 }}
          animate={{
            opacity: 0.05,
            y: -80,
            x: Math.random() * 40 - 20,
            rotate: Math.random() * 20 - 10
          }}
          transition={{
            duration: animationDuration / 5, // Speed up the initial animation significantly
            delay: delay / 10, // Reduce the delay drastically for almost immediate appearance
            repeat: Infinity,
            repeatType: "reverse",
            ease: "easeInOut"
          }}
          style={{
            left: `${leftPos}%`,
            top: `${topPos}%`,
            fontSize: `${size}px`,
            zIndex: 0,
            filter: 'grayscale(70%) brightness(1.5)',
          }}
        >
          {randomEmoji}
        </motion.div>
      );
    }

    return emojis;
  };

  return (
    <>
      {/* Floating animated emojis in the background on all pages */}
      {renderFloatingEmojis()}

      {/* Original layout with children */}
      <OriginalLayout {...props} />
    </>
  );
};

export default Layout;