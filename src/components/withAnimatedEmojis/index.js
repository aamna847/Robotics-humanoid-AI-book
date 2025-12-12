import React, { useEffect, useState } from 'react';
import { motion } from 'framer-motion';

// A HOC that wraps components with animated tech emojis
const withAnimatedEmojis = (WrappedComponent, level = 1) => {
  return (props) => {
    const [showAnimations, setShowAnimations] = useState(false);
    
    useEffect(() => {
      // Apply to all pages that are part of the book/website
      setShowAnimations(
        typeof window !== 'undefined' &&
        (window.location.pathname === '/' || 
         window.location.pathname.includes('/docs/') ||
         window.location.pathname.includes('/blog/'))
      );
    }, []);

    // Tech-style emojis to randomly select from
    const techEmojis = ['💻', '🖥️', '🛠️', '📡', '⚙️', '📱', '🔋', '🔌', '🔍', '🚀', '💡', '🌐', '💾', '💿', '⚛️', '⚡', '🔬', '🔭', '📡', '🧩', '🔄', '📦', '📋', '📌', '🔔', '🔒', '🔓', '🔑', '🔨', '🔧', '🧪', '🎨', '⚙️'];

    if (!showAnimations) {
      return <WrappedComponent {...props} />;
    }

    // Function to generate random emojis
    const getRandomEmojis = (count) => {
      const shuffled = [...techEmojis].sort(() => 0.5 - Math.random());
      return shuffled.slice(0, count);
    };

    // Generate 2-4 random emojis for this element
    const randomEmojis = getRandomEmojis(Math.floor(Math.random() * 3) + 2);

    return (
      <div className="relative">
        {/* Render animated emojis behind the content */}
        {randomEmojis.map((emoji, index) => (
          <motion.span
            key={index}
            className="absolute opacity-10 z-0"
            initial={{ opacity: 0, y: 20, x: 0, rotate: 0 }}
            animate={{ opacity: 0.1, y: -10, x: Math.random() * 20 - 10, rotate: Math.random() * 10 - 5 }}
            transition={{
              duration: 1.8,
              delay: index * 0.1,
              ease: "easeOut"
            }}
            style={{
              fontSize: level === 1 ? '2.5rem' : level === 2 ? '2rem' : '1.5rem',
              left: `${Math.random() * 20 - 10}%`,
              top: `${Math.random() * 20 - 10}px`,
            }}
          >
            {emoji}
          </motion.span>
        ))}
        
        {/* The wrapped component */}
        <div className="relative z-10">
          <WrappedComponent {...props} />
        </div>
      </div>
    );
  };
};

export default withAnimatedEmojis;