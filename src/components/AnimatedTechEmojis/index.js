import React from 'react';
import { motion } from 'framer-motion';

// Reusable HOC to add animated tech emojis behind any component
const withAnimatedTechEmojis = (WrappedComponent, options = {}) => {
  const { 
    emojiCount = 3, 
    animationType = 'fade-up', 
    size = 'medium',
    includeRotation = true 
  } = options;

  // Tech-style emojis
  const techEmojis = [
    '💻', '🖥️', '🛠️', '📡', '⚙️', '📱', '🔋', '🔌', '🔍', 
    '🚀', '💡', '🌐', '💾', '💿', '⚛️', '⚡', '🔬', '🔭', 
    '🧩', '🔄', '📦', '📋', '📌', '🔔', '🔒', '🔓', '🔑', 
    '🔨', '🔧', '🧪', '🎨', '⚙️', '⌨️', '🖱️', '🖨️', '💾', '💿'
  ];

  const AnimatedWrapper = (props) => {
    // Randomly select emojis for this instance
    const randomEmojis = Array.from({ length: emojiCount }, () => 
      techEmojis[Math.floor(Math.random() * techEmojis.length)]
    );

    // Size mapping
    const sizeMap = {
      small: '1rem',
      medium: '1.5rem',
      large: '2rem',
      xlarge: '2.5rem'
    };
    
    const emojiSize = sizeMap[size] || sizeMap.medium;

    return (
      <div className="relative inline-block w-full">
        {/* Animated Emoji Layer Behind Content */}
        {randomEmojis.map((emoji, index) => {
          // Random positions relative to the element
          const leftOffset = (Math.random() * 40 - 20); // -20% to +20% horizontally
          const topOffset = (Math.random() * 40 - 20);  // -20px to +20px vertically
          const delay = index * 0.2; // Stagger animations

          return (
            <motion.span
              key={index}
              className="absolute opacity-15 z-0 pointer-events-none"
              initial={{ 
                opacity: 0, 
                y: 30, 
                x: leftOffset, 
                rotate: includeRotation ? 0 : 0 
              }}
              animate={{ 
                opacity: 0.15, 
                y: -15, 
                x: leftOffset / 2, 
                rotate: includeRotation ? (index % 2 === 0 ? 15 : -15) : 0
              }}
              transition={{
                duration: 1.5,
                delay: delay,
                ease: "easeOut",
                repeat: Infinity,
                repeatType: "reverse",
                repeatDelay: Math.random() * 2
              }}
              style={{
                fontSize: emojiSize,
                left: `${50 + leftOffset}%`,
                top: `${topOffset}px`,
                transform: 'translateX(-50%)',
              }}
            >
              {emoji}
            </motion.span>
          );
        })}
        
        {/* Wrapped Component */}
        <div className="relative z-10">
          <WrappedComponent {...props} />
        </div>
      </div>
    );
  };

  // Maintain the original component's name for debugging
  AnimatedWrapper.displayName = `withAnimatedTechEmojis(${WrappedComponent.displayName || WrappedComponent.name || 'Component'})`;
  
  return AnimatedWrapper;
};

// Standalone component for direct use without HOC
const AnimatedTechEmojiContainer = ({ children, emojiCount = 3, size = 'medium', includeRotation = true }) => {
  // Tech-style emojis
  const techEmojis = [
    '💻', '🖥️', '🛠️', '📡', '⚙️', '📱', '🔋', '🔌', '🔍', 
    '🚀', '💡', '🌐', '💾', '💿', '⚛️', '⚡', '🔬', '.telescope', 
    '🧩', '🔄', '📦', '📋', '📌', '🔔', '🔒', '🔓', '🔑', 
    '🔨', '🔧', '🧪', '🎨', '⚙️', '⌨️', '🖱️', '🖨️', '💾', '💿'
  ];

  // Size mapping
  const sizeMap = {
    small: '1rem',
    medium: '1.5rem',
    large: '2rem',
    xlarge: '2.5rem'
  };
  
  const emojiSize = sizeMap[size] || sizeMap.medium;

  // Randomly select emojis
  const randomEmojis = Array.from({ length: emojiCount }, () => 
    techEmojis[Math.floor(Math.random() * techEmojis.length)]
  );

  return (
    <div className="relative w-full">
      {/* Animated Emoji Layer Behind Content */}
      {randomEmojis.map((emoji, index) => {
        // Random positions relative to the element
        const leftOffset = (Math.random() * 40 - 20); // -20% to +20% horizontally
        const topOffset = (Math.random() * 40 - 20);  // -20px to +20px vertically
        const delay = index * 0.2; // Stagger animations

        return (
          <motion.span
            key={index}
            className="absolute opacity-15 z-0 pointer-events-none"
            initial={{ 
              opacity: 0, 
              y: 30, 
              x: leftOffset, 
              rotate: includeRotation ? 0 : 0 
            }}
            animate={{ 
              opacity: 0.15, 
              y: -15, 
              x: leftOffset / 2, 
              rotate: includeRotation ? (index % 2 === 0 ? 15 : -15) : 0
            }}
            transition={{
              duration: 1.5,
              delay: delay,
              ease: "easeOut",
              repeat: Infinity,
              repeatType: "reverse",
              repeatDelay: Math.random() * 2
            }}
            style={{
              fontSize: emojiSize,
              left: `${50 + leftOffset}%`,
              top: `${topOffset}px`,
              transform: 'translateX(-50%)',
            }}
          >
            {emoji}
          </motion.span>
        );
      })}
      
      {/* Children Content */}
      <div className="relative z-10">
        {children}
      </div>
    </div>
  );
};

export { withAnimatedTechEmojis, AnimatedTechEmojiContainer };