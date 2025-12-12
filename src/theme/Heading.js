import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import OriginalHeading from '@theme-original/Heading';

const MarkdownHeading = (props) => {
  const [shouldAnimate, setShouldAnimate] = useState(false);
  const [loaded, setLoaded] = useState(false);
  
  useEffect(() => {
    // Apply to all pages within the site
    const path = typeof window !== 'undefined' ? window.location.pathname : '';
    const isSitePage = path !== undefined;
    
    setShouldAnimate(isSitePage);
    setLoaded(true);
  }, []);

  // Only animate for headings up to level 3 and when component is loaded
  if (!shouldAnimate || props.level > 3 || !loaded) {
    return <OriginalHeading {...props} />;
  }

  // Tech-style emojis to randomly select from
  const techEmojis = ['💻', '🖥️', '🛠️', '📡', '⚙️', '📱', '🔋', '🔌', '🔍', '🚀', '💡', '🌐', '💾', '💿', '⚛️', '⚡', '🔬', '.telescope', '📡', '🧩', '🔄', '📦', '📋', '📌', '🔔', '🔒', '🔓', '🔑', '🔨', '🔧', '🧪', '🎨', '⚙️'];

  // Generate 2-4 random emojis for this heading
  const randomEmojis = Array.from({ length: Math.floor(Math.random() * 3) + 2 }, () =>
    techEmojis[Math.floor(Math.random() * techEmojis.length)]
  );

  return (
    <div className="relative inline-block w-full">
      {/* Render animated emojis behind the heading */}
      {randomEmojis.map((emoji, index) => {
        const leftOffset = Math.random() * 40 - 20; // Random position between -20% and +20%
        const topOffset = Math.random() * 30 - 15;  // Random position between -15px and +15px

        // Make animations more subtle on smaller screens
        const isMobile = typeof window !== 'undefined' && window.innerWidth < 768;
        const opacity = isMobile ? 0.05 : 0.1;
        const duration = isMobile ? 3 : 2.5;
        const fontSize = props.level === 1 ? '3rem' : props.level === 2 ? '2.5rem' : '2rem';
        const positionDivisor = isMobile ? 3 : 2;
        const rotationAngle = isMobile ? (index % 2 === 0 ? 5 : -5) : (index % 2 === 0 ? 10 : -10);

        return (
          <motion.span
            key={index}
            className="absolute z-0 tech-emoji"
            initial={{ opacity: 0, y: 30, x: leftOffset, rotate: 0 }}
            animate={{
              opacity: opacity,
              y: isMobile ? -10 : -20,
              x: leftOffset / positionDivisor,
              rotate: rotationAngle,
              scale: isMobile ? [1, 1.05, 1] : [1, 1.1, 1]  // Gentle pulsing effect
            }}
            transition={{
              duration: duration,
              delay: index * 0.1,
              ease: "easeOut",
              repeat: isMobile ? Infinity : Infinity, // Keep animations for both mobile and desktop
              repeatType: "reverse",
              repeatDelay: Math.random() * 2
            }}
            style={{
              fontSize: fontSize,
              left: `${50 + leftOffset}%`,
              top: `${topOffset}px`,
              transform: 'translateX(-50%)',
            }}
          >
            {emoji}
          </motion.span>
        );
      })}
      
      {/* Render the original heading on top */}
      <OriginalHeading {...props} className={`relative z-10 ${props.className || ''}`} />
    </div>
  );
};

export default MarkdownHeading;