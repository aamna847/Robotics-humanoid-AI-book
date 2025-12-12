import React, { useEffect, useState } from 'react';
import { motion } from 'framer-motion';

const AnimatedEmoji = ({ emoji, delay = 0 }) => {
  return (
    <motion.span
      className="absolute opacity-20 text-lg z-0"
      initial={{ opacity: 0, y: 20, rotate: 0 }}
      animate={{ opacity: 0.2, y: -10, rotate: 5 }}
      transition={{
        duration: 0.8,
        delay: delay,
        ease: "easeOut"
      }}
      style={{
        fontSize: '1.5rem',
      }}
    >
      {emoji}
    </motion.span>
  );
};

const ChapterTitleWithEmojis = ({ children, ...props }) => {
  const techEmojis = ['💻', '🖥️', '🛠️', '📡', '⚙️', '📱', '🔋', '🔌', '🔍', '🚀', '💡', '🌐', '💾', '💿', '📡'];
  
  // Get current path to determine if we're on a chapter page
  const [isChapterPage, setIsChapterPage] = useState(false);
  
  useEffect(() => {
    // Check if current path includes /docs/ but not /docs/ (homepage)
    setIsChapterPage(
      typeof window !== 'undefined' &&
      window.location.pathname.includes('/docs/') &&
      window.location.pathname !== '/docs/'
    );
  }, []);

  if (!isChapterPage || !children) {
    return <>{children}</>;
  }

  // Select random emojis for this instance
  const randomEmojis = Array.from({ length: 3 }, () => 
    techEmojis[Math.floor(Math.random() * techEmojis.length)]
  );

  return (
    <div className="relative">
      {randomEmojis.map((emoji, index) => (
        <AnimatedEmoji 
          key={index} 
          emoji={emoji} 
          delay={index * 0.2} 
        />
      ))}
      <span className="relative z-10">{children}</span>
    </div>
  );
};

export default ChapterTitleWithEmojis;