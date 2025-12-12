import React, { Component } from 'react';
import { motion } from 'framer-motion';

// Component to add animated tech-style emojis to any content
class AnimatedEmojiBackground extends Component {
  constructor(props) {
    super(props);
    this.techEmojis = ['💻', '🖥️', '🛠️', '📡', '⚙️', '📱', '🔋', '🔌', '🔍', '🚀', '💡', '🌐', '💾', '💿', '⚛️', '⚡', '🔬', '🔭', '📡', '🧩', '🔄', '📦', '📋', '📌', '🔔', '🔒', '🔓', '🔑', '.hammer', '🔧', '🧪', '🎨', '⚙️'];
  }

  getRandomEmojis = (count) => {
    const shuffled = [...this.techEmojis].sort(() => 0.5 - Math.random());
    return shuffled.slice(0, count);
  };

  render() {
    const { children, className = '', style = {}, emojiCount = 5 } = this.props;
    
    const randomEmojis = this.getRandomEmojis(Math.min(emojiCount, this.techEmojis.length));

    return (
      <div className={`relative ${className}`} style={style}>
        {/* Animated Emoji Layer Behind Content */}
        {randomEmojis.map((emoji, index) => {
          const leftPos = Math.random() * 100; // 0-100% horizontal position
          const topPos = Math.random() * 100;  // 0-100% vertical position
          const size = Math.random() * 24 + 16; // 16-40px random size
          const delay = Math.random() * 2;      // Random delay up to 2s
          
          return (
            <motion.div
              key={index}
              className="absolute pointer-events-none z-0"
              initial={{ opacity: 0, y: 20, x: -10, rotate: 0 }}
              animate={{ opacity: 0.15, y: -30, x: 10, rotate: 5 }}
              transition={{
                duration: 2,
                delay: delay,
                repeat: Infinity,
                repeatType: "reverse",
                ease: "easeOut"
              }}
              style={{
                left: `${leftPos}%`,
                top: `${topPos}%`,
                fontSize: `${size}px`,
                filter: 'grayscale(100%)',
              }}
            >
              {emoji}
            </motion.div>
          );
        })}
        
        {/* Actual Content Layer */}
        <div className="relative z-10">
          {children}
        </div>
      </div>
    );
  }
}

export default AnimatedEmojiBackground;