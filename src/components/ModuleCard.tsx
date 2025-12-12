import React from 'react';

interface ModuleCardProps {
  moduleNumber: string;
  title: string;
  description: string;
  icon: string;
  glowColor: string;
}

const ModuleCard = ({ moduleNumber, title, description, icon, glowColor }: ModuleCardProps) => {
  return (
    <div 
      className={`
        bg-gray-900/50 backdrop-blur-xs border border-gray-800 rounded-xl p-6 
        transition-all duration-300 hover:bg-gray-900/70 
        hover:scale-[1.03] ${glowColor} hover:shadow-2xl
        transform hover:-translate-y-1
      `}
      style={{ 
        boxShadow: `0 0 15px ${glowColor.replace('hover:', '').includes('cyan') ? 'rgba(0, 255, 255, 0.3)' : 'rgba(127, 79, 255, 0.3)'}`
      }}
    >
      <div className="flex items-start space-x-4">
        <div className={`p-3 rounded-lg ${glowColor.includes('cyan') ? 'bg-cyan-900/30 text-electric-cyan' : 'bg-purple-900/30 text-purple-accent'}`}>
          {icon}
        </div>
        <div>
          <div 
            className={`text-3xl font-bold mb-1 ${glowColor.includes('cyan') ? 'text-electric-cyan' : 'text-purple-accent'}`}
            style={{ 
              textShadow: `${glowColor.includes('cyan') ? '0 0 8px rgba(0, 255, 255, 0.7)' : '0 0 8px rgba(127, 79, 255, 0.7)'}`
            }}
          >
            {moduleNumber}
          </div>
          <h3 
            className="text-xl font-bold mb-2 bg-clip-text text-transparent"
            style={{ 
              backgroundImage: 'linear-gradient(to right, #00FFFF, #7F4FFF)',
              textShadow: '0 0 10px rgba(127, 79, 255, 0.3)'
            }}
          >
            {title}
          </h3>
          <p className="text-gray-400">{description}</p>
        </div>
      </div>
    </div>
  );
};

export default ModuleCard;