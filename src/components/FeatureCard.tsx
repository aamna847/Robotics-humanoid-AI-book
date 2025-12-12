import React from 'react';
import { FeatureCardProps } from '../types';

const FeatureCard = ({ title, description, icon }: FeatureCardProps) => {
  return (
    <div className="bg-gray-900/50 backdrop-blur-xs border border-gray-800 rounded-xl p-6 transition-all duration-300 hover:bg-gray-900/70 hover:shadow-[0_0_20px_rgba(127,79,255,0.3)] hover:scale-[1.02]">
      <div className="flex items-start space-x-4">
        <div className="p-3 rounded-lg bg-purple-900/30 text-purple-accent">
          {icon}
        </div>
        <div>
          <h3 className="text-xl font-bold mb-2">{title}</h3>
          <p className="text-gray-400">{description}</p>
        </div>
      </div>
    </div>
  );
};

export default FeatureCard;