import React from 'react';
import { StatCardProps } from '../types';

const StatCard = ({ value, label, icon }: StatCardProps) => {
  return (
    <div className="bg-gray-900/50 backdrop-blur-xs border border-gray-800 rounded-xl p-6 transition-all duration-300 hover:bg-gray-900/70 hover:shadow-[0_0_20px_rgba(0,255,255,0.3)] hover:scale-[1.02]">
      <div className="flex items-center justify-between">
        <div>
          <div className="text-3xl font-bold text-electric-cyan mb-1">{value}</div>
          <div className="text-gray-400">{label}</div>
        </div>
        <div className="p-3 rounded-lg bg-gray-800/50 text-electric-cyan">
          {icon}
        </div>
      </div>
    </div>
  );
};

export default StatCard;