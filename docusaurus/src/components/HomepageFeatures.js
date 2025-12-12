import React from 'react';
import clsx from 'clsx';

const FeatureList = [
  {
    title: '🤖 Embodied Intelligence',
    description: (
      <>
        Learn how AI operates in the physical world through embodied intelligence.
        Bridge the gap between digital AI (LLMs, Computer Vision, Reinforcement Learning)
        and physical robotics.
      </>
    ),
  },
  {
    title: '⚡ Full Pipeline Coverage',
    description: (
      <>
        From ROS 2 fundamentals to Gazebo/Unity simulation, NVIDIA Isaac Sim/ROS for AI-powered
        perception, to Vision-Language-Action models for natural commands.
      </>
    ),
  },
  {
    title: '🧩 Hands-on Implementation',
    description: (
      <>
        Build a humanoid robot that hears voice commands, plans, navigates, identifies objects,
        and manipulates them with 85% success rate.
      </>
    ),
  },
];

function Feature({Svg, title, description}) {
  return (
    <div className={clsx('col col--4')}>
      <div className="text--center padding-horiz--md">
        <h3 style={{
          fontSize: '1.5rem',
          fontWeight: '700',
          paddingBottom: '6px',
          borderBottom: '1px solid #00FFFF80'
        }}>{title}</h3>
        <div style={{ marginTop: '8px' }}>
          <p style={{ lineHeight: '1.7' }}>{description}</p>
        </div>
      </div>
    </div>
  );
}

export default function HomepageFeatures() {
  return (
    <section className="features">
      <div className="container">
        <div className="row">
          {FeatureList.map((props, idx) => (
            <Feature key={idx} {...props} />
          ))}
        </div>
      </div>
    </section>
  );
}