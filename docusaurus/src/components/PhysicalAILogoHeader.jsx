import React from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import useBaseUrl from '@docusaurus/useBaseUrl';

// Inline SVG for Physical AI Logo
const PhysicalAILogo = () => (
  <svg 
    xmlns="http://www.w3.org/2000/svg" 
    viewBox="0 0 100 100" 
    className="physical-ai-logo"
    width="100%" 
    height="100%"
  >
    <defs>
      <filter id="neonGlow" x="-50%" y="-50%" width="200%" height="200%">
        <feGaussianBlur in="SourceAlpha" stdDeviation="3" result="blur"/>
        <feFlood flood-color="#00FFFF" flood-opacity="0.8" result="glowColor"/>
        <feComposite in="glowColor" in2="blur" operator="in" result="glow"/>
        <feMerge>
          <feMergeNode in="glow"/>
          <feMergeNode in="SourceGraphic"/>
        </feMerge>
      </filter>
      <linearGradient id="cyanGradient" x1="0%" y1="0%" x2="100%" y2="100%">
        <stop offset="0%" stopColor="#00FFFF" />
        <stop offset="100%" stopColor="#009999" />
      </linearGradient>
    </defs>
    
    {/* Main body path - single continuous line that transforms into circuits and joints */}
    <path 
      d="M50,25 
         C55,25 55,30 50,30 
         C45,30 45,25 50,25 
         L50,45 
         M50,45 L35,55 
         M50,45 L65,55 
         M50,45 L50,70 
         M50,70 L40,85 
         M50,70 L60,85
         M35,55 
         Q30,50 32,45
         Q34,40 37,42
         Q40,44 38,48
         Q36,52 35,55
         M65,55
         Q70,50 68,45
         Q66,40 63,42
         Q60,44 62,48
         Q64,52 65,55
         M40,85
         Q35,80 37,75
         Q39,70 42,72
         Q45,74 43,78
         Q41,82 40,85
         M60,85
         Q65,80 63,75
         Q61,70 58,72
         Q55,74 57,78
         Q59,82 60,85" 
      stroke="url(#cyanGradient)"
      strokeWidth="1.5"
      fill="none"
      strokeLinecap="round"
      strokeLinejoin="round"
      filter="url(#neonGlow)"
      className="logo-stroke"
    />
    
    {/* Circuit details */}
    <circle cx="50" cy="28" r="1" fill="none" stroke="url(#cyanGradient)" strokeWidth="0.5" filter="url(#neonGlow)" />
    <circle cx="45" cy="53" r="0.7" fill="none" stroke="url(#cyanGradient)" strokeWidth="0.4" filter="url(#neonGlow)" />
    <circle cx="55" cy="53" r="0.7" fill="none" stroke="url(#cyanGradient)" strokeWidth="0.4" filter="url(#neonGlow)" />
    <path d="M48,65 L52,65 M49,63 L51,63" stroke="url(#cyanGradient)" strokeWidth="0.5" filter="url(#neonGlow)" />
    
    {/* Text */}
    <text 
      x="50" 
      y="95" 
      textAnchor="middle" 
      fontSize="8" 
      fontWeight="bold"
      fill="url(#cyanGradient)"
      filter="url(#neonGlow)"
      fontFamily="system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif"
      className="logo-text"
    >
      Physical AI
    </text>
  </svg>
);

const PhysicalAILogoHeader = () => {
  return (
    <header className="navbar navbar--fixed-top">
      <div className="navbar__inner">
        <div className="navbar__items">
          <Link className="navbar__brand" to={useBaseUrl('/')} aria-label="Physical AI Logo">
            <div className="logo-wrapper">
              <PhysicalAILogo />
            </div>
          </Link>
        </div>
        <div className="navbar__items navbar__items--right">
          {/* Navigation items would go here */}
        </div>
      </div>
    </header>
  );
};

export default PhysicalAILogoHeader;

// CSS rules for the logo
const styles = `
  .logo-wrapper {
    width: 120px;
    height: 120px;
    transition: transform 0.3s ease;
  }
  
  .logo-wrapper:hover {
    transform: scale(1.05);
  }
  
  .logo-stroke {
    transition: all 0.3s ease;
  }
  
  .logo-text {
    transition: all 0.3s ease;
  }
  
  .logo-wrapper:hover .logo-stroke,
  .logo-wrapper:hover .logo-text {
    filter: 
      drop-shadow(0 0 4px #00FFFF)
      drop-shadow(0 0 8px #00FFFF)
      drop-shadow(0 0 12px #00FFFF);
  }
  
  /* Responsive design */
  @media (max-width: 996px) {
    .logo-wrapper {
      width: 100px;
      height: 100px;
    }
  }
  
  @media (max-width: 768px) {
    .logo-wrapper {
      width: 80px;
      height: 80px;
    }
  }
  
  @media (max-width: 480px) {
    .logo-wrapper {
      width: 60px;
      height: 60px;
    }
  }
`;

// Inject the CSS into the document
if (typeof document !== 'undefined') {
  const styleElement = document.createElement('style');
  styleElement.innerHTML = styles;
  document.head.appendChild(styleElement);
}