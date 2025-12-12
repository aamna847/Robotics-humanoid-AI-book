import React from 'react';
import clsx from 'clsx';
import {translate} from '@docusaurus/Translate';
import styles from './Footer.module.css';

function Footer() {
  return (
    <footer
      className={clsx('footer footer--dark', styles.footer)}>
      <div className="container">
        <div className="text--center">
          <div className={styles.footerContent}>
            <span>Built with </span>
            <span 
              className={styles.heart}
              style={{ 
                color: '#FF3333', 
                animation: 'heartbeat 1.5s infinite',
                display: 'inline-block',
                fontSize: '1.2em'
              }}
            >
              ❤️
            </span>
            <span> by </span>
            <span style={{ fontWeight: '600', color: '#e2e2e2' }}>Aamna Rana</span>
          </div>
        </div>
      </div>
      <style jsx>{`
        @keyframes heartbeat {
          0% { transform: scale(1); }
          25% { transform: scale(1.1); }
          50% { transform: scale(1); }
          75% { transform: scale(1.1); }
          100% { transform: scale(1); }
        }
      `}</style>
    </footer>
  );
}

export default React.memo(Footer);