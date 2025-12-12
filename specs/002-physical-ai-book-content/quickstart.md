# Quickstart Guide: Accessible Search for Physical AI & Humanoid Robotics Documentation

## Overview
This guide provides a quick walkthrough of how to implement an accessible search bar for the Physical AI & Humanoid Robotics textbook documentation site using Docusaurus.

## Prerequisites
- Node.js 18 or higher
- Docusaurus v3.1 or higher
- Access to the `002-physical-ai-book-content` branch

## Setup Steps

### 1. Install Dependencies
```bash
npm install @docusaurus/theme-search-algolia
```

### 2. Configure Docusaurus for Search
Update your `docusaurus.config.js` to include the Algolia search plugin:

```javascript
// docusaurus.config.js
module.exports = {
  // ... existing config
  themes: [
    // ... other themes
    [
      '@docusaurus/theme-classic',
      /** @type {import('@docusaurus/theme-classic').Options} */
      ({
        customCss: require.resolve('./src/css/custom.css'),
      }),
    ],
    [
      '@docusaurus/theme-search-algolia',
      /** @type {import('@docusaurus/theme-search-algolia').Options} */
      ({
        // The application ID provided by Algolia
        appId: 'YOUR_APP_ID',
        // Public API key: it is safe to commit it
        apiKey: 'YOUR_SEARCH_API_KEY',
        indexName: 'your-index-name',
        // Optional: see doc section below
        contextualSearch: true,
        // Optional: path for search page that enabled by default (`false` to disable it)
        searchPagePath: 'search',
        // ... other options
      }),
    ],
  ],
};
```

### 3. Implement Accessible Search Component
Create an accessible search component in `src/components/AccessibleSearchBar/index.js`:

```jsx
import React, { useState, useRef, useEffect } from 'react';
import { useSearchPage } from '@docusaurus/theme-common';
import { translate } from '@docusaurus/Translate';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import clsx from 'clsx';

const AccessibleSearchBar = () => {
  const isSearchPage = useSearchPage();
  const [isOpen, setIsOpen] = useState(false);
  const [query, setQuery] = useState('');
  const inputRef = useRef(null);
  const containerRef = useRef(null);

  const { siteConfig } = useDocusaurusContext();
  const searchConfig = siteConfig.customFields?.searchConfig || {};

  // Focus the search input when opening the search bar
  useEffect(() => {
    if (isOpen && inputRef.current) {
      inputRef.current.focus();
    }
  }, [isOpen]);

  // Close search when clicking outside
  useEffect(() => {
    const handleClickOutside = (event) => {
      if (containerRef.current && !containerRef.current.contains(event.target)) {
        setIsOpen(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, []);

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (event) => {
      // Ctrl + K to focus search
      if ((event.ctrlKey || event.metaKey) && event.key === 'k') {
        event.preventDefault();
        setIsOpen(true);
      }
      
      // Escape to close search
      if (event.key === 'Escape') {
        setIsOpen(false);
      }
    };

    document.addEventListener('keydown', handleKeyDown);
    return () => {
      document.removeEventListener('keydown', handleKeyDown);
    };
  }, []);

  return (
    <div className="navbar__search" ref={containerRef}>
      {!isOpen && (
        <button
          type="button"
          className="navbar__search-button"
          aria-label={translate({
            id: 'theme.SearchBar.seeAll',
            message: searchConfig.ariaLabels?.searchButton || 'Open Search',
          })}
          onClick={() => setIsOpen(true)}
        >
          <svg className="navbar__search-icon" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <circle cx="11" cy="11" r="8"></circle>
            <line x1="21" y1="21" x2="16.65" y2="16.65"></line>
          </svg>
        </button>
      )}
      
      {(isOpen || isSearchPage) && (
        <div className={clsx('navbar__search-input-container', { 'navbar__search-input-container--open': isOpen })}>
          <input
            ref={inputRef}
            type="search"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder={translate({
              id: 'theme.SearchBar.placeholder',
              message: searchConfig.placeholderText || 'Search Physical AI & Robotics Documentation',
            })}
            aria-label={translate({
              id: 'theme.SearchBar.seeAll',
              message: searchConfig.ariaLabels?.searchInput || 'Search Input',
            })}
            className="navbar__search-input"
            role="combobox"
            aria-expanded={isOpen}
            aria-owns="search-results"
            autoComplete="off"
          />
          <button
            type="button"
            className="navbar__search-clear-button"
            aria-label={translate({
              id: 'theme.SearchBar.clearButton',
              message: searchConfig.ariaLabels?.clearSearch || 'Clear Search',
            })}
            onClick={() => {
              setQuery('');
              inputRef.current?.focus();
            }}
          >
            ✕
          </button>
        </div>
      )}
    </div>
  );
};

export default AccessibleSearchBar;
```

### 4. Add CSS for Accessibility
Add accessibility-focused CSS in `src/css/custom.css`:

```css
/* Accessible Search Styles */
.navbar__search {
  position: relative;
}

.navbar__search-input-container {
  display: flex;
  align-items: center;
  position: relative;
}

.navbar__search-input {
  padding: 8px 30px 8px 40px;
  width: 200px;
  border: 2px solid #ddd;
  border-radius: 4px;
  font-size: 16px;
  transition: width 0.2s ease;
}

.navbar__search-input:focus {
  outline: 3px solid #007bff;
  outline-offset: 1px;
  border-color: #007bff;
}

.navbar__search-input-container--open .navbar__search-input {
  width: 300px;
}

.navbar__search-icon {
  position: absolute;
  left: 10px;
  color: #666;
  pointer-events: none;
}

.navbar__search-clear-button {
  position: absolute;
  right: 10px;
  background: none;
  border: none;
  cursor: pointer;
  color: #999;
  padding: 4px;
}

.navbar__search-clear-button:hover,
.navbar__search-clear-button:focus {
  color: #333;
  outline: 2px solid #007bff;
  outline-offset: 1px;
}

/* Keyboard navigation focus indicators */
.navbar__search-button:focus {
  outline: 2px solid #007bff;
  outline-offset: 1px;
}

/* Screen reader only text */
.sr-only {
  position: absolute;
  width: 1px;
  height: 1px;
  padding: 0;
  margin: -1px;
  overflow: hidden;
  clip: rect(0, 0, 0, 0);
  white-space: nowrap;
  border: 0;
}

/* Search results styling */
.search-result-item:focus {
  outline: 2px solid #007bff;
  outline-offset: 1px;
}

/* High contrast mode support */
@media (prefers-contrast: high) {
  .navbar__search-input {
    border-width: 3px;
  }
  
  .navbar__search-input:focus {
    outline-width: 4px;
  }
}
```

### 5. Update Navigation Bar
Update your navigation bar in `src/components/Navbar/index.js` to include the accessible search bar:

```jsx
import React from 'react';
import AccessibleSearchBar from '../AccessibleSearchBar';

const Navbar = () => {
  return (
    <nav className="navbar navbar--fixed-top">
      {/* ... other navbar elements */}
      <div className="navbar__items navbar__items--right">
        {/* ... other items */}
        <AccessibleSearchBar />
      </div>
    </nav>
  );
};

export default Navbar;
```

## Testing Accessibility

### 1. Keyboard Navigation
- Use `Ctrl+K` to focus the search input
- Navigate search results with arrow keys
- Press Enter to select a result
- Press Escape to close the search panel

### 2. Screen Reader Compatibility
- Verify all search elements have appropriate ARIA labels
- Test with screen readers like NVDA or JAWS
- Ensure search results are announced properly

### 3. Automated Testing
```bash
# Run accessibility tests
npm run test:accessibility
```

## Verification Steps

1. The search bar is visible in the navigation bar
2. The search bar is focusable using keyboard navigation
3. Pressing `Ctrl+K` focuses the search input
4. Search results appear as you type
5. Results are properly linked to relevant content
6. All elements have appropriate ARIA attributes
7. Screen readers properly announce search elements and results
8. High contrast mode works correctly