import React, { useState, useRef, useEffect } from 'react';
import { useSearchPage, useLocationChange } from '@docusaurus/theme-common';
import { translate } from '@docusaurus/Translate';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import clsx from 'clsx';

const AccessibleSearchBar = React.memo(() => {
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
        setQuery('');
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
      // Ctrl + K or Cmd + K to focus search
      if ((event.ctrlKey || event.metaKey) && event.key === 'k') {
        event.preventDefault();
        setIsOpen(true);
        if (inputRef.current) {
          inputRef.current.focus();
        }
      }
      
      // Escape to close search
      if (event.key === 'Escape' && isOpen) {
        setIsOpen(false);
        setQuery('');
        if (document.activeElement === inputRef.current) {
          document.activeElement.blur();
        }
      }
    };

    document.addEventListener('keydown', handleKeyDown);
    return () => {
      document.removeEventListener('keydown', handleKeyDown);
    };
  }, [isOpen]);

  // Handle location changes to clear the search state
  useLocationChange(() => {
    setIsOpen(false);
    setQuery('');
  });

  return (
    <div className="navbar__search" ref={containerRef}>
      {!isOpen && !isSearchPage && (
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
          <svg className="navbar__search-icon" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <circle cx="11" cy="11" r="8"></circle>
            <line x1="21" y1="21" x2="16.65" y2="16.65"></line>
          </svg>
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
            // ARIA attributes for accessibility
            aria-autocomplete="list"
            aria-haspopup="listbox"
          />
          {query && (
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
          )}
        </div>
      )}
      
      {/* Display keyboard shortcut hint */}
      {!isSearchPage && (
        <span className="sr-only">
          {translate({
            id: 'theme.SearchBar.shortcutLabel',
            message: 'Press Ctrl+K to focus search',
          })}
        </span>
      )}
    </div>
  );
});

export default AccessibleSearchBar;