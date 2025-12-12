/**
 * @jest-environment jsdom
 */
import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import '@testing-library/jest-dom';
import AccessibleSearchBar from '../index';

// Mock the Docusaurus hooks and modules
jest.mock('@docusaurus/theme-common', () => ({
  useSearchPage: jest.fn(() => false),
  useLocationChange: jest.fn(() => {}),
}));

jest.mock('@docusaurus/Translate', () => ({
  translate: jest.fn((args) => {
    if (args.id === 'theme.SearchBar.placeholder') {
      return 'Search Physical AI & Robotics Documentation';
    }
    if (args.id === 'theme.SearchBar.seeAll') {
      return 'Search Input';
    }
    return args.message || 'Default Translation';
  }),
}));

jest.mock('@docusaurus/useDocusaurusContext', () => ({
  default: jest.fn(() => ({
    siteConfig: {
      customFields: {
        searchConfig: {
          ariaLabels: {
            searchInput: 'Search Input',
            searchButton: 'Open Search',
            clearSearch: 'Clear Search',
          },
          placeholderText: 'Search Physical AI & Robotics Documentation',
        },
      },
    },
  })),
}));

// Mock clsx
jest.mock('clsx', () => jest.fn((...args) => args.join(' ')));

describe('AccessibleSearchBar', () => {
  beforeEach(() => {
    // Reset all mocks before each test
    jest.clearAllMocks();
  });

  test('renders search button when not on search page and search is closed', () => {
    const { useSearchPage } = require('@docusaurus/theme-common');
    useSearchPage.mockReturnValue(false);

    render(<AccessibleSearchBar />);
    
    const searchButton = screen.getByLabelText(/Open Search/i);
    expect(searchButton).toBeInTheDocument();
  });

  test('renders search input when search is open', () => {
    const { useSearchPage } = require('@docusaurus/theme-common');
    useSearchPage.mockReturnValue(false);

    render(<AccessibleSearchBar />);
    
    // Initially should show the search button
    const searchButton = screen.getByLabelText(/Open Search/i);
    fireEvent.click(searchButton);
    
    // After clicking, the search input should appear
    const searchInput = screen.getByPlaceholderText(/Search Physical AI & Robotics Documentation/i);
    expect(searchInput).toBeInTheDocument();
  });

  test('has proper ARIA attributes for accessibility', () => {
    render(<AccessibleSearchBar />);
    
    const searchButton = screen.getByLabelText(/Open Search/i);
    expect(searchButton).toHaveAttribute('aria-label');
  });

  test('toggles search visibility when clicking the search button', () => {
    const { useSearchPage } = require('@docusaurus/theme-common');
    useSearchPage.mockReturnValue(false);

    render(<AccessibleSearchBar />);
    
    // Initially should show the search button
    let searchButton = screen.getByLabelText(/Open Search/i);
    let searchInput = screen.queryByPlaceholderText(/Search Physical AI & Robotics Documentation/i);
    expect(searchButton).toBeInTheDocument();
    expect(searchInput).not.toBeInTheDocument();
    
    // Click the search button
    fireEvent.click(searchButton);
    
    // Now the search input should be visible and the button hidden
    searchInput = screen.getByPlaceholderText(/Search Physical AI & Robotics Documentation/i);
    searchButton = screen.queryByLabelText(/Open Search/i);
    expect(searchInput).toBeInTheDocument();
    expect(searchButton).not.toBeInTheDocument();
  });
});