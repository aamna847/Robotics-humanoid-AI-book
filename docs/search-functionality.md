# ℹ️ Search Functionality Documentation ℹ️

## 📊 Overview 📊

The Physical AI & Humanoid Robotics documentation site includes an accessible search feature that allows users to search across all book content, including headings, sections, and modules. The search functionality is built using Docusaurus with Algolia DocSearch and includes several accessibility features.

## ℹ️ Features ℹ️

### ℹ️ Search Capabilities ℹ️
- Search across all 12 book chapters (2000+ words each)
- Search in document titles, headings, and content
- Instant search results as you type
- Keyboard-friendly navigation
- Screen reader support

### ♿ Accessibility Features ♿
- Full keyboard navigation support
- ARIA labels and attributes for screen readers
- Proper focus management
- High contrast mode support
- Reduced motion support for users with vestibular disorders

## ℹ️ Usage ℹ️

### ℹ️ Keyboard Shortcuts ℹ️
- `Ctrl+K` (or `Cmd+K` on Mac) - Focus the search input
- `Escape` - Close the search panel
- `Tab` - Navigate between search results
- `Enter` - Select a search result

### 🧭 Screen Reader Navigation 🧭
The search is fully compatible with popular screen readers like NVDA, JAWS, and VoiceOver. All search elements have appropriate ARIA labels and roles.

## 🔨 Technical Implementation 🔨

### 🧩 Components 🧩
The search functionality is implemented using:
- Docusaurus theme-search-algolia
- Custom AccessibleSearchBar component with enhanced accessibility features
- Custom CSS for improved accessibility and styling

### ℹ️ Indexing ℹ️
All documentation pages are indexed automatically by Algolia. The search index includes:
- Page titles
- Headings (H1-H6)
- Content text
- Metadata

## ℹ️ For Developers ℹ️

### ℹ️ Configuring the Search ℹ️

The search functionality is configured in `docusaurus.config.js`:

```js
themes: [
  [
    '@docusaurus/theme-search-algolia',
    {
      // The application ID provided by Algolia
      appId: 'YOUR_APP_ID',
      // Public API key: it is safe to commit it
      apiKey: 'YOUR_SEARCH_API_KEY',
      indexName: 'physical-ai-book',
      // Optional: see doc section below
      contextualSearch: true,
      // Optional: path for search page that enabled by default (`false` to disable it)
      searchPagePath: 'search',
    },
  ],
],
```

### ℹ️ Styling ℹ️

The search component uses custom CSS for accessibility enhancements located in `src/css/custom.css` under the "Accessible Search Styles" section.

## ℹ️ For Users ℹ️

### 💡 Searching Tips 💡
- Use specific keywords for better results
- Search queries are not case-sensitive
- Use quotes for exact phrase matching
- Results are displayed in order of relevance

### ♿ Accessibility Settings ♿
- The search automatically adapts to system high contrast settings
- Animations can be disabled at the system level for reduced motion support
- All functionality is available via keyboard alone

## 🔧 Troubleshooting 🔧

### ℹ️ Search not working ℹ️
- Check that you have a stable internet connection
- Clear your browser cache and try again
- Ensure JavaScript is enabled in your browser

### ♿ Accessibility issues ♿
- If you encounter any accessibility issues, please report them via our GitHub repository
- Try using a different browser if compatibility issues arise