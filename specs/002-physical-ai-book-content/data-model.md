# Data Model: Accessible Search Bar for Docusaurus

## Overview
This document defines the data models needed for implementing an accessible search bar in the Docusaurus-based documentation site for the Physical AI & Humanoid Robotics textbook.

## Search Input Model

### SearchInput
Represents the search input field in the UI

**Fields:**
- id: string (unique identifier)
- query: string (the user's search query)
- placeholder: string (placeholder text in the search field)
- label: string (accessible label for screen readers)
- isFocused: boolean (whether the search field has focus)
- isExpanded: boolean (whether search results panel is open)
- searchResults: SearchResult[] (list of matching results)
- lastUpdated: timestamp (when the search was last executed)

## Search Result Model

### SearchResult
Represents an individual search result

**Fields:**
- id: string (unique identifier for the result)
- title: string (title of the document/page containing the result)
- contentSnippet: string (excerpts from the document that match the query)
- url: string (URL to the relevant page/section)
- docSection: string (name of the section in which the result appears)
- docPart: string (name of the part/parent section)
- relevanceScore: number (score indicating how relevant the result is to the query)
- resultType: SearchResultType (enum: 'heading', 'section', 'content', 'module')

### SearchResultType
Enumeration of possible result types

**Values:**
- heading (results from page headings)
- section (results from document sections)
- content (results from within content body)
- module (results from course modules)

## Search Index Model

### SearchIndex
Represents the searchable content index

**Fields:**
- id: string (unique identifier)
- indexedPages: IndexedPage[] (list of all indexed pages)
- lastIndexed: timestamp (when the index was last updated)
- totalDocuments: number (count of indexed documents)
- indexVersion: string (version of the search index)

### IndexedPage
Represents a single indexed page

**Fields:**
- pageId: string (unique identifier for the page)
- title: string (title of the page)
- url: string (URL of the page)
- headings: IndexedHeading[] (list of headings in the page)
- content: string (full text content of the page, processed)
- lastUpdated: timestamp (when the page content was last indexed)
- part: string (course part the page belongs to, e.g. "Part 1: Foundations")
- chapter: string (chapter the page belongs to, e.g. "Chapter 1: Introduction")

### IndexedHeading
Represents a heading that is indexed for search

**Fields:**
- headingId: string (unique identifier for the heading)
- text: string (text of the heading)
- level: number (heading level: 1-6)
- url: string (specific URL fragment to the heading)
- pageId: string (reference to the parent page)

## Accessibility State Model

### AccessibilityState
Represents the accessibility state of the search interface

**Fields:**
- keyboardNavigationActive: boolean (whether keyboard navigation is currently active)
- screenReaderMode: boolean (whether screen reader is detected)
- focusElement: string (the currently focused UI element)
- lastAnnouncement: string (last message announced to screen readers)
- announcementTimestamp: timestamp (when the last announcement was made)

## Search Configuration Model

### SearchConfig
Configures the search functionality

**Fields:**
- enabled: boolean (whether search is enabled)
- minQueryLength: number (minimum query length before search starts)
- debounceTime: number (time to wait after user stops typing before searching)
- maxResults: number (maximum number of results to display)
- searchIn: SearchScope[] (scopes to search in: 'title', 'content', 'headings')
- accessibilityFeatures: AccessibilityConfig (accessibility-specific settings)

### AccessibilityConfig
Configures accessibility-specific search features

**Fields:**
- ariaLabels: AriaLabels (object containing all ARIA labels)
- keyboardShortcuts: KeyboardShortcuts (object defining keyboard shortcuts)
- focusManagement: FocusManagementConfig (configuration for managing focus)
- screenReaderSupport: boolean (whether to optimize for screen readers)

### AriaLabels
Contains all ARIA labels for accessibility

**Fields:**
- searchInput: string (label for search input)
- searchButton: string (label for search button)
- searchResults: string (label for search results container)
- resultItem: string (template for labeling each result item)
- clearSearch: string (label for clear search button)

### KeyboardShortcuts
Defines keyboard shortcuts for search

**Fields:**
- focusSearch: string (key combination to focus search, e.g. 'Ctrl+K')
- selectResult: string (key to select a search result, e.g. 'Enter')
- navigateResultsUp: string (key to move up in results, e.g. 'ArrowUp')
- navigateResultsDown: string (key to move down in results, e.g. 'ArrowDown')
- closeResults: string (key to close results panel, e.g. 'Escape')