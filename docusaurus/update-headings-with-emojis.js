const fs = require('fs');
const path = require('path');

// Mapping of keywords to emojis
const emojiMap = {
  'introduction': '👋',
  'learning': '🎯',
  'objectives': '🎯',
  'overview': '📊',
  'summary': '📝',
  'conclusion': '🔚',
  'chapter': '📚',
  'table of contents': '📋',
  'physical ai': '🤖',
  'embodied intelligence': '🧠',
  'ai': '🤖',
  'robotics': '🤖',
  'robot': '🤖',
  'nervous system': '⚡',
  'digital twin': '🎮',
  'simulation': '🎮',
  'ai-brain': '🧠',
  'vision': '👁️',
  'language': '💬',
  'action': '⚡',
  'vslam': '👁️',
  'navigation': '🧭',
  'slam': '👁️',
  'perception': '👁️',
  'sensors': '📡',
  'lidar': '📡',
  'cameras': '📷',
  'imu': '⚖️',
  'fusion': '🔗',
  'modules': '🧩',
  'curriculum': '🎓',
  'applications': '🛠️',
  'foundations': '基石', // Foundation emoji
  'historical': '📜',
  'principles': '📐',
  'characteristics': '✅',
  'environments': '🌍',
  'affordances': '🤝',
  'computation': '⚙️',
  'manufacturing': '🏭',
  'automation': '⚙️',
  'healthcare': '💉',
  'assistive': '🆘',
  'exploration': '🌏',
  'discovery': '🔍',
  'theoretical': '📘',
  'theory': '🧮',
  'inference': '🧠',
  'processing': '🧠',
  'knowledge': '🤔',
  'questions': '💬',
  'discussion': '🗣️',
  'applications': '🚀',
  'focus': '🎯',
  'middleware': '⚙️',
  'control': '🎛️',
  'architecture': '🏗️',
  'components': '🧩',
  'systems': '⚙️',
  'integration': '🔗',
  'design': '🎨',
  'development': '🛠️',
  'implementation': '🔨',
  'deployment': '🚚',
  'testing': '🧪',
  'evaluation': '📊',
  'performance': '📈',
  'optimization': '⚙️',
  'troubleshooting': '🔧',
  'maintenance': '🔄',
  'security': '🔒',
  'privacy': '🔐',
  'ethics': '⚖️',
  'future': '🔮',
  'trends': '📊',
  'challenges': '⚠️',
  'solutions': '💡',
  'case studies': '📖',
  'examples': '💡',
  'exercises': '✍️',
  'activities': '🎯',
  'projects': '🏗️',
  'resources': '📚',
  'references': '🔗',
  'bibliography': '📚',
  'appendix': '📋',
  'glossary': '📖',
  'terminology': '🔤',
  'acronyms': '🔤',
  'faq': '❓',
  'troubleshooting': '🔧',
  'errors': '⚠️',
  'warnings': '⚠️',
  'notes': '📝',
  'tips': '💡',
  'best practices': '✅',
  'patterns': '🔄',
  'anti-patterns': '❌',
  'algorithms': '🔢',
  'data': '📊',
  'structures': '🏗️',
  'models': '🏗️',
  'frameworks': ' setFrame',
  'libraries': '📚',
  'tools': '🛠️',
  'techniques': ' 🔧',
  'methods': '🔧',
  'protocols': '📋',
  'standards': '📏',
  'specifications': '📋',
  'requirements': '📋',
  'constraints': '⚠️',
  'assumptions': '💭',
  'dependencies': '🔗',
  'compatibility': '✅',
  'scalability': '📈',
  'reliability': '✅',
  'availability': '✅',
  'maintainability': '🔄',
  'usability': '👍',
  'accessibility': '♿'
};

// Keywords that should have a default emoji
const defaultEmoji = 'ℹ️';

// Function to add emojis to headings in a markdown string
function addEmojisToHeadings(content) {
  // Split content by lines
  const lines = content.split('\n');
  
  // Process each line
  const updatedLines = lines.map(line => {
    // Check if the line is a heading (starts with #)
    if (line.trim().startsWith('#')) {
      // Extract the heading text (everything after the #s and space)
      const headingMatch = line.match(/^(#+)\s+(.*)/);
      if (headingMatch) {
        const hashes = headingMatch[1];
        const headingText = headingMatch[2].toLowerCase();
        
        // Try to find an appropriate emoji based on keywords in the heading
        let emoji = defaultEmoji;
        
        // Look for keywords in the mapping
        for (const [keyword, keywordEmoji] of Object.entries(emojiMap)) {
          if (headingText.includes(keyword.toLowerCase())) {
            emoji = keywordEmoji;
            break;
          }
        }
        
        // Special handling for chapter headings
        if (/^chapter\s+\d+/i.test(headingText)) {
          emoji = '📚';
        }
        
        // Return the heading with the emoji appended
        return `${hashes} ${emoji} ${headingMatch[2]} ${emoji}`;
      }
    }
    
    // Return the line unchanged if it's not a heading
    return line;
  });
  
  // Join the lines back together
  return updatedLines.join('\n');
}

// Function to process all markdown files in a directory recursively
function processDirectory(dirPath) {
  const items = fs.readdirSync(dirPath);
  
  for (const item of items) {
    const fullPath = path.join(dirPath, item);
    const stat = fs.statSync(fullPath);
    
    if (stat.isDirectory()) {
      // Recursively process subdirectories
      processDirectory(fullPath);
    } else if (item.endsWith('.md')) {
      // Process markdown files
      console.log(`Processing: ${fullPath}`);
      
      try {
        // Read the file content
        const content = fs.readFileSync(fullPath, 'utf8');
        
        // Add emojis to headings
        const updatedContent = addEmojisToHeadings(content);
        
        // Write the updated content back to the file
        fs.writeFileSync(fullPath, updatedContent);
        
        console.log(`✓ Updated: ${fullPath}`);
      } catch (error) {
        console.error(`✗ Error processing ${fullPath}:`, error.message);
      }
    }
  }
}

// Start processing from the docs directory
const docsDir = './docs';
if (fs.existsSync(docsDir)) {
  console.log('Starting emoji update process...');
  processDirectory(docsDir);
  console.log('Emoji update process completed!');
} else {
  console.error('Docs directory not found!');
}