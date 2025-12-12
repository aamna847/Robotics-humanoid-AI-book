const path = require('path');

module.exports = {
  title: 'Physical AI & Humanoid Robotics',
  tagline: 'Bridging the gap between digital AI and the physical world',
  url: 'https://your-docusaurus-site.com',
  baseUrl: '/',
  onBrokenLinks: 'warn',
  onBrokenMarkdownLinks: 'warn',
  favicon: 'img/favicon.ico',
  organizationName: 'Humanoid Robotics',
  projectName: 'Physical AI Book',
  trailingSlash: false,

  presets: [
    [
      '@docusaurus/preset-classic',
      {
        docs: {
          sidebarPath: require.resolve('./sidebars.js'),
          editUrl: 'https://github.com/humanoid-robotics/physical-ai-book/edit/main/',
          showLastUpdateAuthor: false,
          showLastUpdateTime: false,
        },
        theme: {
          customCss: require.resolve('./src/css/custom.css'),
        },
        blog: {
          showReadingTime: true,
          // Please change this to your repo.
          editUrl:
            'https://github.com/facebook/docusaurus/edit/main/packages/create-docusaurus/templates/shared/',
        },
        sitemap: {
          changefreq: 'weekly',
          priority: 0.5,
          ignorePatterns: ['/tags/**'],
          filename: 'sitemap.xml',
        },
      },
    ],
  ],

  themeConfig: {
    navbar: {
      // Hide the default navbar by making it transparent and minimal
      style: 'dark',
      title: ' ',
      logo: {
        alt: 'Logo',
        src: 'img/logo.svg',  // Use the same logo but hide it in CSS
      },
      items: [], // Empty items array to ensure no navbar items appear
    },
    footer: {
      style: 'dark',
      links: [
        {
          title: 'Docs',
          items: [
            {
              label: 'Introduction',
              to: '/docs/introduction',
            },
            {
              label: 'Physical AI Foundations',
              to: '/docs/part-1-foundations/chapter-1-introduction-to-physical-ai',
            },
            {
              label: 'ROS 2 Fundamentals',
              to: '/docs/part-2-nervous-system/chapter-3-ros2-architecture-core-concepts',
            },
          ],
        },
        {
          title: 'Community',
          items: [
            {
              label: 'Stack Overflow',
              href: 'https://stackoverflow.com/questions/tagged/humanoid-robotics',
            },
            {
              label: 'Discord',
              href: 'https://discordapp.com/invite/humanoid-robotics',
            },
          ],
        },
        {
          title: 'More',
          items: [
            {
              label: 'GitHub',
              href: 'https://github.com/humanoid-robotics/physical-ai-book',
            },
          ],
        },
      ],
      copyright: `Copyright © ${new Date().getFullYear()} Humanoid Robotics. Built with ❤️ by Aamna Rana`,
    },
    prism: {
      theme: undefined,
      darkTheme: undefined,
    },
  },

  plugins: [
    // Tailwind CSS plugin
    async function myPlugin(context, options) {
      return {
        name: 'tailwind-plugin',
        configurePostCss(postcssOptions) {
          postcssOptions.plugins.push(require('tailwindcss'));
          postcssOptions.plugins.push(require('autoprefixer'));
          return postcssOptions;
        },
      };
    },
  ],
};