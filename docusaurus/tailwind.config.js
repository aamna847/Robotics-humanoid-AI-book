/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./src/**/*.{js,jsx,ts,tsx}",
    "./docs/**/*.{md,mdx}",
    "./blog/**/*.{md,mdx}",
    "./pages/**/*.{md,mdx,js,jsx,ts,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        'navy-black': '#0a0a12',
        'electric-cyan': '#00FFFF',
        'purple-accent': '#7F4FFF',
      },
      backdropBlur: {
        'xs': '2px',
        'sm': '4px',
        'md': '8px',
        'lg': '12px',
        'xl': '16px',
      },
      boxShadow: {
        'neon': '0 0 15px rgba(0, 255, 255, 0.5)',
        'neon-purple': '0 0 15px rgba(127, 79, 255, 0.5)',
        'glass': '0 8px 32px 0 rgba(31, 38, 135, 0.37)',
      },
      animation: {
        'pulse-cyan': 'pulse-cyan 2s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'pulse-purple': 'pulse-purple 2s cubic-bezier(0.4, 0, 0.6, 1) infinite',
      },
      keyframes: {
        'pulse-cyan': {
          '0%, 100%': { boxShadow: '0 0 0 0 rgba(0, 255, 255, 0.4)' },
          '50%': { boxShadow: '0 0 0 10px rgba(0, 255, 255, 0)' },
        },
        'pulse-purple': {
          '0%, 100%': { boxShadow: '0 0 0 0 rgba(127, 79, 255, 0.4)' },
          '50%': { boxShadow: '0 0 0 10px rgba(127, 79, 255, 0)' },
        }
      }
    },
  },
  plugins: [
    require('@tailwindcss/forms')({
      strategy: 'class',
    }),
    require('@tailwindcss/typography'),
    require('@tailwindcss/aspect-ratio'),
  ],
}