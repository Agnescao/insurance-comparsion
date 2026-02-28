/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      colors: {
        panel: '#f8fafc',
        ink: '#0f172a',
        accent: '#0f766e',
        warn: '#f59e0b',
      },
      fontFamily: {
        sans: ['"Noto Sans SC"', '"Source Han Sans SC"', 'ui-sans-serif', 'system-ui'],
      },
    },
  },
  plugins: [],
}
