/** @type {import('tailwindcss').Config} */
const v = (name) => `rgb(var(--${name}) / <alpha-value>)`;
export default {
  darkMode: 'class',
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        bg: v('bg'), surface: v('surface'), raised: v('raised'), line: v('line'),
        ink: v('ink'), muted: v('muted'),
        hivis: v('hivis'), hazard: v('hazard'),
        ok: v('ok'), danger: v('danger'), unsure: v('unsure'),
      },
      fontFamily: {
        display: ['"Space Grotesk Variable"', 'system-ui', 'sans-serif'],
        sans: ['"Inter Variable"', 'system-ui', 'sans-serif'],
        mono: ['"JetBrains Mono Variable"', 'ui-monospace', 'monospace'],
      },
      borderRadius: { DEFAULT: '6px', lg: '10px', xl: '14px' },
      keyframes: {
        hazardPulse: { '0%,100%': { opacity: '1' }, '50%': { opacity: '.35' } },
        stripes: { to: { backgroundPosition: '56px 0' } },
      },
      animation: {
        hazard: 'hazardPulse 1s ease-in-out infinite',
        stripes: 'stripes 1.2s linear infinite',
      },
    },
  },
  plugins: [],
};
