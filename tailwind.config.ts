import type { Config } from "tailwindcss";

const config: Config = {
  darkMode: "class",
  content: [
    "./src/pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        background: "var(--background)",
        foreground: "var(--foreground)",
        radiology: {
          dark: '#070a13',
          card: '#0d1527',
          border: '#1e293b',
          tumor: '#ef4444',
          benign: '#10b981',
          caution: '#f59e0b',
          cyan: '#06b6d4'
        }
      },
      animation: {
        'scanner-sweep': 'scannerSweep 3s ease-in-out infinite',
        'pulse-danger': 'pulseDanger 1.2s ease-in-out infinite',
      },
      keyframes: {
        scannerSweep: {
          '0%, 100%': { top: '0%' },
          '50%': { top: '100%' },
        },
        pulseDanger: {
          '0%, 100%': { opacity: '0.4' },
          '50%': { opacity: '1' },
        }
      }
    },
  },
  plugins: [],
};
export default config;
