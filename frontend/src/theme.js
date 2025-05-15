// theme.js - Central theme configuration for the SafeGram application
const theme = {
  colors: {
    primary: {
      light: '#F9A8D4', // pink-300
      main: '#EC4899',  // pink-500
      dark: '#BE185D',  // pink-700
      gradient: 'linear-gradient(to right, #EC4899, #8B5CF6)' // pink to purple gradient
    },
    secondary: {
      light: '#C4B5FD', // violet-300
      main: '#8B5CF6',  // violet-500
      dark: '#6D28D9',  // violet-700
    },
    success: {
      light: '#A7F3D0', // green-200
      main: '#10B981',  // green-500
      dark: '#047857',  // green-700
    },
    error: {
      light: '#FECACA', // red-200
      main: '#EF4444',  // red-500
      dark: '#B91C1C',  // red-700
    },
    warning: {
      light: '#FDE68A', // yellow-200
      main: '#F59E0B',  // amber-500
      dark: '#B45309',  // amber-700
    },
    info: {
      light: '#BFDBFE', // blue-200
      main: '#3B82F6',  // blue-500
      dark: '#1D4ED8',  // blue-700
    },
    neutral: {
      white: '#FFFFFF',
      background: '#F9FAFB', // gray-50
      card: '#FFFFFF',
      divider: '#E5E7EB', // gray-200
      outline: '#D1D5DB', // gray-300
      input: '#F3F4F6', // gray-100
      inputFocus: '#FFFFFF',
      text: {
        primary: '#1F2937', // gray-800
        secondary: '#4B5563', // gray-600
        disabled: '#9CA3AF', // gray-400
      }
    }
  },
  shadows: {
    sm: '0 1px 2px 0 rgba(0, 0, 0, 0.05)',
    md: '0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)',
    lg: '0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05)',
    xl: '0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04)',
  },
  borderRadius: {
    sm: '0.125rem',
    md: '0.375rem',
    lg: '0.5rem',
    xl: '0.75rem',
    full: '9999px',
  },
  typography: {
    fontFamily: {
      sans: 'Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif',
      serif: 'ui-serif, Georgia, Cambria, "Times New Roman", Times, serif',
      mono: 'ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace',
    },
    fontWeight: {
      light: 300,
      normal: 400,
      medium: 500,
      semibold: 600,
      bold: 700,
    },
    fontSize: {
      xs: '0.75rem',
      sm: '0.875rem',
      base: '1rem',
      lg: '1.125rem',
      xl: '1.25rem',
      '2xl': '1.5rem',
      '3xl': '1.875rem',
      '4xl': '2.25rem',
      '5xl': '3rem',
    }
  },
  spacing: {
    0: '0',
    1: '0.25rem',
    2: '0.5rem',
    3: '0.75rem',
    4: '1rem',
    5: '1.25rem',
    6: '1.5rem',
    8: '2rem',
    10: '2.5rem',
    12: '3rem',
    16: '4rem',
    20: '5rem',
    24: '6rem',
    32: '8rem',
    40: '10rem',
    48: '12rem',
    56: '14rem',
    64: '16rem',
  },
  transitions: {
    default: 'all 0.3s ease',
    fast: 'all 0.15s ease',
    slow: 'all 0.5s ease',
  },
  zIndex: {
    dropdown: 1000,
    sticky: 1020,
    fixed: 1030,
    modal: 1040,
    tooltip: 1050,
  }
};

export default theme;
