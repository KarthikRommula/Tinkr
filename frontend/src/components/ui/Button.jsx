import React from 'react';

/**
 * Button component with various variants and sizes
 */
const Button = ({ 
  children, 
  variant = 'primary', 
  size = 'md', 
  fullWidth = false,
  disabled = false,
  onClick,
  type = 'button',
  className = '',
  ...props 
}) => {
  // Base classes for all buttons
  const baseClasses = 'font-medium rounded-lg transition-all focus:outline-none focus:ring-2 focus:ring-offset-2';
  
  // Size classes
  const sizeClasses = {
    sm: 'py-1.5 px-3 text-sm',
    md: 'py-2 px-4 text-base',
    lg: 'py-3 px-6 text-lg',
  };
  
  // Variant classes
  const variantClasses = {
    primary: `bg-gradient-to-r from-pink-500 to-purple-600 hover:from-pink-600 hover:to-purple-700 text-white focus:ring-pink-500 ${disabled ? 'opacity-50 cursor-not-allowed' : ''}`,
    secondary: `bg-purple-100 text-purple-700 hover:bg-purple-200 focus:ring-purple-500 ${disabled ? 'opacity-50 cursor-not-allowed' : ''}`,
    outline: `border border-pink-500 text-pink-500 hover:bg-pink-50 focus:ring-pink-500 ${disabled ? 'opacity-50 cursor-not-allowed' : ''}`,
    text: `text-pink-500 hover:bg-pink-50 focus:ring-pink-500 ${disabled ? 'opacity-50 cursor-not-allowed' : ''}`,
    success: `bg-green-500 hover:bg-green-600 text-white focus:ring-green-500 ${disabled ? 'opacity-50 cursor-not-allowed' : ''}`,
    danger: `bg-red-500 hover:bg-red-600 text-white focus:ring-red-500 ${disabled ? 'opacity-50 cursor-not-allowed' : ''}`,
  };
  
  // Width classes
  const widthClasses = fullWidth ? 'w-full' : '';
  
  // Combine all classes
  const buttonClasses = `${baseClasses} ${sizeClasses[size]} ${variantClasses[variant]} ${widthClasses} ${className}`;
  
  return (
    <button
      type={type}
      className={buttonClasses}
      onClick={onClick}
      disabled={disabled}
      {...props}
    >
      {children}
    </button>
  );
};

export default Button;
