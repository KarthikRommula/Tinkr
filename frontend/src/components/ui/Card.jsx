import React from 'react';

/**
 * Card component for containing content with various elevation levels
 */
const Card = ({ 
  children, 
  elevation = 'md',
  className = '',
  ...props 
}) => {
  // Base classes for all cards
  const baseClasses = 'bg-white rounded-xl overflow-hidden';
  
  // Elevation classes
  const elevationClasses = {
    none: '',
    sm: 'shadow-sm',
    md: 'shadow-md',
    lg: 'shadow-lg',
    xl: 'shadow-xl',
  };
  
  // Combine all classes
  const cardClasses = `${baseClasses} ${elevationClasses[elevation]} ${className}`;
  
  return (
    <div
      className={cardClasses}
      {...props}
    >
      {children}
    </div>
  );
};

// Card subcomponents
Card.Header = ({ children, className = '', ...props }) => (
  <div className={`px-6 py-4 border-b border-gray-100 ${className}`} {...props}>
    {children}
  </div>
);

Card.Body = ({ children, className = '', ...props }) => (
  <div className={`p-6 ${className}`} {...props}>
    {children}
  </div>
);

Card.Footer = ({ children, className = '', ...props }) => (
  <div className={`px-6 py-4 bg-gray-50 border-t border-gray-100 ${className}`} {...props}>
    {children}
  </div>
);

export default Card;
