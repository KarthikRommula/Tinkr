import React from 'react';

/**
 * Input component with various styles and states
 */
const Input = ({
  label,
  type = 'text',
  id,
  name,
  value,
  onChange,
  placeholder,
  error,
  helperText,
  fullWidth = false,
  disabled = false,
  required = false,
  className = '',
  ...props
}) => {
  // Base classes for all inputs
  const baseClasses = 'block rounded-lg transition-all focus:outline-none focus:ring-2 focus:ring-pink-500 focus:border-transparent';
  
  // Error classes
  const errorClasses = error 
    ? 'border-red-500 bg-red-50 text-red-900 placeholder-red-300 focus:ring-red-500' 
    : 'border-gray-300 bg-gray-50 focus:bg-white';
  
  // Width classes
  const widthClasses = fullWidth ? 'w-full' : '';
  
  // Disabled classes
  const disabledClasses = disabled ? 'opacity-50 cursor-not-allowed' : '';
  
  // Combine all classes
  const inputClasses = `${baseClasses} ${errorClasses} ${widthClasses} ${disabledClasses} py-2 px-4 ${className}`;
  
  return (
    <div className={`${fullWidth ? 'w-full' : ''} mb-4`}>
      {label && (
        <label 
          htmlFor={id} 
          className="block text-sm font-medium text-gray-700 mb-1"
        >
          {label}
          {required && <span className="text-red-500 ml-1">*</span>}
        </label>
      )}
      
      <input
        type={type}
        id={id}
        name={name}
        value={value}
        onChange={onChange}
        placeholder={placeholder}
        disabled={disabled}
        required={required}
        className={inputClasses}
        {...props}
      />
      
      {(error || helperText) && (
        <p className={`mt-1 text-sm ${error ? 'text-red-600' : 'text-gray-500'}`}>
          {error || helperText}
        </p>
      )}
    </div>
  );
};

export default Input;
