import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { AuthContext } from './AuthContextUtils';

export const AuthProvider = ({ children }) => {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);
  
  useEffect(() => {
    // Check if user is logged in
    const token = localStorage.getItem('token');
    if (token) {
      axios.get('http://localhost:8000/users/me', {
        headers: {
          Authorization: `Bearer ${token}`
        }
      })
      .then(response => {
        setUser(response.data);
      })
      .catch(error => {
        console.error('Error fetching user:', error);
        localStorage.removeItem('token');
      })
      .finally(() => {
        setLoading(false);
      });
    } else {
      setLoading(false);
    }
  }, []);
  
  const login = async (username, password) => {
    try {
      const formData = new FormData();
      formData.append('username', username);
      formData.append('password', password);
      
      const response = await axios.post('http://localhost:8000/token', formData);
      const { access_token } = response.data;
      
      localStorage.setItem('token', access_token);
      
      // Get user data
      const userResponse = await axios.get('http://localhost:8000/users/me', {
        headers: {
          Authorization: `Bearer ${access_token}`
        }
      });
      
      setUser(userResponse.data);
      return true;
    } catch (error) {
      console.error('Login error:', error);
      return false;
    }
  };
  
  const register = async (userData) => {
    try {
      await axios.post('http://localhost:8000/users/', userData);
      return { success: true };
    } catch (error) {
      console.error('Registration error:', error);
      let errorMessage = 'Registration failed. Please try again.';
      
      // Handle specific error responses from the backend
      if (error.response) {
        // The request was made and the server responded with a status code
        // that falls out of the range of 2xx
        if (error.response.status === 400) {
          if (error.response.data.detail) {
            errorMessage = error.response.data.detail;
          } else if (typeof error.response.data === 'object') {
            // Extract field-specific errors
            const fieldErrors = [];
            for (const [field, msgs] of Object.entries(error.response.data)) {
              if (Array.isArray(msgs)) {
                fieldErrors.push(`${field}: ${msgs.join(', ')}`);
              } else {
                fieldErrors.push(`${field}: ${msgs}`);
              }
            }
            if (fieldErrors.length > 0) {
              errorMessage = fieldErrors.join('\n');
            }
          }
        } else if (error.response.status === 409) {
          errorMessage = 'Username or email already exists.';
        } else if (error.response.status === 422) {
          errorMessage = 'Invalid input data. Please check your information.';
        }
      } else if (error.request) {
        // The request was made but no response was received
        errorMessage = 'No response from server. Please check your connection.';
      }
      
      return { success: false, message: errorMessage };
    }
  };
  
  const logout = () => {
    localStorage.removeItem('token');
    setUser(null);
  };
  
  const value = {
    user,
    loading,
    login,
    register,
    logout
  };
  
  return (
    <AuthContext.Provider value={value}>
      {!loading && children}
    </AuthContext.Provider>
  );
};
