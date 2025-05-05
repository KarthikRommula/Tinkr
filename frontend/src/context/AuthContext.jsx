import React, { createContext, useState, useContext, useEffect } from 'react';
import axios from 'axios';

const AuthContext = createContext();

export const useAuth = () => useContext(AuthContext);

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
      return true;
    } catch (error) {
      console.error('Registration error:', error);
      return false;
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