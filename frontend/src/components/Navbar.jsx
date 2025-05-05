import React from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { useAuth } from '../context/Authcontext';

function Navbar() {
  const { user, logout } = useAuth();
  const navigate = useNavigate();
  
  const handleLogout = () => {
    logout();
    navigate('/login');
  };
  
  return (
    <nav className="bg-white shadow-lg">
      <div className="max-w-6xl mx-auto px-4">
        <div className="flex justify-between">
          <div className="flex space-x-7">
            <div>
              <Link to="/" className="flex items-center py-4">
                <span className="font-semibold text-gray-500 text-lg">SafeGram</span>
              </Link>
            </div>
          </div>
          
          {user ? (
            <div className="flex items-center space-x-3">
              <Link to="/upload" className="py-2 px-3 bg-blue-500 hover:bg-blue-600 text-white rounded-md shadow">
                Upload
              </Link>
              <button
                onClick={handleLogout}
                className="py-2 px-3 bg-gray-200 hover:bg-gray-300 text-gray-800 rounded-md shadow"
              >
                Logout
              </button>
              <span className="text-gray-700">
                {user.username}
              </span>
            </div>
          ) : (
            <div className="flex items-center space-x-3">
              <Link to="/login" className="py-2 px-3 text-gray-700 hover:text-gray-900">
                Login
              </Link>
              <Link to="/register" className="py-2 px-3 bg-blue-500 hover:bg-blue-600 text-white rounded-md shadow">
                Register
              </Link>
            </div>
          )}
        </div>
      </div>
    </nav>
  );
}

export default Navbar;