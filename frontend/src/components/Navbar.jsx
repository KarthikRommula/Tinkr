import React, { useState, useEffect } from 'react';
import { Link, useNavigate, useLocation } from 'react-router-dom';
import { useAuth } from '../context/AuthContextUtils';

function Navbar() {
  const { user, logout } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();
  const [searchQuery, setSearchQuery] = useState('');
  const [isScrolled, setIsScrolled] = useState(false);
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);
  
  // Handle scroll effect for navbar
  useEffect(() => {
    const handleScroll = () => {
      if (window.scrollY > 10) {
        setIsScrolled(true);
      } else {
        setIsScrolled(false);
      }
    };
    
    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);
  
  const handleLogout = () => {
    logout();
    navigate('/login');
  };
  
  const handleSearch = (e) => {
    e.preventDefault();
    // Implement search functionality here
    
    // For now, just clear the search box
    setSearchQuery('');
  };
  
  const isActive = (path) => {
    return location.pathname === path;
  };
  
  return (
    <nav className={`sticky top-0 z-50 transition-all duration-300 ${isScrolled ? 'bg-white shadow-md' : 'bg-white/80 backdrop-blur-md'}`}>
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center h-16">
          {/* Logo */}
          <div className="flex-shrink-0">
            <Link to="/" className="flex items-center">
              <svg xmlns="http://www.w3.org/2000/svg" className="h-8 w-8 text-pink-500" viewBox="0 0 20 20" fill="currentColor">
                <path fillRule="evenodd" d="M4 5a2 2 0 00-2 2v8a2 2 0 002 2h12a2 2 0 002-2V7a2 2 0 00-2-2h-1.586a1 1 0 01-.707-.293l-1.121-1.121A2 2 0 0011.172 3H8.828a2 2 0 00-1.414.586L6.293 4.707A1 1 0 015.586 5H4zm6 9a3 3 0 100-6 3 3 0 000 6z" clipRule="evenodd" />
              </svg>
              <span className="font-bold text-xl ml-2 bg-gradient-to-r from-pink-500 to-purple-600 text-transparent bg-clip-text">Tinkr</span>
            </Link>
          </div>
          
          {/* Mobile menu button */}
          <div className="-mr-2 flex md:hidden">
            <button 
              onClick={() => setIsMobileMenuOpen(!isMobileMenuOpen)}
              className="inline-flex items-center justify-center p-2 rounded-md text-gray-400 hover:text-gray-500 hover:bg-gray-100 focus:outline-none focus:ring-2 focus:ring-inset focus:ring-pink-500"
            >
              <span className="sr-only">Open main menu</span>
              {isMobileMenuOpen ? (
                <svg className="block h-6 w-6" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor" aria-hidden="true">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M6 18L18 6M6 6l12 12" />
                </svg>
              ) : (
                <svg className="block h-6 w-6" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor" aria-hidden="true">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 6h16M4 12h16M4 18h16" />
                </svg>
              )}
            </button>
          </div>
          
          {/* Desktop menu */}
          <div className="hidden md:flex md:items-center md:space-x-4">
            {/* Search bar - only show when logged in */}
            {user && (
              <div className="relative max-w-md mx-4">
                <form onSubmit={handleSearch} className="relative">
                  <input
                    type="text"
                    placeholder="Search..."
                    className="w-full bg-gray-100 rounded-full py-2 px-4 pl-10 focus:outline-none focus:ring-2 focus:ring-pink-500 focus:bg-white transition-all"
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                  />
                  <div className="absolute left-3 top-2.5 text-gray-400">
                    <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                    </svg>
                  </div>
                </form>
              </div>
            )}
            
            {/* Navigation */}
            {user ? (
              <div className="flex items-center space-x-6">
                {/* Home */}
                <Link 
                  to="/" 
                  className={`flex items-center space-x-1 ${isActive('/') ? 'text-pink-500' : 'text-gray-600 hover:text-pink-500'} transition-colors`}
                >
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 12l2-2m0 0l7-7 7 7m-7-7v14" />
                  </svg>
                  <span className="font-medium text-sm">Home</span>
                </Link>
                
                {/* Upload */}
                <Link 
                  to="/upload" 
                  className={`flex items-center space-x-1 ${isActive('/upload') ? 'text-pink-500' : 'text-gray-600 hover:text-pink-500'} transition-colors`}
                >
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v3m0 0v3m0-3h3m-3 0H9m12 0a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                  <span className="font-medium text-sm">Upload</span>
                </Link>
                
                {/* Notifications */}
                <button className="flex items-center space-x-1 text-gray-600 hover:text-pink-500 transition-colors relative">
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 17h5l-1.405-1.405A2.032 2.032 0 0118 14.158V11a6.002 6.002 0 00-4-5.659V5a2 2 0 10-4 0v.341C7.67 6.165 6 8.388 6 11v3.159c0 .538-.214 1.055-.595 1.436L4 17h5m6 0v1a3 3 0 11-6 0v-1m6 0H9" />
                  </svg>
                  <span className="font-medium text-sm">Alerts</span>
                  <span className="absolute -top-1 -right-1 bg-red-500 text-white rounded-full w-4 h-4 text-xs flex items-center justify-center">3</span>
                </button>
                
                {/* User profile */}
                <div className="relative group">
                  <button className="flex items-center space-x-2 focus:outline-none">
                    <div className="h-8 w-8 rounded-full bg-gradient-to-r from-pink-500 to-purple-600 flex items-center justify-center text-white font-bold shadow-md">
                      {user.username ? user.username.charAt(0).toUpperCase() : '?'}
                    </div>
                    <span className="font-medium text-sm text-gray-700">{user.username}</span>
                    <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                    </svg>
                  </button>
                  
                  <div className="absolute right-0 mt-2 w-48 bg-white rounded-xl shadow-lg py-1 z-50 hidden group-hover:block transform transition-all opacity-0 group-hover:opacity-100 scale-95 group-hover:scale-100">
                    <div className="px-4 py-3 text-sm text-gray-700 border-b">
                      <div className="font-medium">{user.username}</div>
                      <div className="text-xs text-gray-500 truncate">{user.email}</div>
                    </div>
                    <Link to="/profile" className="block px-4 py-2 text-sm text-gray-700 hover:bg-gray-100 transition-colors">
                      Your Profile
                    </Link>
                    <Link to="/settings" className="block px-4 py-2 text-sm text-gray-700 hover:bg-gray-100 transition-colors">
                      Settings
                    </Link>
                    <button
                      onClick={handleLogout}
                      className="block w-full text-left px-4 py-2 text-sm text-gray-700 hover:bg-gray-100 transition-colors"
                    >
                      Sign out
                    </button>
                  </div>
                </div>
              </div>
            ) : (
              <div className="flex items-center space-x-4">
                <Link to="/login" className="text-gray-600 hover:text-pink-500 font-medium transition-colors">
                  Log in
                </Link>
                <Link to="/register" className="bg-gradient-to-r from-pink-500 to-purple-600 text-white px-4 py-2 rounded-md font-medium hover:from-pink-600 hover:to-purple-700 shadow-md hover:shadow-lg transition-all">
                  Sign up
                </Link>
              </div>
            )}
          </div>
        </div>
      </div>
      
      {/* Mobile menu, show/hide based on menu state */}
      <div className={`md:hidden transition-all duration-300 ease-in-out ${isMobileMenuOpen ? 'max-h-screen opacity-100' : 'max-h-0 opacity-0 overflow-hidden'}`}>
        <div className="px-2 pt-2 pb-3 space-y-1 sm:px-3 bg-white shadow-lg rounded-b-lg">
          {user ? (
            <>
              <div className="px-4 py-2">
                <form onSubmit={handleSearch} className="relative">
                  <input
                    type="text"
                    placeholder="Search..."
                    className="w-full bg-gray-100 rounded-full py-2 px-4 pl-10 focus:outline-none focus:ring-2 focus:ring-pink-500 focus:bg-white"
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                  />
                  <div className="absolute left-3 top-2.5 text-gray-400">
                    <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                    </svg>
                  </div>
                </form>
              </div>
              <Link 
                to="/" 
                className={`block px-3 py-2 rounded-md ${isActive('/') ? 'bg-pink-50 text-pink-500' : 'text-gray-700 hover:bg-gray-50'}`}
                onClick={() => setIsMobileMenuOpen(false)}
              >
                <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 12l2-2m0 0l7-7 7 7m-7-7v14" />
                </svg>
                <span className="font-medium text-sm">Home</span>
              </Link>

              {/* Upload */}
              <Link 
                to="/upload" 
                className={`flex items-center space-x-1 ${isActive('/upload') ? 'text-pink-500' : 'text-gray-600 hover:text-pink-500'} transition-colors`}
              >
                <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v3m0 0v3m0-3h3m-3 0H9m12 0a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
                <span className="font-medium text-sm">Upload</span>
              </Link>

              {/* Classify */}
              <Link 
                to="/classify" 
                className={`flex items-center space-x-1 ${isActive('/classify') ? 'text-pink-500' : 'text-gray-600 hover:text-pink-500'} transition-colors`}
              >
                <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v3m0 0v3m0-3h3m-3 0H9m12 0a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
                <span className="font-medium text-sm">Classify</span>
              </Link>
              <Link 
                to="/profile" 
                className={`block px-3 py-2 rounded-md ${isActive('/profile') ? 'bg-pink-50 text-pink-500' : 'text-gray-700 hover:bg-gray-50'}`}
                onClick={() => setIsMobileMenuOpen(false)}
              >
                Profile
              </Link>
              <Link 
                to="/settings" 
                className={`block px-3 py-2 rounded-md ${isActive('/settings') ? 'bg-pink-50 text-pink-500' : 'text-gray-700 hover:bg-gray-50'}`}
                onClick={() => setIsMobileMenuOpen(false)}
              >
                Settings
              </Link>
              <button
                onClick={() => {
                  handleLogout();
                  setIsMobileMenuOpen(false);
                }}
                className="block w-full text-left px-3 py-2 rounded-md text-gray-700 hover:bg-gray-50"
              >
                Sign out
              </button>
            </>
          ) : (
            <>
              <Link 
                to="/login" 
                className={`block px-3 py-2 rounded-md ${isActive('/login') ? 'bg-pink-50 text-pink-500' : 'text-gray-700 hover:bg-gray-50'}`}
                onClick={() => setIsMobileMenuOpen(false)}
              >
                Log in
              </Link>
              <Link 
                to="/register" 
                className={`block px-3 py-2 rounded-md ${isActive('/register') ? 'bg-pink-50 text-pink-500' : 'text-gray-700 hover:bg-gray-50'}`}
                onClick={() => setIsMobileMenuOpen(false)}
              >
                Sign up
              </Link>
            </>
          )}
        </div>
      </div>
    </nav>
  );
}

export default Navbar;