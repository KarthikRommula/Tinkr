import React, { useState, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';

function PostCard({ post, currentUser }) {
  const navigate = useNavigate();
  const [liked, setLiked] = useState(false);
  const [showComments, setShowComments] = useState(false);
  const [commentText, setCommentText] = useState('');
  const [showOptionsMenu, setShowOptionsMenu] = useState(false);
  const [showShareOptions, setShowShareOptions] = useState(false);
  const [isSaved, setIsSaved] = useState(false);
  const [comments, setComments] = useState([
    // Dummy comments for UI demonstration
    { id: 1, username: 'user123', text: 'Great photo!', timestamp: new Date(Date.now() - 3600000).toISOString() },
    { id: 2, username: 'photoLover', text: 'The lighting is perfect!', timestamp: new Date(Date.now() - 7200000).toISOString() }
  ]);
  
  // Get current user ID from localStorage or props
  const userId = currentUser || localStorage.getItem('userId') || 'user123';
  const isOwnPost = post.user_id === userId;
  
  // Refs for click outside detection
  const optionsMenuRef = useRef(null);
  const shareMenuRef = useRef(null);
  
  // Format date
  const formatDate = (dateString) => {
    const date = new Date(dateString);
    const now = new Date();
    const diffMs = now - date;
    const diffSecs = Math.floor(diffMs / 1000);
    const diffMins = Math.floor(diffSecs / 60);
    const diffHours = Math.floor(diffMins / 60);
    const diffDays = Math.floor(diffHours / 24);
    
    if (diffSecs < 60) {
      return 'just now';
    } else if (diffMins < 60) {
      return `${diffMins}m ago`;
    } else if (diffHours < 24) {
      return `${diffHours}h ago`;
    } else if (diffDays < 7) {
      return `${diffDays}d ago`;
    } else {
      return date.toLocaleDateString('en-US', {
        month: 'short',
        day: 'numeric'
      });
    }
  };
  
  const handleLike = () => {
    setLiked(!liked);
  };
  
  const handleComment = (e) => {
    e.preventDefault();
    if (commentText.trim()) {
      const newComment = {
        id: comments.length + 1,
        username: post.user_id, // In a real app, this would be the current user
        text: commentText,
        timestamp: new Date().toISOString()
      };
      setComments([...comments, newComment]);
      setCommentText('');
    }
  };

  // Animation for like button
  const [animateLike, setAnimateLike] = useState(false);

  useEffect(() => {
    if (liked) {
      setAnimateLike(true);
      const timer = setTimeout(() => setAnimateLike(false), 1000);
      return () => clearTimeout(timer);
    }
  }, [liked]);

  // Close menus when clicking outside
  useEffect(() => {
    const handleClickOutside = (event) => {
      if (optionsMenuRef.current && !optionsMenuRef.current.contains(event.target)) {
        setShowOptionsMenu(false);
      }
      if (shareMenuRef.current && !shareMenuRef.current.contains(event.target)) {
        setShowShareOptions(false);
      }
    };
    
    document.addEventListener('mousedown', handleClickOutside);
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, []);

  // Handle post actions
  const handleEditPost = () => {
    setShowOptionsMenu(false);
    // Navigate to edit page or open edit modal
    console.log('Edit post:', post.id);
    // Implement your edit functionality here
    // navigate(`/edit-post/${post.id}`);
  };

  const handleDeletePost = () => {
    setShowOptionsMenu(false);
    if (window.confirm('Are you sure you want to delete this post?')) {
      console.log('Delete post:', post.id);
      // Implement your delete functionality here
      // Example: axios.delete(`http://localhost:8000/posts/${post.id}`)
      //   .then(() => { /* handle success */ })
      //   .catch(error => { /* handle error */ });
    }
  };

  const handleViewProfile = () => {
    setShowOptionsMenu(false);
    navigate(`/profile/${post.user_id}`);
  };

  const handleReport = () => {
    setShowOptionsMenu(false);
    console.log('Report post:', post.id);
    // Implement your report functionality here
    alert('Post reported. Thank you for helping keep Tinkr safe.');
  };
  
  const handleSavePost = () => {
    setShowOptionsMenu(false);
    setIsSaved(!isSaved);
    console.log(`${isSaved ? 'Unsave' : 'Save'} post:`, post.id);
    // Implement your save post functionality here
    if (!isSaved) {
      // Show a subtle toast notification instead of an alert
      const toast = document.createElement('div');
      toast.className = 'fixed bottom-4 right-4 bg-black bg-opacity-80 text-white px-4 py-2 rounded-lg shadow-lg z-50 animate-fade-in-up';
      toast.textContent = 'Post saved to your collection';
      document.body.appendChild(toast);
      setTimeout(() => {
        toast.classList.add('animate-fade-out');
        setTimeout(() => document.body.removeChild(toast), 500);
      }, 2000);
    }
  };
  
  // Share functionality
  const handleShare = () => {
    setShowShareOptions(!showShareOptions);
  };
  
  const shareToSocial = (platform) => {
    setShowShareOptions(false);
    const postUrl = `http://localhost:3000/post/${post.id}`;
    let shareUrl = '';
    
    switch (platform) {
      case 'facebook':
        shareUrl = `https://www.facebook.com/sharer/sharer.php?u=${encodeURIComponent(postUrl)}`;
        break;
      case 'twitter':
        shareUrl = `https://twitter.com/intent/tweet?url=${encodeURIComponent(postUrl)}&text=${encodeURIComponent('Check out this post on Tinkr!')}`;
        break;
      case 'whatsapp':
        shareUrl = `https://api.whatsapp.com/send?text=${encodeURIComponent('Check out this post on Tinkr! ' + postUrl)}`;
        break;
      case 'copy':
        navigator.clipboard.writeText(postUrl)
          .then(() => {
            // Show a subtle toast notification
            const toast = document.createElement('div');
            toast.className = 'fixed bottom-4 right-4 bg-black bg-opacity-80 text-white px-4 py-2 rounded-lg shadow-lg z-50 animate-fade-in-up';
            toast.textContent = 'Link copied to clipboard';
            document.body.appendChild(toast);
            setTimeout(() => {
              toast.classList.add('animate-fade-out');
              setTimeout(() => document.body.removeChild(toast), 500);
            }, 2000);
          })
          .catch(err => console.error('Failed to copy: ', err));
        return;
      default:
        return;
    }
    
    window.open(shareUrl, '_blank', 'width=600,height=400');
  };

  return (
    <div className="bg-white dark:bg-gray-800 border dark:border-gray-700 rounded-xl overflow-hidden shadow-lg mb-8 transition-all duration-300 hover:shadow-xl transform hover:-translate-y-1">
      {/* Post header with user info */}
      <div className="p-4 border-b dark:border-gray-700">
        <div className="flex items-center justify-between">
          <div className="flex items-center">
            <div className="h-12 w-12 rounded-full bg-gradient-to-r from-pink-500 to-purple-600 flex items-center justify-center text-white font-bold shadow-md">
              {post.user_id.charAt(0).toUpperCase()}
            </div>
            <div className="ml-3">
              <p className="font-semibold text-gray-800 dark:text-gray-200">{post.user_id}</p>
              <p className="text-gray-500 dark:text-gray-400 text-xs">{formatDate(post.created_at)}</p>
            </div>
          </div>
          <div className="relative" ref={optionsMenuRef}>
            <button 
              onClick={() => setShowOptionsMenu(!showOptionsMenu)}
              className="text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-200 transition-colors duration-200 rounded-full p-1 hover:bg-gray-100 dark:hover:bg-gray-700"
              aria-label="Post options"
            >
              <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
                <path d="M10 6a2 2 0 110-4 2 2 0 010 4zM10 12a2 2 0 110-4 2 2 0 010 4zM10 18a2 2 0 110-4 2 2 0 010 4z" />
              </svg>
            </button>
            
            {/* Options dropdown menu */}
            {showOptionsMenu && (
              <div className="absolute right-0 mt-1 w-48 bg-white dark:bg-gray-800 rounded-md shadow-lg py-1 z-10 border border-gray-200 dark:border-gray-700 transform origin-top-right transition-all duration-200">
                {isOwnPost && (
                  <>
                    <button
                      onClick={handleEditPost}
                      className="w-full text-left px-4 py-2 text-sm text-gray-700 dark:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-700 flex items-center"
                    >
                      <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z" />
                      </svg>
                      Edit Post
                    </button>
                    <button
                      onClick={handleDeletePost}
                      className="w-full text-left px-4 py-2 text-sm text-red-600 dark:text-red-400 hover:bg-gray-100 dark:hover:bg-gray-700 flex items-center"
                    >
                      <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                      </svg>
                      Delete Post
                    </button>
                    <div className="border-t border-gray-200 dark:border-gray-700 my-1"></div>
                  </>
                )}
                
                <button
                  onClick={handleViewProfile}
                  className="w-full text-left px-4 py-2 text-sm text-gray-700 dark:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-700 flex items-center"
                >
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
                  </svg>
                  View Profile
                </button>
                
                <button
                  onClick={handleSavePost}
                  className="w-full text-left px-4 py-2 text-sm text-gray-700 dark:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-700 flex items-center"
                >
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 mr-2" fill={isSaved ? "currentColor" : "none"} viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={isSaved ? 0 : 2} d="M5 5a2 2 0 012-2h10a2 2 0 012 2v16l-7-3.5L5 21V5z" />
                  </svg>
                  {isSaved ? 'Unsave Post' : 'Save Post'}
                </button>
                
                {!isOwnPost && (
                  <button
                    onClick={handleReport}
                    className="w-full text-left px-4 py-2 text-sm text-gray-700 dark:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-700 flex items-center"
                  >
                    <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                    </svg>
                    Report Post
                  </button>
                )}
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Post image with enhanced container */}
      <div className="relative overflow-hidden group">
        <img
          src={`http://localhost:8000/uploads/${post.image_url.split('/').pop()}`}
          alt={post.caption || "User post"}
          className="w-full h-auto object-cover max-h-[600px] transition-transform duration-700 group-hover:scale-105"
          loading="lazy"
        />
        {/* Double-tap overlay for likes */}
        <div
          className="absolute inset-0 cursor-pointer flex items-center justify-center"
          onDoubleClick={handleLike}
        >
          {animateLike && (
            <div className="animate-ping-once">
              <svg xmlns="http://www.w3.org/2000/svg" className="h-24 w-24 text-white drop-shadow-lg" fill="currentColor" viewBox="0 0 24 24">
                <path d="M4.318 6.318a4.5 4.5 0 000 6.364L12 20.364l7.682-7.682a4.5 4.5 0 00-6.364-6.364L12 7.636l-1.318-1.318a4.5 4.5 0 00-6.364 0z" />
              </svg>
            </div>
          )}
        </div>
      </div>

      {/* Action buttons with enhanced styling */}
      <div className="p-4 flex items-center space-x-6 border-b dark:border-gray-700">
        <button
          onClick={handleLike}
          className={`flex items-center space-x-2 transition-colors duration-200 ${liked ? 'text-red-500' : 'text-gray-500 dark:text-gray-400 hover:text-red-500 dark:hover:text-red-400'}`}
        >
          <svg xmlns="http://www.w3.org/2000/svg"
            className={`h-7 w-7 ${animateLike ? 'animate-heartbeat' : ''}`}
            fill={liked ? "currentColor" : "none"}
            viewBox="0 0 24 24"
            stroke="currentColor"
          >
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={liked ? 0 : 2} d="M4.318 6.318a4.5 4.5 0 000 6.364L12 20.364l7.682-7.682a4.5 4.5 0 00-6.364-6.364L12 7.636l-1.318-1.318a4.5 4.5 0 00-6.364 0z" />
          </svg>
          <span className="font-medium">{liked ? 'Liked' : 'Like'}</span>
        </button>

        <button
          onClick={() => setShowComments(!showComments)}
          className="flex items-center space-x-2 text-gray-500 dark:text-gray-400 hover:text-blue-500 dark:hover:text-blue-400 transition-colors duration-200"
        >
          <svg xmlns="http://www.w3.org/2000/svg" className="h-7 w-7" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
          </svg>
          <span className="font-medium">Comment{comments.length > 0 ? ` (${comments.length})` : ''}</span>
        </button>

        <div className="relative ml-auto" ref={shareMenuRef}>
          <button 
            onClick={handleShare}
            className="flex items-center space-x-2 text-gray-500 dark:text-gray-400 hover:text-green-500 dark:hover:text-green-400 transition-colors duration-200"
          >
            <svg xmlns="http://www.w3.org/2000/svg" className="h-7 w-7" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8.684 13.342C8.886 12.938 9 12.482 9 12c0-.482-.114-.938-.316-1.342m0 2.684a3 3 0 110-2.684m0 2.684l6.632 3.316m-6.632-6l6.632-3.316m0 0a3 3 0 105.367-2.684 3 3 0 00-5.367 2.684zm0 9.316a3 3 0 105.368 2.684 3 3 0 00-5.368-2.684z" />
            </svg>
            <span className="font-medium">Share</span>
          </button>
          
          {/* Share options dropdown */}
          {showShareOptions && (
            <div className="absolute right-0 mt-1 w-48 bg-white dark:bg-gray-800 rounded-md shadow-lg py-1 z-10 border border-gray-200 dark:border-gray-700 transform origin-top-right transition-all duration-200">
              <button
                onClick={() => shareToSocial('facebook')}
                className="w-full text-left px-4 py-2 text-sm text-gray-700 dark:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-700 flex items-center"
              >
                <svg className="h-4 w-4 mr-2 text-blue-600" fill="currentColor" viewBox="0 0 24 24">
                  <path d="M22 12c0-5.523-4.477-10-10-10S2 6.477 2 12c0 4.991 3.657 9.128 8.438 9.878v-6.987h-2.54V12h2.54V9.797c0-2.506 1.492-3.89 3.777-3.89 1.094 0 2.238.195 2.238.195v2.46h-1.26c-1.243 0-1.63.771-1.63 1.562V12h2.773l-.443 2.89h-2.33v6.988C18.343 21.128 22 16.991 22 12z" />
                </svg>
                Facebook
              </button>
              <button
                onClick={() => shareToSocial('twitter')}
                className="w-full text-left px-4 py-2 text-sm text-gray-700 dark:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-700 flex items-center"
              >
                <svg className="h-4 w-4 mr-2 text-blue-400" fill="currentColor" viewBox="0 0 24 24">
                  <path d="M23.953 4.57a10 10 0 01-2.825.775 4.958 4.958 0 002.163-2.723 10.054 10.054 0 01-3.127 1.184 4.92 4.92 0 00-8.384 4.482C7.69 8.095 4.067 6.13 1.64 3.162a4.822 4.822 0 00-.666 2.475c0 1.71.87 3.213 2.188 4.096a4.904 4.904 0 01-2.228-.616v.06a4.923 4.923 0 003.946 4.827 4.996 4.996 0 01-2.212.085 4.937 4.937 0 004.604 3.417 9.868 9.868 0 01-6.102 2.105c-.39 0-.779-.023-1.17-.067a13.995 13.995 0 007.557 2.209c9.054 0 13.999-7.496 13.999-13.986 0-.209 0-.42-.015-.63a9.936 9.936 0 002.46-2.548l-.047-.02z" />
                </svg>
                Twitter
              </button>
              <button
                onClick={() => shareToSocial('whatsapp')}
                className="w-full text-left px-4 py-2 text-sm text-gray-700 dark:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-700 flex items-center"
              >
                <svg className="h-4 w-4 mr-2 text-green-500" fill="currentColor" viewBox="0 0 24 24">
                  <path d="M17.472 14.382c-.297-.149-1.758-.867-2.03-.967-.273-.099-.471-.148-.67.15-.197.297-.767.966-.94 1.164-.173.199-.347.223-.644.075-.297-.15-1.255-.463-2.39-1.475-.883-.788-1.48-1.761-1.653-2.059-.173-.297-.018-.458.13-.606.134-.133.298-.347.446-.52.149-.174.198-.298.298-.497.099-.198.05-.371-.025-.52-.075-.149-.669-1.612-.916-2.207-.242-.579-.487-.5-.669-.51-.173-.008-.371-.01-.57-.01-.198 0-.52.074-.792.372-.272.297-1.04 1.016-1.04 2.479 0 1.462 1.065 2.875 1.213 3.074.149.198 2.096 3.2 5.077 4.487.709.306 1.262.489 1.694.625.712.227 1.36.195 1.871.118.571-.085 1.758-.719 2.006-1.413.248-.694.248-1.289.173-1.413-.074-.124-.272-.198-.57-.347m-5.421 7.403h-.004a9.87 9.87 0 01-5.031-1.378l-.361-.214-3.741.982.998-3.648-.235-.374a9.86 9.86 0 01-1.51-5.26c.001-5.45 4.436-9.884 9.888-9.884 2.64 0 5.122 1.03 6.988 2.898a9.825 9.825 0 012.893 6.994c-.003 5.45-4.437 9.884-9.885 9.884m8.413-18.297A11.815 11.815 0 0012.05 0C5.495 0 .16 5.335.157 11.892c0 2.096.547 4.142 1.588 5.945L.057 24l6.305-1.654a11.882 11.882 0 005.683 1.448h.005c6.554 0 11.89-5.335 11.893-11.893a11.821 11.821 0 00-3.48-8.413z" />
                </svg>
                WhatsApp
              </button>
              <div className="border-t border-gray-200 dark:border-gray-700 my-1"></div>
              <button
                onClick={() => shareToSocial('copy')}
                className="w-full text-left px-4 py-2 text-sm text-gray-700 dark:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-700 flex items-center"
              >
                <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 5H6a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2v-1M8 5a2 2 0 002 2h2a2 2 0 002-2M8 5a2 2 0 012-2h2a2 2 0 012 2m0 0h2a2 2 0 012 2v3m2 4H10m0 0l3-3m-3 3l3 3" />
                </svg>
                Copy Link
              </button>
            </div>
          )}
        </div>
      </div>

      {/* Likes count with enhanced styling */}
      <div className="px-4 pt-3 pb-1">
        <p className="text-sm font-semibold text-gray-800 dark:text-gray-200">
          {liked ? '1 like' : '0 likes'}
        </p>
      </div>

      {/* Caption with enhanced styling */}
      {post.caption && (
        <div className="px-4 pb-3">
          <p className="text-gray-800 dark:text-gray-200">
            <span className="font-semibold mr-2">{post.user_id}</span>
            <span className="text-gray-700 dark:text-gray-300">{post.caption}</span>
          </p>
        </div>
      )}

      {/* Comments section with enhanced styling */}
      {showComments && (
        <div className="border-t dark:border-gray-700 bg-gray-50 dark:bg-gray-900 rounded-b-xl">
          <div className="max-h-48 overflow-y-auto px-4 py-3 scrollbar-thin scrollbar-thumb-gray-300 dark:scrollbar-thumb-gray-600">
            {comments.length > 0 ? (
              comments.map(comment => (
                <div key={comment.id} className="mb-3 last:mb-0">
                  <div className="flex items-start">
                    <div className="flex-shrink-0 mr-2">
                      <div className="h-8 w-8 rounded-full bg-gradient-to-r from-blue-400 to-indigo-500 flex items-center justify-center text-white text-xs font-bold shadow-sm">
                        {comment.username.charAt(0).toUpperCase()}
                      </div>
                    </div>
                    <div className="flex-1">
                      <div className="bg-white dark:bg-gray-800 rounded-2xl px-4 py-2 shadow-sm">
                        <p className="text-sm text-gray-800 dark:text-gray-200">
                          <span className="font-semibold mr-2">{comment.username}</span>
                          {comment.text}
                        </p>
                      </div>
                      <p className="text-xs text-gray-500 dark:text-gray-400 mt-1 ml-2">{formatDate(comment.timestamp)}</p>
                    </div>
                  </div>
                </div>
              ))
            ) : (
              <p className="text-center text-gray-500 dark:text-gray-400 py-3 text-sm">No comments yet. Be the first to comment!</p>
            )}
          </div>

          {/* Add comment form with enhanced styling */}
          <form onSubmit={handleComment} className="flex items-center px-4 py-3 border-t dark:border-gray-700">
            <input
              type="text"
              placeholder="Add a comment..."
              className="flex-1 bg-transparent border-none focus:outline-none focus:ring-0 text-gray-700 dark:text-gray-300 placeholder-gray-500 dark:placeholder-gray-400 text-sm"
              value={commentText}
              onChange={(e) => setCommentText(e.target.value)}
            />
            <button
              type="submit"
              disabled={!commentText.trim()}
              className={`text-blue-500 dark:text-blue-400 font-semibold text-sm px-2 py-1 rounded-full transition-colors duration-200 ${!commentText.trim() ? 'opacity-50 cursor-not-allowed' : 'hover:bg-blue-50 dark:hover:bg-blue-900/30'}`}
            >
              Post
            </button>
          </form>
        </div>
      )}
    </div>
  );
}

// Add these styles to your global CSS or tailwind.config.js
// @keyframes heartbeat {
//   0% { transform: scale(1); }
//   25% { transform: scale(1.2); }
//   50% { transform: scale(1); }
//   75% { transform: scale(1.2); }
//   100% { transform: scale(1); }
// }
// @keyframes ping-once {
//   0% { transform: scale(0.5); opacity: 0; }
//   50% { transform: scale(1.5); opacity: 0.5; }
//   100% { transform: scale(2); opacity: 0; }
// }
// .animate-heartbeat { animation: heartbeat 0.5s ease-in-out; }
// .animate-ping-once { animation: ping-once 1s cubic-bezier(0, 0, 0.2, 1) forwards; }

export default PostCard;