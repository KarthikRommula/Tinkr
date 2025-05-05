import React, { useState } from 'react';

function PostCard({ post }) {
  const [liked, setLiked] = useState(false);
  const [showComments, setShowComments] = useState(false);
  const [commentText, setCommentText] = useState('');
  const [comments, setComments] = useState([
    // Dummy comments for UI demonstration
    { id: 1, username: 'user123', text: 'Great photo!', timestamp: new Date(Date.now() - 3600000).toISOString() },
    { id: 2, username: 'photoLover', text: 'The lighting is perfect!', timestamp: new Date(Date.now() - 7200000).toISOString() }
  ]);
  
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
  
  return (
    <div className="bg-white border rounded-lg overflow-hidden shadow-md mb-6 transition-shadow duration-300 hover:shadow-lg">
      {/* Post header with user info */}
      <div className="p-3 border-b">
        <div className="flex items-center justify-between">
          <div className="flex items-center">
            <div className="h-10 w-10 rounded-full bg-gradient-to-r from-pink-500 to-purple-600 flex items-center justify-center text-white font-bold">
              {post.user_id.charAt(0).toUpperCase()}
            </div>
            <div className="ml-3">
              <p className="font-semibold text-gray-800">{post.user_id}</p>
              <p className="text-gray-500 text-xs">{formatDate(post.created_at)}</p>
            </div>
          </div>
          <button className="text-gray-500 hover:text-gray-700">
            <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
              <path d="M10 6a2 2 0 110-4 2 2 0 010 4zM10 12a2 2 0 110-4 2 2 0 010 4zM10 18a2 2 0 110-4 2 2 0 010 4z" />
            </svg>
          </button>
        </div>
      </div>
      
      {/* Post image */}
      <div className="relative">
        <img
          src={`http://localhost:8000/uploads/${post.image_url.split('/').pop()}`}
          alt={post.caption || "User post"}
          className="w-full h-auto object-cover max-h-[600px]"
          loading="lazy"
        />
      </div>
      
      {/* Action buttons */}
      <div className="p-3 flex items-center space-x-4 border-b">
        <button 
          onClick={handleLike}
          className={`flex items-center space-x-1 ${liked ? 'text-red-500' : 'text-gray-500 hover:text-red-500'}`}
        >
          <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" 
            fill={liked ? "currentColor" : "none"} 
            viewBox="0 0 24 24" 
            stroke="currentColor"
          >
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={liked ? 0 : 2} d="M4.318 6.318a4.5 4.5 0 000 6.364L12 20.364l7.682-7.682a4.5 4.5 0 00-6.364-6.364L12 7.636l-1.318-1.318a4.5 4.5 0 00-6.364 0z" />
          </svg>
          <span>{liked ? 'Liked' : 'Like'}</span>
        </button>
        
        <button 
          onClick={() => setShowComments(!showComments)}
          className="flex items-center space-x-1 text-gray-500 hover:text-blue-500"
        >
          <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
          </svg>
          <span>Comment{comments.length > 0 ? ` (${comments.length})` : ''}</span>
        </button>
        
        <button className="flex items-center space-x-1 text-gray-500 hover:text-green-500 ml-auto">
          <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8.684 13.342C8.886 12.938 9 12.482 9 12c0-.482-.114-.938-.316-1.342m0 2.684a3 3 0 110-2.684m0 2.684l6.632 3.316m-6.632-6l6.632-3.316m0 0a3 3 0 105.367-2.684 3 3 0 00-5.367 2.684zm0 9.316a3 3 0 105.368 2.684 3 3 0 00-5.368-2.684z" />
          </svg>
          <span>Share</span>
        </button>
      </div>
      
      {/* Likes count */}
      <div className="px-4 py-2">
        <p className="text-sm font-semibold">{liked ? '1 like' : '0 likes'}</p>
      </div>
      
      {/* Caption */}
      {post.caption && (
        <div className="px-4 pb-2">
          <p>
            <span className="font-semibold mr-2">{post.user_id}</span>
            {post.caption}
          </p>
        </div>
      )}
      
      {/* Comments section */}
      {showComments && (
        <div className="border-t">
          <div className="max-h-40 overflow-y-auto px-4 py-2">
            {comments.map(comment => (
              <div key={comment.id} className="mb-2">
                <div className="flex items-start">
                  <div className="flex-shrink-0 mr-2">
                    <div className="h-7 w-7 rounded-full bg-gray-200 flex items-center justify-center text-xs font-bold">
                      {comment.username.charAt(0).toUpperCase()}
                    </div>
                  </div>
                  <div className="flex-1 bg-gray-100 rounded-2xl px-3 py-2">
                    <p className="text-sm">
                      <span className="font-semibold mr-2">{comment.username}</span>
                      {comment.text}
                    </p>
                    <p className="text-xs text-gray-500 mt-1">{formatDate(comment.timestamp)}</p>
                  </div>
                </div>
              </div>
            ))}
          </div>
          
          {/* Add comment form */}
          <form onSubmit={handleComment} className="flex items-center px-4 py-3 border-t">
            <input
              type="text"
              placeholder="Add a comment..."
              className="flex-1 bg-transparent border-none focus:outline-none text-sm"
              value={commentText}
              onChange={(e) => setCommentText(e.target.value)}
            />
            <button 
              type="submit" 
              disabled={!commentText.trim()}
              className={`text-blue-500 font-semibold text-sm ${!commentText.trim() ? 'opacity-50 cursor-not-allowed' : 'hover:text-blue-600'}`}
            >
              Post
            </button>
          </form>
        </div>
      )}
    </div>
  );
}

export default PostCard;