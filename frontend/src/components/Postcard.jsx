import React from 'react';

function PostCard({ post }) {
  // Format date
  const formatDate = (dateString) => {
    const date = new Date(dateString);
    return date.toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'long',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });
  };
  
  return (
    <div className="bg-white border rounded-lg overflow-hidden shadow-md">
      <div className="p-4 border-b">
        <div className="flex items-center">
          <div className="h-10 w-10 rounded-full bg-gray-300 flex items-center justify-center text-gray-700 font-bold">
            {post.user_id.charAt(0).toUpperCase()}
          </div>
          <div className="ml-3">
            <p className="font-bold">{post.user_id}</p>
            <p className="text-gray-500 text-sm">{formatDate(post.created_at)}</p>
          </div>
        </div>
      </div>
      
      <img
        src={`http://localhost:8000/uploads/${post.image_url}`}
        alt={post.caption || "User post"}
        className="w-full h-auto"
      />
      
      {post.caption && (
        <div className="p-4">
          <p>{post.caption}</p>
        </div>
      )}
      
      <div className="p-4 border-t flex space-x-4">
        <button className="flex items-center space-x-1 text-gray-500 hover:text-red-500">
          <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4.318 6.318a4.5 4.5 0 000 6.364L12 20.364l7.682-7.682a4.5 4.5 0 00-6.364-6.364L12 7.636l-1.318-1.318a4.5 4.5 0 00-6.364 0z" />
          </svg>
          <span>Like</span>
        </button>
        
        <button className="flex items-center space-x-1 text-gray-500 hover:text-blue-500">
          <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
          </svg>
          <span>Comment</span>
        </button>
      </div>
    </div>
  );
}

export default PostCard;