import React, { useState, useEffect } from 'react';
import axios from 'axios';
import PostCard from '../components/Postcard';
import { useAuth } from '../context/AuthContextUtils';
import { Link } from 'react-router-dom';

function Home() {
  const { user } = useAuth();
  const [posts, setPosts] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [suggestedUsers] = useState([
    { id: 1, username: 'photography_lover', fullName: 'Alex Johnson', avatar: 'A' },
    { id: 2, username: 'travel_enthusiast', fullName: 'Maya Patel', avatar: 'M' },
    { id: 3, username: 'food_blogger', fullName: 'Sam Wilson', avatar: 'S' },
    { id: 4, username: 'fitness_guru', fullName: 'Taylor Reed', avatar: 'T' },
    { id: 5, username: 'tech_geek', fullName: 'Jordan Lee', avatar: 'J' }
  ]);

  // Mock stories data
  const [stories] = useState([
    { id: 1, username: 'travel_enthusiast', avatar: 'M', hasUnseenStory: true },
    { id: 2, username: 'food_blogger', avatar: 'S', hasUnseenStory: true },
    { id: 3, username: 'photography_lover', avatar: 'A', hasUnseenStory: true },
    { id: 4, username: 'fitness_guru', avatar: 'T', hasUnseenStory: false },
    { id: 5, username: 'tech_geek', avatar: 'J', hasUnseenStory: true },
    { id: 6, username: 'art_creator', avatar: 'C', hasUnseenStory: true },
    { id: 7, username: 'music_fan', avatar: 'R', hasUnseenStory: false },
    { id: 8, username: 'book_worm', avatar: 'B', hasUnseenStory: true }
  ]);

  useEffect(() => {
    const fetchPosts = async () => {
      try {
        const token = localStorage.getItem('token');
        const response = await axios.get('http://localhost:8000/feed/', {
          headers: {
            Authorization: `Bearer ${token}`
          }
        });

        setPosts(response.data);
      } catch (error) {
        console.error('Error fetching posts:', error);
        setError('Failed to load posts. Please try again later.');
      } finally {
        setLoading(false);
      }
    };

    fetchPosts();
  }, []);

  if (loading) {
    return (
      <div className="flex justify-center items-center h-64">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-pink-500 mx-auto"></div>
          <p className="mt-4 text-gray-600">Loading your feed...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="max-w-7xl mx-auto px-4 py-6">
        <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded mb-4">
          {error}
        </div>
      </div>
    );
  }

  return (
    <div className="max-w-7xl mx-auto px-4 py-6">
      <div className="flex flex-col md:flex-row gap-6">
        {/* Main content - Feed */}
        <div className="md:w-8/12 lg:w-7/12">
          {/* Stories section */}
          <div className="bg-white rounded-lg shadow mb-6 p-4 overflow-hidden">
            <h2 className="font-semibold text-gray-800 mb-4">Stories</h2>
            <div className="flex space-x-4 overflow-x-auto pb-2 scrollbar-hide">
              {/* Add story button */}
              <div className="flex flex-col items-center space-y-1 flex-shrink-0">
                <div className="w-16 h-16 rounded-full flex items-center justify-center bg-gray-100 border-2 border-dashed border-gray-300">
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6v6m0 0v6m0-6h6m-6 0H6" />
                  </svg>
                </div>
                <span className="text-xs text-gray-500">Add Story</span>
              </div>

              {/* Stories */}
              {stories.map(story => (
                <div key={story.id} className="flex flex-col items-center space-y-1 flex-shrink-0">
                  <div className={`w-16 h-16 rounded-full flex items-center justify-center ${story.hasUnseenStory ? 'border-2 border-pink-500 p-0.5' : ''}`}>
                    <div className="w-full h-full rounded-full bg-gradient-to-r from-purple-500 to-pink-500 flex items-center justify-center text-white font-bold">
                      {story.avatar}
                    </div>
                  </div>
                  <span className="text-xs truncate w-16 text-center">{story.username}</span>
                </div>
              ))}
            </div>
          </div>

          {/* Posts */}
          {posts.length === 0 ? (
            <div className="text-center p-8 bg-white rounded-lg shadow">
              <svg xmlns="http://www.w3.org/2000/svg" className="h-16 w-16 text-gray-300 mx-auto mb-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
              </svg>
              <p className="text-lg text-gray-600 mb-2">No posts in your feed yet</p>
              <p className="text-sm text-gray-500 mb-4">Follow some users or upload your first post!</p>
              <Link to="/upload" className="px-4 py-2 bg-gradient-to-r from-pink-500 to-purple-600 text-white rounded-full font-medium hover:shadow-lg transition duration-300">
                Create Post
              </Link>
            </div>
          ) : (
            <div>
              {posts.map((post) => (
                <PostCard key={post.id} post={post} />
              ))}
            </div>
          )}
        </div>

        {/* Sidebar */}
        <div className="md:w-4/12 lg:w-5/12">
          <div className="sticky top-24">
            {/* User profile card */}
            <div className="bg-white rounded-lg shadow p-4 mb-6">
              <div className="flex items-center">
                <div className="h-14 w-14 rounded-full bg-gradient-to-r from-pink-500 to-purple-600 flex items-center justify-center text-white font-bold text-xl">
                  {user?.username ? user.username.charAt(0).toUpperCase() : '?'}
                </div>
                <div className="ml-4">
                  <p className="font-bold text-gray-800">{user?.username}</p>
                  <p className="text-gray-500 text-sm">{user?.email}</p>
                </div>
              </div>
              <div className="mt-4 grid grid-cols-3 gap-2 text-center">
                <div className="bg-gray-50 rounded p-2">
                  <p className="font-bold text-gray-800">0</p>
                  <p className="text-gray-500 text-xs">Posts</p>
                </div>
                <div className="bg-gray-50 rounded p-2">
                  <p className="font-bold text-gray-800">0</p>
                  <p className="text-gray-500 text-xs">Followers</p>
                </div>
                <div className="bg-gray-50 rounded p-2">
                  <p className="font-bold text-gray-800">0</p>
                  <p className="text-gray-500 text-xs">Following</p>
                </div>
              </div>
              <div className="mt-4">
                <Link to="/profile" className="block w-full py-2 bg-gray-100 text-gray-700 text-center rounded-md hover:bg-gray-200 transition duration-200">
                  Edit Profile
                </Link>
              </div>
            </div>

            {/* Suggested users */}
            <div className="bg-white rounded-lg shadow p-4">
              <div className="flex justify-between items-center mb-4">
                <h3 className="font-semibold text-gray-500">Suggested For You</h3>
                <button className="text-sm font-medium text-gray-700 hover:text-gray-900">See All</button>
              </div>

              {suggestedUsers.map(user => (
                <div key={user.id} className="flex items-center justify-between py-2">
                  <div className="flex items-center">
                    <div className="h-9 w-9 rounded-full bg-gradient-to-r from-blue-400 to-purple-500 flex items-center justify-center text-white font-bold">
                      {user.avatar}
                    </div>
                    <div className="ml-3">
                      <p className="text-sm font-semibold">{user.username}</p>
                      <p className="text-xs text-gray-500">{user.fullName}</p>
                    </div>
                  </div>
                  <button className="text-xs font-semibold text-pink-500 hover:text-pink-600">
                    Follow
                  </button>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default Home;