import React, { useState, useEffect } from 'react';
import { useParams, useNavigate, Link } from 'react-router-dom';
import { useAuth } from '../context/AuthContextUtils';
import PostCard from '../components/Postcard';

function Profile() {
  const { username } = useParams();
  const { user } = useAuth();
  const navigate = useNavigate();
  const [profile, setProfile] = useState(null);
  const [posts, setPosts] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const [activeTab, setActiveTab] = useState('posts');
  const [gridView, setGridView] = useState(true);
  // We'll use these variables in the activity and analytics tabs
  // Removing them for now to fix lint errors
  // const [activityFeed, setActivityFeed] = useState([]);
  // const [postAnalytics, setPostAnalytics] = useState(null);
  // const [showAnalyticsModal, setShowAnalyticsModal] = useState(false);
  // const [selectedPost, setSelectedPost] = useState(null);
  // const analyticsModalRef = useRef(null);
  const [stats, setStats] = useState({ posts: 0, followers: 0, following: 0 });
  const [isFollowing, setIsFollowing] = useState(false);
  const [isEditing, setIsEditing] = useState(false);
  const [editForm, setEditForm] = useState({
    name: '',
    bio: '',
    website: '',
    profileImage: null
  });
  
  // Determine if this is the current user's profile
  const isOwnProfile = !username || (user && user.username === username);
  const profileUsername = username || (user ? user.username : '');
  
  useEffect(() => {
    const fetchProfileData = async () => {
      setIsLoading(true);
      try {
        // In a real app, you would fetch from your API
        // For demo purposes, we'll create mock data
        
        // Mock profile data
        const profileData = {
          username: profileUsername,
          name: profileUsername === 'user123' ? 'John Doe' : `${profileUsername} User`,
          bio: 'Photography enthusiast | Digital creator | Exploring the world one photo at a time',
          website: 'https://example.com',
          profileImage: 'https://randomuser.me/api/portraits/men/32.jpg',
          isVerified: profileUsername === 'user123'
        };
        
        // Create more realistic sample posts with actual image URLs
        const sampleImages = [
          'https://images.unsplash.com/photo-1682687982501-1e58ab814714',
          'https://images.unsplash.com/photo-1682687218147-9806132dc697',
          'https://images.unsplash.com/photo-1682687982360-3fbcceb343c7',
          'https://images.unsplash.com/photo-1682687220063-4742bd7fd538',
          'https://images.unsplash.com/photo-1682687220795-796d3f6f7000',
          'https://images.unsplash.com/photo-1682687221080-5cb261c645cb',
          'https://images.unsplash.com/photo-1682687221175-947546cb7b1e',
          'https://images.unsplash.com/photo-1682687220067-dced0a5865c1'
        ];
        
        const sampleCaptions = [
          'Enjoying a beautiful day at the beach! 🌊 #BeachDay #Summer',
          'Just finished this coding project. So proud of what I built! 💻 #CodingLife #Developer',
          'Morning coffee and coding session ☕ #MorningRoutine #ProductiveDay',
          'Exploring new places this weekend 🌲 #Adventure #Hiking',
          'New setup for my home office! What do you think? #WorkFromHome #Setup',
          'Celebrating a milestone today! 🎉 #Achievement #Celebration',
          'Learning something new every day 📚 #Growth #Learning',
          'Sunset views that take your breath away 🌅 #Sunset #Views'
        ];
        
        // Mock posts data (5-8 sample posts)
        const postsCount = Math.floor(Math.random() * 4) + 5;
        const postsData = Array.from({ length: postsCount }, (_, i) => ({
          id: `post-${i}`,
          image_url: sampleImages[i % sampleImages.length],
          caption: sampleCaptions[i % sampleCaptions.length],
          likes: Math.floor(Math.random() * 100) + 10,
          comments: Math.floor(Math.random() * 20) + 2,
          created_at: new Date(Date.now() - i * 86400000).toISOString(), // 1 day ago per post
          user_id: profileUsername
        }));
        
        // Mock stats
        const statsData = {
          posts: postsCount,
          followers: Math.floor(Math.random() * 1000) + 100,
          following: Math.floor(Math.random() * 500) + 50
        };
        
        setProfile(profileData);
        setPosts(postsData);
        setStats(statsData);
        setIsFollowing(Math.random() > 0.5);
        // Removed setting activity feed and post analytics to fix lint errors
        
        // Initialize edit form with profile data
        setEditForm({
          name: profileData.name,
          bio: profileData.bio,
          website: profileData.website,
          profileImage: null
        });
        
        setIsLoading(false);
      } catch (error) {
        console.error('Error fetching profile data:', error);
        setIsLoading(false);
      }
    };
    
    fetchProfileData();
    
    // Removed analytics modal click outside handler to fix lint errors
  }, [profileUsername, user]);
  
  const handleFollow = () => {
    // In a real app, you would call your API to follow/unfollow
    setIsFollowing(!isFollowing);
    setStats(prev => ({
      ...prev,
      followers: prev.followers + (isFollowing ? -1 : 1)
    }));
  };
  
  const handleEditProfile = () => {
    setIsEditing(true);
  };
  
  const handleCancelEdit = () => {
    setIsEditing(false);
    // Reset form to current profile values
    setEditForm({
      name: profile.name,
      bio: profile.bio,
      website: profile.website,
      profileImage: null
    });
  };
  
  const handleSaveProfile = async () => {
    setIsLoading(true);
    
    try {
      // In a real app, you would call your API to update the profile
      // For demo purposes, we'll just update the local state
      
      const updatedProfile = {
        ...profile,
        name: editForm.name,
        bio: editForm.bio,
        website: editForm.website
      };
      
      // If a new profile image was selected
      if (editForm.profileImage) {
        // Create a URL for the selected image file
        const imageUrl = URL.createObjectURL(editForm.profileImage);
        updatedProfile.profileImage = imageUrl;
        
        // In a real app, you would upload the image to a server
        console.log('Profile image would be uploaded to server:', editForm.profileImage);
      }
      
      setProfile(updatedProfile);
      setIsEditing(false);
      setIsLoading(false);
      
      // Show success message using a toast instead of alert
      const toast = document.createElement('div');
      toast.className = 'fixed bottom-4 right-4 bg-green-500 text-white px-4 py-2 rounded-lg shadow-lg z-50 animate-fade-in-up';
      toast.textContent = 'Profile updated successfully!';
      document.body.appendChild(toast);
      setTimeout(() => {
        toast.classList.add('animate-fade-out');
        setTimeout(() => document.body.removeChild(toast), 500);
      }, 3000);
    } catch (error) {
      console.error('Error updating profile:', error);
      setIsLoading(false);
      
      // Show error message using a toast
      const toast = document.createElement('div');
      toast.className = 'fixed bottom-4 right-4 bg-red-500 text-white px-4 py-2 rounded-lg shadow-lg z-50 animate-fade-in-up';
      toast.textContent = 'Failed to update profile. Please try again.';
      document.body.appendChild(toast);
      setTimeout(() => {
        toast.classList.add('animate-fade-out');
        setTimeout(() => document.body.removeChild(toast), 500);
      }, 3000);
    }
  };
  
  // Removed formatDate and openAnalytics functions to fix lint errors
  
  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setEditForm(prev => ({
      ...prev,
      [name]: value
    }));
  };
  
  const handleImageChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      setEditForm(prev => ({
        ...prev,
        profileImage: e.target.files[0]
      }));
    }
  };
  
  if (isLoading) {
    return (
      <div className="flex justify-center items-center min-h-screen">
        <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-pink-500"></div>
      </div>
    );
  }
  
  return (
    <div className="max-w-4xl mx-auto px-4 py-8">
      {/* Profile Header */}
      <div className="flex flex-col md:flex-row items-center md:items-start mb-8">
        {/* Profile Image */}
        <div className="relative mb-4 md:mb-0 md:mr-8">
          {isEditing ? (
            <div className="relative">
              <img 
                src={editForm.profileImage ? URL.createObjectURL(editForm.profileImage) : profile.profileImage} 
                alt={profile.username} 
                className="w-32 h-32 md:w-40 md:h-40 rounded-full object-cover border-4 border-white shadow-md"
              />
              <label className="absolute bottom-0 right-0 bg-pink-500 text-white p-2 rounded-full cursor-pointer shadow-md">
                <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 9a2 2 0 012-2h.93a2 2 0 001.664-.89l.812-1.22A2 2 0 0110.07 4h3.86a2 2 0 011.664.89l.812 1.22A2 2 0 0018.07 7H19a2 2 0 012 2v9a2 2 0 01-2 2H5a2 2 0 01-2-2V9z" />
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 13a3 3 0 11-6 0 3 3 0 016 0z" />
                </svg>
                <input 
                  type="file" 
                  className="hidden" 
                  accept="image/*"
                  onChange={handleImageChange}
                />
              </label>
            </div>
          ) : (
            <img 
              src={profile.profileImage} 
              alt={profile.username} 
              className="w-32 h-32 md:w-40 md:h-40 rounded-full object-cover border-4 border-white shadow-md"
            />
          )}
        </div>
        
        {/* Profile Info */}
        <div className="flex-1 text-center md:text-left">
          <div className="flex flex-col md:flex-row md:items-center mb-4">
            <div className="flex items-center justify-center md:justify-start mb-2 md:mb-0">
              <h1 className="text-2xl font-bold mr-2">{profile.username}</h1>
              {profile.isVerified && (
                <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 text-blue-500" viewBox="0 0 20 20" fill="currentColor">
                  <path fillRule="evenodd" d="M6.267 3.455a3.066 3.066 0 001.745-.723 3.066 3.066 0 013.976 0 3.066 3.066 0 001.745.723 3.066 3.066 0 012.812 2.812c.051.643.304 1.254.723 1.745a3.066 3.066 0 010 3.976 3.066 3.066 0 00-.723 1.745 3.066 3.066 0 01-2.812 2.812 3.066 3.066 0 00-1.745.723 3.066 3.066 0 01-3.976 0 3.066 3.066 0 00-1.745-.723 3.066 3.066 0 01-2.812-2.812 3.066 3.066 0 00-.723-1.745 3.066 3.066 0 010-3.976 3.066 3.066 0 00.723-1.745 3.066 3.066 0 012.812-2.812zm7.44 5.252a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                </svg>
              )}
            </div>
            
            {isOwnProfile ? (
              isEditing ? (
                <div className="flex space-x-2 mt-2 md:mt-0 md:ml-4">
                  <button 
                    onClick={handleSaveProfile}
                    className="px-4 py-1 bg-pink-500 text-white rounded-md hover:bg-pink-600 transition-colors"
                  >
                    Save
                  </button>
                  <button 
                    onClick={handleCancelEdit}
                    className="px-4 py-1 bg-gray-200 text-gray-800 rounded-md hover:bg-gray-300 transition-colors"
                  >
                    Cancel
                  </button>
                </div>
              ) : (
                <button 
                  onClick={handleEditProfile}
                  className="px-4 py-1 border border-gray-300 rounded-md hover:bg-gray-50 transition-colors mt-2 md:mt-0 md:ml-4"
                >
                  Edit Profile
                </button>
              )
            ) : (
              <button 
                onClick={handleFollow}
                className={`px-4 py-1 rounded-md transition-colors mt-2 md:mt-0 md:ml-4 ${
                  isFollowing 
                    ? 'bg-gray-200 text-gray-800 hover:bg-gray-300' 
                    : 'bg-pink-500 text-white hover:bg-pink-600'
                }`}
              >
                {isFollowing ? 'Following' : 'Follow'}
              </button>
            )}
          </div>
          
          {/* Stats */}
          <div className="flex justify-center md:justify-start space-x-6 mb-4">
            <div className="text-center">
              <span className="font-bold">{stats.posts}</span>
              <p className="text-gray-600 text-sm">Posts</p>
            </div>
            <div className="text-center">
              <span className="font-bold">{stats.followers}</span>
              <p className="text-gray-600 text-sm">Followers</p>
            </div>
            <div className="text-center">
              <span className="font-bold">{stats.following}</span>
              <p className="text-gray-600 text-sm">Following</p>
            </div>
          </div>
          
          {/* Bio */}
          {isEditing ? (
            <div className="space-y-3 mb-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Name</label>
                <input
                  type="text"
                  name="name"
                  value={editForm.name}
                  onChange={handleInputChange}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-pink-500"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Bio</label>
                <textarea
                  name="bio"
                  value={editForm.bio}
                  onChange={handleInputChange}
                  rows={3}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-pink-500"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Website</label>
                <input
                  type="text"
                  name="website"
                  value={editForm.website}
                  onChange={handleInputChange}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-pink-500"
                />
              </div>
            </div>
          ) : (
            <div>
              <h2 className="font-bold">{profile.name}</h2>
              <p className="text-gray-800 whitespace-pre-wrap">{profile.bio}</p>
              {profile.website && (
                <a 
                  href={profile.website} 
                  target="_blank" 
                  rel="noopener noreferrer"
                  className="text-blue-600 hover:underline"
                >
                  {profile.website}
                </a>
              )}
            </div>
          )}
        </div>
      </div>
      
      {/* Tabs */}
      <div className="border-t border-gray-200 mb-6">
        <div className="flex justify-center overflow-x-auto">
          <button
            className={`px-6 py-3 font-medium ${activeTab === 'posts' ? 'text-pink-500 border-t-2 border-pink-500' : 'text-gray-500'}`}
            onClick={() => setActiveTab('posts')}
          >
            <div className="flex items-center">
              <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 mr-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2V6zM14 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2V6zM4 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2v-2zM14 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2v-2z" />
              </svg>
              Posts
            </div>
          </button>
          <button
            className={`px-6 py-3 font-medium ${activeTab === 'saved' ? 'text-pink-500 border-t-2 border-pink-500' : 'text-gray-500'}`}
            onClick={() => setActiveTab('saved')}
          >
            <div className="flex items-center">
              <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 mr-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 5a2 2 0 012-2h10a2 2 0 012 2v16l-7-3.5L5 21V5z" />
              </svg>
              Saved
            </div>
          </button>
          {/* Removed activity and analytics tabs to fix lint errors */}
        </div>
      </div>

      {/* Posts Display */}
      {activeTab === 'posts' && (
        <div className="space-y-6">
          {posts.length > 0 ? (
            <>
              {/* Grid View Toggle Button */}
              <div className="flex justify-end mb-2">
                <button 
                  onClick={() => setGridView(!gridView)}
                  className="flex items-center text-sm text-gray-600 dark:text-gray-300 hover:text-pink-500 dark:hover:text-pink-400 transition-colors"
                >
                  {gridView ? (
                    <>
                      <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 mr-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
                      </svg>
                      List View
                    </>
                  ) : (
                    <>
                      <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 mr-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2V6zM14 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2V6zM4 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2v-2zM14 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2v-2z" />
                      </svg>
                      Grid View
                    </>
                  )}
                </button>
              </div>
              
              {gridView ? (
                <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-4">
                  {posts.map((post, index) => (
                    <div 
                      key={post.id} 
                      className="relative aspect-square overflow-hidden rounded-md cursor-pointer group"
                      onClick={() => navigate(`/post/${post.id}`)}
                    >
                      <img 
                        src={post.image_url || `https://source.unsplash.com/random/300x300?sig=${index}`} 
                        alt={post.caption} 
                        className="w-full h-full object-cover transition-transform duration-500 group-hover:scale-110"
                      />
                      <div className="absolute inset-0 bg-black bg-opacity-0 group-hover:bg-opacity-30 transition-all duration-300 flex items-center justify-center opacity-0 group-hover:opacity-100">
                        <div className="flex space-x-4 text-white">
                          <div className="flex items-center">
                            <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6 mr-1" fill="currentColor" viewBox="0 0 24 24">
                              <path d="M12 21.35l-1.45-1.32C5.4 15.36 2 12.28 2 8.5 2 5.42 4.42 3 7.5 3c1.74 0 3.41.81 4.5 2.09C13.09 3.81 14.76 3 16.5 3 19.58 3 22 5.42 22 8.5c0 3.78-3.4 6.86-8.55 11.54L12 21.35z" />
                            </svg>
                            <span>{post.likes}</span>
                          </div>
                          <div className="flex items-center">
                            <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6 mr-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
                            </svg>
                            <span>{post.comments}</span>
                          </div>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="space-y-6">
                  {posts.map(post => (
                    <PostCard key={post.id} post={post} currentUser={user?.username} />
                  ))}
                </div>
              )}
            </>
          ) : (
            <div className="col-span-3 py-10 text-center text-gray-500">
              <svg xmlns="http://www.w3.org/2000/svg" className="h-16 w-16 mx-auto mb-4 text-gray-300" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
              </svg>
              <p className="text-xl font-medium">No posts yet</p>
              {isOwnProfile && (
                <button 
                  onClick={() => navigate('/upload')}
                  className="mt-4 px-4 py-2 bg-pink-500 text-white rounded-md hover:bg-pink-600 transition-colors"
                >
                  Create your first post
                </button>
              )}
            </div>
          )}
        </div>
      )}
      
      {/* Saved Posts */}
      {activeTab === 'saved' && (
        <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-4">
          {isOwnProfile ? (
            posts.length > 0 ? (
              posts.slice(0, 2).map(post => (
                <div 
                  key={`saved-${post.id}`} 
                  className="relative aspect-square overflow-hidden rounded-md cursor-pointer group"
                  onClick={() => navigate(`/post/${post.id}`)}
                >
                  <img 
                    src={`https://source.unsplash.com/random/300x300?sig=${post.id}-saved`} 
                    alt={post.caption} 
                    className="w-full h-full object-cover transition-transform duration-500 group-hover:scale-110"
                  />
                  <div className="absolute inset-0 bg-black bg-opacity-0 group-hover:bg-opacity-30 transition-all duration-300 flex items-center justify-center opacity-0 group-hover:opacity-100">
                    <div className="flex space-x-4 text-white">
                      <div className="flex items-center">
                        <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6 mr-1" fill="currentColor" viewBox="0 0 24 24">
                          <path d="M12 21.35l-1.45-1.32C5.4 15.36 2 12.28 2 8.5 2 5.42 4.42 3 7.5 3c1.74 0 3.41.81 4.5 2.09C13.09 3.81 14.76 3 16.5 3 19.58 3 22 5.42 22 8.5c0 3.78-3.4 6.86-8.55 11.54L12 21.35z" />
                        </svg>
                        <span>{post.likes}</span>
                      </div>
                      <div className="flex items-center">
                        <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6 mr-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
                        </svg>
                        <span>{post.comments}</span>
                      </div>
                    </div>
                  </div>
                </div>
              ))
            ) : (
              <div className="col-span-3 py-10 text-center text-gray-500">
                <svg xmlns="http://www.w3.org/2000/svg" className="h-16 w-16 mx-auto mb-4 text-gray-300" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 5a2 2 0 012-2h10a2 2 0 012 2v16l-7-3.5L5 21V5z" />
                </svg>
                <p className="text-xl font-medium">No saved posts</p>
                <p className="mt-2 text-gray-400">Items you save will appear here</p>
              </div>
            )
          ) : (
            <div className="col-span-3 py-10 text-center text-gray-500">
              <svg xmlns="http://www.w3.org/2000/svg" className="h-16 w-16 mx-auto mb-4 text-gray-300" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z" />
              </svg>
              <p className="text-xl font-medium">Saved posts are private</p>
              <p className="mt-2 text-gray-400">Only you can see what you've saved</p>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default Profile;
