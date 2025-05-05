import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';

function Upload() {
  const [file, setFile] = useState(null);
  const [caption, setCaption] = useState('');
  const [preview, setPreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState(false);
  const navigate = useNavigate();
  
  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    setError('');
    
    if (!selectedFile) {
      setFile(null);
      setPreview(null);
      return;
    }
    
    // Check if file is an image
    if (!selectedFile.type.match('image.*')) {
      setError('Please select an image file (png, jpg, jpeg)');
      setFile(null);
      setPreview(null);
      return;
    }
    
    // Create preview
    const reader = new FileReader();
    reader.onload = () => {
      setPreview(reader.result);
    };
    reader.readAsDataURL(selectedFile);
    
    setFile(selectedFile);
  };
  
  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    setSuccess(false);
    
    if (!file) {
      setError('Please select an image to upload');
      return;
    }
    
    setLoading(true);
    
    try {
      const token = localStorage.getItem('token');
      const formData = new FormData();
      formData.append('file', file);
      
      if (caption) {
        formData.append('caption', caption);
      }
      
      await axios.post('http://localhost:8000/upload/', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
          Authorization: `Bearer ${token}`
        }
      });
      
      setSuccess(true);
      
      // Reset form
      setFile(null);
      setCaption('');
      setPreview(null);
      
      // Redirect to home after a short delay
      setTimeout(() => {
        navigate('/');
      }, 2000);
    } catch (error) {
      console.error('Upload error:', error);
      
      if (error.response && error.response.data && error.response.data.detail) {
        setError(error.response.data.detail);
      } else {
        setError('Failed to upload image. Please try again.');
      }
    } finally {
      setLoading(false);
    }
  };
  
  return (
    <div className="max-w-2xl mx-auto">
      <h1 className="text-2xl font-bold mb-6">Upload Image</h1>
      
      {error && (
        <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded mb-4">
          {error}
        </div>
      )}
      
      {success && (
        <div className="bg-green-100 border border-green-400 text-green-700 px-4 py-3 rounded mb-4">
          Image uploaded successfully! Redirecting to feed...
        </div>
      )}
      
      <form onSubmit={handleSubmit} className="bg-white shadow-md rounded px-8 pt-6 pb-8 mb-4">
        <div className="mb-4">
          <label className="block text-gray-700 text-sm font-bold mb-2" htmlFor="image">
            Select Image
          </label>
          <input
            className="shadow appearance-none border rounded w-full py-2 px-3 text-gray-700 leading-tight focus:outline-none focus:shadow-outline"
            id="image"
            type="file"
            accept="image/*"
            onChange={handleFileChange}
            disabled={loading}
          />
        </div>
        
        {preview && (
          <div className="mb-4">
            <p className="block text-gray-700 text-sm font-bold mb-2">Preview</p>
            <div className="border rounded p-2">
              <img
                src={preview}
                alt="Preview"
                className="max-h-64 mx-auto"
              />
            </div>
          </div>
        )}
        
        <div className="mb-6">
          <label className="block text-gray-700 text-sm font-bold mb-2" htmlFor="caption">
            Caption (optional)
          </label>
          <textarea
            className="shadow appearance-none border rounded w-full py-2 px-3 text-gray-700 leading-tight focus:outline-none focus:shadow-outline"
            id="caption"
            placeholder="Add a caption to your image..."
            value={caption}
            onChange={(e) => setCaption(e.target.value)}
            disabled={loading}
            rows="3"
          ></textarea>
        </div>
        
        <div className="flex items-center justify-between">
          <button
            className={`bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded focus:outline-none focus:shadow-outline ${
              loading ? 'opacity-50 cursor-not-allowed' : ''
            }`}
            type="submit"
            disabled={loading}
          >
            {loading ? 'Uploading...' : 'Upload'}
          </button>
        </div>
      </form>
      
      <div className="bg-blue-100 border-l-4 border-blue-500 text-blue-700 p-4 mb-4">
        <p className="font-bold">AI-Powered Image Moderation</p>
        <p>SafeGram automatically scans all uploaded images for:</p>
        <ul className="list-disc ml-5 mt-2">
          <li>Deepfake content</li>
          <li>Explicit (NSFW) content</li>
        </ul>
        <p className="mt-2">Images that violate our community standards will be automatically rejected.</p>
      </div>
    </div>
  );
}

export default Upload;