import React, { useState, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';

function Classify() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [result, setResult] = useState(null);
  const [dragActive, setDragActive] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const fileInputRef = useRef(null);
  const navigate = useNavigate();
  
  // Maximum file size (5MB)
  const MAX_FILE_SIZE = 5 * 1024 * 1024;
  // Allowed file types
  const ALLOWED_TYPES = ['image/jpeg', 'image/png', 'image/jpg', 'image/gif'];
  
  // Handle file selection from input
  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      processFile(e.target.files[0]);
    }
  };

  // Handle drag events
  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  // Handle drop event
  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      processFile(e.dataTransfer.files[0]);
    }
  };

  // Process the selected file
  const processFile = (selectedFile) => {
    // Reset states
    setError('');
    setResult(null);
    
    // Validate file type
    if (!ALLOWED_TYPES.includes(selectedFile.type)) {
      setError('Please select a valid image file (JPEG, PNG, or GIF)');
      return;
    }
    
    // Validate file size
    if (selectedFile.size > MAX_FILE_SIZE) {
      setError(`File size exceeds the limit (${formatFileSize(MAX_FILE_SIZE)})`);
      return;
    }
    
    // Set the file and create preview
    setFile(selectedFile);
    
    // Create preview
    const reader = new FileReader();
    reader.onload = () => {
      setPreview(reader.result);
    };
    reader.readAsDataURL(selectedFile);
  };

  // Handle form submission
  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (!file) {
      setError('Please select an image to classify');
      return;
    }
    
    setLoading(true);
    setError('');
    setUploadProgress(0);
    
    // Create form data
    const formData = new FormData();
    formData.append('file', file);
    
    try {
      // Get the token from localStorage
      const token = localStorage.getItem('token');
      
      if (!token) {
        setError('You must be logged in to classify images');
        setLoading(false);
        return;
      }
      
      // Upload the image for classification
      const response = await axios.post(
        'http://localhost:8000/classify-image/',
        formData,
        {
          headers: {
            'Content-Type': 'multipart/form-data',
            'Authorization': `Bearer ${token}`
          },
          onUploadProgress: (progressEvent) => {
            const percentCompleted = Math.round((progressEvent.loaded * 100) / progressEvent.total);
            setUploadProgress(percentCompleted);
          }
        }
      );
      
      // Set the classification result
      setResult(response.data);
      setLoading(false);
      
    } catch (error) {
      console.error('Error classifying image:', error);
      
      if (error.response && error.response.status === 401) {
        setError('Your session has expired. Please log in again.');
        // Redirect to login page after a short delay
        setTimeout(() => navigate('/login'), 2000);
      } else if (error.response && error.response.data) {
        setError(error.response.data.detail || 'Failed to classify image');
      } else {
        setError('An error occurred while classifying the image');
      }
      
      setLoading(false);
    }
  };

  // Function to format file size
  const formatFileSize = (bytes) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  return (
    <div className="max-w-4xl mx-auto p-4 sm:p-6 lg:p-8">
      <h1 className="text-3xl font-bold mb-6 text-center text-gray-800">
        Image Classification
      </h1>
      
      <div className="mb-6 text-center">
        <p className="text-gray-600">
          Upload an image to classify it as real or fake
        </p>
      </div>
      
      <form onSubmit={handleSubmit} className="space-y-6">
        {/* Drag and drop area */}
        <div 
          className={`border-2 border-dashed rounded-lg p-6 text-center cursor-pointer transition-colors duration-300 ${
            dragActive ? 'border-pink-500 bg-pink-50' : 'border-gray-300 hover:border-pink-400'
          }`}
          onClick={() => fileInputRef.current.click()}
          onDragEnter={handleDrag}
          onDragLeave={handleDrag}
          onDragOver={handleDrag}
          onDrop={handleDrop}
        >
          <input
            type="file"
            ref={fileInputRef}
            onChange={handleFileChange}
            accept="image/*"
            className="hidden"
          />
          
          {preview ? (
            <div className="relative">
              <img 
                src={preview} 
                alt="Preview" 
                className="max-h-64 mx-auto rounded-lg shadow-md"
              />
              <div className="mt-2 text-sm text-gray-500">
                {file && (
                  <p>{file.name} ({formatFileSize(file.size)})</p>
                )}
              </div>
              <button
                type="button"
                onClick={(e) => {
                  e.stopPropagation();
                  setFile(null);
                  setPreview('');
                }}
                className="absolute top-2 right-2 bg-red-500 text-white rounded-full p-1 hover:bg-red-600 focus:outline-none"
              >
                <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
                  <path fillRule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clipRule="evenodd" />
                </svg>
              </button>
            </div>
          ) : (
            <div className="py-8">
              <svg className="mx-auto h-12 w-12 text-gray-400" stroke="currentColor" fill="none" viewBox="0 0 48 48" aria-hidden="true">
                <path d="M28 8H12a4 4 0 00-4 4v20m32-12v8m0 0v8a4 4 0 01-4 4H12a4 4 0 01-4-4v-4m32-4l-3.172-3.172a4 4 0 00-5.656 0L28 28M8 32l9.172-9.172a4 4 0 015.656 0L28 28m0 0l4 4m4-24h8m-4-4v8m-12 4h.02" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
              </svg>
              <p className="mt-2 text-sm text-gray-500">
                Click to upload or drag and drop
              </p>
              <p className="text-xs text-gray-500">
                PNG, JPG, GIF up to 5MB
              </p>
            </div>
          )}
        </div>
        
        {/* Error message */}
        {error && (
          <div className="p-4 bg-red-50 border border-red-200 rounded-lg">
            <p className="text-red-600">{error}</p>
          </div>
        )}
        
        {/* Classification result */}
        {result && (
          <div className={`p-6 rounded-lg shadow-md ${
            result.is_real ? 'bg-green-50 border border-green-200' : 'bg-red-50 border border-red-200'
          }`}>
            <h3 className={`text-xl font-bold mb-2 ${
              result.is_real ? 'text-green-700' : 'text-red-700'
            }`}>
              {result.is_real ? 'Real Image' : 'Fake Image'}
            </h3>
            <p className="text-gray-700 mb-2">
              Confidence: <span className="font-semibold">{(result.confidence * 100).toFixed(2)}%</span>
            </p>
            <p className="text-sm text-gray-500">
              The image has been saved to the {result.is_real ? 'real' : 'fake'} folder
            </p>
          </div>
        )}
        
        {/* Upload progress */}
        {loading && (
          <div className="mb-6">
            <div className="relative pt-1">
              <div className="flex mb-2 items-center justify-between">
                <div>
                  <span className="text-xs font-semibold inline-block py-1 px-2 uppercase rounded-full text-pink-600 bg-pink-200">
                    Uploading and classifying...
                  </span>
                </div>
                <div className="text-xs text-gray-500">
                  {uploadProgress}%
                </div>
              </div>
              <div className="overflow-hidden h-2 mb-4 text-xs flex rounded bg-pink-200">
                <div 
                  className="shadow-none flex flex-col text-center whitespace-nowrap text-white justify-center bg-pink-500 transition-all duration-300" 
                  style={{ width: `${uploadProgress}%` }}
                ></div>
              </div>
            </div>
          </div>
        )}
        
        {/* Submit button */}
        <div className="flex items-center justify-between">
          <button
            className={`w-full bg-gradient-to-r from-pink-500 to-purple-600 hover:from-pink-600 hover:to-purple-700 text-white font-bold py-3 px-6 rounded-full focus:outline-none focus:shadow-outline transition duration-300 ${
              loading || !file ? 'opacity-50 cursor-not-allowed' : ''
            }`}
            type="submit"
            disabled={loading || !file}
          >
            {loading ? 'Processing...' : 'Classify Image'}
          </button>
        </div>
      </form>
    </div>
  );
}

export default Classify;
