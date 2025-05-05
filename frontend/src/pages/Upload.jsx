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
  const [fileSize, setFileSize] = useState(0);
  const [fileType, setFileType] = useState('');
  const [scanningStage, setScanningStage] = useState(''); // for UI feedback
  const navigate = useNavigate();
  
  // Maximum file size (5MB)
  const MAX_FILE_SIZE = 5 * 1024 * 1024;
  // Allowed file types
  const ALLOWED_TYPES = ['image/jpeg', 'image/png', 'image/jpg', 'image/gif'];
  
  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    setError('');
    setScanningStage('');
    
    if (!selectedFile) {
      setFile(null);
      setPreview(null);
      setFileSize(0);
      setFileType('');
      return;
    }
    
    // Check file size
    if (selectedFile.size > MAX_FILE_SIZE) {
      setError(`File size exceeds the maximum limit of ${MAX_FILE_SIZE / (1024 * 1024)}MB`);
      setFile(null);
      setPreview(null);
      return;
    }
    
    // Check file type
    if (!ALLOWED_TYPES.includes(selectedFile.type)) {
      setError('Please select an image file (JPEG, PNG, or GIF)');
      setFile(null);
      setPreview(null);
      return;
    }
    
    // Store file details
    setFileSize(selectedFile.size);
    setFileType(selectedFile.type);
    
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
    setScanningStage('starting');
    
    if (!file) {
      setError('Please select an image to upload');
      setScanningStage('');
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
      
      // Update scanning stage for UI feedback
      setScanningStage('uploading');
      setTimeout(() => setScanningStage('analyzing'), 1000);
      
      // Upload the image
      await axios.post('http://localhost:8000/upload/', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
          Authorization: `Bearer ${token}`
        }
      });
      
      setScanningStage('complete');
      setSuccess(true);
      
      // Reset form
      setFile(null);
      setCaption('');
      setPreview(null);
      setFileSize(0);
      setFileType('');
      
      // Redirect to home after a short delay
      setTimeout(() => {
        navigate('/');
      }, 2000);
    } catch (error) {
      console.error('Upload error:', error);
      setScanningStage('failed');
      
      if (error.response && error.response.data && error.response.data.detail) {
        setError(error.response.data.detail);
      } else {
        setError('Failed to upload image. Please try again.');
      }
    } finally {
      setLoading(false);
    }
  };
  
  // Function to format file size
  const formatFileSize = (bytes) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };
  
  // UI feedback for scanning stage
  const getScanningStageText = () => {
    switch (scanningStage) {
      case 'starting':
        return 'Preparing upload...';
      case 'uploading':
        return 'Uploading image...';
      case 'analyzing':
        return 'Analyzing image for deepfakes and NSFW content...';
      case 'complete':
        return 'Analysis complete - Image is safe!';
      case 'failed':
        return 'Analysis complete - Image rejected';
      default:
        return '';
    }
  };
  
  return (
    <div className="max-w-2xl mx-auto">
      <h1 className="text-2xl font-bold mb-6">Upload Image</h1>
      
      {error && (
        <div className="bg-red-100 border-l-4 border-red-500 text-red-700 p-4 mb-4 rounded">
          <div className="flex">
            <div className="flex-shrink-0">
              <svg className="h-5 w-5 text-red-500" viewBox="0 0 20 20" fill="currentColor">
                <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
              </svg>
            </div>
            <div className="ml-3">
              <p className="text-sm font-medium">{error}</p>
            </div>
          </div>
        </div>
      )}
      
      {success && (
        <div className="bg-green-100 border-l-4 border-green-500 text-green-700 p-4 mb-4 rounded">
          <div className="flex">
            <div className="flex-shrink-0">
              <svg className="h-5 w-5 text-green-500" viewBox="0 0 20 20" fill="currentColor">
                <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
              </svg>
            </div>
            <div className="ml-3">
              <p className="text-sm font-medium">Image uploaded successfully! Redirecting to feed...</p>
            </div>
          </div>
        </div>
      )}
      
      <form onSubmit={handleSubmit} className="bg-white shadow-md rounded-lg px-8 pt-6 pb-8 mb-4">
        <div className="mb-4">
          <label className="block text-gray-700 text-sm font-bold mb-2" htmlFor="image">
            Select Image
          </label>
          <div className="border-2 border-dashed border-gray-300 rounded-lg p-6 text-center hover:border-pink-500 transition duration-300">
            <input
              className="hidden"
              id="image"
              type="file"
              accept="image/*"
              onChange={handleFileChange}
              disabled={loading}
              ref={(fileInput) => fileInput && (window.fileInput = fileInput)}
            />
            {!preview ? (
              <div onClick={() => window.fileInput.click()} className="cursor-pointer">
                <svg xmlns="http://www.w3.org/2000/svg" className="mx-auto h-12 w-12 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                </svg>
                <p className="mt-1 text-sm text-gray-600">Click to browse or drag and drop</p>
                <p className="mt-1 text-xs text-gray-500">PNG, JPG, GIF up to 5MB</p>
              </div>
            ) : (
              <div className="relative">
                <img
                  src={preview}
                  alt="Preview"
                  className="max-h-64 mx-auto rounded"
                />
                <button
                  type="button"
                  onClick={() => {
                    setFile(null);
                    setPreview(null);
                    setFileSize(0);
                    setFileType('');
                  }}
                  className="absolute top-2 right-2 bg-red-500 text-white rounded-full p-1 hover:bg-red-600 focus:outline-none"
                >
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
                    <path fillRule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clipRule="evenodd" />
                  </svg>
                </button>
              </div>
            )}
          </div>
          
          {fileSize > 0 && (
            <div className="mt-2 text-sm text-gray-600">
              File details: {formatFileSize(fileSize)} • {fileType.split('/')[1].toUpperCase()}
            </div>
          )}
        </div>
        
        <div className="mb-6">
          <label className="block text-gray-700 text-sm font-bold mb-2" htmlFor="caption">
            Caption (optional)
          </label>
          <textarea
            className="shadow appearance-none border rounded w-full py-2 px-3 text-gray-700 leading-tight focus:outline-none focus:ring-2 focus:ring-pink-500 focus:border-transparent"
            id="caption"
            placeholder="Add a caption to your image..."
            value={caption}
            onChange={(e) => setCaption(e.target.value)}
            disabled={loading}
            rows="3"
          ></textarea>
        </div>
        
        {scanningStage && (
          <div className="mb-6">
            <div className="relative pt-1">
              <div className="flex mb-2 items-center justify-between">
                <div>
                  <span className="text-xs font-semibold inline-block py-1 px-2 uppercase rounded-full text-pink-600 bg-pink-200">
                    {getScanningStageText()}
                  </span>
                </div>
              </div>
              <div className="overflow-hidden h-2 mb-4 text-xs flex rounded bg-pink-200">
                <div className={`shadow-none flex flex-col text-center whitespace-nowrap text-white justify-center bg-pink-500 ${scanningStage === 'complete' ? 'w-full' : scanningStage === 'failed' ? 'w-full bg-red-500' : 'w-3/4 animate-pulse'}`}></div>
              </div>
            </div>
          </div>
        )}
        
        <div className="flex items-center justify-between">
          <button
            className={`w-full bg-gradient-to-r from-pink-500 to-purple-600 hover:from-pink-600 hover:to-purple-700 text-white font-bold py-3 px-6 rounded-full focus:outline-none focus:shadow-outline transition duration-300 ${
              loading ? 'opacity-50 cursor-not-allowed' : ''
            }`}
            type="submit"
            disabled={loading || !file}
          >
            {loading ? 'Processing...' : 'Upload Image'}
          </button>
        </div>
      </form>
      
      <div className="bg-blue-100 border-l-4 border-blue-500 text-blue-700 p-4 mb-4 rounded-lg shadow">
        <h3 className="font-bold mb-2">AI-Powered Image Moderation</h3>
        <p className="mb-3">SafeGram automatically analyzes all uploaded images for:</p>
        <div className="grid grid-cols-2 gap-3 mb-3">
          <div className="flex items-start">
            <div className="flex-shrink-0 mt-1">
              <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 text-blue-500" viewBox="0 0 20 20" fill="currentColor">
                <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
              </svg>
            </div>
            <p className="ml-2 text-sm">
              <span className="font-semibold">Deepfake Detection:</span> Identifies manipulated or synthetic faces
            </p>
          </div>
          <div className="flex items-start">
            <div className="flex-shrink-0 mt-1">
              <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 text-blue-500" viewBox="0 0 20 20" fill="currentColor">
                <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
              </svg>
            </div>
            <p className="ml-2 text-sm">
              <span className="font-semibold">NSFW Screening:</span> Filters explicit or inappropriate content
            </p>
          </div>
        </div>
        <p className="text-sm">Images that violate our community standards will be automatically rejected to maintain a safe environment for all users.</p>
      </div>
      
      <div className="text-sm text-gray-500 mt-4">
        <p>Having trouble uploading? Make sure your image meets these requirements:</p>
        <ul className="list-disc ml-5 mt-1">
          <li>File must be a PNG, JPG, or GIF</li>
          <li>Maximum file size is 5MB</li>
          <li>Image must not contain manipulated faces or explicit content</li>
        </ul>
      </div>
    </div>
  );
}

export default Upload;