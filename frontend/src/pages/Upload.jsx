import React, { useState, useRef, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';

function Upload() {
  const [files, setFiles] = useState([]);
  const [captions, setCaptions] = useState({});
  const [previews, setPreviews] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState(false);
  const [filesInfo, setFilesInfo] = useState([]);
  const [scanningStage, setScanningStage] = useState(''); // for UI feedback
  const [detailedExplanation, setDetailedExplanation] = useState(null); // for AI explanation
  const [visualAreas, setVisualAreas] = useState(null); // for visualization of problematic areas
  const [dragActive, setDragActive] = useState(false); // for drag and drop interface
  const [uploadProgress, setUploadProgress] = useState(0); // for upload progress tracking
  const [rejectedImages, setRejectedImages] = useState([]); // for tracking rejected images
  const [showFilters, setShowFilters] = useState(false); // for showing/hiding filters
  const fileInputRef = useRef(null);
  const navigate = useNavigate();
  
  // Maximum file size (5MB)
  const MAX_FILE_SIZE = 5 * 1024 * 1024;
  // Allowed file types
  const ALLOWED_TYPES = ['image/jpeg', 'image/png', 'image/jpg', 'image/gif'];
  
  // Helper function to process error responses
  const processErrorResponse = (error) => {
    if (error.response && error.response.data && error.response.data.detail) {
      // Check if the detail is an object with explanation data
      if (typeof error.response.data.detail === 'object') {
        const detailData = error.response.data.detail;
        
        // Set the basic error message
        setError(detailData.message || 'Image rejected - See detailed explanation below');
        
        // Create a detailed explanation object
        const explanation = {};
        
        // Handle deepfake detection
        if (detailData.deepfake_detected) {
          explanation.deepfake = {
            detected: true,
            confidence: detailData.deepfake_confidence || 0.75,
            details: detailData.deepfake_explanation || 'Our AI system detected signs of facial manipulation in this image.',
            common_indicators: detailData.deepfake_indicators || [
              'Inconsistent facial features',
              'Unnatural skin texture',
              'Irregular lighting on face',
              'Blurry or distorted areas around facial features'
            ]
          };
        }
        
        // Handle NSFW detection
        if (detailData.nsfw_detected) {
          explanation.nsfw = {
            detected: true,
            confidence: detailData.nsfw_confidence || 0.8,
            details: detailData.nsfw_explanation || 'Our AI system detected inappropriate content that violates our community guidelines.',
            content_type: detailData.content_type || 'Potentially inappropriate content'
          };
        }
        
        // Handle technical issues
        if (detailData.technical_issue) {
          explanation.technical_issue = {
            details: detailData.technical_message || 'Our system encountered a technical issue while processing your image.'
          };
        }
        
        // If we have any explanation data, set it
        if (Object.keys(explanation).length > 0) {
          setDetailedExplanation(explanation);
        }
        
        // Set visual areas if available
        if (detailData.visual_areas) {
          setVisualAreas(detailData.visual_areas);
        }
      } else {
        // If detail is just a string, create a technical issue explanation
        setError('Image upload failed');
        setDetailedExplanation({
          technical_issue: {
            details: error.response.data.detail
          }
        });
      }
    } else if (error.response && error.response.status === 400) {
      setError('Image rejected - See detailed explanation below');
      // Create a generic explanation when specific details aren't available
      setDetailedExplanation({
        deepfake: {
          detected: true,
          confidence: 0.65,
          details: 'Our AI system detected potential manipulation in this image that violates our community guidelines.',
          common_indicators: ['The image contains patterns consistent with synthetic or manipulated content']
        }
      });
    } else if (error.response && error.response.status === 500) {
      setError('Server error');
      setDetailedExplanation({
        technical_issue: {
          details: 'Our AI image detection system is currently unavailable. For safety reasons, we cannot accept uploads at this time. Please try again later.'
        }
      });
    } else {
      setError('Upload failed');
      setDetailedExplanation({
        technical_issue: {
          details: 'We encountered an unexpected error while processing your image. Please try again later or contact support if the problem persists.'
        }
      });
    }
  };
  
  // Handle file selection from input
  const handleFileChange = (e) => {
    const selectedFiles = Array.from(e.target.files);
    selectedFiles.forEach(processFile);
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
    
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      const droppedFiles = Array.from(e.dataTransfer.files);
      droppedFiles.forEach(processFile);
    }
  };
  
  // Process the selected file
  const processFile = (selectedFile) => {
    setError('');
    setScanningStage('');
    setDetailedExplanation(null);
    setVisualAreas(null);
    setShowFilters(false);
    
    if (!selectedFile) {
      return;
    }
    
    // Check file size
    if (selectedFile.size > MAX_FILE_SIZE) {
      setError(`File size exceeds the maximum limit of ${MAX_FILE_SIZE / (1024 * 1024)}MB`);
      return;
    }
    
    // Check file type
    if (!ALLOWED_TYPES.includes(selectedFile.type)) {
      setError('Please select an image file (JPEG, PNG, or GIF)');
      return;
    }
    
    // Create a unique ID for this file
    const fileId = Date.now() + '-' + Math.random().toString(36).substr(2, 9);
    
    // Create preview
    const reader = new FileReader();
    reader.onload = () => {
      // Add to files array
      setFiles(prevFiles => [...prevFiles, selectedFile]);
      
      // Add to previews array
      setPreviews(prevPreviews => [...prevPreviews, {
        id: fileId,
        file: selectedFile,
        preview: reader.result,
        size: selectedFile.size,
        type: selectedFile.type,
        filter: null
      }]);
      
      // Add to filesInfo array
      setFilesInfo(prevFilesInfo => [...prevFilesInfo, {
        id: fileId,
        size: selectedFile.size,
        type: selectedFile.type
      }]);
    };
    reader.readAsDataURL(selectedFile);
  };
  
  // Apply image filter to a specific image
  const applyFilter = (fileId, filterName) => {
    setPreviews(prevPreviews => 
      prevPreviews.map(item => 
        item.id === fileId ? { ...item, filter: filterName } : item
      )
    );
  };
  
  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    setSuccess(false);
    setScanningStage('starting');
    setDetailedExplanation(null);
    setVisualAreas(null);
    setUploadProgress(0);
    
    if (files.length === 0) {
      setError('Please select at least one image to upload');
      setScanningStage('');
      return;
    }
    
    setLoading(true);
    
    try {
      const token = localStorage.getItem('token');
      
      // Create an array to store all upload promises
      const uploadPromises = [];
      
      // Update scanning stage for UI feedback
      setScanningStage('uploading');
      
      // Simulate upload progress
      const progressInterval = setInterval(() => {
        setUploadProgress(prev => {
          if (prev >= 95) {
            clearInterval(progressInterval);
            return 95;
          }
          return prev + 5;
        });
      }, 200);
      
      // Process each file
      for (const fileItem of previews) {
        const formData = new FormData();
        formData.append('file', fileItem.file);
        
        // Add caption if available for this file
        if (captions[fileItem.id]) {
          formData.append('caption', captions[fileItem.id]);
        }
        
        // If we have applied filters, add that information
        if (fileItem.filter) {
          formData.append('filter', fileItem.filter);
        }
        
        // Create upload promise with individual file tracking
        const uploadPromise = axios.post('http://localhost:8000/upload/', formData, {
          headers: {
            'Content-Type': 'multipart/form-data',
            Authorization: `Bearer ${token}`
          },
          onUploadProgress: (progressEvent) => {
            const percentCompleted = Math.round((progressEvent.loaded * 100) / progressEvent.total);
            setUploadProgress(percentCompleted > 95 ? 95 : percentCompleted);
          }
        })
        .then(response => {
          return { id: fileItem.id, success: true, response };
        })
        .catch(error => {
          return { id: fileItem.id, success: false, error, file: fileItem };
        });
        
        uploadPromises.push(uploadPromise);
      }
      
      // Wait for all uploads to complete and process results
      const results = await Promise.all(uploadPromises);
      
      // Clear the progress interval if it's still running
      clearInterval(progressInterval);
      setUploadProgress(100);
      
      // Process results to identify rejected images
      const rejected = results.filter(result => !result.success).map(result => result.file);
      setRejectedImages(rejected);
      
      if (rejected.length > 0) {
        // Some images were rejected
        setScanningStage('partial');
        setError(`${rejected.length} image${rejected.length > 1 ? 's were' : ' was'} rejected - See details below`);
        
        // Process the first rejection for detailed explanation
        const firstRejection = results.find(result => !result.success);
        if (firstRejection && firstRejection.error) {
          processErrorResponse(firstRejection.error);
        }
        
        // Remove rejected images from previews but keep successful ones
        const successfulImageIds = results.filter(result => result.success).map(result => result.id);
        setPreviews(prev => prev.filter(item => successfulImageIds.includes(item.id)));
      } else {
        // All images were successful
        setScanningStage('complete');
        setSuccess(true);
        
        // Reset form and explanations
        setFiles([]);
        setCaptions({});
        setPreviews([]);
        setFilesInfo([]);
        setDetailedExplanation(null);
        setVisualAreas(null);
        
        // Only redirect to home if all uploads were successful
        setTimeout(() => {
          navigate('/');
        }, 2000);
      }
    } catch (error) {
      console.error('Upload error:', error);
      setScanningStage('failed');
      processErrorResponse(error);
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
        return `Uploading image... ${uploadProgress}%`;
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
  
  // Effect to update scanning stage based on upload progress
  useEffect(() => {
    if (uploadProgress >= 95 && scanningStage === 'uploading') {
      setScanningStage('analyzing');
    }
  }, [uploadProgress, scanningStage]);
  
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
      
      {/* Detailed AI Explanation Panel */}
      {detailedExplanation && (
        <div className="bg-yellow-50 border-l-4 border-yellow-500 p-4 mb-6 rounded-lg shadow-md">
          <h3 className="text-lg font-semibold text-yellow-800 mb-2">Why This Image Was Rejected</h3>
          
          {/* Deepfake Explanation */}
          {detailedExplanation.deepfake && detailedExplanation.deepfake.detected && (
            <div className="mb-4">
              <h4 className="font-bold text-yellow-700 mb-1">Deepfake Detection</h4>
              <div className="flex items-center mb-2">
                <div className="w-full bg-gray-200 rounded-full h-2.5">
                  <div 
                    className="bg-red-600 h-2.5 rounded-full" 
                    style={{ width: `${Math.min(detailedExplanation.deepfake.confidence * 100, 100)}%` }}
                  ></div>
                </div>
                <span className="ml-2 text-sm font-medium text-gray-700">
                  {(detailedExplanation.deepfake.confidence * 100).toFixed(1)}% confidence
                </span>
              </div>
              <p className="text-sm text-gray-700 mb-2">{detailedExplanation.deepfake.details || 'Our AI system detected signs of facial manipulation in this image.'}</p>
              
              {detailedExplanation.deepfake.common_indicators && detailedExplanation.deepfake.common_indicators.length > 0 && (
                <div className="mt-2">
                  <p className="text-sm font-medium text-gray-700">Detected indicators:</p>
                  <ul className="list-disc ml-5 mt-1 text-sm text-gray-600">
                    {detailedExplanation.deepfake.common_indicators.map((indicator, index) => (
                      <li key={index}>{indicator}</li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          )}
          
          {/* NSFW Explanation */}
          {detailedExplanation.nsfw && detailedExplanation.nsfw.detected && (
            <div>
              <h4 className="font-bold text-yellow-700 mb-1">Inappropriate Content Detection</h4>
              <div className="flex items-center mb-2">
                <div className="w-full bg-gray-200 rounded-full h-2.5">
                  <div 
                    className="bg-red-600 h-2.5 rounded-full" 
                    style={{ width: `${Math.min(detailedExplanation.nsfw.confidence * 100, 100)}%` }}
                  ></div>
                </div>
                <span className="ml-2 text-sm font-medium text-gray-700">
                  {(detailedExplanation.nsfw.confidence * 100).toFixed(1)}% confidence
                </span>
              </div>
              <p className="text-sm text-gray-700 mb-2">{detailedExplanation.nsfw.details || 'Our AI system detected inappropriate content in this image that violates our community guidelines.'}</p>
              
              {detailedExplanation.nsfw.content_type && (
                <p className="text-sm text-gray-600">
                  <span className="font-medium">Content type:</span> {detailedExplanation.nsfw.content_type}
                </p>
              )}
            </div>
          )}
          
          {/* Technical Issues Explanation */}
          {detailedExplanation.technical_issue && (
            <div className="mb-4">
              <h4 className="font-bold text-yellow-700 mb-1">Technical Issue</h4>
              <p className="text-sm text-gray-700">{detailedExplanation.technical_issue.details || 'Our system encountered a technical issue while processing your image.'}</p>
            </div>
          )}
          
          {/* Visual Areas - Shows areas of concern in the image */}
          {visualAreas && visualAreas.length > 0 && previews.length > 0 && (
            <div className="mt-4">
              <p className="text-sm font-medium text-gray-700">Areas of concern detected in image</p>
              <div className="relative mt-2 border border-gray-300 rounded-md overflow-hidden">
                {previews[0] && (
                  <img src={previews[0].preview} alt="Preview with issues" className="w-full max-h-64 object-contain" />
                )}
                {visualAreas.map((area, index) => (
                  <div 
                    key={index}
                    className="absolute border-2 border-red-500 bg-red-200 bg-opacity-30"
                    style={{
                      left: `${area.x * 100}%`,
                      top: `${area.y * 100}%`,
                      width: `${area.width * 100}%`,
                      height: `${area.height * 100}%`
                    }}
                  ></div>
                ))}
              </div>
            </div>
          )}
          
          <div className="mt-4 text-sm text-gray-600">
            <p className="font-medium">What to do next:</p>
            <ul className="list-disc ml-5 mt-1">
              <li>Try uploading a different image that meets our community guidelines</li>
              <li>If you believe this is a mistake, please contact our support team</li>
              <li>Review our <a href="#" className="text-blue-600 hover:underline">content policy</a> for more information</li>
            </ul>
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
              multiple
              onChange={handleFileChange}
              disabled={loading}
              ref={fileInputRef}
            />
            <div 
              onDragEnter={handleDrag}
              onDragLeave={handleDrag}
              onDragOver={handleDrag}
              onDrop={handleDrop}
              className={`transition-all duration-300 ${dragActive ? 'border-pink-500 bg-pink-50' : ''}`}
            >
              {previews.length === 0 ? (
                <div onClick={() => fileInputRef.current.click()} className="cursor-pointer">
                  <svg xmlns="http://www.w3.org/2000/svg" className="mx-auto h-12 w-12 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                  </svg>
                  <p className="mt-1 text-sm text-gray-600">Click to browse or drag and drop</p>
                  <p className="mt-1 text-xs text-gray-500">PNG, JPG, GIF up to 5MB</p>
                  <p className="mt-1 text-xs font-semibold text-pink-500">Multiple files supported!</p>
                  {dragActive && (
                    <div className="absolute inset-0 bg-pink-100 bg-opacity-50 flex items-center justify-center rounded-lg">
                      <p className="text-lg font-medium text-pink-600">Drop images here</p>
                    </div>
                  )}
                </div>
              ) : (
                <div>
                  <div className="flex justify-between items-center mb-3">
                    <h3 className="text-sm font-medium text-gray-700">{previews.length} image{previews.length !== 1 ? 's' : ''} selected</h3>
                    <button 
                      onClick={() => fileInputRef.current.click()} 
                      className="text-xs bg-pink-500 text-white px-2 py-1 rounded hover:bg-pink-600"
                    >
                      Add More
                    </button>
                  </div>
                  
                  <div className="grid grid-cols-2 sm:grid-cols-3 gap-3 max-h-80 overflow-y-auto p-2">
                    {previews.map((item) => (
                      <div key={item.id} className="relative group">
                        <img
                          src={item.preview}
                          alt={`Preview ${item.id}`}
                          className="w-full h-32 object-cover rounded border border-gray-200"
                          style={{
                            filter: item.filter === 'grayscale' ? 'grayscale(100%)' : 
                                  item.filter === 'sepia' ? 'sepia(100%)' : 
                                  item.filter === 'blur' ? 'blur(2px)' : 
                                  item.filter === 'brightness' ? 'brightness(150%)' : 
                                  item.filter === 'contrast' ? 'contrast(150%)' : ''
                          }}
                        />
                        <div className="absolute top-1 right-1 flex space-x-1 opacity-0 group-hover:opacity-100 transition-opacity">
                          {/* Filter button */}
                          <button
                            type="button"
                            onClick={() => setShowFilters(item.id)}
                            className="bg-blue-500 text-white rounded-full p-1 hover:bg-blue-600 focus:outline-none"
                            title="Apply filters"
                          >
                            <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" viewBox="0 0 20 20" fill="currentColor">
                              <path fillRule="evenodd" d="M3 3a1 1 0 011-1h12a1 1 0 011 1v3a1 1 0 01-.293.707L12 11.414V15a1 1 0 01-.293.707l-2 2A1 1 0 018 17v-5.586L3.293 6.707A1 1 0 013 6V3z" clipRule="evenodd" />
                            </svg>
                          </button>
                          
                          {/* Remove button */}
                          <button
                            type="button"
                            onClick={() => {
                              setPreviews(prevPreviews => prevPreviews.filter(p => p.id !== item.id));
                              setFiles(prevFiles => prevFiles.filter((_, index) => 
                                prevFiles[index] !== item.file
                              ));
                              setCaptions(prevCaptions => {
                                const newCaptions = {...prevCaptions};
                                delete newCaptions[item.id];
                                return newCaptions;
                              });
                            }}
                            className="bg-red-500 text-white rounded-full p-1 hover:bg-red-600 focus:outline-none"
                            title="Remove image"
                          >
                            <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" viewBox="0 0 20 20" fill="currentColor">
                              <path fillRule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clipRule="evenodd" />
                            </svg>
                          </button>
                        </div>
                        
                        {/* Caption input for each image */}
                        <input 
                          type="text" 
                          placeholder="Add caption..."
                          className="mt-1 w-full text-xs p-1 border border-gray-200 rounded"
                          value={captions[item.id] || ''}
                          onChange={(e) => setCaptions(prev => ({...prev, [item.id]: e.target.value}))}
                        />
                        
                        {/* Filter options */}
                        {showFilters === item.id && (
                          <div className="absolute left-1 top-1 bg-white p-2 rounded-lg shadow-md z-10">
                            <div className="flex flex-col space-y-1">
                              <button 
                                onClick={() => applyFilter(item.id, null)} 
                                className={`px-2 py-1 text-xs rounded ${!item.filter ? 'bg-blue-500 text-white' : 'bg-gray-200 hover:bg-gray-300'}`}
                              >
                                Normal
                              </button>
                              <button 
                                onClick={() => applyFilter(item.id, 'grayscale')} 
                                className={`px-2 py-1 text-xs rounded ${item.filter === 'grayscale' ? 'bg-blue-500 text-white' : 'bg-gray-200 hover:bg-gray-300'}`}
                              >
                                Grayscale
                              </button>
                              <button 
                                onClick={() => applyFilter(item.id, 'sepia')} 
                                className={`px-2 py-1 text-xs rounded ${item.filter === 'sepia' ? 'bg-blue-500 text-white' : 'bg-gray-200 hover:bg-gray-300'}`}
                              >
                                Sepia
                              </button>
                              <button 
                                onClick={() => applyFilter(item.id, 'brightness')} 
                                className={`px-2 py-1 text-xs rounded ${item.filter === 'brightness' ? 'bg-blue-500 text-white' : 'bg-gray-200 hover:bg-gray-300'}`}
                              >
                                Brighten
                              </button>
                              <button 
                                onClick={() => applyFilter(item.id, 'contrast')} 
                                className={`px-2 py-1 text-xs rounded ${item.filter === 'contrast' ? 'bg-blue-500 text-white' : 'bg-gray-200 hover:bg-gray-300'}`}
                              >
                                Contrast
                              </button>
                            </div>
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>
          
          {filesInfo.length > 0 && (
            <div className="mt-2 text-sm text-gray-600">
              Total files: {files.length} • Total size: {formatFileSize(files.reduce((total, file) => total + file.size, 0))}
            </div>
          )}
        </div>
        
        {/* Display rejected images if any */}
        {rejectedImages.length > 0 && (
          <div className="mb-6 mt-4">
            <h3 className="text-lg font-semibold text-red-600 mb-2">Rejected Images</h3>
            <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-4">
              {rejectedImages.map((item, index) => (
                <div key={index} className="relative border border-red-300 rounded-lg overflow-hidden bg-red-50">
                  <div className="relative pt-[100%] overflow-hidden">
                    <img 
                      src={item.preview} 
                      alt={`Rejected image ${index + 1}`} 
                      className="absolute inset-0 w-full h-full object-cover"
                      style={{
                        filter: item.filter === 'grayscale' ? 'grayscale(100%)' :
                                item.filter === 'sepia' ? 'sepia(100%)' :
                                item.filter === 'brightness' ? 'brightness(130%)' :
                                item.filter === 'contrast' ? 'contrast(150%)' : 'none'
                      }}
                    />
                    <div className="absolute inset-0 bg-red-900 bg-opacity-30 flex items-center justify-center">
                      <span className="text-white font-bold text-sm px-2 py-1 bg-red-600 rounded-full">Rejected</span>
                    </div>
                  </div>
                  <div className="p-2 text-xs text-gray-700">
                    <p className="truncate">{item.file.name}</p>
                    <p>{formatFileSize(item.size)}</p>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
        
        {/* Caption inputs are now per image */}
        
        {scanningStage && (
          <div className="mb-6">
            <div className="relative pt-1">
              <div className="flex mb-2 items-center justify-between">
                <div>
                  <span className="text-xs font-semibold inline-block py-1 px-2 uppercase rounded-full text-pink-600 bg-pink-200">
                    {getScanningStageText()}
                  </span>
                </div>
                {scanningStage === 'uploading' && (
                  <div className="text-xs text-gray-500">
                    {uploadProgress}%
                  </div>
                )}
              </div>
              <div className="overflow-hidden h-2 mb-4 text-xs flex rounded bg-pink-200">
                {scanningStage === 'uploading' ? (
                  <div 
                    className="shadow-none flex flex-col text-center whitespace-nowrap text-white justify-center bg-pink-500 transition-all duration-300" 
                    style={{ width: `${uploadProgress}%` }}
                  ></div>
                ) : (
                  <div className={`shadow-none flex flex-col text-center whitespace-nowrap text-white justify-center bg-pink-500 ${scanningStage === 'complete' ? 'w-full' : scanningStage === 'failed' ? 'w-full bg-red-500' : 'w-3/4 animate-pulse'}`}></div>
                )}
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
            disabled={loading || files.length === 0}
          >
            {loading ? 'Processing...' : `Upload ${files.length} Image${files.length !== 1 ? 's' : ''}`}
          </button>
        </div>
      </form>
    </div>
  );
}

export default Upload;