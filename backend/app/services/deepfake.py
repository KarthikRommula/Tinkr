import cv2
import os
import numpy as np
import logging
import time

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='deepfake_detection.log'
)
logger = logging.getLogger("DeepfakeDetector")

# Try to import TensorFlow, but provide fallback if not available
try:
    import tensorflow as tf
    from tensorflow.keras.preprocessing.image import img_to_array
    from tensorflow.keras.applications.xception import preprocess_input
    TENSORFLOW_AVAILABLE = True
    logger.info("TensorFlow successfully loaded")
except ImportError:
    TENSORFLOW_AVAILABLE = False
    logger.warning("TensorFlow not available. Using fallback deepfake detection.")

class DeepfakeDetector:
    def __init__(self, model_path=None):
        """
        Initialize the deepfake detector with a pre-trained model.
        If model_path is None or TensorFlow is not available, use a placeholder implementation.
        """
        self.model = None
        self.model_path = model_path
        self.face_cascade = None
        
        # Default model path if none provided
        if model_path is None:
            self.model_path = os.path.join(os.path.dirname(__file__), 'models', 'deepfake_model.h5')
            logger.info(f"Using default model path: {self.model_path}")
        
        # Initialize face detection with OpenCV
        try:
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            if os.path.exists(cascade_path):
                self.face_cascade = cv2.CascadeClassifier(cascade_path)
                logger.info(f"Loaded face detection model from {cascade_path}")
            else:
                logger.error(f"Face cascade file not found at: {cascade_path}")
        except Exception as e:
            logger.error(f"Error initializing face detector: {str(e)}")
        
        # Initialize model if TensorFlow is available and path exists
        if TENSORFLOW_AVAILABLE:
            try:
                if os.path.exists(self.model_path):
                    self.model = tf.keras.models.load_model(self.model_path)
                    logger.info(f"Loaded deepfake detection model from {self.model_path}")
                else:
                    logger.warning(f"Model file not found at: {self.model_path}. Using fallback detection.")
            except Exception as e:
                logger.error(f"Error loading model: {str(e)}")
                logger.warning("Using placeholder implementation instead")
    
    def detect(self, image_path, threshold=0.5):
        """
        Detect if an image contains deepfake content.
        
        Args:
            image_path: Path to the image file
            threshold: Confidence threshold for classification
            
        Returns:
            bool: True if deepfake is detected, False otherwise
        """
        start_time = time.time()
        logger.info(f"Checking image for deepfakes: {image_path}")
        
        # If no model and no TensorFlow, reject by default for safety
        if not TENSORFLOW_AVAILABLE or self.model is None:
            logger.warning("No deepfake detection model available - rejecting for safety")
            return True  # Reject the image to be safe
            
        # Load image and check if it's valid
        try:
            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"Failed to load image: {image_path}")
                return True  # Reject on error to be safe
            
            has_faces = False
            faces = []
            
            # Detect faces with OpenCV
            if self.face_cascade:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
                
                if len(faces) > 0:
                    has_faces = True
                    logger.info(f"Detected {len(faces)} faces in the image")
                else:
                    logger.info("No faces detected in the image")
            
            # If we have a model, check for deepfakes
            if TENSORFLOW_AVAILABLE and self.model:
                # If we found faces, check each one
                if has_faces:
                    max_confidence = 0.0
                    
                    for (x, y, w, h) in faces:
                        # Extract face region with some margin
                        face = image[max(0, y-40):min(image.shape[0], y+h+40), 
                                     max(0, x-40):min(image.shape[1], x+w+40)]
                        
                        if face.size == 0:
                            continue
                        
                        # Resize to model input size
                        face = cv2.resize(face, (299, 299))
                        
                        # Preprocess for model
                        face = img_to_array(face)
                        face = preprocess_input(face)
                        face = np.expand_dims(face, axis=0)
                        
                        # Predict
                        prediction = self.model.predict(face)
                        confidence = prediction[0][0]
                        max_confidence = max(max_confidence, confidence)
                        
                        logger.info(f"Face detection confidence: {confidence:.4f}")
                        
                        # Check if probability exceeds threshold
                        if confidence > threshold:
                            logger.warning(f"Deepfake detected with confidence: {confidence:.4f}")
                            return True
                    
                    logger.info(f"No deepfakes detected. Max confidence: {max_confidence:.4f}")
                    return False
                else:
                    # If no faces detected, process the whole image
                    resized_image = cv2.resize(image, (299, 299))
                    img_array = img_to_array(resized_image)
                    img_array = preprocess_input(img_array)
                    img_array = np.expand_dims(img_array, axis=0)
                    
                    # Predict
                    prediction = self.model.predict(img_array)
                    confidence = prediction[0][0]
                    
                    logger.info(f"Whole image deepfake confidence: {confidence:.4f}")
                    
                    # Check if probability exceeds threshold
                    if confidence > threshold:
                        logger.warning(f"Deepfake detected in image with confidence: {confidence:.4f}")
                        return True
                    
                    return False
            
            # If we reach here, we couldn't properly check the image
            logger.warning("Could not properly analyze image - rejecting for safety")
            return True  # Reject to be safe
            
        except Exception as e:
            logger.error(f"Error in deepfake detection: {str(e)}")
            return True  # On error, reject the image to be safe
        finally:
            processing_time = time.time() - start_time
            logger.info(f"Deepfake detection processing time: {processing_time:.2f} seconds")
            
    def _alternative_detection(self, image_path):
        """
        A simple alternative detection method based on image properties
        Used as a fallback when main detection isn't available
        """
        try:
            # Simple heuristics that might indicate manipulation
            # This is very basic and should be replaced with a real model
            image = cv2.imread(image_path)
            if image is None:
                return True  # Reject
                
            # Check for unusually perfect skin - might indicate deepfake
            # This is just a placeholder - not a real detection method
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            skin_lower = np.array([0, 20, 70], np.uint8)
            skin_upper = np.array([20, 150, 255], np.uint8)
            skin_mask = cv2.inRange(hsv, skin_lower, skin_upper)
            
            skin_percentage = (np.sum(skin_mask > 0) / (skin_mask.shape[0] * skin_mask.shape[1])) * 100
            
            # Unusually high percentage of skin tones might be suspicious
            if skin_percentage > 70:
                logger.warning(f"High skin percentage detected: {skin_percentage:.2f}%")
                return True
                
            return False
            
        except Exception as e:
            logger.error(f"Error in alternative detection: {str(e)}")
            return True  # Reject on error