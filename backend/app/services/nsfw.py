import cv2
import os
import numpy as np
import logging
import time

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='nsfw_detection.log'
)
logger = logging.getLogger("NSFWDetector")

# Try to import TensorFlow, but provide fallback if not available
try:
    import tensorflow as tf
    # Check if tf.keras is available
    try:
        from tensorflow.keras.preprocessing import image as tf_image
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
        TENSORFLOW_AVAILABLE = True
        logger.info("TensorFlow successfully loaded")
    except ImportError:
        # Alternative import paths for different TensorFlow versions
        try:
            from keras.preprocessing import image as tf_image
            from keras.applications.mobilenet_v2 import preprocess_input
            TENSORFLOW_AVAILABLE = True
            logger.info("Keras successfully loaded as alternative")
        except ImportError:
            TENSORFLOW_AVAILABLE = False
            logger.warning("TensorFlow keras modules not properly installed. Using fallback NSFW detection.")
except ImportError:
    TENSORFLOW_AVAILABLE = False
    logger.warning("TensorFlow not available. Using fallback NSFW detection.")

class NSFWDetector:
    def __init__(self, model_path=None):
        """
        Initialize the NSFW content detector with a pre-trained model.
        If model_path is None or TensorFlow is not available, use a placeholder implementation.
        """
        self.model = None
        
        # Default model path if none provided
        if model_path is None:
            self.model_path = os.path.join(os.path.dirname(__file__), 'models', 'nsfw_model.h5')
            logger.info(f"Using default model path: {self.model_path}")
        else:
            self.model_path = model_path
        
        # Initialize model if TensorFlow is available and path exists
        if TENSORFLOW_AVAILABLE:
            try:
                if os.path.exists(self.model_path):
                    self.model = tf.keras.models.load_model(self.model_path)
                    logger.info(f"Loaded NSFW detection model from {self.model_path}")
                else:
                    logger.warning(f"Model file not found at: {self.model_path}. Using fallback detection.")
            except Exception as e:
                logger.error(f"Error loading model: {str(e)}")
                logger.warning("Using placeholder implementation instead")
    
    def detect(self, image_path, threshold=0.5):
        """
        Detect if an image contains NSFW content.
        
        Args:
            image_path: Path to the image file
            threshold: Confidence threshold for classification
            
        Returns:
            bool: True if NSFW content is detected, False otherwise
        """
        start_time = time.time()
        logger.info(f"Checking image for NSFW content: {image_path}")
        
        # If no model and no TensorFlow, reject by default for safety
        if not TENSORFLOW_AVAILABLE or self.model is None:
            logger.warning("No NSFW detection model available - using fallback detection")
            return self._alternative_detection(image_path)
            
        try:
            # Load image and check if it's valid
            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"Failed to load image: {image_path}")
                return True  # Reject on error to be safe
            
            # Resize to model input size (assuming MobileNetV2 input shape)
            resized_image = cv2.resize(image, (224, 224))
            
            # Preprocess for model
            img_array = tf_image.img_to_array(resized_image)
            img_array = preprocess_input(img_array)
            img_array = np.expand_dims(img_array, axis=0)
            
            # Predict
            prediction = self.model.predict(img_array)
            
            # Assuming binary classification (SFW vs NSFW)
            # For multi-class, we'd need to adjust this logic
            nsfw_probability = prediction[0][0]
            
            logger.info(f"NSFW probability: {nsfw_probability:.4f}")
            
            # If probability > threshold, classify as NSFW
            if nsfw_probability > threshold:
                logger.warning(f"NSFW content detected with confidence: {nsfw_probability:.4f}")
                return True
                
            return False
            
        except Exception as e:
            logger.error(f"Error in NSFW detection: {str(e)}")
            return True  # On error, reject the image to be safe
        finally:
            processing_time = time.time() - start_time
            logger.info(f"NSFW detection processing time: {processing_time:.2f} seconds")
            
    def _alternative_detection(self, image_path):
        """
        A simple alternative detection method based on image properties
        Used as a fallback when main detection isn't available
        """
        try:
            # Simple heuristics that might indicate NSFW content
            # This is very basic and should be replaced with a real model
            image = cv2.imread(image_path)
            if image is None:
                return True  # Reject
                
            # Check for skin tone percentage - high percentage might indicate NSFW
            # This is just a placeholder - not a real detection method
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            skin_lower = np.array([0, 20, 70], np.uint8)
            skin_upper = np.array([20, 150, 255], np.uint8)
            skin_mask = cv2.inRange(hsv, skin_lower, skin_upper)
            
            skin_percentage = (np.sum(skin_mask > 0) / (skin_mask.shape[0] * skin_mask.shape[1])) * 100
            
            # Unusually high percentage of skin tones might be suspicious
            if skin_percentage > 60:
                logger.warning(f"High skin percentage detected: {skin_percentage:.2f}%")
                return True
                
            return False
            
        except Exception as e:
            logger.error(f"Error in alternative detection: {str(e)}")
            return True  # Reject on error