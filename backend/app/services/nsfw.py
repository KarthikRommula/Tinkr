import cv2
import os
import numpy as np

# Try to import TensorFlow, but provide fallback if not available
try:
    import tensorflow as tf
    # Check if tf.keras is available (in some versions it might be structured differently)
    try:
        from tensorflow.keras.preprocessing import image as tf_image
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
        TENSORFLOW_AVAILABLE = True
    except ImportError:
        # Alternative import paths for different TensorFlow versions
        try:
            from keras.preprocessing import image as tf_image
            from keras.applications.mobilenet_v2 import preprocess_input
            TENSORFLOW_AVAILABLE = True
        except ImportError:
            TENSORFLOW_AVAILABLE = False
            print("TensorFlow keras modules not properly installed. Using fallback NSFW detection.")
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("TensorFlow not available. Using fallback NSFW detection.")

class NSFWDetector:
    def __init__(self, model_path=None):
        """
        Initialize the NSFW content detector with a pre-trained model.
        If model_path is None or TensorFlow is not available, use a placeholder implementation.
        """
        self.model = None
        self.model_path = model_path
        
        # Initialize model if TensorFlow is available and path is provided
        if TENSORFLOW_AVAILABLE and self.model_path and os.path.exists(self.model_path):
            try:
                self.model = tf.keras.models.load_model(self.model_path)
                print(f"Loaded NSFW detection model from {model_path}")
            except Exception as e:
                print(f"Error loading model: {str(e)}")
                print("Using placeholder implementation instead")
    
    def detect(self, image_path, threshold=0.5):
        """
        Detect if an image contains NSFW content.
        
        Args:
            image_path: Path to the image file
            threshold: Confidence threshold for classification
            
        Returns:
            bool: True if NSFW content is detected, False otherwise
        """
        print(f"Checking image for NSFW content: {image_path}")
        
        try:
            # Load image and check if it's valid
            image = cv2.imread(image_path)
            if image is None:
                print(f"Failed to load image: {image_path}")
                return False
            
            # If we have a TensorFlow model, use it for NSFW detection
            if TENSORFLOW_AVAILABLE and self.model:
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
                
                print(f"NSFW probability: {nsfw_probability}")
                
                # If probability > threshold, classify as NSFW
                return nsfw_probability > threshold
            
            # For prototype without a model, always return False (not NSFW)
            return False
            
        except Exception as e:
            print(f"Error in NSFW detection: {str(e)}")
            # On error, default to allowing the image
            return False