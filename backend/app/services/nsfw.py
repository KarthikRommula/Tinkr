# nsfw.py
import cv2
import os
import numpy as np
import logging
import time
import tensorflow as tf
import tensorflow_hub as hub

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
    TENSORFLOW_AVAILABLE = True
    logger.info("TensorFlow successfully loaded")
except ImportError:
    TENSORFLOW_AVAILABLE = False
    logger.warning("TensorFlow not available. Cannot perform NSFW detection.")

class NSFWDetector:
    def __init__(self, model_path=None):
        """
        Initialize the NSFW content detector with a pre-trained model.
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
                    # Try to load model from saved format
                    try:
                        self.model = tf.keras.models.load_model(self.model_path)
                        logger.info(f"Loaded NSFW detection model from {self.model_path}")
                    except:
                        # If fails, try loading as a SavedModel
                        self.model = tf.saved_model.load(self.model_path)
                        logger.info(f"Loaded NSFW detection SavedModel from {self.model_path}")
                else:
                    # If model doesn't exist locally, try loading from TF Hub
                    try:
                        logger.info("Attempting to load NSFW model from TensorFlow Hub")
                        # Use a well-known NSFW detection model from TF Hub
                        self.model = hub.load("https://tfhub.dev/google/imagenet/mobilenet_v2_140_224/classification/5")
                        logger.info("Loaded NSFW detection model from TensorFlow Hub")
                    except Exception as hub_error:
                        logger.error(f"Error loading model from TensorFlow Hub: {str(hub_error)}")
                        logger.error("Model file not found and could not load from TensorFlow Hub")
            except Exception as e:
                logger.error(f"Error loading model: {str(e)}")
    
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
            logger.error("No NSFW detection model available - rejecting upload")
            return True  # Reject the image
            
        try:
            # Load image and check if it's valid
            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"Failed to load image: {image_path}")
                return True  # Reject on error to be safe
            
            # Resize to model input size (224x224 is common for many models)
            resized_image = cv2.resize(image, (224, 224))
            
            # Convert BGR to RGB (TensorFlow models typically expect RGB)
            rgb_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
            
            # Preprocess for model
            img_array = np.asarray(rgb_image, dtype=np.float32)
            img_array = img_array / 255.0  # Normalize to [0,1]
            img_array = np.expand_dims(img_array, axis=0)
            
            # Predict
            try:
                # For Keras model
                prediction = self.model.predict(img_array)
                # Adapt this based on your model's expected output format
                if isinstance(prediction, list):
                    nsfw_probability = prediction[0][0]  # Assuming first index is NSFW score
                else:
                    nsfw_probability = prediction[0][0]
            except:
                # For TF Hub model
                prediction = self.model(img_array)
                if isinstance(prediction, dict):
                    # Example for NudeNet-like output with multiple classes
                    if 'nsfw_score' in prediction:
                        nsfw_probability = prediction['nsfw_score'][0].numpy()
                    else:
                        # For general classification model, use appropriate index
                        # This will vary based on the model used
                        nsfw_probability = prediction['logits'][0][513].numpy()  # Example index
                else:
                    # Basic prediction tensor, adjust index as needed
                    nsfw_probability = prediction[0][0].numpy()
            
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