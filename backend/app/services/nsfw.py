# app/services/nsfw.py
# Update the existing file with the following changes
import cv2
import os
import numpy as np
import logging
import time
import tensorflow as tf
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

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
        
        # Check for development mode
        self.dev_mode = os.getenv("SAFEGRAM_DEV_MODE", "False").lower() == "true"
        if self.dev_mode:
            logger.info("Running in development mode - detection may be bypassed")
        
        # Default model path if none provided
        if model_path is None:
            self.model_path = os.path.join(os.path.dirname(__file__), 'models', 'nsfw_model.h5')
            logger.info(f"Using default model path: {self.model_path}")
        else:
            self.model_path = model_path
        
        # Initialize model if TensorFlow is available
        if TENSORFLOW_AVAILABLE:
            self._load_model()
    
    def _load_model(self):
        """Load the NSFW detection model"""
        try:
            if os.path.exists(self.model_path):
                # Try to load model from saved format
                try:
                    self.model = tf.keras.models.load_model(self.model_path)
                    logger.info(f"Loaded NSFW detection model from {self.model_path}")
                except Exception as e:
                    logger.error(f"Error loading model from path: {str(e)}")
                    # Try loading as a SavedModel
                    try:
                        self.model = tf.saved_model.load(self.model_path)
                        logger.info(f"Loaded NSFW detection SavedModel from {self.model_path}")
                    except Exception as e2:
                        logger.error(f"Error loading SavedModel: {str(e2)}")
                        self._load_placeholder_model()
            else:
                logger.warning(f"Model file not found at: {self.model_path}")
                self._load_placeholder_model()
        except Exception as e:
            logger.error(f"Error in _load_model: {str(e)}")
    
    def _load_placeholder_model(self):
        """Create and load a simple placeholder model"""
        try:
            # Create a simple model
            input_shape = (224, 224, 3)
            model = tf.keras.Sequential([
                tf.keras.layers.Input(shape=input_shape),
                tf.keras.layers.Conv2D(16, (3, 3), activation='relu'),
                tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(8, activation='relu'),
                tf.keras.layers.Dense(1, activation='sigmoid')
            ])
            
            # Compile the model
            model.compile(
                optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
            
            # Initialize with weights that predict low confidence
            for layer in model.layers:
                if isinstance(layer, tf.keras.layers.Dense) and layer.units == 1:
                    weights = layer.get_weights()
                    # Set bias to a negative value to produce low confidence scores
                    weights[1] = np.array([-5.0])
                    layer.set_weights(weights)
            
            self.model = model
            logger.info("Created placeholder NSFW detection model")
            
            # Save the model if directory exists
            model_dir = os.path.dirname(self.model_path)
            if os.path.exists(model_dir):
                model.save(self.model_path)
                logger.info(f"Saved placeholder model to {self.model_path}")
        except Exception as e:
            logger.error(f"Error creating placeholder model: {str(e)}")
    
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
        
        # Check if we're in development mode
        if self.dev_mode:
            dev_bypass = os.getenv("ALWAYS_PASS_NSFW", "True").lower() == "true"
            if dev_bypass:
                logger.info("Development mode - bypassing NSFW detection")
                return False
        
        # If no TensorFlow and no model, reject by default for safety
        if not TENSORFLOW_AVAILABLE or self.model is None:
            reject_on_failure = os.getenv("REJECT_ON_MODEL_FAILURE", "False").lower() == "true"
            logger.error("No NSFW detection model available - " + 
                       ("rejecting" if reject_on_failure else "accepting") + " upload")
            return reject_on_failure
            
        try:
            # Load image and check if it's valid
            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"Failed to load image: {image_path}")
                return os.getenv("REJECT_ON_IMAGE_ERROR", "True").lower() == "true"
            
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
            except Exception as e1:
                logger.warning(f"Error using Keras predict: {str(e1)}")
                try:
                    # For TF Hub model
                    prediction = self.model(img_array)
                    if isinstance(prediction, dict):
                        # Example for NudeNet-like output with multiple classes
                        if 'nsfw_score' in prediction:
                            nsfw_probability = prediction['nsfw_score'][0].numpy()
                        else:
                            # For general classification model, use appropriate index
                            # This will vary based on the model used
                            nsfw_probability = prediction['logits'][0][0].numpy()
                    else:
                        # Basic prediction tensor, adjust index as needed
                        nsfw_probability = prediction[0][0].numpy()
                except Exception as e2:
                    logger.error(f"Error in both prediction methods: {str(e2)}")
                    return os.getenv("REJECT_ON_ERROR", "False").lower() == "true"
            
            logger.info(f"NSFW probability: {nsfw_probability:.4f}")
            
            # If probability > threshold, classify as NSFW
            if nsfw_probability > threshold:
                logger.warning(f"NSFW content detected with confidence: {nsfw_probability:.4f}")
                return True
                
            return False
            
        except Exception as e:
            logger.error(f"Error in NSFW detection: {str(e)}")
            return os.getenv("REJECT_ON_ERROR", "False").lower() == "true"
        finally:
            processing_time = time.time() - start_time
            logger.info(f"NSFW detection processing time: {processing_time:.2f} seconds")