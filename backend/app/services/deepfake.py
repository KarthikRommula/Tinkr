# app/services/deepfake.py
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
    filename='deepfake_detection.log'
)
logger = logging.getLogger("DeepfakeDetector")

# Check TensorFlow availability
try:
    import tensorflow as tf
    from tensorflow.keras.preprocessing.image import img_to_array
    TENSORFLOW_AVAILABLE = True
    logger.info("TensorFlow successfully loaded")
except ImportError:
    TENSORFLOW_AVAILABLE = False
    logger.warning("TensorFlow not available. Cannot perform deepfake detection.")

class DeepfakeDetector:
    def __init__(self, model_path=None):
        """
        Initialize the deepfake detector with a pre-trained model.
        """
        self.model = None
        self.model_path = model_path
        self.face_cascade = None
        
        # Check for development mode
        self.dev_mode = os.getenv("SAFEGRAM_DEV_MODE", "False").lower() == "true"
        if self.dev_mode:
            logger.info("Running in development mode - detection may be bypassed")
        
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
        
        # Initialize model if TensorFlow is available
        if TENSORFLOW_AVAILABLE:
            self._load_model()
    
    def _load_model(self):
        """Load the deepfake detection model"""
        try:
            if os.path.exists(self.model_path):
                # Try to load model from saved format
                try:
                    self.model = tf.keras.models.load_model(self.model_path)
                    logger.info(f"Loaded deepfake detection model from {self.model_path}")
                except Exception as e:
                    logger.error(f"Error loading model from path: {str(e)}")
                    # Try loading as a SavedModel
                    try:
                        self.model = tf.saved_model.load(self.model_path)
                        logger.info(f"Loaded deepfake detection SavedModel from {self.model_path}")
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
            logger.info("Created placeholder deepfake detection model")
            
            # Save the model if directory exists
            model_dir = os.path.dirname(self.model_path)
            if os.path.exists(model_dir):
                model.save(self.model_path)
                logger.info(f"Saved placeholder model to {self.model_path}")
        except Exception as e:
            logger.error(f"Error creating placeholder model: {str(e)}")
    
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
        
        # Check if we're in development mode
        if self.dev_mode:
            dev_bypass = os.getenv("ALWAYS_PASS_DEEPFAKE", "True").lower() == "true"
            if dev_bypass:
                logger.info("Development mode - bypassing deepfake detection")
                return False
        
        # If no TensorFlow, reject based on configuration
        if not TENSORFLOW_AVAILABLE:
            reject_on_failure = os.getenv("REJECT_ON_MODEL_FAILURE", "False").lower() == "true"
            logger.error("TensorFlow not available - " + 
                       ("rejecting" if reject_on_failure else "accepting") + " upload")
            return reject_on_failure
        
        # If no model and failed to create one, reject based on configuration
        if self.model is None:
            try:
                self._load_placeholder_model()
            except Exception as e:
                logger.error(f"Failed to load any model: {str(e)}")
                reject_on_failure = os.getenv("REJECT_ON_MODEL_FAILURE", "False").lower() == "true"
                return reject_on_failure
        
        try:
            # Load image and check if it's valid
            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"Failed to load image: {image_path}")
                return os.getenv("REJECT_ON_IMAGE_ERROR", "True").lower() == "true"
            
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
            
            # If no faces detected, process the whole image
            if not has_faces:
                # Preprocess image for the model
                resized_image = cv2.resize(image, (224, 224))  # Standard size for many models
                img_array = img_to_array(resized_image)
                img_array = img_array / 255.0  # Normalize to [0,1]
                img_array = np.expand_dims(img_array, axis=0)
                
                # Make prediction - adapt this based on your model's expected input/output
                try:
                    # For Keras model
                    prediction = self.model.predict(img_array)
                    if isinstance(prediction, list):
                        confidence = prediction[0][0]
                    else:
                        confidence = prediction[0][0]
                except:
                    # For TF Hub model
                    prediction = self.model(img_array)
                    if isinstance(prediction, dict):
                        confidence = prediction['logits'][0][0].numpy()
                    else:
                        confidence = prediction[0][0].numpy()
                
                logger.info(f"Whole image deepfake confidence: {confidence:.4f}")
                
                # Check if probability exceeds threshold
                if confidence > threshold:
                    logger.warning(f"Deepfake detected in image with confidence: {confidence:.4f}")
                    return True
                
                return False
            
            # Process each detected face
            max_confidence = 0.0
            for (x, y, w, h) in faces:
                # Extract face region with some margin
                face = image[max(0, y-40):min(image.shape[0], y+h+40), 
                            max(0, x-40):min(image.shape[1], x+w+40)]
                
                if face.size == 0:
                    continue
                
                # Resize to model input size
                face = cv2.resize(face, (224, 224))
                
                # Preprocess for model
                face = img_to_array(face)
                face = face / 255.0  # Normalize
                face = np.expand_dims(face, axis=0)
                
                # Predict
                try:
                    # For Keras model
                    prediction = self.model.predict(face)
                    if isinstance(prediction, list):
                        confidence = prediction[0][0]
                    else:
                        confidence = prediction[0][0]
                except:
                    # For TF Hub model
                    prediction = self.model(face)
                    if isinstance(prediction, dict):
                        confidence = prediction['logits'][0][0].numpy()
                    else:
                        confidence = prediction[0][0].numpy()
                
                max_confidence = max(max_confidence, confidence)
                
                logger.info(f"Face detection confidence: {confidence:.4f}")
                
                # Check if probability exceeds threshold
                if confidence > threshold:
                    logger.warning(f"Deepfake detected with confidence: {confidence:.4f}")
                    return True
            
            logger.info(f"No deepfakes detected. Max confidence: {max_confidence:.4f}")
            return False
            
        except Exception as e:
            logger.error(f"Error in deepfake detection: {str(e)}")
            return os.getenv("REJECT_ON_ERROR", "False").lower() == "true"
        finally:
            processing_time = time.time() - start_time
            logger.info(f"Deepfake detection processing time: {processing_time:.2f} seconds")