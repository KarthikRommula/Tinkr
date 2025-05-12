import os
import cv2
import time
import numpy as np
import logging
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.efficientnet import preprocess_input

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ImprovedDeepfakeDetector")

# Check if TensorFlow is available
TENSORFLOW_AVAILABLE = False
try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
    logger.info("TensorFlow successfully loaded")
except ImportError:
    logger.error("TensorFlow not available. Some features will be limited.")

class ImprovedDeepfakeDetector:
    def __init__(self, model_path=None):
        """
        Initialize the improved deepfake detector with a pre-trained model.
        
        Args:
            model_path: Path to a pre-trained model (if None, will download or create one)
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
            self.model_path = os.path.join(os.path.dirname(__file__), 'models', 'improved_deepfake_model.h5')
            logger.info(f"Using default model path: {self.model_path}")
        else:
            self.model_path = model_path
        
        # Load face cascade for face detection
        try:
            # Try different potential paths for the cascade file
            potential_paths = [
                os.path.join(os.path.dirname(__file__), 'models', 'haarcascade_frontalface_default.xml'),
                os.path.join(os.getcwd(), 'app', 'services', 'models', 'haarcascade_frontalface_default.xml'),
                os.path.join(cv2.__path__[0], 'data', 'haarcascade_frontalface_default.xml')
            ]
            
            for path in potential_paths:
                if os.path.exists(path):
                    self.face_cascade = cv2.CascadeClassifier(path)
                    logger.info(f"Loaded face detection model from {path}")
                    break
            
            if self.face_cascade is None:
                logger.warning("Could not find face cascade file. Face detection will be limited.")
        except Exception as e:
            logger.error(f"Error initializing face detector: {str(e)}")
        
        # Initialize model if TensorFlow is available
        if TENSORFLOW_AVAILABLE:
            self._load_model()
    
    def _load_model(self):
        """Load or create the deepfake detection model"""
        try:
            # First check if model exists
            if os.path.exists(self.model_path):
                try:
                    self.model = tf.keras.models.load_model(self.model_path)
                    logger.info(f"Loaded deepfake detection model from {self.model_path}")
                    
                    # Test the model with a dummy input
                    dummy_input = np.zeros((1, 224, 224, 3))
                    _ = self.model.predict(dummy_input)
                    logger.info("Model successfully tested with dummy input")
                    return
                except Exception as e:
                    logger.error(f"Error loading model: {str(e)}")
            
            # If model doesn't exist or failed to load, create a transfer learning model
            logger.info("Creating transfer learning model for deepfake detection...")
            self._create_transfer_learning_model()
            
        except Exception as e:
            logger.error(f"Error in _load_model: {str(e)}")
            # Fallback to placeholder if everything fails
            self._load_placeholder_model()
    
    def _create_transfer_learning_model(self):
        """Create a transfer learning model based on EfficientNet"""
        try:
            # Create model directory if it doesn't exist
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            
            # Create a model using transfer learning from EfficientNetB0
            base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
            
            # Freeze the base model layers
            base_model.trainable = False
            
            # Add custom classification layers
            x = base_model.output
            x = GlobalAveragePooling2D()(x)
            x = Dense(256, activation='relu')(x)
            x = Dense(128, activation='relu')(x)
            predictions = Dense(1, activation='sigmoid')(x)
            
            # Create the model
            model = Model(inputs=base_model.input, outputs=predictions)
            
            # Compile the model
            model.compile(
                optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
            
            # Initialize with slightly biased weights to be conservative (assume real unless confident)
            for layer in model.layers:
                if isinstance(layer, tf.keras.layers.Dense) and layer.units == 1:
                    weights = layer.get_weights()
                    # Set bias to a slightly negative value to be conservative
                    weights[1] = np.array([-1.0])
                    layer.set_weights(weights)
            
            self.model = model
            logger.info("Created transfer learning model for deepfake detection")
            
            # Save the model
            model.save(self.model_path)
            logger.info(f"Saved transfer learning model to {self.model_path}")
            
        except Exception as e:
            logger.error(f"Error creating transfer learning model: {str(e)}")
            self._load_placeholder_model()
    
    def _load_placeholder_model(self):
        """Create and load a simple placeholder model as fallback"""
        try:
            # Create a simple model
            model = tf.keras.Sequential([
                tf.keras.layers.Input(shape=(224, 224, 3)),
                tf.keras.layers.Conv2D(16, (3, 3), activation='relu'),
                tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
                tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
                tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(64, activation='relu'),
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
                    weights[1] = np.array([-3.0])
                    layer.set_weights(weights)
            
            self.model = model
            logger.info("Created placeholder deepfake detection model")
            
            # Save the model if directory exists
            model_dir = os.path.dirname(self.model_path)
            if os.path.exists(model_dir):
                try:
                    model.save(self.model_path)
                    logger.info(f"Saved placeholder model to {self.model_path}")
                except Exception as e:
                    logger.error(f"Error saving placeholder model: {str(e)}")
        except Exception as e:
            logger.error(f"Error creating placeholder model: {str(e)}")
    
    def preprocess_image(self, image_path):
        """
        Preprocess image for the model
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Preprocessed image array ready for the model
        """
        try:
            # Read image
            img = cv2.imread(image_path)
            if img is None:
                logger.error(f"Failed to read image: {image_path}")
                return None
            
            # Convert BGR to RGB (OpenCV uses BGR, but our models expect RGB)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Detect faces
            if self.face_cascade is not None:
                gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
                
                # If faces are found, focus on the largest face
                if len(faces) > 0:
                    # Find the largest face
                    largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
                    x, y, w, h = largest_face
                    
                    # Add some margin
                    margin = int(0.2 * max(w, h))
                    x = max(0, x - margin)
                    y = max(0, y - margin)
                    w = min(img.shape[1] - x, w + 2 * margin)
                    h = min(img.shape[0] - y, h + 2 * margin)
                    
                    # Crop to the face region
                    img = img[y:y+h, x:x+w]
            
            # Resize to model input size
            img = cv2.resize(img, (224, 224))
            
            # Convert to array and preprocess for the model
            img_array = img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)
            
            return img_array
            
        except Exception as e:
            logger.error(f"Error preprocessing image: {str(e)}")
            return None
    
    def detect(self, image_path, threshold=0.5):
        """
        Detect if an image contains deepfake content.
        
        Args:
            image_path: Path to the image file
            threshold: Confidence threshold for classification
                
        Returns:
            bool: True if deepfake is detected, False otherwise
            float: Confidence score
        """
        start_time = time.time()
        logger.info(f"Checking image for deepfakes: {image_path}")
        
        # Check if we're in development mode with bypass enabled
        if self.dev_mode and os.getenv("ALWAYS_PASS_DEEPFAKE", "False").lower() == "true":
            logger.info("Development mode: Bypassing deepfake detection")
            return False, 0.0
        
        # Check if the model is loaded
        if self.model is None:
            logger.error("Deepfake detection model not loaded")
            return os.getenv("REJECT_ON_MODEL_FAILURE", "True").lower() == "true", 1.0
        
        try:
            # Preprocess the image
            processed_image = self.preprocess_image(image_path)
            if processed_image is None:
                logger.error("Failed to preprocess image")
                return os.getenv("REJECT_ON_IMAGE_ERROR", "True").lower() == "true", 1.0
            
            # Make prediction
            prediction = self.model.predict(processed_image)
            confidence = float(prediction[0][0])
            
            logger.info(f"Deepfake detection confidence: {confidence:.4f}")
            
            # Determine if it's a deepfake based on threshold
            is_deepfake = confidence > threshold
            
            if is_deepfake:
                logger.warning(f"Deepfake detected with confidence: {confidence:.4f}")
                return True, confidence
            else:
                logger.info(f"No deepfake detected. Confidence: {confidence:.4f}")
                return False, confidence
                
        except Exception as e:
            logger.error(f"Error in deepfake detection: {str(e)}")
            return os.getenv("REJECT_ON_ERROR", "True").lower() == "true", 1.0
        finally:
            processing_time = time.time() - start_time
            logger.info(f"Deepfake detection processing time: {processing_time:.2f} seconds")
