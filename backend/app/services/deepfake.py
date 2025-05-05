import cv2
import os
import numpy as np

# Try to import TensorFlow, but provide fallback if not available
try:
    import tensorflow as tf
    from tensorflow.keras.preprocessing.image import img_to_array
    from tensorflow.keras.applications.xception import preprocess_input
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("TensorFlow not available. Using fallback deepfake detection.")

class DeepfakeDetector:
    def __init__(self, model_path=None):
        """
        Initialize the deepfake detector with a pre-trained model.
        If model_path is None or TensorFlow is not available, use a placeholder implementation.
        """
        self.model = None
        self.model_path = model_path
        self.face_cascade = None
        
        # Initialize face detection with OpenCV
        try:
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            if os.path.exists(cascade_path):
                self.face_cascade = cv2.CascadeClassifier(cascade_path)
                print(f"Loaded face detection model")
            else:
                print(f"Face cascade file not found at: {cascade_path}")
        except Exception as e:
            print(f"Error initializing face detector: {str(e)}")
        
        # Initialize model if TensorFlow is available and path is provided
        if TENSORFLOW_AVAILABLE and self.model_path and os.path.exists(self.model_path):
            try:
                self.model = tf.keras.models.load_model(self.model_path)
                print(f"Loaded deepfake detection model from {model_path}")
            except Exception as e:
                print(f"Error loading model: {str(e)}")
                print("Using placeholder implementation instead")
    
    def detect(self, image_path, threshold=0.5):
        """
        Detect if an image contains deepfake content.
        
        Args:
            image_path: Path to the image file
            threshold: Confidence threshold for classification
            
        Returns:
            bool: True if deepfake is detected, False otherwise
        """
        print(f"Checking image for deepfakes: {image_path}")
        
        # Load image and check if it's valid
        try:
            image = cv2.imread(image_path)
            if image is None:
                print(f"Failed to load image: {image_path}")
                return False
                
            # Detect faces with OpenCV
            if self.face_cascade:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
                print(f"Detected {len(faces)} faces in the image")
                
                # If no faces detected, we can't check for deepfakes
                if len(faces) == 0:
                    print("No faces detected in the image")
                    return False
            
            # If we have a TensorFlow model, use it for deepfake detection
            if TENSORFLOW_AVAILABLE and self.model:
                for (x, y, w, h) in faces:
                    # Extract face region with some margin
                    face = image[max(0, y-40):min(image.shape[0], y+h+40), 
                                 max(0, x-40):min(image.shape[1], x+w+40)]
                    
                    if face.size == 0:
                        continue
                    
                    # Resize to model input size (assuming Xception input shape)
                    face = cv2.resize(face, (299, 299))
                    
                    # Preprocess for model
                    face = img_to_array(face)
                    face = preprocess_input(face)
                    face = np.expand_dims(face, axis=0)
                    
                    # Predict
                    prediction = self.model.predict(face)
                    
                    # Check if probability exceeds threshold
                    if prediction[0][0] > threshold:
                        print(f"Deepfake detected with confidence: {prediction[0][0]}")
                        return True
            
            # For prototype without a model, always return False (not a deepfake)
            return False
            
        except Exception as e:
            print(f"Error in deepfake detection: {str(e)}")
            # On error, default to allowing the image
            return False