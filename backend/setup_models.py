import os
import urllib.request
import zipfile
import logging
import tensorflow as tf
import numpy as np

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ModelSetup")

def setup_models():
    """Set up directory structure and create placeholder models if needed"""
    # Create model directories
    base_dir = os.path.join('app', 'services', 'models')
    os.makedirs(base_dir, exist_ok=True)
    
    logger.info(f"Created models directory at {base_dir}")
    
    # Check if TensorFlow is available
    try:
        import tensorflow as tf
        logger.info(f"TensorFlow version: {tf.__version__}")
        
        # Check Keras
        try:
            from tensorflow import keras
            logger.info(f"Keras is available")
            
            # Create placeholder models if they don't exist
            deepfake_model_path = os.path.join(base_dir, 'deepfake_model.h5')
            nsfw_model_path = os.path.join(base_dir, 'nsfw_model.h5')
            
            # Create placeholder deepfake model
            if not os.path.exists(deepfake_model_path):
                logger.info(f"Creating placeholder deepfake model at {deepfake_model_path}")
                
                # Simple model that always returns 0 (not a deepfake)
                model = tf.keras.Sequential([
                    tf.keras.layers.Input(shape=(299, 299, 3)),
                    tf.keras.layers.Flatten(),
                    tf.keras.layers.Dense(64, activation='relu'),
                    tf.keras.layers.Dense(1, activation='sigmoid')
                ])
                
                model.compile(optimizer='adam', loss='binary_crossentropy')
                
                # Initialize with weights that will predict very low probability
                for layer in model.layers:
                    if isinstance(layer, tf.keras.layers.Dense):
                        weights = layer.get_weights()
                        # Set weights close to zero
                        weights[0] = np.zeros_like(weights[0])
                        weights[1] = np.zeros_like(weights[1])
                        layer.set_weights(weights)
                
                # Save model
                model.save(deepfake_model_path)
                logger.info("Placeholder deepfake model created")
            
            # Create placeholder NSFW model
            if not os.path.exists(nsfw_model_path):
                logger.info(f"Creating placeholder NSFW model at {nsfw_model_path}")
                
                # Simple model that always returns 0 (not NSFW)
                model = tf.keras.Sequential([
                    tf.keras.layers.Input(shape=(224, 224, 3)),
                    tf.keras.layers.Flatten(),
                    tf.keras.layers.Dense(64, activation='relu'),
                    tf.keras.layers.Dense(1, activation='sigmoid')
                ])
                
                model.compile(optimizer='adam', loss='binary_crossentropy')
                
                # Initialize with weights that will predict very low probability
                for layer in model.layers:
                    if isinstance(layer, tf.keras.layers.Dense):
                        weights = layer.get_weights()
                        # Set weights close to zero
                        weights[0] = np.zeros_like(weights[0])
                        weights[1] = np.zeros_like(weights[1])
                        layer.set_weights(weights)
                
                # Save model
                model.save(nsfw_model_path)
                logger.info("Placeholder NSFW model created")
            
        except ImportError as e:
            logger.error(f"Keras not available: {e}")
            create_placeholder_files(base_dir)
            
    except ImportError as e:
        logger.error(f"TensorFlow not available: {e}")
        create_placeholder_files(base_dir)
    
    logger.info("Model setup complete.")
    logger.info("IMPORTANT: Replace the placeholder models with actual trained models for production use.")

def create_placeholder_files(base_dir):
    """Create placeholder files when TensorFlow is not available"""
    deepfake_model_path = os.path.join(base_dir, 'deepfake_model.h5')
    nsfw_model_path = os.path.join(base_dir, 'nsfw_model.h5')
    
    if not os.path.exists(deepfake_model_path):
        logger.info(f"Creating placeholder file for deepfake model at {deepfake_model_path}")
        with open(deepfake_model_path, 'w') as f:
            f.write("# This is a placeholder. Replace with actual model file.")
            
    if not os.path.exists(nsfw_model_path):
        logger.info(f"Creating placeholder file for NSFW model at {nsfw_model_path}")
        with open(nsfw_model_path, 'w') as f:
            f.write("# This is a placeholder. Replace with actual model file.")

if __name__ == "__main__":
    setup_models()