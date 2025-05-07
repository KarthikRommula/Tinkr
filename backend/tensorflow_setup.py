# tensorflow_setup.py
import subprocess
import sys
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TensorFlowSetup")

def check_tensorflow():
    """Check if TensorFlow is properly installed"""
    try:
        import tensorflow as tf
        logger.info(f"TensorFlow version: {tf.__version__}")
        
        # Test TensorFlow functionality
        logger.info("Testing TensorFlow...")
        test_tensor = tf.constant([[1, 2], [3, 4]])
        logger.info(f"Test tensor shape: {test_tensor.shape}")
        
        # Test Keras
        try:
            from tensorflow import keras
            logger.info("Keras is available")
            
            # Test model creation
            model = keras.Sequential([
                keras.layers.Dense(10, activation='relu', input_shape=(10,)),
                keras.layers.Dense(1, activation='sigmoid')
            ])
            model.compile(optimizer='adam', loss='binary_crossentropy')
            logger.info("Successfully created a test model")
            
            return True
        except ImportError:
            logger.error("Keras not available in TensorFlow")
            return False
    except ImportError:
        logger.error("TensorFlow not installed")
        return False
    except Exception as e:
        logger.error(f"TensorFlow error: {str(e)}")
        return False

def install_tensorflow():
    """Install TensorFlow and dependencies"""
    logger.info("Installing TensorFlow...")
    
    # Try to install TensorFlow with pip
    try:
        # Install dependencies first
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "numpy"])
        
        # Install TensorFlow
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "tensorflow"])
        
        # Verify installation
        if check_tensorflow():
            logger.info("TensorFlow installation successful")
            return True
        else:
            logger.error("TensorFlow installation verification failed")
            return False
    except Exception as e:
        logger.error(f"Error installing TensorFlow: {str(e)}")
        return False

if __name__ == "__main__":
    if check_tensorflow():
        logger.info("TensorFlow is already properly installed")
    else:
        logger.info("TensorFlow is not properly installed. Attempting installation...")
        install_tensorflow()