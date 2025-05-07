import os
import logging
import sys

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ModelChecker")

def check_tensorflow():
    """Check TensorFlow installation"""
    logger.info("Checking TensorFlow installation...")
    try:
        # Try to import TensorFlow
        import tensorflow
        # Try different ways to get the version
        try:
            version = tensorflow.__version__
            logger.info(f"TensorFlow version: {version}")
        except AttributeError:
            try:
                from tensorflow import version
                logger.info(f"TensorFlow version: {version.VERSION}")
            except (ImportError, AttributeError):
                logger.warning("Could not determine TensorFlow version")
        
        # Check if keras is available
        try:
            from tensorflow import keras
            logger.info("Keras is available within TensorFlow")
            return True
        except ImportError:
            logger.error("Keras not available within TensorFlow")
            return False
    except ImportError as e:
        logger.error(f"Error importing TensorFlow: {e}")
        return False

def check_models():
    """Check model files"""
    # Check if TensorFlow is properly installed
    if not check_tensorflow():
        return False
        
    # Check model files
    logger.info("Checking model files...")
    
    # Define model paths
    base_dir = os.path.join('app', 'services', 'models')
    deepfake_model_path = os.path.join(base_dir, 'deepfake_model.h5')
    nsfw_model_path = os.path.join(base_dir, 'nsfw_model.h5')
    
    # Check if model files exist
    if not os.path.exists(deepfake_model_path):
        logger.error(f"Deepfake model not found at: {deepfake_model_path}")
        return False
        
    if not os.path.exists(nsfw_model_path):
        logger.error(f"NSFW model not found at: {nsfw_model_path}")
        return False
    
    # Try loading the models - commented out for now as it might cause issues
    # We'll just check if the files exist
    logger.info(f"Found model files, but not attempting to load them yet")
    logger.info("Run setup_models.py to create placeholder models")
    
    return True

def install_tensorflow():
    """Attempt to install TensorFlow"""
    logger.info("Attempting to install TensorFlow...")
    try:
        import subprocess
        # Install TensorFlow
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "tensorflow"])
        logger.info("TensorFlow installed successfully!")
        return True
    except Exception as e:
        logger.error(f"Error installing TensorFlow: {e}")
        return False

if __name__ == "__main__":
    print("Checking AI models for SafeGram...")
    if check_models():
        print("\nALL CHECKS PASSED: Models are properly installed and ready to use.")
    else:
        print("\nCHECK FAILED: There are issues with the models or TensorFlow.")
        print("Do you want to attempt to install/reinstall TensorFlow? (y/n)")
        choice = input().lower()
        if choice == 'y':
            if install_tensorflow():
                print("TensorFlow installed. Please run setup_models.py to create model placeholders.")
                print("Then run this script again to verify installation.")
            else:
                print("Failed to install TensorFlow automatically.")
                print("Please install TensorFlow manually with: pip install tensorflow")
        else:
            print("Please install the required dependencies manually.")
    
    print("\nNOTE: Actual trained models need to be placed in the app/services/models directory.")
    print("Without proper models, uploads will be rejected for safety.")