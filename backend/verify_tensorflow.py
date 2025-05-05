try:
    import tensorflow as tf
    print(f"TensorFlow package is installed")
    print(f"TensorFlow package location: {tf.__file__}")
    
    # Try to access the version in a different way
    try:
        print(f"TensorFlow version: {tf.version.VERSION}")
    except:
        print("Could not access tf.version.VERSION")
    
    # Check if keras is available
    try:
        from tensorflow import keras
        print("Keras is available within TensorFlow")
    except:
        print("Keras not available within TensorFlow")
        
    print("TensorFlow is successfully installed!")
except ImportError as e:
    print(f"Error importing TensorFlow: {e}")
    print("TensorFlow may not be properly installed.")
except Exception as e:
    print(f"Unexpected error: {e}")

