import os
import logging
import tensorflow as tf
import numpy as np

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ModelDownloader")

def create_deepfake_model():
    """Create a simple placeholder model for deepfake detection"""
    models_dir = os.path.join('app', 'services', 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    model_path = os.path.join(models_dir, 'deepfake_model.h5')
    
    if os.path.exists(model_path):
        logger.info(f"Deepfake model already exists at {model_path}")
        return
    
    logger.info(f"Creating deepfake detection model at {model_path}")
    
    # Create a simple convolutional model
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(224, 224, 3)),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')  # Binary classification
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # Initialize with bias toward negative (not deepfake) predictions
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Dense) and layer.units == 1:
            weights = layer.get_weights()
            # Set bias to negative value
            weights[1] = np.array([-3.0])
            layer.set_weights(weights)
    
    # Save model
    model.save(model_path)
    logger.info("Deepfake detection model created successfully")

def create_nsfw_model():
    """Create a simple placeholder model for NSFW detection"""
    models_dir = os.path.join('app', 'services', 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    model_path = os.path.join(models_dir, 'nsfw_model.h5')
    
    if os.path.exists(model_path):
        logger.info(f"NSFW model already exists at {model_path}")
        return
    
    logger.info(f"Creating NSFW detection model at {model_path}")
    
    # Create a simple convolutional model
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(224, 224, 3)),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')  # Binary classification
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # Initialize with bias toward negative (not NSFW) predictions
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Dense) and layer.units == 1:
            weights = layer.get_weights()
            # Set bias to negative value
            weights[1] = np.array([-3.0])
            layer.set_weights(weights)
    
    # Save model
    model.save(model_path)
    logger.info("NSFW detection model created successfully")

if __name__ == "__main__":
    try:
        create_deepfake_model()
        create_nsfw_model()
        logger.info("All models created successfully!")
    except Exception as e:
        logger.error(f"Error creating models: {str(e)}")