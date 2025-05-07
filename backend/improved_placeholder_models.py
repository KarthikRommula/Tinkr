# improved_placeholder_models.py (updated version)
import os
import logging
import numpy as np
import tensorflow as tf

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("PlaceholderModelCreator")

def create_placeholder_model(name, input_shape, model_dir="app/services/models"):
    """
    Create and save a simple placeholder model that returns low confidence scores
    
    Args:
        name: Model name (e.g., 'deepfake_model' or 'nsfw_model')
        input_shape: Input shape for the model (e.g., (224, 224, 3))
        model_dir: Directory to save the model
    
    Returns:
        Path to the saved model
    """
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"{name}.h5")
    
    # Check if model already exists
    if os.path.exists(model_path):
        logger.info(f"Model already exists at {model_path}")
        try:
            # Try to load the model to verify it's valid
            test_model = tf.keras.models.load_model(model_path)
            # Test with a dummy input
            dummy_input = np.zeros((1,) + input_shape)
            _ = test_model.predict(dummy_input)
            logger.info(f"Successfully verified existing model at {model_path}")
            return model_path
        except Exception as e:
            logger.warning(f"Existing model at {model_path} is invalid, recreating: {str(e)}")
            # Continue to create a new model
    
    logger.info(f"Creating placeholder model: {name}")
    
    # Create a very simple model
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same'),
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
    
    # Test the model with dummy input
    dummy_input = np.zeros((1,) + input_shape)
    prediction = model.predict(dummy_input)
    logger.info(f"Test prediction for {name}: {prediction[0][0]}")
    
    # Save the model
    try:
        model.save(model_path)
        logger.info(f"Successfully created and saved {name} at {model_path}")
        return model_path
    except Exception as e:
        logger.error(f"Error saving model: {str(e)}")
        # Try saving in SavedModel format instead
        try:
            saved_model_dir = os.path.join(model_dir, f"{name}_saved_model")
            tf.saved_model.save(model, saved_model_dir)
            logger.info(f"Saved model in SavedModel format at {saved_model_dir}")
            return saved_model_dir
        except Exception as e2:
            logger.error(f"Error saving in SavedModel format: {str(e2)}")
            return None

def setup_models():
    """Create both placeholder models"""
    models_created = []
    
    # Create deepfake detection model
    deepfake_path = create_placeholder_model(
        "deepfake_model", 
        (224, 224, 3)  # Standard input shape for many image models
    )
    if deepfake_path:
        models_created.append(("deepfake_model", deepfake_path))
    
    # Create NSFW detection model
    nsfw_path = create_placeholder_model(
        "nsfw_model", 
        (224, 224, 3)  # Standard input shape for many image models
    )
    if nsfw_path:
        models_created.append(("nsfw_model", nsfw_path))
    
    return models_created

if __name__ == "__main__":
    models = setup_models()
    if models:
        logger.info(f"Successfully created {len(models)} models:")
        for name, path in models:
            logger.info(f"  - {name}: {path}")
    else:
        logger.error("Failed to create placeholder models")