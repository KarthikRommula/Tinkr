import os
import sys
import logging
import numpy as np
import tensorflow as tf
import argparse
import time
from datetime import datetime
from pathlib import Path
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("safegram_training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("ModelTraining")

# Import our improved detectors
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
try:
    from app.services.improved_deepfake import ImprovedDeepfakeDetector
    from app.services.improved_nsfw import ImprovedNSFWDetector
    MODELS_AVAILABLE = True
except ImportError as e:
    MODELS_AVAILABLE = False
    logger.error(f"Could not import improved models: {str(e)}")

def setup_training_directories(base_dir="training_data"):
    """
    Set up directory structure for training data
    
    Returns:
        dict: Dictionary containing paths to training directories
    """
    dirs = {
        "base": base_dir,
        "deepfake": {
            "train": {
                "real": os.path.join(base_dir, "deepfake", "train", "real"),
                "fake": os.path.join(base_dir, "deepfake", "train", "fake")
            },
            "val": {
                "real": os.path.join(base_dir, "deepfake", "val", "real"),
                "fake": os.path.join(base_dir, "deepfake", "val", "fake")
            }
        },
        "nsfw": {
            "train": {
                "safe": os.path.join(base_dir, "nsfw", "train", "safe"),
                "nsfw": os.path.join(base_dir, "nsfw", "train", "nsfw")
            },
            "val": {
                "safe": os.path.join(base_dir, "nsfw", "val", "safe"),
                "nsfw": os.path.join(base_dir, "nsfw", "val", "nsfw")
            }
        }
    }
    
    # Create directories
    for category in ["deepfake", "nsfw"]:
        for split in ["train", "val"]:
            for class_name in dirs[category][split].keys():
                os.makedirs(dirs[category][split][class_name], exist_ok=True)
                logger.info(f"Created directory: {dirs[category][split][class_name]}")
    
    return dirs

def train_deepfake_model(training_dirs, epochs=20, batch_size=32):
    """
    Train the deepfake detection model using transfer learning
    
    Args:
        training_dirs: Dictionary containing paths to training directories
        epochs: Number of epochs to train for
        batch_size: Batch size for training
        
    Returns:
        str: Path to the trained model
    """
    logger.info("Starting deepfake model training...")
    
    # Initialize the detector to get the base model
    detector = ImprovedDeepfakeDetector()
    
    # Create a new model for training
    base_model = tf.keras.applications.EfficientNetB0(
        weights='imagenet', 
        include_top=False, 
        input_shape=(224, 224, 3)
    )
    
    # Unfreeze some layers for fine-tuning
    for layer in base_model.layers[-20:]:
        layer.trainable = True
    
    # Add custom classification layers
    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    predictions = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    
    # Create the model
    model = tf.keras.models.Model(inputs=base_model.input, outputs=predictions)
    
    # Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
    
    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    # Only rescaling for validation
    val_datagen = ImageDataGenerator(rescale=1./255)
    
    # Create train generators
    train_generator = train_datagen.flow_from_directory(
        os.path.join(training_dirs["deepfake"]["train"]["real"], ".."),
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='binary',
        classes=["real", "fake"]
    )
    
    # Create validation generators
    val_generator = val_datagen.flow_from_directory(
        os.path.join(training_dirs["deepfake"]["val"]["real"], ".."),
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='binary',
        classes=["real", "fake"]
    )
    
    # Create model directory if it doesn't exist
    model_dir = os.path.dirname(detector.model_path)
    os.makedirs(model_dir, exist_ok=True)
    
    # Set up callbacks
    callbacks = [
        ModelCheckpoint(
            detector.model_path,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=3,
            min_lr=1e-6,
            verbose=1
        ),
        TensorBoard(
            log_dir=os.path.join('logs', 'deepfake', datetime.now().strftime("%Y%m%d-%H%M%S")),
            histogram_freq=1
        )
    ]
    
    # Check if we have enough training data
    if len(os.listdir(training_dirs["deepfake"]["train"]["real"])) < 10 or \
       len(os.listdir(training_dirs["deepfake"]["train"]["fake"])) < 10:
        logger.warning("Not enough training data for deepfake detection. Using placeholder model.")
        detector._load_placeholder_model()
        model.save(detector.model_path)
        return detector.model_path
    
    # Train the model
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // batch_size,
        epochs=epochs,
        validation_data=val_generator,
        validation_steps=val_generator.samples // batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save the final model
    model.save(detector.model_path)
    logger.info(f"Deepfake model trained and saved to {detector.model_path}")
    
    return detector.model_path

def train_nsfw_model(training_dirs, epochs=20, batch_size=32):
    """
    Train the NSFW detection model using transfer learning
    
    Args:
        training_dirs: Dictionary containing paths to training directories
        epochs: Number of epochs to train for
        batch_size: Batch size for training
        
    Returns:
        str: Path to the trained model
    """
    logger.info("Starting NSFW model training...")
    
    # Initialize the detector to get the base model
    detector = ImprovedNSFWDetector()
    
    # Create a new model for training
    base_model = tf.keras.applications.MobileNetV2(
        weights='imagenet', 
        include_top=False, 
        input_shape=(224, 224, 3)
    )
    
    # Unfreeze some layers for fine-tuning
    for layer in base_model.layers[-20:]:
        layer.trainable = True
    
    # Add custom classification layers
    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    predictions = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    
    # Create the model
    model = tf.keras.models.Model(inputs=base_model.input, outputs=predictions)
    
    # Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
    
    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    # Only rescaling for validation
    val_datagen = ImageDataGenerator(rescale=1./255)
    
    # Create train generators
    train_generator = train_datagen.flow_from_directory(
        os.path.join(training_dirs["nsfw"]["train"]["safe"], ".."),
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='binary',
        classes=["safe", "nsfw"]
    )
    
    # Create validation generators
    val_generator = val_datagen.flow_from_directory(
        os.path.join(training_dirs["nsfw"]["val"]["safe"], ".."),
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='binary',
        classes=["safe", "nsfw"]
    )
    
    # Create model directory if it doesn't exist
    model_dir = os.path.dirname(detector.model_path)
    os.makedirs(model_dir, exist_ok=True)
    
    # Set up callbacks
    callbacks = [
        ModelCheckpoint(
            detector.model_path,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=3,
            min_lr=1e-6,
            verbose=1
        ),
        TensorBoard(
            log_dir=os.path.join('logs', 'nsfw', datetime.now().strftime("%Y%m%d-%H%M%S")),
            histogram_freq=1
        )
    ]
    
    # Check if we have enough training data
    if len(os.listdir(training_dirs["nsfw"]["train"]["safe"])) < 10 or \
       len(os.listdir(training_dirs["nsfw"]["train"]["nsfw"])) < 10:
        logger.warning("Not enough training data for NSFW detection. Using placeholder model.")
        detector._load_placeholder_model()
        model.save(detector.model_path)
        return detector.model_path
    
    # Train the model
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // batch_size,
        epochs=epochs,
        validation_data=val_generator,
        validation_steps=val_generator.samples // batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save the final model
    model.save(detector.model_path)
    logger.info(f"NSFW model trained and saved to {detector.model_path}")
    
    return detector.model_path

def download_sample_training_data(training_dirs):
    """
    Download sample training data for demonstration purposes
    This is a placeholder - in a real application, you would download actual datasets
    
    Args:
        training_dirs: Dictionary containing paths to training directories
    """
    logger.info("Creating sample training data...")
    
    # Create a few sample images for demonstration
    # In a real application, you would download actual datasets
    
    # Create sample deepfake training data
    for category in ["real", "fake"]:
        for split in ["train", "val"]:
            dir_path = training_dirs["deepfake"][split][category]
            for i in range(20):  # Create 20 sample images per category
                img_path = os.path.join(dir_path, f"sample_{i}.jpg")
                if not os.path.exists(img_path):
                    # Create a sample image
                    if category == "real":
                        # Create a face-like pattern for "real" images
                        arr = np.zeros((224, 224, 3), dtype=np.uint8)
                        # Draw a circle for a face
                        cv2.circle(arr, (112, 112), 80, (200, 200, 200), -1)
                        # Draw eyes
                        cv2.circle(arr, (85, 90), 10, (50, 50, 50), -1)
                        cv2.circle(arr, (140, 90), 10, (50, 50, 50), -1)
                        # Draw mouth
                        cv2.ellipse(arr, (112, 140), (30, 20), 0, 0, 180, (50, 50, 50), -1)
                        # Add some random variation
                        arr = arr + np.random.randint(-20, 20, size=arr.shape, dtype=np.int16).clip(0, 255).astype(np.uint8)
                    else:
                        # Create a distorted face-like pattern for "fake" images
                        arr = np.zeros((224, 224, 3), dtype=np.uint8)
                        # Draw a distorted circle for a face
                        cv2.ellipse(arr, (112, 112), (90, 70), 30, 0, 360, (200, 200, 200), -1)
                        # Draw distorted eyes
                        cv2.ellipse(arr, (85, 90), (15, 10), 20, 0, 360, (50, 50, 50), -1)
                        cv2.ellipse(arr, (140, 90), (15, 10), 20, 0, 360, (50, 50, 50), -1)
                        # Draw distorted mouth
                        cv2.ellipse(arr, (112, 140), (40, 15), 10, 0, 180, (50, 50, 50), -1)
                        # Add some random variation and artifacts
                        arr = arr + np.random.randint(-30, 30, size=arr.shape, dtype=np.int16).clip(0, 255).astype(np.uint8)
                    
                    # Save the image
                    cv2.imwrite(img_path, arr)
    
    # Create sample NSFW training data
    for category in ["safe", "nsfw"]:
        for split in ["train", "val"]:
            dir_path = training_dirs["nsfw"][split][category]
            for i in range(20):  # Create 20 sample images per category
                img_path = os.path.join(dir_path, f"sample_{i}.jpg")
                if not os.path.exists(img_path):
                    # Create a sample image
                    if category == "safe":
                        # Create a landscape-like pattern for "safe" images
                        arr = np.zeros((224, 224, 3), dtype=np.uint8)
                        # Sky
                        arr[:112, :, 0] = 100
                        arr[:112, :, 1] = 150
                        arr[:112, :, 2] = 200
                        # Ground
                        arr[112:, :, 0] = 100
                        arr[112:, :, 1] = 120
                        arr[112:, :, 2] = 80
                        # Add some random variation
                        arr = arr + np.random.randint(-20, 20, size=arr.shape, dtype=np.int16).clip(0, 255).astype(np.uint8)
                    else:
                        # Create a pattern with skin-tone colors for "nsfw" images
                        arr = np.zeros((224, 224, 3), dtype=np.uint8)
                        # Fill with skin-tone color
                        arr[:, :, 0] = 200
                        arr[:, :, 1] = 150
                        arr[:, :, 2] = 140
                        # Add some random variation
                        arr = arr + np.random.randint(-30, 30, size=arr.shape, dtype=np.int16).clip(0, 255).astype(np.uint8)
                    
                    # Save the image
                    cv2.imwrite(img_path, arr)
    
    logger.info("Sample training data created successfully")

def main():
    """Train the improved AI models"""
    print("=" * 80)
    print("SAFEGRAM MODEL TRAINING")
    print("=" * 80)
    
    # Check if TensorFlow is available
    if not MODELS_AVAILABLE:
        logger.error("Models not available. Cannot proceed with training.")
        return 1
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train improved AI models')
    parser.add_argument('--deepfake', action='store_true', help='Train deepfake detection only')
    parser.add_argument('--nsfw', action='store_true', help='Train NSFW detection only')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs to train for')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--data-dir', type=str, default='training_data', help='Directory for training data')
    parser.add_argument('--download-samples', action='store_true', help='Download sample training data')
    args = parser.parse_args()
    
    # Set up training directories
    training_dirs = setup_training_directories(args.data_dir)
    
    # Download sample training data if requested
    if args.download_samples:
        download_sample_training_data(training_dirs)
    
    # Train deepfake detection model
    if not args.nsfw or args.deepfake:
        deepfake_model_path = train_deepfake_model(
            training_dirs, 
            epochs=args.epochs, 
            batch_size=args.batch_size
        )
        logger.info(f"Deepfake model trained and saved to: {deepfake_model_path}")
    
    # Train NSFW detection model
    if not args.deepfake or args.nsfw:
        nsfw_model_path = train_nsfw_model(
            training_dirs, 
            epochs=args.epochs, 
            batch_size=args.batch_size
        )
        logger.info(f"NSFW model trained and saved to: {nsfw_model_path}")
    
    print("\n" + "=" * 80)
    print("Training completed successfully.")
    print("=" * 80)
    return 0

if __name__ == "__main__":
    sys.exit(main())
