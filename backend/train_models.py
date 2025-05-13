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

def train_deepfake_model(training_dirs, epochs=50, batch_size=32):
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
    
    # Create a new model for training - using a more powerful base model
    base_model = tf.keras.applications.EfficientNetB3(
        weights='imagenet', 
        include_top=False, 
        input_shape=(224, 224, 3)
    )
    
    # First freeze the entire base model for initial training
    base_model.trainable = False
    
    # Add custom classification layers with stronger regularization
    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    # Add batch normalization to help with training stability
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    x = tf.keras.layers.Dropout(0.6)(x)  # Increased dropout for better regularization
    x = tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    predictions = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    
    # Create the model
    model = tf.keras.models.Model(inputs=base_model.input, outputs=predictions)
    
    # Check class balance in training data
    real_count = len(os.listdir(training_dirs["deepfake"]["train"]["real"]))
    fake_count = len(os.listdir(training_dirs["deepfake"]["train"]["fake"]))
    total = real_count + fake_count
    
    logger.info(f"Training data distribution: {real_count} real images ({real_count/total:.1%}), "
               f"{fake_count} fake images ({fake_count/total:.1%})")
    
    # Calculate class weights to handle imbalance
    class_weight = None
    if abs(real_count - fake_count) > 0.1 * total:  # If imbalance is more than 10%
        weight_for_0 = (1 / real_count) * total / 2
        weight_for_1 = (1 / fake_count) * total / 2
        class_weight = {0: weight_for_0, 1: weight_for_1}
        logger.info(f"Using class weights due to imbalance: {class_weight}")
    
    # Compile the model with a lower initial learning rate
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.00005),
        loss='binary_crossentropy',
        metrics=['accuracy', 
                tf.keras.metrics.AUC(), 
                tf.keras.metrics.Precision(), 
                tf.keras.metrics.Recall(),
                tf.keras.metrics.FalsePositives(),
                tf.keras.metrics.FalseNegatives()]
    )
    
    # Enhanced data augmentation for training to improve generalization
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,
        width_shift_range=0.3,
        height_shift_range=0.3,
        shear_range=0.3,
        zoom_range=0.3,
        horizontal_flip=True,
        vertical_flip=False,  # Faces are usually upright
        brightness_range=[0.7, 1.3],  # Vary brightness
        channel_shift_range=30,  # Slight color variations
        fill_mode='nearest',
        validation_split=0.1  # Use 10% of training data as additional validation
    )
    
    # Add some minimal augmentation for validation to better match real-world conditions
    val_datagen = ImageDataGenerator(
        rescale=1./255,
        width_shift_range=0.1,  # Slight shifts to simulate real-world variations
        height_shift_range=0.1,
        brightness_range=[0.9, 1.1]  # Slight brightness variations
    )
    
    # Create train generators with shuffle=True to ensure random order
    train_generator = train_datagen.flow_from_directory(
        os.path.join(training_dirs["deepfake"]["train"]["real"], ".."),
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='binary',
        classes=["real", "fake"],
        shuffle=True,
        seed=42  # Set seed for reproducibility
    )
    
    # Create validation generators
    val_generator = val_datagen.flow_from_directory(
        os.path.join(training_dirs["deepfake"]["val"]["real"], ".."),
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='binary',
        classes=["real", "fake"],
        shuffle=True,
        seed=42  # Set seed for reproducibility
    )
    
    # Create model directory if it doesn't exist
    model_dir = os.path.dirname(detector.model_path)
    os.makedirs(model_dir, exist_ok=True)
    
    # Set up callbacks with improved parameters
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
            patience=10,  # Increased patience to allow more training time
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,  # Increased patience before reducing learning rate
            min_lr=1e-7,
            verbose=1
        ),
        TensorBoard(
            log_dir=os.path.join('logs', 'deepfake', datetime.now().strftime("%Y%m%d-%H%M%S")),
            histogram_freq=1,
            update_freq='epoch',
            profile_batch=0  # Disable profiling to reduce memory usage
        )
    ]
    
    # Check if we have enough training data
    if real_count < 50 or fake_count < 50:
        logger.warning("Not enough training data for deepfake detection. Using placeholder model.")
        detector._load_placeholder_model()
        model.save(detector.model_path)
        return detector.model_path
    
    # First phase: Train with frozen base model
    logger.info("Phase 1: Training with frozen base model...")
    # Try to use multiprocessing if available, otherwise fall back to standard training
    try:
        history_phase1 = model.fit(
            train_generator,
            steps_per_epoch=train_generator.samples // batch_size,
            epochs=min(10, epochs // 3),  # Train for 1/3 of total epochs or 10, whichever is smaller
            validation_data=val_generator,
            validation_steps=val_generator.samples // batch_size,
            callbacks=callbacks,
            verbose=1,
            class_weight=class_weight
            # Removed multiprocessing parameters that caused errors
        )
    except Exception as e:
        logger.error(f"Error in phase 1 training: {str(e)}")
        raise
    
    # Second phase: Fine-tune the top layers of the base model
    logger.info("Phase 2: Fine-tuning top layers of base model...")
    # Unfreeze the top layers of the base model
    for layer in base_model.layers[-30:]:  # Unfreeze more layers for better fine-tuning
        layer.trainable = True
    
    # Recompile the model with a lower learning rate for fine-tuning
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),  # Lower learning rate for fine-tuning
        loss='binary_crossentropy',
        metrics=['accuracy', 
                tf.keras.metrics.AUC(), 
                tf.keras.metrics.Precision(), 
                tf.keras.metrics.Recall(),
                tf.keras.metrics.FalsePositives(),
                tf.keras.metrics.FalseNegatives()]
    )
    
    # Continue training with fine-tuning
    history_phase2 = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // batch_size,
        epochs=epochs,
        initial_epoch=min(10, epochs // 3),  # Continue from where phase 1 left off
        validation_data=val_generator,
        validation_steps=val_generator.samples // batch_size,
        callbacks=callbacks,
        verbose=1,
        class_weight=class_weight
        # Removed multiprocessing parameters that caused errors
    )
    
    # Save the final model
    model.save(detector.model_path)
    logger.info(f"Deepfake model trained and saved to {detector.model_path}")
    
    # Plot training history if matplotlib is available
    try:
        import matplotlib.pyplot as plt
        
        # Combine histories from both phases
        acc = history_phase1.history['accuracy'] + history_phase2.history['accuracy']
        val_acc = history_phase1.history['val_accuracy'] + history_phase2.history['val_accuracy']
        loss = history_phase1.history['loss'] + history_phase2.history['loss']
        val_loss = history_phase1.history['val_loss'] + history_phase2.history['val_loss']
        
        epochs_range = range(len(acc))
        
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, acc, label='Training Accuracy')
        plt.plot(epochs_range, val_acc, label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.title('Training and Validation Accuracy')
        
        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, loss, label='Training Loss')
        plt.plot(epochs_range, val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss')
        
        # Save the plot
        plot_path = os.path.join(model_dir, 'training_history.png')
        plt.savefig(plot_path)
        logger.info(f"Training history plot saved to {plot_path}")
    except ImportError:
        logger.warning("Matplotlib not available. Skipping training history plot.")
    
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

def validate_dataset(training_dirs):
    """Validate the dataset structure and balance"""
    logger.info("Validating dataset...")
    
    # Check for deepfake dataset
    real_train = os.path.join(training_dirs["deepfake"]["train"]["real"])
    fake_train = os.path.join(training_dirs["deepfake"]["train"]["fake"])
    real_val = os.path.join(training_dirs["deepfake"]["val"]["real"])
    fake_val = os.path.join(training_dirs["deepfake"]["val"]["fake"])
    
    # Count images in each directory
    real_train_count = len([f for f in os.listdir(real_train) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    fake_train_count = len([f for f in os.listdir(fake_train) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    real_val_count = len([f for f in os.listdir(real_val) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    fake_val_count = len([f for f in os.listdir(fake_val) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    logger.info(f"Deepfake dataset statistics:")
    logger.info(f"  Training: {real_train_count} real, {fake_train_count} fake")
    logger.info(f"  Validation: {real_val_count} real, {fake_val_count} fake")
    
    # Check for potential issues
    issues = []
    
    # Check for empty directories
    if real_train_count == 0:
        issues.append(f"No real images found in training directory: {real_train}")
    if fake_train_count == 0:
        issues.append(f"No fake images found in training directory: {fake_train}")
    if real_val_count == 0:
        issues.append(f"No real images found in validation directory: {real_val}")
    if fake_val_count == 0:
        issues.append(f"No fake images found in validation directory: {fake_val}")
    
    # Check for class imbalance
    if real_train_count > 0 and fake_train_count > 0:
        train_ratio = max(real_train_count, fake_train_count) / min(real_train_count, fake_train_count)
        if train_ratio > 3:
            issues.append(f"Severe class imbalance in training data: ratio {train_ratio:.1f}:1")
        elif train_ratio > 1.5:
            issues.append(f"Moderate class imbalance in training data: ratio {train_ratio:.1f}:1")
    
    if real_val_count > 0 and fake_val_count > 0:
        val_ratio = max(real_val_count, fake_val_count) / min(real_val_count, fake_val_count)
        if val_ratio > 3:
            issues.append(f"Severe class imbalance in validation data: ratio {val_ratio:.1f}:1")
        elif val_ratio > 1.5:
            issues.append(f"Moderate class imbalance in validation data: ratio {val_ratio:.1f}:1")
    
    # Check for sufficient data
    if real_train_count + fake_train_count < 200:
        issues.append(f"Limited training data: only {real_train_count + fake_train_count} images")
    if real_val_count + fake_val_count < 50:
        issues.append(f"Limited validation data: only {real_val_count + fake_val_count} images")
    
    # Report issues
    if issues:
        logger.warning("Dataset validation found the following issues:")
        for issue in issues:
            logger.warning(f"  - {issue}")
    else:
        logger.info("Dataset validation completed: No issues found.")
    
    return issues

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
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train for')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--data-dir', type=str, default='training_data', help='Directory for training data')
    parser.add_argument('--download-samples', action='store_true', help='Download sample training data')
    parser.add_argument('--validate-only', action='store_true', help='Only validate the dataset without training')
    parser.add_argument('--learning-rate', type=float, default=0.00005, help='Initial learning rate')
    parser.add_argument('--model-type', type=str, default='efficientnetb3', 
                      choices=['efficientnetb0', 'efficientnetb3', 'resnet50', 'xception'], 
                      help='Base model architecture')
    args = parser.parse_args()
    
    # Set up training directories
    training_dirs = setup_training_directories(args.data_dir)
    
    # Download sample training data if requested
    if args.download_samples:
        download_sample_training_data(training_dirs)
    
    # Validate the dataset
    issues = validate_dataset(training_dirs)
    
    # If validate-only flag is set, exit after validation
    if args.validate_only:
        if issues:
            logger.warning("Dataset validation found issues. Please fix them before training.")
            return 1
        logger.info("Dataset validation completed successfully.")
        return 0
    
    # If there are critical issues, ask for confirmation before proceeding
    if any("No " in issue for issue in issues):
        logger.error("Critical dataset issues found. Training may fail or produce poor results.")
        response = input("Do you want to proceed with training anyway? (y/n): ")
        if response.lower() != 'y':
            logger.info("Training aborted by user.")
            return 1
    
    # Set TensorFlow memory growth to avoid OOM errors
    try:
        physical_devices = tf.config.list_physical_devices('GPU')
        if physical_devices:
            for device in physical_devices:
                tf.config.experimental.set_memory_growth(device, True)
            logger.info(f"GPU memory growth enabled for {len(physical_devices)} GPU(s)")
    except Exception as e:
        logger.warning(f"Could not configure GPU memory growth: {str(e)}")
    
    # Train deepfake detection model
    if not args.nsfw or args.deepfake:
        logger.info("Starting deepfake model training...")
        deepfake_model_path = train_deepfake_model(
            training_dirs, 
            epochs=args.epochs, 
            batch_size=args.batch_size
        )
        logger.info(f"Deepfake model trained and saved to: {deepfake_model_path}")
        
        # Test the model on some sample images
        logger.info("Testing the trained deepfake model...")
        detector = ImprovedDeepfakeDetector(model_path=deepfake_model_path)
        
        # Test on training samples
        test_images = []
        for category, subdir in [("real", "real"), ("fake", "fake")]:
            dir_path = training_dirs["deepfake"]["val"][subdir]
            files = [f for f in os.listdir(dir_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if files:
                # Take up to 3 random samples from each category
                import random
                samples = random.sample(files, min(3, len(files)))
                for sample in samples:
                    test_images.append((os.path.join(dir_path, sample), category))
        
        # Run tests and report results
        if test_images:
            logger.info(f"Testing model on {len(test_images)} sample images...")
            for img_path, true_category in test_images:
                is_deepfake, confidence = detector.detect(img_path)
                result = "CORRECT" if (is_deepfake and true_category == "fake") or (not is_deepfake and true_category == "real") else "INCORRECT"
                logger.info(f"Test image: {os.path.basename(img_path)}")
                logger.info(f"  True category: {true_category}")
                logger.info(f"  Prediction: {'fake' if is_deepfake else 'real'} with {confidence:.4f} confidence")
                logger.info(f"  Result: {result}")
    
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
