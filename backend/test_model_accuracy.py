import os
import sys
import logging
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ModelAccuracyTest")

def test_model_accuracy():
    # Path to the trained model
    model_path = os.path.join("app", "services", "models", "improved_deepfake_model.h5")
    
    if not os.path.exists(model_path):
        logger.error(f"Model file not found at {model_path}")
        return
    
    logger.info(f"Loading model from {model_path}")
    model = load_model(model_path)
    
    # Data directory
    data_dir = "training_data"
    validation_dir = os.path.join(data_dir, "deepfake", "val")
    
    # Data generator with preprocessing
    test_datagen = ImageDataGenerator(
        rescale=1./255,
    )
    
    # Load validation data
    validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='binary',
        shuffle=False
    )
    
    # Evaluate the model
    logger.info("Evaluating model on validation data...")
    results = model.evaluate(validation_generator, verbose=1)
    
    # Print results
    metrics = model.metrics_names
    logger.info("Model Evaluation Results:")
    for i, metric in enumerate(metrics):
        logger.info(f"{metric}: {results[i]:.4f}")
    
    # Generate predictions for confusion matrix
    logger.info("Generating predictions for detailed analysis...")
    validation_generator.reset()
    y_pred = model.predict(validation_generator, verbose=1)
    y_pred_binary = (y_pred > 0.5).astype(int)
    
    # Get true labels
    y_true = validation_generator.classes
    
    # Calculate metrics
    true_positives = np.sum((y_true == 1) & (y_pred_binary.flatten() == 1))
    true_negatives = np.sum((y_true == 0) & (y_pred_binary.flatten() == 0))
    false_positives = np.sum((y_true == 0) & (y_pred_binary.flatten() == 1))
    false_negatives = np.sum((y_true == 1) & (y_pred_binary.flatten() == 0))
    
    accuracy = (true_positives + true_negatives) / len(y_true)
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    logger.info("\nDetailed Performance Metrics:")
    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall: {recall:.4f}")
    logger.info(f"F1 Score: {f1_score:.4f}")
    
    logger.info("\nConfusion Matrix:")
    logger.info(f"True Positives: {true_positives}")
    logger.info(f"True Negatives: {true_negatives}")
    logger.info(f"False Positives: {false_positives}")
    logger.info(f"False Negatives: {false_negatives}")
    
    # Calculate confidence distribution
    real_confidences = 1 - y_pred[y_true == 0].flatten()
    fake_confidences = y_pred[y_true == 1].flatten()
    
    logger.info("\nConfidence Distribution:")
    logger.info(f"Real images - Avg confidence: {np.mean(real_confidences):.4f}, Min: {np.min(real_confidences):.4f}, Max: {np.max(real_confidences):.4f}")
    logger.info(f"Fake images - Avg confidence: {np.mean(fake_confidences):.4f}, Min: {np.min(fake_confidences):.4f}, Max: {np.max(fake_confidences):.4f}")

if __name__ == "__main__":
    logger.info("Starting model accuracy test")
    test_model_accuracy()
    logger.info("Model accuracy test completed")
