import os
import logging
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model

# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

def test_model():
    print("\n===== DEEPFAKE DETECTION MODEL ACCURACY TEST =====\n")
    
    # Path to the trained model
    model_path = os.path.join("app", "services", "models", "improved_deepfake_model.h5")
    
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return
    
    print(f"Loading model from {model_path}")
    model = load_model(model_path)
    
    # Data directory
    validation_dir = os.path.join("training_data", "deepfake", "val")
    
    # Data generator with preprocessing
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    # Load validation data
    print("Loading validation data...")
    validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='binary',
        shuffle=False
    )
    
    # Evaluate the model
    print("\nEvaluating model on validation data...")
    results = model.evaluate(validation_generator, verbose=1)
    
    # Print results
    metrics = model.metrics_names
    print("\n----- MODEL EVALUATION RESULTS -----")
    for i, metric in enumerate(metrics):
        print(f"{metric}: {results[i]:.4f}")
    
    # Generate predictions for confusion matrix
    print("\nGenerating predictions for detailed analysis...")
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
    
    print("\n----- DETAILED PERFORMANCE METRICS -----")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1_score:.4f}")
    
    print("\n----- CONFUSION MATRIX -----")
    print(f"True Positives: {true_positives} (Correctly identified fake images)")
    print(f"True Negatives: {true_negatives} (Correctly identified real images)")
    print(f"False Positives: {false_positives} (Real images misclassified as fake)")
    print(f"False Negatives: {false_negatives} (Fake images misclassified as real)")
    
    # Calculate confidence distribution
    real_confidences = 1 - y_pred[y_true == 0].flatten()
    fake_confidences = y_pred[y_true == 1].flatten()
    
    print("\n----- CONFIDENCE DISTRIBUTION -----")
    print(f"Real images - Avg confidence: {np.mean(real_confidences):.4f}, Min: {np.min(real_confidences):.4f}, Max: {np.max(real_confidences):.4f}")
    print(f"Fake images - Avg confidence: {np.mean(fake_confidences):.4f}, Min: {np.min(fake_confidences):.4f}, Max: {np.max(fake_confidences):.4f}")

if __name__ == "__main__":
    test_model()
