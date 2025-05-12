import os
import sys
import logging
import numpy as np
from PIL import Image
import cv2
import time
import argparse
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ImprovedModelTest")

# Try to import TensorFlow
try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
    logger.info(f"TensorFlow version: {tf.__version__}")
    # Suppress TensorFlow warnings
    tf.get_logger().setLevel('ERROR')
except ImportError:
    TENSORFLOW_AVAILABLE = False
    logger.error("TensorFlow not available. Tests will be limited.")

# Import our improved detectors
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
try:
    from app.services.improved_deepfake import ImprovedDeepfakeDetector
    from app.services.improved_nsfw import ImprovedNSFWDetector
    MODELS_AVAILABLE = True
except ImportError as e:
    MODELS_AVAILABLE = False
    logger.error(f"Could not import improved models: {str(e)}")

def create_test_images():
    """Create various test images for detection testing"""
    test_dir = 'test_images'
    os.makedirs(test_dir, exist_ok=True)
    
    test_images = []
    
    # Create a simple gradient image
    gradient_path = os.path.join(test_dir, 'gradient.jpg')
    if not os.path.exists(gradient_path):
        try:
            arr = np.zeros((224, 224, 3), dtype=np.uint8)
            for i in range(224):
                for j in range(224):
                    arr[i, j, 0] = i // 2  # Red
                    arr[i, j, 1] = j // 2  # Green
                    arr[i, j, 2] = 100     # Blue
            
            img = Image.fromarray(arr)
            img.save(gradient_path)
            logger.info(f"Created gradient test image: {gradient_path}")
        except Exception as e:
            logger.error(f"Failed to create gradient test image: {str(e)}")
    
    test_images.append(gradient_path)
    
    # Create a noise image
    noise_path = os.path.join(test_dir, 'noise.jpg')
    if not os.path.exists(noise_path):
        try:
            arr = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
            img = Image.fromarray(arr)
            img.save(noise_path)
            logger.info(f"Created noise test image: {noise_path}")
        except Exception as e:
            logger.error(f"Failed to create noise test image: {str(e)}")
    
    test_images.append(noise_path)
    
    # Create a face-like pattern
    face_path = os.path.join(test_dir, 'face_pattern.jpg')
    if not os.path.exists(face_path):
        try:
            arr = np.zeros((224, 224, 3), dtype=np.uint8)
            # Draw a circle for a face
            cv2.circle(arr, (112, 112), 80, (200, 200, 200), -1)
            # Draw eyes
            cv2.circle(arr, (85, 90), 10, (50, 50, 50), -1)
            cv2.circle(arr, (140, 90), 10, (50, 50, 50), -1)
            # Draw mouth
            cv2.ellipse(arr, (112, 140), (30, 20), 0, 0, 180, (50, 50, 50), -1)
            
            img = Image.fromarray(arr)
            img.save(face_path)
            logger.info(f"Created face pattern test image: {face_path}")
        except Exception as e:
            logger.error(f"Failed to create face pattern test image: {str(e)}")
    
    test_images.append(face_path)
    
    return test_images

def test_deepfake_detector(image_paths):
    """Test the improved deepfake detector on multiple images"""
    if not TENSORFLOW_AVAILABLE or not MODELS_AVAILABLE:
        logger.error("Cannot test deepfake detector - TensorFlow or models not available")
        return False
    
    logger.info("Testing ImprovedDeepfakeDetector...")
    
    try:
        # Initialize the detector
        detector = ImprovedDeepfakeDetector()
        
        # Test each image
        results = []
        for image_path in image_paths:
            start_time = time.time()
            is_deepfake, confidence = detector.detect(image_path)
            processing_time = time.time() - start_time
            
            result = {
                "image": os.path.basename(image_path),
                "is_deepfake": is_deepfake,
                "confidence": confidence,
                "processing_time": processing_time
            }
            results.append(result)
            
            logger.info(f"Image: {os.path.basename(image_path)}")
            logger.info(f"  Deepfake: {'Yes' if is_deepfake else 'No'}")
            logger.info(f"  Confidence: {confidence:.4f}")
            logger.info(f"  Processing time: {processing_time:.2f} seconds")
        
        return results
    except Exception as e:
        logger.error(f"Error testing deepfake detector: {str(e)}")
        return False

def test_nsfw_detector(image_paths):
    """Test the improved NSFW detector on multiple images"""
    if not TENSORFLOW_AVAILABLE or not MODELS_AVAILABLE:
        logger.error("Cannot test NSFW detector - TensorFlow or models not available")
        return False
    
    logger.info("Testing ImprovedNSFWDetector...")
    
    try:
        # Initialize the detector
        detector = ImprovedNSFWDetector()
        
        # Test each image
        results = []
        for image_path in image_paths:
            start_time = time.time()
            is_nsfw, confidence = detector.detect(image_path)
            processing_time = time.time() - start_time
            
            result = {
                "image": os.path.basename(image_path),
                "is_nsfw": is_nsfw,
                "confidence": confidence,
                "processing_time": processing_time
            }
            results.append(result)
            
            logger.info(f"Image: {os.path.basename(image_path)}")
            logger.info(f"  NSFW: {'Yes' if is_nsfw else 'No'}")
            logger.info(f"  Confidence: {confidence:.4f}")
            logger.info(f"  Processing time: {processing_time:.2f} seconds")
        
        return results
    except Exception as e:
        logger.error(f"Error testing NSFW detector: {str(e)}")
        return False

def generate_report(deepfake_results, nsfw_results):
    """Generate a detailed test report"""
    report_dir = 'test_reports'
    os.makedirs(report_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(report_dir, f"model_test_report_{timestamp}.txt")
    
    try:
        with open(report_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("SAFEGRAM IMPROVED MODELS TEST REPORT\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")
            
            # Deepfake detection results
            f.write("DEEPFAKE DETECTION RESULTS\n")
            f.write("-" * 80 + "\n")
            if deepfake_results:
                for result in deepfake_results:
                    f.write(f"Image: {result['image']}\n")
                    f.write(f"  Deepfake: {'Yes' if result['is_deepfake'] else 'No'}\n")
                    f.write(f"  Confidence: {result['confidence']:.4f}\n")
                    f.write(f"  Processing time: {result['processing_time']:.2f} seconds\n\n")
            else:
                f.write("No deepfake detection results available.\n\n")
            
            # NSFW detection results
            f.write("NSFW DETECTION RESULTS\n")
            f.write("-" * 80 + "\n")
            if nsfw_results:
                for result in nsfw_results:
                    f.write(f"Image: {result['image']}\n")
                    f.write(f"  NSFW: {'Yes' if result['is_nsfw'] else 'No'}\n")
                    f.write(f"  Confidence: {result['confidence']:.4f}\n")
                    f.write(f"  Processing time: {result['processing_time']:.2f} seconds\n\n")
            else:
                f.write("No NSFW detection results available.\n\n")
            
            # Summary
            f.write("SUMMARY\n")
            f.write("-" * 80 + "\n")
            
            if deepfake_results:
                avg_deepfake_time = sum(r['processing_time'] for r in deepfake_results) / len(deepfake_results)
                f.write(f"Average deepfake detection time: {avg_deepfake_time:.2f} seconds\n")
            
            if nsfw_results:
                avg_nsfw_time = sum(r['processing_time'] for r in nsfw_results) / len(nsfw_results)
                f.write(f"Average NSFW detection time: {avg_nsfw_time:.2f} seconds\n")
            
            f.write("\nTest completed successfully.\n")
        
        logger.info(f"Test report generated: {report_path}")
        return report_path
    except Exception as e:
        logger.error(f"Error generating report: {str(e)}")
        return None

def main():
    """Run tests on the improved models"""
    print("=" * 80)
    print("SAFEGRAM IMPROVED MODELS TEST")
    print("=" * 80)
    
    # Create test images
    test_images = create_test_images()
    if not test_images:
        logger.error("Failed to create test images")
        return 1
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Test improved AI models')
    parser.add_argument('--deepfake', action='store_true', help='Test deepfake detection only')
    parser.add_argument('--nsfw', action='store_true', help='Test NSFW detection only')
    parser.add_argument('--image', type=str, help='Path to a specific image to test')
    args = parser.parse_args()
    
    # If a specific image is provided, use only that
    if args.image and os.path.exists(args.image):
        test_images = [args.image]
        logger.info(f"Testing with specific image: {args.image}")
    
    # Test deepfake detection
    deepfake_results = None
    if not args.nsfw or args.deepfake:
        deepfake_results = test_deepfake_detector(test_images)
    
    # Test NSFW detection
    nsfw_results = None
    if not args.deepfake or args.nsfw:
        nsfw_results = test_nsfw_detector(test_images)
    
    # Generate report
    report_path = generate_report(deepfake_results, nsfw_results)
    
    if report_path:
        print("\n" + "=" * 80)
        print(f"Test completed successfully. Report saved to: {report_path}")
        print("=" * 80)
        return 0
    else:
        print("\n" + "=" * 80)
        print("Test completed with errors. See log for details.")
        print("=" * 80)
        return 1

if __name__ == "__main__":
    sys.exit(main())
