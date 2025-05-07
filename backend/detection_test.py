# detection_test.py
import os
import argparse
import logging
from app.services.deepfake import DeepfakeDetector
from app.services.nsfw import NSFWDetector
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("DetectionTest")

def test_detection(image_path, model_dir="app/services/models"):
    """Test both detection systems on a single image"""
    # Initialize detectors
    deepfake_detector = DeepfakeDetector(
        model_path=os.path.join(model_dir, "deepfake_model.h5")
    )
    nsfw_detector = NSFWDetector(
        model_path=os.path.join(model_dir, "nsfw_model.h5")
    )
    
    # Check if image exists
    if not os.path.exists(image_path):
        logger.error(f"Image not found: {image_path}")
        return False
    
    # Test deepfake detection
    logger.info("Testing deepfake detection...")
    try:
        is_deepfake = deepfake_detector.detect(image_path)
        logger.info(f"Deepfake detection result: {'Rejected' if is_deepfake else 'Accepted'}")
    except Exception as e:
        logger.error(f"Deepfake detection error: {str(e)}")
        return False
    
    # Test NSFW detection
    logger.info("Testing NSFW detection...")
    try:
        is_nsfw = nsfw_detector.detect(image_path)
        logger.info(f"NSFW detection result: {'Rejected' if is_nsfw else 'Accepted'}")
    except Exception as e:
        logger.error(f"NSFW detection error: {str(e)}")
        return False
    
    # Overall result
    if is_deepfake or is_nsfw:
        logger.info("OVERALL RESULT: Image would be REJECTED")
    else:
        logger.info("OVERALL RESULT: Image would be ACCEPTED")
    
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test image detection systems")
    parser.add_argument("image_path", help="Path to the image to test")
    args = parser.parse_args()
    
    # Run the test
    success = test_detection(args.image_path)
    if not success:
        print("Detection test failed. Check the logs for details.")
        exit(1)