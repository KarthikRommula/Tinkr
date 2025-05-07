# run_model_tests.py (continued)
import os
import sys
import logging
import subprocess
import numpy as np
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ModelTests")

def run_test(test_name, command):
    """Run a test command and log the result"""
    logger.info(f"Running test: {test_name}")
    try:
        result = subprocess.run(
            command,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        logger.info(f"{test_name} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"{test_name} failed: {e}")
        logger.error(f"STDOUT: {e.stdout}")
        logger.error(f"STDERR: {e.stderr}")
        return False

def create_test_image():
    """Create a simple test image for detection testing"""
    test_image = "test_images/sample.jpg"
    
    # Create directory if it doesn't exist
    os.makedirs("test_images", exist_ok=True)
    
    if not os.path.exists(test_image):
        try:
            # Create a simple gradient image
            arr = np.zeros((224, 224, 3), dtype=np.uint8)
            for i in range(224):
                for j in range(224):
                    arr[i, j, 0] = i // 2  # Red
                    arr[i, j, 1] = j // 2  # Green
                    arr[i, j, 2] = 100     # Blue
            
            img = Image.fromarray(arr)
            img.save(test_image)
            logger.info(f"Created test image: {test_image}")
            return test_image
        except Exception as e:
            logger.error(f"Failed to create test image: {str(e)}")
            return None
    return test_image

def main():
    """Run all model-related tests"""
    # Create test image
    test_image = create_test_image()
    if not test_image:
        logger.error("Could not create test image - aborting tests")
        return 1
    
    # Define tests to run
    tests = [
        ("Directory Check", ["python", "check_directories.py"]),
        ("TensorFlow Check", ["python", "tensorflow_setup.py"]),
        ("Create Placeholder Models", ["python", "improved_placeholder_models.py"]),
        ("Detection Test", ["python", "detection_test.py", test_image])
    ]
    
    # Run all tests
    results = []
    for test_name, command in tests:
        result = run_test(test_name, command)
        results.append((test_name, result))
    
    # Print summary
    logger.info("\n===== TEST SUMMARY =====")
    all_passed = True
    for test_name, result in results:
        status = "PASSED" if result else "FAILED"
        logger.info(f"{test_name}: {status}")
        if not result:
            all_passed = False
    
    if all_passed:
        logger.info("\nAll tests passed! AI models should be working correctly.")
        return 0
    else:
        logger.error("\nSome tests failed. Review the logs for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main())