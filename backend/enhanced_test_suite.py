import os
import sys
import logging
import numpy as np
import cv2
import time
import argparse
from datetime import datetime
import json
import matplotlib.pyplot as plt
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("safegram_testing.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("EnhancedModelTest")

# Import our improved detectors
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
try:
    from app.services.improved_deepfake import ImprovedDeepfakeDetector
    from app.services.improved_nsfw import ImprovedNSFWDetector
    MODELS_AVAILABLE = True
except ImportError as e:
    MODELS_AVAILABLE = False
    logger.error(f"Could not import improved models: {str(e)}")

def create_diverse_test_images():
    """Create a diverse set of test images for model evaluation"""
    test_dir = 'test_images'
    os.makedirs(test_dir, exist_ok=True)
    
    test_images = []
    
    # Test image categories
    categories = {
        'gradient': lambda: create_gradient_image(test_dir),
        'noise': lambda: create_noise_image(test_dir),
        'face': lambda: create_face_image(test_dir),
        'text': lambda: create_text_image(test_dir),
        'landscape': lambda: create_landscape_image(test_dir),
        'portrait': lambda: create_portrait_image(test_dir),
        'geometric': lambda: create_geometric_image(test_dir),
        'blurred': lambda: create_blurred_image(test_dir),
        'high_contrast': lambda: create_high_contrast_image(test_dir),
        'low_light': lambda: create_low_light_image(test_dir)
    }
    
    # Create each category of test image
    for name, create_func in categories.items():
        img_path = create_func()
        if img_path:
            test_images.append(img_path)
    
    return test_images

def create_gradient_image(test_dir):
    """Create a gradient test image"""
    path = os.path.join(test_dir, 'gradient.jpg')
    if not os.path.exists(path):
        try:
            arr = np.zeros((224, 224, 3), dtype=np.uint8)
            for i in range(224):
                for j in range(224):
                    arr[i, j, 0] = i // 2  # Red
                    arr[i, j, 1] = j // 2  # Green
                    arr[i, j, 2] = 100     # Blue
            
            cv2.imwrite(path, arr)
            logger.info(f"Created gradient test image: {path}")
        except Exception as e:
            logger.error(f"Failed to create gradient test image: {str(e)}")
            return None
    return path

def create_noise_image(test_dir):
    """Create a noise test image"""
    path = os.path.join(test_dir, 'noise.jpg')
    if not os.path.exists(path):
        try:
            arr = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
            cv2.imwrite(path, arr)
            logger.info(f"Created noise test image: {path}")
        except Exception as e:
            logger.error(f"Failed to create noise test image: {str(e)}")
            return None
    return path

def create_face_image(test_dir):
    """Create a face-like pattern image"""
    path = os.path.join(test_dir, 'face_pattern.jpg')
    if not os.path.exists(path):
        try:
            arr = np.zeros((224, 224, 3), dtype=np.uint8)
            # Draw a circle for a face
            cv2.circle(arr, (112, 112), 80, (200, 200, 200), -1)
            # Draw eyes
            cv2.circle(arr, (85, 90), 10, (50, 50, 50), -1)
            cv2.circle(arr, (140, 90), 10, (50, 50, 50), -1)
            # Draw mouth
            cv2.ellipse(arr, (112, 140), (30, 20), 0, 0, 180, (50, 50, 50), -1)
            
            cv2.imwrite(path, arr)
            logger.info(f"Created face pattern test image: {path}")
        except Exception as e:
            logger.error(f"Failed to create face pattern test image: {str(e)}")
            return None
    return path

def create_text_image(test_dir):
    """Create an image with text"""
    path = os.path.join(test_dir, 'text.jpg')
    if not os.path.exists(path):
        try:
            arr = np.ones((224, 224, 3), dtype=np.uint8) * 255
            # Add text
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(arr, 'SafeGram', (30, 112), font, 1, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(arr, 'Test Image', (40, 142), font, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
            
            cv2.imwrite(path, arr)
            logger.info(f"Created text test image: {path}")
        except Exception as e:
            logger.error(f"Failed to create text test image: {str(e)}")
            return None
    return path

def create_landscape_image(test_dir):
    """Create a simple landscape image"""
    path = os.path.join(test_dir, 'landscape.jpg')
    if not os.path.exists(path):
        try:
            arr = np.zeros((224, 224, 3), dtype=np.uint8)
            # Sky
            arr[:112, :, 0] = 100
            arr[:112, :, 1] = 150
            arr[:112, :, 2] = 200
            # Ground
            arr[112:, :, 0] = 100
            arr[112:, :, 1] = 120
            arr[112:, :, 2] = 80
            
            cv2.imwrite(path, arr)
            logger.info(f"Created landscape test image: {path}")
        except Exception as e:
            logger.error(f"Failed to create landscape test image: {str(e)}")
            return None
    return path

def create_portrait_image(test_dir):
    """Create a simple portrait-like image"""
    path = os.path.join(test_dir, 'portrait.jpg')
    if not os.path.exists(path):
        try:
            arr = np.zeros((224, 224, 3), dtype=np.uint8)
            # Background
            arr[:, :] = [200, 200, 200]
            # Head
            cv2.circle(arr, (112, 80), 60, (220, 180, 160), -1)
            # Body
            cv2.rectangle(arr, (72, 140), (152, 224), (100, 100, 150), -1)
            
            cv2.imwrite(path, arr)
            logger.info(f"Created portrait test image: {path}")
        except Exception as e:
            logger.error(f"Failed to create portrait test image: {str(e)}")
            return None
    return path

def create_geometric_image(test_dir):
    """Create an image with geometric shapes"""
    path = os.path.join(test_dir, 'geometric.jpg')
    if not os.path.exists(path):
        try:
            arr = np.zeros((224, 224, 3), dtype=np.uint8)
            # Background
            arr[:, :] = [240, 240, 240]
            # Shapes
            cv2.rectangle(arr, (50, 50), (100, 100), (255, 0, 0), -1)
            cv2.circle(arr, (150, 75), 25, (0, 255, 0), -1)
            cv2.line(arr, (50, 150), (174, 150), (0, 0, 255), 5)
            triangle_pts = np.array([[112, 120], [150, 170], [74, 170]], np.int32)
            cv2.fillPoly(arr, [triangle_pts], (255, 255, 0))
            
            cv2.imwrite(path, arr)
            logger.info(f"Created geometric test image: {path}")
        except Exception as e:
            logger.error(f"Failed to create geometric test image: {str(e)}")
            return None
    return path

def create_blurred_image(test_dir):
    """Create a blurred test image"""
    path = os.path.join(test_dir, 'blurred.jpg')
    if not os.path.exists(path):
        try:
            # Create a face image first
            face_arr = np.zeros((224, 224, 3), dtype=np.uint8)
            cv2.circle(face_arr, (112, 112), 80, (200, 200, 200), -1)
            cv2.circle(face_arr, (85, 90), 10, (50, 50, 50), -1)
            cv2.circle(face_arr, (140, 90), 10, (50, 50, 50), -1)
            cv2.ellipse(face_arr, (112, 140), (30, 20), 0, 0, 180, (50, 50, 50), -1)
            
            # Apply blur
            blurred = cv2.GaussianBlur(face_arr, (15, 15), 0)
            
            cv2.imwrite(path, blurred)
            logger.info(f"Created blurred test image: {path}")
        except Exception as e:
            logger.error(f"Failed to create blurred test image: {str(e)}")
            return None
    return path

def create_high_contrast_image(test_dir):
    """Create a high contrast test image"""
    path = os.path.join(test_dir, 'high_contrast.jpg')
    if not os.path.exists(path):
        try:
            arr = np.zeros((224, 224, 3), dtype=np.uint8)
            # Create a checkerboard pattern
            tile_size = 28
            for i in range(0, 224, tile_size):
                for j in range(0, 224, tile_size):
                    if (i // tile_size + j // tile_size) % 2 == 0:
                        arr[i:i+tile_size, j:j+tile_size] = [255, 255, 255]
                    else:
                        arr[i:i+tile_size, j:j+tile_size] = [0, 0, 0]
            
            cv2.imwrite(path, arr)
            logger.info(f"Created high contrast test image: {path}")
        except Exception as e:
            logger.error(f"Failed to create high contrast test image: {str(e)}")
            return None
    return path

def create_low_light_image(test_dir):
    """Create a low light test image"""
    path = os.path.join(test_dir, 'low_light.jpg')
    if not os.path.exists(path):
        try:
            # Create a base image
            arr = np.zeros((224, 224, 3), dtype=np.uint8)
            # Add some objects
            cv2.rectangle(arr, (50, 50), (150, 150), (50, 50, 50), -1)
            cv2.circle(arr, (150, 150), 30, (60, 60, 60), -1)
            
            # Darken the image
            arr = (arr * 0.3).astype(np.uint8)
            
            cv2.imwrite(path, arr)
            logger.info(f"Created low light test image: {path}")
        except Exception as e:
            logger.error(f"Failed to create low light test image: {str(e)}")
            return None
    return path

def test_deepfake_detector_comprehensive(image_paths):
    """Test the improved deepfake detector with comprehensive metrics"""
    if not MODELS_AVAILABLE:
        logger.error("Models not available. Cannot proceed with testing.")
        return None
    
    logger.info("Running comprehensive deepfake detection tests...")
    
    try:
        # Initialize the detector
        detector = ImprovedDeepfakeDetector()
        
        # Test each image with different thresholds
        thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
        results = []
        
        for image_path in image_paths:
            image_results = {'image': os.path.basename(image_path), 'thresholds': {}}
            
            for threshold in thresholds:
                start_time = time.time()
                is_deepfake, confidence = detector.detect(image_path, threshold=threshold)
                processing_time = time.time() - start_time
                
                image_results['thresholds'][str(threshold)] = {
                    'is_deepfake': is_deepfake,
                    'confidence': float(confidence),
                    'processing_time': processing_time
                }
            
            # Add overall results
            image_results['avg_processing_time'] = sum(r['processing_time'] for r in image_results['thresholds'].values()) / len(thresholds)
            image_results['confidence'] = image_results['thresholds']['0.5']['confidence']
            image_results['is_deepfake'] = image_results['thresholds']['0.5']['is_deepfake']
            
            results.append(image_results)
            logger.info(f"Tested {os.path.basename(image_path)} - Confidence: {image_results['confidence']:.4f}")
        
        return results
    except Exception as e:
        logger.error(f"Error in deepfake detection testing: {str(e)}")
        return None

def test_nsfw_detector_comprehensive(image_paths):
    """Test the improved NSFW detector with comprehensive metrics"""
    if not MODELS_AVAILABLE:
        logger.error("Models not available. Cannot proceed with testing.")
        return None
    
    logger.info("Running comprehensive NSFW detection tests...")
    
    try:
        # Initialize the detector
        detector = ImprovedNSFWDetector()
        
        # Test each image with different thresholds
        thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
        results = []
        
        for image_path in image_paths:
            image_results = {'image': os.path.basename(image_path), 'thresholds': {}}
            
            for threshold in thresholds:
                start_time = time.time()
                is_nsfw, confidence = detector.detect(image_path, threshold=threshold)
                processing_time = time.time() - start_time
                
                image_results['thresholds'][str(threshold)] = {
                    'is_nsfw': is_nsfw,
                    'confidence': float(confidence),
                    'processing_time': processing_time
                }
            
            # Add overall results
            image_results['avg_processing_time'] = sum(r['processing_time'] for r in image_results['thresholds'].values()) / len(thresholds)
            image_results['confidence'] = image_results['thresholds']['0.5']['confidence']
            image_results['is_nsfw'] = image_results['thresholds']['0.5']['is_nsfw']
            
            results.append(image_results)
            logger.info(f"Tested {os.path.basename(image_path)} - Confidence: {image_results['confidence']:.4f}")
        
        return results
    except Exception as e:
        logger.error(f"Error in NSFW detection testing: {str(e)}")
        return None

def generate_comprehensive_report(deepfake_results, nsfw_results):
    """Generate a comprehensive test report with visualizations"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_dir = os.path.join('test_reports')
    os.makedirs(report_dir, exist_ok=True)
    report_path = os.path.join(report_dir, f'model_test_report_{timestamp}.txt')
    json_report_path = os.path.join(report_dir, f'model_test_report_{timestamp}.json')
    
    # Save results as JSON for later analysis
    if deepfake_results or nsfw_results:
        try:
            with open(json_report_path, 'w') as f:
                json.dump({
                    'timestamp': timestamp,
                    'deepfake_results': deepfake_results,
                    'nsfw_results': nsfw_results
                }, f, indent=2)
            logger.info(f"JSON report saved to {json_report_path}")
        except Exception as e:
            logger.error(f"Error saving JSON report: {str(e)}")
    
    # Generate text report
    try:
        with open(report_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("SAFEGRAM ENHANCED MODEL TEST REPORT\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")
            
            # Deepfake detection results
            f.write("DEEPFAKE DETECTION RESULTS\n")
            f.write("-" * 80 + "\n")
            if deepfake_results:
                # Summary statistics
                avg_confidence = sum(r['confidence'] for r in deepfake_results) / len(deepfake_results)
                avg_processing_time = sum(r['avg_processing_time'] for r in deepfake_results) / len(deepfake_results)
                deepfake_count = sum(1 for r in deepfake_results if r['is_deepfake'])
                
                f.write(f"Number of images tested: {len(deepfake_results)}\n")
                f.write(f"Images detected as deepfakes: {deepfake_count} ({deepfake_count/len(deepfake_results)*100:.1f}%)\n")
                f.write(f"Average confidence score: {avg_confidence:.4f}\n")
                f.write(f"Average processing time: {avg_processing_time:.2f} seconds\n\n")
                
                f.write("Detailed Results:\n")
                for result in deepfake_results:
                    f.write(f"Image: {result['image']}\n")
                    f.write(f"  Deepfake: {'Yes' if result['is_deepfake'] else 'No'}\n")
                    f.write(f"  Confidence: {result['confidence']:.4f}\n")
                    f.write(f"  Avg Processing time: {result['avg_processing_time']:.2f} seconds\n")
                    f.write("  Threshold analysis:\n")
                    for threshold, data in result['thresholds'].items():
                        f.write(f"    Threshold {threshold}: {'Deepfake' if data['is_deepfake'] else 'Not Deepfake'} (Confidence: {data['confidence']:.4f})\n")
                    f.write("\n")
            else:
                f.write("No deepfake detection results available.\n\n")
            
            # NSFW detection results
            f.write("NSFW DETECTION RESULTS\n")
            f.write("-" * 80 + "\n")
            if nsfw_results:
                # Summary statistics
                avg_confidence = sum(r['confidence'] for r in nsfw_results) / len(nsfw_results)
                avg_processing_time = sum(r['avg_processing_time'] for r in nsfw_results) / len(nsfw_results)
                nsfw_count = sum(1 for r in nsfw_results if r['is_nsfw'])
                
                f.write(f"Number of images tested: {len(nsfw_results)}\n")
                f.write(f"Images detected as NSFW: {nsfw_count} ({nsfw_count/len(nsfw_results)*100:.1f}%)\n")
                f.write(f"Average confidence score: {avg_confidence:.4f}\n")
                f.write(f"Average processing time: {avg_processing_time:.2f} seconds\n\n")
                
                f.write("Detailed Results:\n")
                for result in nsfw_results:
                    f.write(f"Image: {result['image']}\n")
                    f.write(f"  NSFW: {'Yes' if result['is_nsfw'] else 'No'}\n")
                    f.write(f"  Confidence: {result['confidence']:.4f}\n")
                    f.write(f"  Avg Processing time: {result['avg_processing_time']:.2f} seconds\n")
                    f.write("  Threshold analysis:\n")
                    for threshold, data in result['thresholds'].items():
                        f.write(f"    Threshold {threshold}: {'NSFW' if data['is_nsfw'] else 'Safe'} (Confidence: {data['confidence']:.4f})\n")
                    f.write("\n")
            else:
                f.write("No NSFW detection results available.\n\n")
            
            # Overall summary
            f.write("OVERALL SUMMARY\n")
            f.write("-" * 80 + "\n")
            
            if deepfake_results and nsfw_results:
                total_avg_time = (
                    sum(r['avg_processing_time'] for r in deepfake_results) + 
                    sum(r['avg_processing_time'] for r in nsfw_results)
                ) / (len(deepfake_results) + len(nsfw_results))
                
                f.write(f"Total images tested: {len(deepfake_results)}\n")
                f.write(f"Average processing time across all tests: {total_avg_time:.2f} seconds\n")
                f.write(f"Deepfake detection rate: {deepfake_count/len(deepfake_results)*100:.1f}%\n")
                f.write(f"NSFW detection rate: {nsfw_count/len(nsfw_results)*100:.1f}%\n")
            
            f.write("\nTest completed successfully.\n")
        
        logger.info(f"Comprehensive test report generated: {report_path}")
        return report_path, json_report_path
    except Exception as e:
        logger.error(f"Error generating report: {str(e)}")
        return None, None

def generate_visualizations(deepfake_results, nsfw_results, report_dir='test_reports'):
    """Generate visualizations for test results"""
    if not deepfake_results and not nsfw_results:
        logger.error("No results to visualize")
        return
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    os.makedirs(report_dir, exist_ok=True)
    
    try:
        # Confidence distribution for deepfake detection
        if deepfake_results:
            plt.figure(figsize=(10, 6))
            confidences = [r['confidence'] for r in deepfake_results]
            plt.hist(confidences, bins=10, alpha=0.7, color='blue')
            plt.title('Deepfake Detection Confidence Distribution')
            plt.xlabel('Confidence Score')
            plt.ylabel('Number of Images')
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(report_dir, f'deepfake_confidence_dist_{timestamp}.png'))
            plt.close()
            
            # Processing time by image
            plt.figure(figsize=(12, 6))
            images = [r['image'] for r in deepfake_results]
            times = [r['avg_processing_time'] for r in deepfake_results]
            plt.bar(range(len(images)), times, alpha=0.7, color='green')
            plt.xticks(range(len(images)), images, rotation=90)
            plt.title('Deepfake Detection Processing Time by Image')
            plt.xlabel('Image')
            plt.ylabel('Processing Time (seconds)')
            plt.tight_layout()
            plt.savefig(os.path.join(report_dir, f'deepfake_processing_time_{timestamp}.png'))
            plt.close()
            
            # Threshold analysis
            plt.figure(figsize=(10, 6))
            thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
            detection_rates = []
            
            for threshold in thresholds:
                threshold_str = str(threshold)
                rate = sum(1 for r in deepfake_results if r['thresholds'][threshold_str]['is_deepfake']) / len(deepfake_results)
                detection_rates.append(rate * 100)
            
            plt.plot(thresholds, detection_rates, 'o-', linewidth=2, markersize=8)
            plt.title('Deepfake Detection Rate by Threshold')
            plt.xlabel('Threshold')
            plt.ylabel('Detection Rate (%)')
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(report_dir, f'deepfake_threshold_analysis_{timestamp}.png'))
            plt.close()
        
        # Confidence distribution for NSFW detection
        if nsfw_results:
            plt.figure(figsize=(10, 6))
            confidences = [r['confidence'] for r in nsfw_results]
            plt.hist(confidences, bins=10, alpha=0.7, color='red')
            plt.title('NSFW Detection Confidence Distribution')
            plt.xlabel('Confidence Score')
            plt.ylabel('Number of Images')
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(report_dir, f'nsfw_confidence_dist_{timestamp}.png'))
            plt.close()
            
            # Processing time by image
            plt.figure(figsize=(12, 6))
            images = [r['image'] for r in nsfw_results]
            times = [r['avg_processing_time'] for r in nsfw_results]
            plt.bar(range(len(images)), times, alpha=0.7, color='purple')
            plt.xticks(range(len(images)), images, rotation=90)
            plt.title('NSFW Detection Processing Time by Image')
            plt.xlabel('Image')
            plt.ylabel('Processing Time (seconds)')
            plt.tight_layout()
            plt.savefig(os.path.join(report_dir, f'nsfw_processing_time_{timestamp}.png'))
            plt.close()
            
            # Threshold analysis
            plt.figure(figsize=(10, 6))
            thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
            detection_rates = []
            
            for threshold in thresholds:
                threshold_str = str(threshold)
                rate = sum(1 for r in nsfw_results if r['thresholds'][threshold_str]['is_nsfw']) / len(nsfw_results)
                detection_rates.append(rate * 100)
            
            plt.plot(thresholds, detection_rates, 'o-', linewidth=2, markersize=8)
            plt.title('NSFW Detection Rate by Threshold')
            plt.xlabel('Threshold')
            plt.ylabel('Detection Rate (%)')
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(report_dir, f'nsfw_threshold_analysis_{timestamp}.png'))
            plt.close()
        
        logger.info(f"Visualizations generated in {report_dir}")
    except Exception as e:
        logger.error(f"Error generating visualizations: {str(e)}")

def main():
    """Run enhanced tests on the improved models"""
    print("=" * 80)
    print("SAFEGRAM ENHANCED MODEL TESTING")
    print("=" * 80)
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run enhanced tests on improved AI models')
    parser.add_argument('--deepfake', action='store_true', help='Test deepfake detection only')
    parser.add_argument('--nsfw', action='store_true', help='Test NSFW detection only')
    parser.add_argument('--visualize', action='store_true', help='Generate visualizations')
    parser.add_argument('--image', type=str, help='Path to a specific image to test')
    args = parser.parse_args()
    
    # Create diverse test images
    test_images = create_diverse_test_images()
    if not test_images:
        logger.error("Failed to create test images")
        return 1
    
    # If a specific image is provided, use only that
    if args.image and os.path.exists(args.image):
        test_images = [args.image]
        logger.info(f"Testing with specific image: {args.image}")
    
    # Test deepfake detection
    deepfake_results = None
    if not args.nsfw or args.deepfake:
        deepfake_results = test_deepfake_detector_comprehensive(test_images)
    
    # Test NSFW detection
    nsfw_results = None
    if not args.deepfake or args.nsfw:
        nsfw_results = test_nsfw_detector_comprehensive(test_images)
    
    # Generate report
    report_path, json_report_path = generate_comprehensive_report(deepfake_results, nsfw_results)
    
    # Generate visualizations if requested
    if args.visualize and (deepfake_results or nsfw_results):
        generate_visualizations(deepfake_results, nsfw_results)
    
    if report_path:
        print("\n" + "=" * 80)
        print(f"Test completed successfully. Report saved to: {report_path}")
        if json_report_path:
            print(f"JSON data saved to: {json_report_path}")
        print("=" * 80)
        return 0
    else:
        print("\n" + "=" * 80)
        print("Test completed with errors. See log for details.")
        print("=" * 80)
        return 1

if __name__ == "__main__":
    sys.exit(main())
