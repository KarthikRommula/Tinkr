import os
import sys
import logging
import argparse
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='safegram.log'
)
logger = logging.getLogger("SafeGramServer")

def check_directories():
    """Check and create necessary directories"""
    dirs = [
        'app/services/models',
        'uploads',
        'temp_uploads',
        'test_images',
        'test_reports'
    ]
    
    for directory in dirs:
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            logger.info(f"Created directory: {directory}")

def check_models():
    """Check if the improved models exist"""
    model_paths = [
        os.path.join('app', 'services', 'models', 'improved_deepfake_model.h5'),
        os.path.join('app', 'services', 'models', 'improved_nsfw_model.h5')
    ]
    
    missing_models = []
    for path in model_paths:
        if not os.path.exists(path):
            missing_models.append(path)
    
    if missing_models:
        logger.warning(f"Missing models: {missing_models}")
        print("Some models are missing. Do you want to create them now? (y/n)")
        choice = input().lower()
        if choice == 'y':
            import download_pretrained_models
            download_pretrained_models.main()
        else:
            print("Warning: Running without proper models may cause issues.")
    else:
        logger.info("All required models are present.")

def run_server(host="0.0.0.0", port=8000):
    """Run the FastAPI server"""
    try:
        import uvicorn
        print(f"Starting SafeGram server on http://{host}:{port}")
        logger.info(f"Starting server on {host}:{port}")
        uvicorn.run("app.main:app", host=host, port=port, reload=True)
        return True
    except Exception as e:
        logger.error(f"Error starting server: {str(e)}")
        return False

def main():
    """Main function to start the SafeGram server"""
    parser = argparse.ArgumentParser(description='SafeGram Server with Improved AI Models')
    parser.add_argument('--host', type=str, default="0.0.0.0", help='Host to run the server on')
    parser.add_argument('--port', type=int, default=8000, help='Port to run the server on')
    args = parser.parse_args()
    
    # Check directories
    check_directories()
    
    # Check models
    check_models()
    
    # Run server
    success = run_server(args.host, args.port)
    if not success:
        logger.error("Failed to start server")
        return 1
    
    return 0

if __name__ == "__main__":
    print("=" * 80)
    print("SafeGram Server with Improved AI Models")
    print("=" * 80)
    sys.exit(main())
