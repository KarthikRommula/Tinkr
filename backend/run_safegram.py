# run_safegram.py
import os
import subprocess
import sys
import time
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("SafegramRunner")

def check_python_version():
    """Check if Python version is compatible"""
    required_version = (3, 7)
    current_version = sys.version_info
    
    if current_version < required_version:
        logger.error(f"Python {required_version[0]}.{required_version[1]} or higher is required. You have {current_version[0]}.{current_version[1]}")
        return False
    
    logger.info(f"Python version: {current_version[0]}.{current_version[1]}.{current_version[2]}")
    return True

def run_setup():
    """Run the setup script"""
    if os.path.exists("setup_safegram.py"):
        try:
            logger.info("Running setup script...")
            result = subprocess.run(
                [sys.executable, "setup_safegram.py"],
                check=True
            )
            logger.info("Setup completed successfully")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Error running setup: {e}")
            return False
    else:
        logger.error("Setup script not found")
        return False

def download_face_cascade():
    """Download the face cascade file"""
    if os.path.exists("download_face_cascade.py"):
        try:
            logger.info("Downloading face cascade file...")
            result = subprocess.run(
                [sys.executable, "download_face_cascade.py"],
                check=True
            )
            logger.info("Face cascade download completed")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Error downloading face cascade: {e}")
            return False
    else:
        logger.error("Face cascade download script not found")
        return False

def check_requirements():
    """Check and install required packages"""
    if os.path.exists("requirements.txt"):
        try:
            logger.info("Installing requirements...")
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
                check=True
            )
            logger.info("Requirements installed successfully")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Error installing requirements: {e}")
            return False
    else:
        logger.warning("requirements.txt not found, skipping package installation")
        return True

def start_server():
    """Start the FastAPI server"""
    try:
        logger.info("Starting the FastAPI server...")
        # Use a non-blocking call to start the server
        process = subprocess.Popen(
            ["uvicorn", "app.main:app", "--reload", "--host", "0.0.0.0", "--port", "8000"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Give it a moment to start
        time.sleep(2)
        
        # Check if the process is still running
        if process.poll() is None:
            logger.info("Server started successfully")
            logger.info("API will be available at: http://localhost:8000")
            logger.info("Press Ctrl+C to stop the server")
            
            # Print server output
            while True:
                output = process.stdout.readline()
                if output:
                    print(output.strip())
                error = process.stderr.readline()
                if error:
                    print(f"ERROR: {error.strip()}")
                
                # Check if the process is still running
                if process.poll() is not None:
                    break
                
                # Check for keyboard interrupt
                try:
                    time.sleep(0.1)
                except KeyboardInterrupt:
                    logger.info("Shutting down server...")
                    process.terminate()
                    break
                    
            return True
        else:
            stdout, stderr = process.communicate()
            logger.error(f"Server failed to start: {stderr}")
            return False
    except Exception as e:
        logger.error(f"Error starting server: {str(e)}")
        return False

def main():
    """Main function to set up and run SafeGram"""
    print("=" * 80)
    print("SafeGram - AI-Powered Image Sharing Platform")
    print("=" * 80)
    
    # Check Python version
    if not check_python_version():
        print("Python version check failed. Exiting.")
        return
    
    # Check and install requirements
    if not check_requirements():
        print("Failed to install requirements. Exiting.")
        return
    
    # Run the setup script
    if not run_setup():
        print("Setup failed. Exiting.")
        return
    
    # Download face cascade file
    if not download_face_cascade():
        print("Face cascade download failed, but continuing anyway...")
    
    # Ask if the user wants to start the server
    print("\nDo you want to start the SafeGram server now? (y/n)")
    choice = input().lower()
    if choice == 'y':
        # Start the server
        start_server()
    else:
        print("\nSetup completed. You can start the server manually with:")
        print("uvicorn app.main:app --reload")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nExiting SafeGram setup...")