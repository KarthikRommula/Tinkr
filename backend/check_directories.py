# check_directories.py
import os
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ModelDiagnostics")

# Define required directories
required_dirs = [
    "app/services/models",
    "uploads",
    "temp_uploads"
]

def check_directories():
    """Check and create necessary directories"""
    for directory in required_dirs:
        if not os.path.exists(directory):
            logger.info(f"Creating missing directory: {directory}")
            os.makedirs(directory, exist_ok=True)
        else:
            logger.info(f"Directory exists: {directory}")
            
    # Check permissions
    for directory in required_dirs:
        try:
            test_file = os.path.join(directory, ".permission_test")
            with open(test_file, "w") as f:
                f.write("test")
            os.remove(test_file)
            logger.info(f"Directory {directory} is writable")
        except Exception as e:
            logger.error(f"Directory {directory} has permission issues: {str(e)}")
            
if __name__ == "__main__":
    check_directories()