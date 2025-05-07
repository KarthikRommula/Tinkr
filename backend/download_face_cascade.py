# download_face_cascade.py
import os
import urllib.request
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("CascadeDownloader")

def download_face_cascade():
    """Download the face cascade file for OpenCV"""
    cascade_dir = os.path.join('app', 'services', 'models')
    os.makedirs(cascade_dir, exist_ok=True)
    
    cascade_file = os.path.join(cascade_dir, 'haarcascade_frontalface_default.xml')
    
    # URL for the Haar cascade file
    url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
    
    if os.path.exists(cascade_file):
        logger.info(f"Face cascade file already exists at {cascade_file}")
        return True
    
    try:
        logger.info(f"Downloading face cascade file from {url}")
        urllib.request.urlretrieve(url, cascade_file)
        logger.info(f"Downloaded face cascade file to {cascade_file}")
        return True
    except Exception as e:
        logger.error(f"Error downloading face cascade file: {str(e)}")
        return False

if __name__ == "__main__":
    if download_face_cascade():
        print("Face cascade file downloaded successfully!")
    else:
        print("Failed to download face cascade file.")