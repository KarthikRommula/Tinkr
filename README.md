# SafeGram Backend

SafeGram is a secure image sharing platform with advanced AI-powered content moderation to detect and filter deepfake and NSFW content. The platform uses state-of-the-art deep learning models to ensure all shared content is authentic and appropriate.

## Features

### AI Models
- **Improved Deepfake Detection**: Using transfer learning with EfficientNetB0 to identify manipulated facial images with high accuracy
- **Improved NSFW Detection**: Using transfer learning with MobileNetV2 for efficient and accurate inappropriate content filtering
- **Custom Model Training**: Support for training models on your own datasets with flexible configuration options
- **Comprehensive Testing**: Enhanced test suite with visualization, reporting, and performance metrics

### Performance
- **Fast Processing**: Optimized inference for quick content moderation (typically <3s per image)
- **Confidence Scoring**: Detailed confidence metrics for both deepfake and NSFW detection
- **Threshold Tuning**: Configurable sensitivity thresholds to balance safety and usability

### Additional Features
- **Comprehensive Logging**: Detailed request and error logging for monitoring and debugging
- **Error Handling**: Robust error handling with informative messages

## Getting Started

### Prerequisites
- Python 3.8+
- TensorFlow 2.16+
- FastAPI

### Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download pre-trained models:
```bash
python download_pretrained_models.py
```

4. Run the server:
```bash
python start_safegram.py
```

## Environment Configuration

Configure the application by setting the following environment variables in the `.env` file:

```
# Development Mode
SAFEGRAM_DEV_MODE=False

# Error Handling
REJECT_ON_MODEL_FAILURE=True
REJECT_ON_IMAGE_ERROR=True
REJECT_ON_ERROR=True

# AI Model Bypass (for testing)
ALWAYS_PASS_DEEPFAKE=False
ALWAYS_PASS_NSFW=False

# Directories
AI_MODEL_DIR=app/services/models
UPLOAD_DIR=uploads
TEMP_DIR=temp_uploads

# AI Model Thresholds
DEEPFAKE_THRESHOLD=0.5
NSFW_THRESHOLD=0.8

# TensorFlow Configuration
TENSORFLOW_LOG_LEVEL=0

# Security
JWT_SECRET_KEY=your-secret-key-here
ACCESS_TOKEN_EXPIRE_MINUTES=30
```

## Model Training

Train the AI models on your own datasets:

```bash
python train_models.py --download-samples
```

Options:
- `--deepfake`: Train only the deepfake detection model
- `--nsfw`: Train only the NSFW detection model
- `--epochs`: Number of epochs to train for (default: 20)
- `--batch-size`: Batch size for training (default: 32)
- `--download-samples`: Download sample training data

### Custom Dataset Structure

To train with your own images, organize them in the following directory structure:

```
training_data/
├── deepfake/
│   ├── train/
│   │   ├── real/     <- Real face images (100-500+ recommended)
│   │   └── fake/     <- Deepfake images (100-500+ recommended)
│   └── val/
│       ├── real/     <- Validation real images (20-30% of dataset)
│       └── fake/     <- Validation fake images (20-30% of dataset)
├── nsfw/
    ├── train/
    │   ├── safe/     <- Safe content images (100-500+ recommended)
    │   └── nsfw/     <- NSFW content images (100-500+ recommended)
    └── val/
        ├── safe/     <- Validation safe images (20-30% of dataset)
        └── nsfw/     <- Validation NSFW images (20-30% of dataset)
```

The training script will automatically resize images to 224x224 pixels and apply data augmentation to improve model robustness.

## Enhanced Testing

Run comprehensive tests on the AI models:

```bash
python enhanced_test_suite.py --visualize
```

Options:
- `--deepfake`: Test only the deepfake detection model
- `--nsfw`: Test only the NSFW detection model
- `--visualize`: Generate visualizations of test results
- `--image`: Path to a specific image to test

## API Endpoints

### Authentication
- `POST /token`: Get access token
- `POST /users/`: Create a new user
- `GET /users/me`: Get current user information

### Content
- `POST /upload/`: Upload an image (with AI safety checks)
- `GET /feed/`: Get image feed

## API Response Codes

- `200 OK`: Request successful
- `400 Bad Request`: Invalid input or rejected content
- `401 Unauthorized`: Authentication required
- `403 Forbidden`: Insufficient permissions
- `404 Not Found`: Resource not found
- `500 Internal Server Error`: Server-side error

## Frontend

SafeGram includes a React-based frontend with the following features:

### Authentication
- User registration and login
- JWT-based authentication
- Protected routes for authenticated users

### User Interface
- Modern, responsive design using Tailwind CSS
- Image upload with real-time feedback
- Feed view for browsing uploaded images
- Detailed error messages for rejected content

### Development

To run the frontend:

```bash
cd frontend
npm install
npm run dev
```

The frontend will be available at http://localhost:5174 (or another port if 5174 is in use).

### Directory Structure

The frontend follows a standard React project structure with special attention to the authentication context:

```
frontend/
├── src/
│   ├── components/    <- Reusable UI components
│   ├── context/       <- Authentication context
│   ├── pages/         <- Page components
│   └── App.jsx        <- Main application component
```

## Future Improvements

1. **Model Training**: Train models on high-quality datasets for better performance
2. **UI Improvements**: Enhance the frontend with additional features and animations
3. **Performance Optimization**: Implement caching and batch processing for AI screening
4. **Security Hardening**: Add more advanced security measures
5. **Distributed Processing**: Scale the application for higher throughput
6. **Mobile App**: Develop native mobile applications
