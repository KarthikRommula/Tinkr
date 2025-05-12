# SafeGram Backend

SafeGram is a secure image sharing platform with advanced AI-powered content moderation to detect and filter deepfake and NSFW content.

## Features

### AI Models
- **Improved Deepfake Detection**: Using transfer learning with EfficientNetB0
- **Improved NSFW Detection**: Using transfer learning with MobileNetV2
- **Model Training**: Support for training models on custom datasets
- **Comprehensive Testing**: Enhanced test suite with visualization and reporting

### Security Enhancements
- **Rate Limiting**: Prevents brute force attacks and API abuse
- **Security Headers**: Protects against common web vulnerabilities
- **Input Validation**: Prevents injection attacks and validates user input
- **Logging**: Comprehensive request and error logging

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
- `--epochs`: Number of epochs to train for
- `--batch-size`: Batch size for training
- `--download-samples`: Download sample training data

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

### Admin
- `GET /admin/rate-limit-status/{client_ip}`: Check rate limit status
- `POST /admin/reset-rate-limit/{client_ip}`: Reset rate limit for a client

## Security Features

### Rate Limiting
The application uses a token bucket algorithm for flexible rate limiting:
- Default: 60 requests per minute
- Burst limit: 10 requests
- IP ban threshold: 100 violations
- Ban duration: 30 minutes

### Input Validation
- Username validation: 3-30 alphanumeric characters and underscores
- Email validation: Standard email format
- Password strength: At least 8 characters with uppercase, lowercase, number, and special character
- Protection against XSS, SQL injection, and command injection

### Security Headers
- X-Content-Type-Options: nosniff
- X-Frame-Options: DENY
- X-XSS-Protection: 1; mode=block
- Strict-Transport-Security: max-age=31536000; includeSubDomains
- Content-Security-Policy: Restrictive policy to prevent XSS
- Referrer-Policy: strict-origin-when-cross-origin
- Permissions-Policy: Restricts browser features

## Future Improvements

1. **Model Training**: Train models on high-quality datasets for better performance
2. **UI Improvements**: Optimize the frontend for mobile responsiveness
3. **Performance Optimization**: Implement caching and batch processing for AI screening
4. **Security Hardening**: Add more advanced security measures
5. **Distributed Processing**: Scale the application for higher throughput
