from fastapi import FastAPI, UploadFile, File, Depends, HTTPException, status, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from datetime import datetime, timedelta
from typing import List, Optional
import jwt
from passlib.context import CryptContext
import os
import shutil
from uuid import uuid4
import logging

# Import AI services
from app.services.deepfake import DeepfakeDetector
from app.services.nsfw import NSFWDetector

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='safegram.log'
)
logger = logging.getLogger("SafeGram")

# Create the FastAPI app
app = FastAPI(title="SafeGram API")

# Set up upload directory
UPLOAD_DIR = "uploads"
TEMP_DIR = "temp_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

# Mount the uploads directory as a static files directory
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")

# Set up CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development - restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security setup
SECRET_KEY = "your-secret-key"  # Change this in production!
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Mock database for demo purposes
fake_users_db = {}
posts_db = []

# Initialize AI detectors
deepfake_detector = DeepfakeDetector()
nsfw_detector = NSFWDetector()

# Models
class User(BaseModel):
    username: str
    email: str
    full_name: Optional[str] = None
    disabled: Optional[bool] = None

class UserInDB(User):
    hashed_password: str

class UserCreate(User):
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None

class Post(BaseModel):
    id: str
    user_id: str
    image_url: str
    caption: Optional[str] = None
    created_at: datetime

class PostCreate(BaseModel):
    caption: Optional[str] = None

class DetectionResult(BaseModel):
    is_safe: bool
    reason: Optional[str] = None

# Helper functions
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def get_user(db, username: str):
    if username in db:
        user_dict = db[username]
        return UserInDB(**user_dict)

def authenticate_user(db, username: str, password: str):
    user = get_user(db, username)
    if not user:
        return False
    if not verify_password(password, user.hashed_password):
        return False
    return user

def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except jwt.PyJWTError:
        raise credentials_exception
    user = get_user(fake_users_db, username=token_data.username)
    if user is None:
        raise credentials_exception
    return user

async def get_current_active_user(current_user: User = Depends(get_current_user)):
    if current_user.disabled:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user

# Content moderation functions
async def check_image_safety(file_path: str) -> DetectionResult:
    """Check if an image is safe (no deepfakes or NSFW content)"""
    logger.info(f"Checking image safety: {file_path}")
    
    # Check for deepfakes
    is_deepfake = deepfake_detector.detect(file_path)
    if is_deepfake:
        logger.warning(f"Deepfake detected in image: {file_path}")
        return DetectionResult(is_safe=False, reason="Detected deepfake image. Upload rejected.")
    
    # Check for NSFW content
    is_nsfw = nsfw_detector.detect(file_path)
    if is_nsfw:
        logger.warning(f"NSFW content detected in image: {file_path}")
        return DetectionResult(is_safe=False, reason="Detected explicit content. Upload rejected.")
    
    # If both checks pass, the image is safe
    logger.info(f"Image passed safety checks: {file_path}")
    return DetectionResult(is_safe=True)

def cleanup_temp_files(file_paths: List[str]):
    """Clean up temporary files after processing"""
    for path in file_paths:
        if os.path.exists(path):
            try:
                os.remove(path)
                logger.info(f"Cleaned up temporary file: {path}")
            except Exception as e:
                logger.error(f"Error cleaning up file {path}: {str(e)}")

# Routes
@app.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(fake_users_db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/users/", response_model=User)
async def create_user(user: UserCreate):
    if user.username in fake_users_db:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already exists"
        )
    hashed_password = get_password_hash(user.password)
    user_dict = user.dict()
    user_dict.pop("password")
    user_dict["hashed_password"] = hashed_password
    fake_users_db[user.username] = user_dict
    return user_dict

@app.get("/users/me", response_model=User)
async def read_users_me(current_user: User = Depends(get_current_active_user)):
    return current_user

@app.post("/upload/", response_model=Post)
async def upload_image(
    file: UploadFile = File(...),
    caption: Optional[str] = None,
    background_tasks: BackgroundTasks = BackgroundTasks(),
    current_user: User = Depends(get_current_active_user)
):
    # Save the uploaded file temporarily
    temp_file_path = f"{TEMP_DIR}/{uuid4().hex}_{file.filename}"
    
    try:
        # Save the file
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        logger.info(f"Temporary file saved: {temp_file_path}")
        
        # Check image safety (deepfake and NSFW detection)
        safety_result = await check_image_safety(temp_file_path)
        
        if not safety_result.is_safe:
            # Schedule cleanup in the background
            background_tasks.add_task(cleanup_temp_files, [temp_file_path])
            
            # Return detailed error
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=safety_result.reason
            )
        
        # If the image passes safety checks, save it permanently
        final_filename = f"{uuid4().hex}_{file.filename}"
        final_path = f"{UPLOAD_DIR}/{final_filename}"
        shutil.copy(temp_file_path, final_path)
        
        # Schedule cleanup of temp file
        background_tasks.add_task(cleanup_temp_files, [temp_file_path])
        
        # Create post
        post_id = str(uuid4())
        new_post = {
            "id": post_id,
            "user_id": current_user.username,
            "image_url": final_path,
            "caption": caption,
            "created_at": datetime.now()
        }
        posts_db.append(new_post)
        
        logger.info(f"Post created successfully: {post_id}")
        return new_post
    
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        # Clean up the temporary file if it exists
        background_tasks.add_task(cleanup_temp_files, [temp_file_path])
        
        logger.error(f"Error processing image: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing image: {str(e)}"
        )

@app.get("/feed/", response_model=List[Post])
async def get_feed(current_user: User = Depends(get_current_active_user)):
    # In a real app, you'd filter the feed based on followed users
    # For this demo, we'll return all posts
    return posts_db

# Root endpoint
@app.get("/")
async def root():
    return {"message": "Welcome to SafeGram API"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)