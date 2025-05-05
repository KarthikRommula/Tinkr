from fastapi import FastAPI, UploadFile, File, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
from datetime import datetime, timedelta
from typing import List, Optional
import jwt
from passlib.context import CryptContext
import os
import shutil
from uuid import uuid4

# Import AI services
from app.services.deepfake import DeepfakeDetector
from app.services.nsfw import NSFWDetector

# Create the FastAPI app
app = FastAPI(title="SafeGram API")

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

# Set up upload directory
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

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
    current_user: User = Depends(get_current_active_user)
):
    # Save the uploaded file temporarily
    temp_file_path = f"{UPLOAD_DIR}/{uuid4().hex}_{file.filename}"
    with open(temp_file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
        # Check for deepfakes
        is_deepfake = deepfake_detector.detect(temp_file_path)
        if is_deepfake:
            os.remove(temp_file_path)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Detected deepfake image. Upload rejected."
            )
        
        # Check for NSFW content
        is_nsfw = nsfw_detector.detect(temp_file_path)
        if is_nsfw:
            os.remove(temp_file_path)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Detected explicit content. Upload rejected."
            )
        
        # If both checks pass, save the image
        final_path = f"{UPLOAD_DIR}/{uuid4().hex}_{file.filename}"
        os.rename(temp_file_path, final_path)
        
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
        
        return new_post
    
    except Exception as e:
        # Clean up the temporary file if it exists
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
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