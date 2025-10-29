# api_server.py - ADD THESE IMPORTS AND MODELS
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import backend
import os, json
import hashlib
import secrets
from datetime import datetime, timedelta

app = FastAPI(title="HR Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# User models
class UserRegister(BaseModel):
    username: str
    email: str
    password: str
    employee_id: Optional[str] = None

class UserLogin(BaseModel):
    username: str
    password: str

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: str
    chat_history: Optional[List[Message]] = []

# User data storage
USERS_FILE = "data/users.json"
SESSIONS_FILE = "data/sessions.json"

def load_users():
    """Load users from JSON file"""
    try:
        if os.path.exists(USERS_FILE):
            with open(USERS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        return []
    except Exception:
        return []

def save_users(users):
    """Save users to JSON file"""
    os.makedirs("data", exist_ok=True)
    with open(USERS_FILE, "w", encoding="utf-8") as f:
        json.dump(users, f, indent=4, ensure_ascii=False)

def load_sessions():
    """Load sessions from JSON file"""
    try:
        if os.path.exists(SESSIONS_FILE):
            with open(SESSIONS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}
    except Exception:
        return {}

def save_sessions(sessions):
    """Save sessions to JSON file"""
    os.makedirs("data", exist_ok=True)
    with open(SESSIONS_FILE, "w", encoding="utf-8") as f:
        json.dump(sessions, f, indent=4, ensure_ascii=False)

def hash_password(password: str) -> str:
    """Hash password using SHA-256"""
    return hashlib.sha256(password.encode()).hexdigest()

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify password against hash"""
    return hash_password(plain_password) == hashed_password

def create_session(user_id: str) -> str:
    """Create a new session token"""
    sessions = load_sessions()
    token = secrets.token_hex(32)
    
    # Store session with expiry (24 hours)
    sessions[token] = {
        "user_id": user_id,
        "created_at": datetime.now().isoformat(),
        "expires_at": (datetime.now() + timedelta(hours=24)).isoformat()
    }
    
    save_sessions(sessions)
    return token

def verify_session(token: str) -> Optional[str]:
    """Verify session token and return user_id if valid"""
    sessions = load_sessions()
    
    if token not in sessions:
        return None
    
    session = sessions[token]
    expires_at = datetime.fromisoformat(session["expires_at"])
    
    if datetime.now() > expires_at:
        # Session expired
        del sessions[token]
        save_sessions(sessions)
        return None
    
    return session["user_id"]

# Authentication endpoints
@app.post("/register")
async def register(user_data: UserRegister):
    """Register a new user"""
    users = load_users()
    
    # Check if username already exists
    if any(user["username"] == user_data.username for user in users):
        raise HTTPException(status_code=400, detail="Username already exists")
    
    # Check if email already exists
    if any(user["email"] == user_data.email for user in users):
        raise HTTPException(status_code=400, detail="Email already registered")
    
    # Create new user
    new_user = {
        "id": secrets.token_hex(16),
        "username": user_data.username,
        "email": user_data.email,
        "password_hash": hash_password(user_data.password),
        "employee_id": user_data.employee_id,
        "created_at": datetime.now().isoformat(),
        "is_active": True
    }
    
    users.append(new_user)
    save_users(users)
    
    return {
        "status": "success",
        "message": "User registered successfully",
        "user_id": new_user["id"]
    }

@app.post("/login")
async def login(login_data: UserLogin):
    """User login"""
    users = load_users()
    
    # Find user by username
    user = next((u for u in users if u["username"] == login_data.username and u["is_active"]), None)
    
    if not user or not verify_password(login_data.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid username or password")
    
    # Create session
    token = create_session(user["id"])
    
    return {
        "status": "success",
        "message": "Login successful",
        "token": token,
        "user": {
            "id": user["id"],
            "username": user["username"],
            "email": user["email"],
            "employee_id": user.get("employee_id")
        }
    }

@app.post("/logout")
async def logout(token: str):
    """User logout"""
    sessions = load_sessions()
    
    if token in sessions:
        del sessions[token]
        save_sessions(sessions)
    
    return {"status": "success", "message": "Logout successful"}

@app.get("/verify-session")
async def verify_session_endpoint(token: str):
    """Verify session token"""
    user_id = verify_session(token)
    
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid or expired session")
    
    users = load_users()
    user = next((u for u in users if u["id"] == user_id), None)
    
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    
    return {
        "status": "success",
        "user": {
            "id": user["id"],
            "username": user["username"],
            "email": user["email"],
            "employee_id": user.get("employee_id")
        }
    }

# Existing endpoints
@app.get("/")
def root():
    return {"status":"HR Chatbot API Running"}

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        reply = backend.ask_hr_bot(user_input=request.message, chat_history=[{"role":m.role,"content":m.content} for m in request.chat_history])
        return {"response": reply}
    except Exception as e:
        print("❌ Chat Error:", e)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health():
    return {"status":"ok"}
