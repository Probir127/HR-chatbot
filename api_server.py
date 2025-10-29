# api_server.py - HR Chatbot API Server
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import backend
import os, json, hashlib, secrets
from datetime import datetime, timedelta

app = FastAPI(title="HR Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- User Models ---
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
    session_token: Optional[str] = None

# --- Data Files ---
USERS_FILE = "data/users.json"
SESSIONS_FILE = "data/sessions.json"

# --- Utility Functions ---
def load_json(path, default):
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return default

def save_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return hash_password(plain_password) == hashed_password

# --- User Management ---
def load_users(): return load_json(USERS_FILE, [])
def save_users(users): save_json(USERS_FILE, users)

def load_sessions(): return load_json(SESSIONS_FILE, {})
def save_sessions(sessions): save_json(SESSIONS_FILE, sessions)

def create_session(user_id: str) -> str:
    sessions = load_sessions()
    token = secrets.token_hex(32)
    sessions[token] = {
        "user_id": user_id,
        "created_at": datetime.now().isoformat(),
        "expires_at": (datetime.now() + timedelta(hours=24)).isoformat()
    }
    save_sessions(sessions)
    return token

def verify_session(token: str) -> Optional[str]:
    sessions = load_sessions()
    if token not in sessions:
        return None
    session = sessions[token]
    if datetime.now() > datetime.fromisoformat(session["expires_at"]):
        del sessions[token]
        save_sessions(sessions)
        return None
    return session["user_id"]

# --- In-Memory Short-Term Context Cache ---
SESSION_CONTEXTS = {}
CONTEXT_EXPIRY_MINUTES = 10
MAX_CONTEXT_MESSAGES = 3

def update_session_context(session_id: str, role: str, message: str):
    """Stores short-term conversational context per session."""
    now = datetime.now()
    if session_id not in SESSION_CONTEXTS:
        SESSION_CONTEXTS[session_id] = {"messages": [], "last_active": now}
    ctx = SESSION_CONTEXTS[session_id]
    ctx["messages"].append({"role": role, "content": message})
    ctx["messages"] = ctx["messages"][-MAX_CONTEXT_MESSAGES:]  # keep last few
    ctx["last_active"] = now

def get_recent_context(session_id: str):
    """Return recent messages if session still active."""
    ctx = SESSION_CONTEXTS.get(session_id)
    if not ctx:
        return []
    if datetime.now() - ctx["last_active"] > timedelta(minutes=CONTEXT_EXPIRY_MINUTES):
        del SESSION_CONTEXTS[session_id]
        return []
    return ctx["messages"]

# --- Authentication Endpoints ---
@app.post("/register")
async def register(user_data: UserRegister):
    users = load_users()
    if any(u["username"] == user_data.username for u in users):
        raise HTTPException(status_code=400, detail="Username already exists")
    if any(u["email"] == user_data.email for u in users):
        raise HTTPException(status_code=400, detail="Email already registered")

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
    return {"status": "success", "message": "User registered successfully", "user_id": new_user["id"]}

@app.post("/login")
async def login(login_data: UserLogin):
    users = load_users()
    user = next((u for u in users if u["username"] == login_data.username and u["is_active"]), None)
    if not user or not verify_password(login_data.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid username or password")

    token = create_session(user["id"])
    return {
        "status": "success",
        "message": "Login successful",
        "token": token,
        "user": {"id": user["id"], "username": user["username"], "email": user["email"], "employee_id": user.get("employee_id")}
    }

@app.post("/logout")
async def logout(token: str):
    sessions = load_sessions()
    if token in sessions:
        del sessions[token]
        save_sessions(sessions)
    return {"status": "success", "message": "Logout successful"}

@app.get("/verify-session")
async def verify_session_endpoint(token: str):
    user_id = verify_session(token)
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid or expired session")
    users = load_users()
    user = next((u for u in users if u["id"] == user_id), None)
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    return {
        "status": "success",
        "user": {"id": user["id"], "username": user["username"], "email": user["email"], "employee_id": user.get("employee_id")}
    }

# --- Chat Endpoints ---
@app.get("/")
def root():
    return {"status": "HR Chatbot API Running"}

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        # Determine session context (anonymous or user-based)
        session_id = request.session_token or "anon"

        # Load previous short-term context
        recent_context = get_recent_context(session_id)
        combined_history = recent_context + [{"role": "user", "content": request.message}]

        # Generate reply
        reply = backend.ask_hr_bot(user_input=request.message, chat_history=combined_history, session_id=session_id)

        # Store this exchange for next query
        update_session_context(session_id, "user", request.message)
        update_session_context(session_id, "assistant", reply)

        return {"response": reply, "context": combined_history[-MAX_CONTEXT_MESSAGES:]}
    except Exception as e:
        print("❌ Chat Error:", e)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health():
    return {"status": "ok"}
