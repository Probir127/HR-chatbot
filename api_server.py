# api_server.py - ULTRA-FAST HR Chatbot API Server with Multi-threading
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from collections import OrderedDict
import backend
import os
import json
import hashlib
import secrets
from datetime import datetime, timedelta
import threading
import time
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
import asyncio

# --- PERFORMANCE: Thread pool for parallel processing ---
THREAD_POOL = ThreadPoolExecutor(max_workers=10)

# --- STARTUP/SHUTDOWN LIFECYCLE ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("\n" + "="*70)
    print("🚀 STARTING ULTRA-FAST HR CHATBOT API SERVER")
    print("="*70)
    
    print("📚 Preloading FAISS knowledge base...")
    start = time.time()
    backend.load_faiss_index()
    elapsed = time.time() - start
    print(f"✅ FAISS loaded in {elapsed:.2f}s")
    
    cleanup_thread = threading.Thread(target=periodic_cleanup, daemon=True)
    cleanup_thread.start()
    print("🧹 Background session cleanup started")
    
    print("="*70)
    print("✅ SERVER READY - ULTRA-FAST MODE")
    print("="*70 + "\n")
    
    yield
    
    print("\n🛑 Shutting down gracefully...")
    THREAD_POOL.shutdown(wait=True)

app = FastAPI(
    title="HR Chatbot API - Ultra-Fast",
    version="3.0-FAST",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- THREAD SAFETY LOCKS ---
USER_LOCK = threading.Lock()
SESSION_LOCK = threading.Lock()
CONTEXT_LOCK = threading.Lock()

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

# --- DATA & CACHE ---
USERS_FILE = "data/users.json"
SESSIONS_FILE = "data/sessions.json"
USER_CACHE = {}
SESSION_CACHE = {}
CACHE_EXPIRY = 300

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
        json.dump(data, f, indent=2, ensure_ascii=False)

def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return hash_password(plain_password) == hashed_password

# --- OPTIMIZED: User Management with Caching ---
def load_users():
    global USER_CACHE
    if USER_CACHE:
        return USER_CACHE
    with USER_LOCK:
        USER_CACHE = load_json(USERS_FILE, [])
        return USER_CACHE

def save_users(users):
    global USER_CACHE
    USER_CACHE = users
    with USER_LOCK:
        save_json(USERS_FILE, users)

def load_sessions():
    global SESSION_CACHE
    if SESSION_CACHE:
        return SESSION_CACHE
    with SESSION_LOCK:
        SESSION_CACHE = load_json(SESSIONS_FILE, {})
        return SESSION_CACHE

def save_sessions(sessions):
    global SESSION_CACHE
    SESSION_CACHE = sessions
    with SESSION_LOCK:
        save_json(SESSIONS_FILE, sessions)

def create_session(user_id: str) -> str:
    sessions = load_sessions()
    token = secrets.token_hex(16)
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
        with SESSION_LOCK:
            del sessions[token]
            save_sessions(sessions)
        return None
    return session["user_id"]

# --- ULTRA-OPTIMIZED: In-Memory Context with Fast Access (Multi-User Safety) ---
SESSION_CONTEXTS = OrderedDict()
MAX_SESSIONS = 200
MAX_CONTEXT_MESSAGES = 4  # Kept low for speed
CONTEXT_EXPIRY_MINUTES = 30

def update_session_context(session_id: str, role: str, message: str):
    """Lightning-fast session context update, thread-safe."""
    now = datetime.now()
    
    with CONTEXT_LOCK:
        if session_id not in SESSION_CONTEXTS:
            SESSION_CONTEXTS[session_id] = {
                "messages": [],
                "last_active": now
            }
        
        ctx = SESSION_CONTEXTS[session_id]
        ctx["messages"].append({"role": role, "content": message})
        ctx["messages"] = ctx["messages"][-MAX_CONTEXT_MESSAGES:]
        ctx["last_active"] = now
        SESSION_CONTEXTS.move_to_end(session_id)
        
        if len(SESSION_CONTEXTS) > MAX_SESSIONS:
            SESSION_CONTEXTS.popitem(last=False)

def get_recent_context(session_id: str) -> List[dict]:
    """Fast context retrieval, thread-safe."""
    with CONTEXT_LOCK:
        ctx = SESSION_CONTEXTS.get(session_id)
        if not ctx:
            return []
        
        if datetime.now() - ctx["last_active"] > timedelta(minutes=CONTEXT_EXPIRY_MINUTES):
            del SESSION_CONTEXTS[session_id]
            return []
        
        return ctx["messages"]

def cleanup_expired_sessions():
    """Fast cleanup"""
    now = datetime.now()
    with CONTEXT_LOCK:
        expired = [
            sid for sid, ctx in list(SESSION_CONTEXTS.items())
            if now - ctx["last_active"] > timedelta(minutes=CONTEXT_EXPIRY_MINUTES)
        ]
        for sid in expired:
            del SESSION_CONTEXTS[sid]

def periodic_cleanup():
    """Background cleanup task"""
    while True:
        time.sleep(600)
        cleanup_expired_sessions()

# --- Authentication Endpoints (Kept minimal for core API function) ---
@app.post("/register")
async def register(user_data: UserRegister):
    users = load_users()
    if any(u["username"] == user_data.username for u in users):
        raise HTTPException(status_code=400, detail="Username already exists")
    if any(u["email"] == user_data.email for u in users):
        raise HTTPException(status_code=400, detail="Email already registered")
    new_user = {
        "id": secrets.token_hex(8),
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
    return {"status": "success", "message": "Login successful", "token": token, "user": {"id": user["id"], "username": user["username"]}}

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
    return {"status": "success", "user": {"id": user["id"], "username": user["username"]}}

# --- ULTRA-FAST: Async Chat Endpoint ---
@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    start_time = time.time()
    
    try:
        session_id = request.session_token or "anon"
        # Get context specific to this user/session
        recent_context = get_recent_context(session_id)
        
        # Prepare history with minimal overhead
        combined_history = recent_context if recent_context else [
            {"role": msg.role, "content": msg.content} 
            for msg in (request.chat_history or [])
        ]
        
        # Run backend query in thread pool for true async
        loop = asyncio.get_event_loop()
        reply = await loop.run_in_executor(
            THREAD_POOL,
            backend.ask_hr_bot,
            request.message,
            combined_history,
            session_id
        )
        
        # Store context asynchronously for THIS session_id
        update_session_context(session_id, "user", request.message)
        update_session_context(session_id, "assistant", reply)
        
        elapsed = time.time() - start_time
        
        print(f"⚡ Chat response in {elapsed:.3f}s | Session: {session_id[:8]}...")
        
        return {
            "response": reply,
            "response_time": round(elapsed, 3),
            "session_id": session_id
        }
        
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"❌ Chat error after {elapsed:.3f}s: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# --- Health & Status ---
@app.get("/")
def root():
    return {"status": "HR Chatbot API Running - ULTRA-FAST MODE", "version": "3.0-FAST", "active_sessions": len(SESSION_CONTEXTS)}

@app.get("/health")
def health():
    return {"status": "ok", "mode": "ultra-fast", "active_sessions": len(SESSION_CONTEXTS), "thread_pool_workers": 10}
