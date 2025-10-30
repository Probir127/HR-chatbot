# backend.py  -- Clean, minimal HR backend (no spaCy, no translator)
import os
import re
import json
import threading 
import requests
from dotenv import load_dotenv
from typing import List, Dict, Optional # Added for clarity

# --- NEW IMPORT: Use external prompt configuration ---
import prompt_config 

# Optional imports (only used if FAISS index exists)
try:
    import numpy as np
    import faiss
    from sentence_transformers import SentenceTransformer
    HAS_FAISS = True
except Exception:
    HAS_FAISS = False

load_dotenv()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:latest")
EMPLOYEE_JSON_PATH = os.getenv("EMPLOYEE_JSON_PATH", "data/employees.json")
INDEX_PATH = os.getenv("INDEX_PATH", "vectorstores/db_faiss/index.faiss")
TEXTS_PATH = os.getenv("TEXTS_PATH", "vectorstores/db_faiss/texts.npy")

# --- Load employees.json (once) ---
if not os.path.isabs(EMPLOYEE_JSON_PATH):
    EMPLOYEE_JSON_PATH = os.path.join(os.path.dirname(__file__), EMPLOYEE_JSON_PATH)

if os.path.exists(EMPLOYEE_JSON_PATH):
    try:
        with open(EMPLOYEE_JSON_PATH, "r", encoding="utf-8") as f:
            EMPLOYEES = json.load(f)
        print(f"✅ Loaded {len(EMPLOYEES)} employees from {EMPLOYEE_JSON_PATH}")
    except Exception as e:
        EMPLOYEES = []
        print(f"⚠️ Failed to load employees.json: {e}")
else:
    EMPLOYEES = []
    print(f"⚠️ employees.json not found at {EMPLOYEE_JSON_PATH}")

# --- FAISS / embeddings optional ---
if HAS_FAISS and os.path.exists(INDEX_PATH) and os.path.exists(TEXTS_PATH):
    try:
        index = faiss.read_index(INDEX_PATH)
        texts = np.load(TEXTS_PATH, allow_pickle=True)
        # Optimization: Default SentenceTransformer model loading is often efficient enough
        embedding_model = SentenceTransformer("all-MiniLM-L6-v2") 
        print(f"✅ FAISS knowledge base loaded ({len(texts)} entries)")
    except Exception as e:
        print(f"⚠️ Could not load FAISS index: {e}")
        index, texts, embedding_model = None, [], None
else:
    index, texts, embedding_model = None, [], None
    if not HAS_FAISS:
        print("⚠️ FAISS/embeddings not available (optional)")

# --- Utility helpers ---
def contains_bangla(text: str) -> bool:
    return any("\u0980" <= ch <= "\u09FF" for ch in text)

def is_probable_banglish(text: str) -> bool:
    # stricter romanized bangla heuristic — require at least 2 token matches
    tokens = re.findall(r"\b[a-zA-Z]+\b", text)
    if len(tokens) < 2:
        return False
    banglish_patterns = [
        r"\bkemon\b", r"\bbhalo\b", r"\bami\b", r"\btumi\b", r"\bache\b",
        r"\bhobe\b", r"\bkorbo\b", r"\bekhane\b", r"\bbhalo?\b", r"\btomar\b",
        r"\bamar\b", r"\bbujh\b", r"\bki\b"
    ]
    text_l = text.lower()
    matches = sum(1 for p in banglish_patterns if re.search(p, text_l))
    return matches >= 2

def extract_person_name_simple(text: str):
    """Very small, dependency-free name heuristic:
       pick consecutive capitalized words or words next to 'who is' / 'about'"""
    # look for explicit phrase
    m = re.search(r"(?:who is|who's|who are|about|details of)\s+([A-Za-z .'-]{2,})", text, re.I)
    if m:
        return m.group(1).strip()
    # fallback: find capitalized sequences (English names)
    caps = re.findall(r"\b([A-Z][a-z]{1,}\s(?:[A-Z][a-z]{1,}\s?)*)\b", text)
    if caps:
        return caps[0].strip()
    return None

# --- NEW: Simple Intent Classification for better prompt guidance ---
def simple_intent_classifier(text: str) -> str:
    text_l = text.lower().strip()
    
    # 1. Identity / Greeting
    if any(phrase in text_l for phrase in ["who are you", "what can you do", "hello", "hi", "salam", "kemon achen"]):
        return "greeting"
    
    # 2. Employee Lookup (already done by name extraction, but good to check)
    if any(phrase in text_l for phrase in ["who is", "who's", "details of", "contact for"]):
        return "employee_lookup"

    # 3. Salary & Benefits
    if any(word in text_l for word in ["salary", "paycheck", "bonus", "increment", "pf", "allowance", "benefits", "loss hour", "deduction"]):
        return "salary_benefits"
        
    # 4. Procedure / How-To (Resignation, leave, onboarding steps)
    if any(word in text_l for word in ["how to", "procedure", "process for", "resign", "apply for leave", "onboarding", "how can i"]):
        return "procedure_howto"
        
    # 5. Office Logistics / Rules
    if any(word in text_l for word in ["office time", "address", "washroom", "kitchen", "AC", "rules for", "dress code", "emergency", "facilities"]):
        return "office_logistics"

    # 6. Default to Policy Question
    return "policy_question"

def get_employee_by_name(name: str):
    if not name: return None
    name_l = name.lower().strip()
    for emp in EMPLOYEES:
        if name_l in emp.get("name","").lower():
            return emp
    return None

def format_employee(emp):
    if not emp:
        return "❌ Employee not found in HR records."
    # Matches the required format in prompt_config's Rule 5/Example
    return (
        f"👤 {emp.get('name','N/A')}\n"
        f"🏢 Position: {emp.get('position','N/A')}\n"
        f"📧 Email: {emp.get('email','N/A')}\n"
    )

# --- FAISS search (optional) ---
def search_knowledge_base(query, top_k=3):
    if index is None or embedding_model is None or len(texts)==0:
        return None, 0.0
    q_emb = embedding_model.encode([query])
    faiss.normalize_L2(q_emb)
    D,I = index.search(q_emb, top_k)
    results = []
    for rank, i in enumerate(I[0]):
        if i < len(texts):
            results.append((str(texts[i]), float(D[0][rank])))
    if not results:
        return None, 0.0
    ctx = "\n\n".join([r[0] for r in results])
    score = results[0][1]
    return ctx, score

# --- Ollama (synchronous call for performance) ---
def call_ollama(payload_json, timeout=60):
    try:
        url = f"{OLLAMA_BASE_URL}/api/chat"
        # Using requests.post synchronously, relying on FastAPI/Uvicorn to handle concurrency
        r = requests.post(url, json=payload_json, timeout=timeout) 
        if r.status_code == 404:
            # some older ollama versions use /api/generate
            url = f"{OLLAMA_BASE_URL}/api/generate"
            r = requests.post(url, json=payload_json, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print("❌ Ollama call failed:", e)
        return {"error": str(e)}

def call_ollama_synchronous(system_prompt: str, messages_history: List[Dict], temperature=0.4, max_tokens=300):
    """
    Synchronous blocking call to Ollama that sends the entire message history.
    """
    # Construct the full messages payload including system prompt
    messages_payload = [{"role": "system", "content": system_prompt}] + messages_history 
    
    payload = {
        "model": OLLAMA_MODEL,
        "messages": messages_payload, # <-- MODIFIED: Use the full message history
        "stream": False,
        "options": {"temperature": temperature, "num_predict": max_tokens}
    }
    resp = call_ollama(payload)
    
    # prefer message.content if present
    if isinstance(resp, dict) and "message" in resp and isinstance(resp["message"], dict):
        return resp["message"].get("content","").strip()
    elif isinstance(resp, dict) and "response" in resp:
        return str(resp.get("response","")).strip()
    else:
        return str(resp) if resp else "⚠️ No response."


# --- Main public API: handle query ---
def handle_hr_query(user_input: str, chat_history: Optional[List[Dict]] = None, session_id: str = None):
    chat_history = chat_history or []
    user_input = (user_input or "").strip()
    
    # 1) language detection (strict)
    if contains_bangla(user_input):
        lang = "bn"
    elif is_probable_banglish(user_input):
        lang = "banglish"
    else:
        lang = "en"
        
    # 2) Intent Classification (for prompt selection)
    intent = simple_intent_classifier(user_input)

    # 3) Immediate Employee Lookup (Bypasses LLM - Fastest Path)
    name = extract_person_name_simple(user_input)
    if name:
        emp = get_employee_by_name(name)
        if emp:
            # Employee info found: return formatted static response immediately
            return format_employee(emp)
        elif intent == "employee_lookup":
            # If explicit lookup failed, fallback static message to save LLM call
            return "❌ Employee not found in HR records."


    # 4) Search knowledge base
    context, score = search_knowledge_base(user_input)
    
    # 5) Handle Missing Context
    # If no context found AND not a simple greeting (which needs no KB)
    if not context and intent not in ["greeting"]:
        # Static fallback aligned with prompt_config rule 1
        return "⚠️ I couldn't find that in HR policies. Please contact HR at people@acmeai.tech for details."

    # 6) Build prompts and call model
    
    system_prompt = prompt_config.get_system_prompt(language=lang, context_type=intent) 
    
    messages_for_llm = []
    
    # Separate the current user message from the rest of the history
    current_user_message = chat_history[-1]["content"] if chat_history else user_input
    previous_history = chat_history[:-1] if chat_history else []
    
    # Add previous conversation messages (retains turns for context/memory)
    messages_for_llm.extend(previous_history)
    
    # Prepend RAG context to the *current* user message's content
    if context:
        rag_prefix = f"Context from HR Knowledge Base:\n{context}\n\n"
        current_user_message = rag_prefix + current_user_message
    
    # Add the (now potentially context-prefixed) current user message
    messages_for_llm.append({"role": "user", "content": current_user_message})

    # Add instruction on how to reply
    messages_for_llm.append({"role": "user", "content": "Please answer concisely. Do not invent facts."})


    reply = call_ollama_synchronous(system_prompt, messages_for_llm)
    return reply

# Backwards-compatibility alias used by api_server.py
def ask_hr_bot(user_input, chat_history=None, session_id=None):
    # Pass the full chat_history object (list of {"role": str, "content": str})
    return handle_hr_query(user_input, chat_history=chat_history, session_id=session_id)

# Simple command-line debug
if __name__ == "__main__":
    print("HR backend running (debug). Type 'exit' to quit.")
    while True:
        q = input("You: ").strip()
        # For debug, we simulate a small history based on the previous query
        # This is a basic simulation and may not perfectly reflect the API server's context window logic
        simulated_history = []
        if 'q' in locals() and q.lower() not in ("exit", "quit"):
            simulated_history = [{"role": "user", "content": q}, {"role": "assistant", "content": "Bot simulated previous reply."}]
        
        if q.lower() in ("exit","quit"):
            break
        print("Bot:", ask_hr_bot(q, chat_history=simulated_history))
