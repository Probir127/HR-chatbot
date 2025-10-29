# backend.py  -- Clean, minimal HR backend (no spaCy, no translator)
import os
import re
import json
import threading
import requests
from uuid import uuid4
from dotenv import load_dotenv

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
    # lightweight romanized bangla heuristic (only when many roman tokens)
    tokens = re.findall(r"\b[a-zA-Z]+\b", text)
    if len(tokens) < 2:
        return False
    banglish_patterns = [r"kemon|bhalo|ami|tumi|kore|hobe|ache|kothay|khobor"]
    text_l = text.lower()
    count = sum(1 for p in banglish_patterns if re.search(r"\b"+p+r"\b", text_l))
    return count >= 1

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

# --- Ollama (thread-safe wrapper) ---
def call_ollama(payload_json, timeout=60):
    try:
        url = f"{OLLAMA_BASE_URL}/api/chat"
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

def call_ollama_threaded(system_prompt, user_prompt, temperature=0.4, max_tokens=300):
    out = []
    def worker():
        payload = {
            "model": OLLAMA_MODEL,
            "messages": [
                {"role":"system", "content": system_prompt},
                {"role":"user", "content": user_prompt}
            ],
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens}
        }
        resp = call_ollama(payload)
        # prefer message.content if present
        if isinstance(resp, dict) and "message" in resp and isinstance(resp["message"], dict):
            out.append(resp["message"].get("content","").strip())
        elif isinstance(resp, dict) and "response" in resp:
            out.append(str(resp.get("response","")).strip())
        else:
            out.append(str(resp))
    t = threading.Thread(target=worker)
    t.start()
    t.join()
    return out[0] if out else "⚠️ No response."

# --- Chain-of-thought style system prompt (hidden reasoning) ---
def build_system_prompt(lang_label):
    return f"""
You are HR Chatbot — a helpful HR assistant for the company.
Before answering, think step-by-step internally: (1) understand the user intent, (2) use the provided context, (3) produce a concise final answer.
Do NOT reveal internal steps. Output only the final answer in the user's language: {lang_label}.
Keep replies short (1–3 sentences), friendly, and professional.
"""

# --- Main public API: handle query ---
def handle_hr_query(user_input: str, session_id: str = None):
    # 1) language detection (strict)
    user_input = (user_input or "").strip()
    if contains_bangla(user_input):
        lang = "bn"
    elif is_probable_banglish(user_input):
        lang = "bn"   # treat as bangla if strong romanized signal
    else:
        lang = "en"

    # 2) employee lookup only when name present
    name = extract_person_name_simple(user_input)
    if name:
        emp = get_employee_by_name(name)
        if emp:
            return format_employee(emp)

    # 3) search knowledge base
    context, score = search_knowledge_base(user_input)
    if not context:
        # fallback short answer
        return "⚠️ I couldn't find that in HR policies. Please contact HR for details."

    # 4) build prompts and call model
    system_prompt = build_system_prompt("Bangla" if lang=="bn" else "English")
    # encourage internal step-by-step reasoning but show only final
    user_prompt = f"Context:\n{context}\n\nUser question: {user_input}\n\nPlease answer concisely."

    reply = call_ollama_threaded(system_prompt, user_prompt)
    return reply

# Backwards-compatibility alias used by api_server.py
def ask_hr_bot(user_input, chat_history=None, session_id=None):
    return handle_hr_query(user_input, session_id=session_id)

# Simple command-line debug
if __name__ == "__main__":
    print("HR backend running (debug). Type 'exit' to quit.")
    while True:
        q = input("You: ").strip()
        if q.lower() in ("exit","quit"):
            break
        print("Bot:", handle_hr_query(q))

