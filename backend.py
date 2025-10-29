# backend.py  -- Clean, minimal HR backend (no spaCy, no translator)
import os
import re
import json
import threading
import requests
from uuid import uuid4
from dotenv import load_dotenv
from datetime import datetime, timedelta

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

# ---------------- Session short-term memory ----------------
# Keeps last N messages per session in memory (simple, in-memory)
SESSIONS = {}  # session_id -> {"messages": [{"role":"user"/"assistant","text":...}], "last_active": datetime}
SESSION_TTL_MINUTES = 10
MAX_SESSION_MESSAGES = 6  # keep a few messages (user+assistant pairs)

def _now(): return datetime.utcnow()

def get_session(session_id: str):
    if not session_id:
        return None
    s = SESSIONS.get(session_id)
    if not s:
        return None
    # expire old session
    if (_now() - s["last_active"]) > timedelta(minutes=SESSION_TTL_MINUTES):
        del SESSIONS[session_id]
        return None
    return s

def touch_session(session_id: str):
    if not session_id:
        return
    if session_id not in SESSIONS:
        SESSIONS[session_id] = {"messages": [], "last_active": _now()}
    else:
        SESSIONS[session_id]["last_active"] = _now()

def append_session_message(session_id: str, role: str, text: str):
    if not session_id:
        return
    touch_session(session_id)
    msgs = SESSIONS[session_id]["messages"]
    msgs.append({"role": role, "text": text, "time": _now().isoformat()})
    # trim
    if len(msgs) > MAX_SESSION_MESSAGES:
        SESSIONS[session_id]["messages"] = msgs[-MAX_SESSION_MESSAGES:]

def summarize_session(session_id: str, max_chars: int = 600):
    """Return a short human-readable summary of recent session messages (for model context)."""
    s = get_session(session_id)
    if not s:
        return ""
    # build a short concatenation, respecting max_chars
    pieces = []
    for m in s["messages"][-MAX_SESSION_MESSAGES:]:
        prefix = "User: " if m["role"] == "user" else "Bot: "
        pieces.append(prefix + m["text"].replace("\n", " "))
    joined = " | ".join(pieces)
    return joined[:max_chars]

# ---------------- Utility helpers ----------------
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
    m = re.search(r"(?:who is|who's|who are|about|details of)\s+([A-Za-z .'-]{2,})", text, re.I)
    if m:
        return m.group(1).strip()
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

# ---------------- FAISS search (optional) ----------------
def search_knowledge_base(query, top_k=3):
    if index is None or embedding_model is None or len(texts) == 0:
        return None, 0.0
    q_emb = embedding_model.encode([query])
    faiss.normalize_L2(q_emb)
    D, I = index.search(q_emb, top_k)
    results = []
    for rank, i in enumerate(I[0]):
        if i < len(texts):
            results.append((str(texts[i]), float(D[0][rank])))
    if not results:
        return None, 0.0
    ctx = "\n\n".join([r[0] for r in results])
    score = results[0][1]
    return ctx, score

# ---------------- Ollama wrapper ----------------
def call_ollama(payload_json, timeout=60):
    try:
        url = f"{OLLAMA_BASE_URL}/api/chat"
        r = requests.post(url, json=payload_json, timeout=timeout)
        if r.status_code == 404:
            url = f"{OLLAMA_BASE_URL}/api/generate"
            r = requests.post(url, json=payload_json, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print("❌ Ollama call failed:", e)
        return {"error": str(e)}

def call_ollama_threaded(system_prompt, user_prompt, temperature=0.35, max_tokens=300):
    out = []
    def worker():
        payload = {
            "model": OLLAMA_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens}
        }
        resp = call_ollama(payload)
        if isinstance(resp, dict) and "message" in resp and isinstance(resp["message"], dict):
            out.append(resp["message"].get("content", "").strip())
        elif isinstance(resp, dict) and "response" in resp:
            out.append(str(resp.get("response", "")).strip())
        else:
            out.append(str(resp))
    t = threading.Thread(target=worker)
    t.start()
    t.join()
    return out[0] if out else "⚠️ No response."

# ---------------- Improved chain-of-thought system prompt ----------------
def build_system_prompt(lang_label, source_label: str = "HR knowledge base"):
    """
    Deeper hidden-reasoning system prompt:
    - instructs model how to reason internally,
    - enforces tone and language,
    - asks model to ground answer on provided context and session summary.
    """
    if lang_label == "bn":
        lang_instruction = "Reply fully in Bangla (বাংলা ভাষায় উত্তর দিন)।"
        example_note = "উত্তরটি বাংলা লিপিতে লিখুন, সংক্ষিপ্ত কিন্তু আন্তরিকভাবে।"
    elif lang_label == "banglish":
        lang_instruction = "Reply in Banglish (Romanized Bangla, e.g., 'Tumi kemon aso?')."
        example_note = "উত্তরটি রোমান অক্ষরে দিন — বাংলা লিপি ব্যবহার করবেন না।"
    else:
        lang_instruction = "Reply in English."
        example_note = "Answer in clear, natural English — concise, polite, and conversational."

    return f"""
You are HR Chatbot — a friendly, context-aware HR assistant.
Before answering, perform hidden internal reasoning steps (do not reveal them):
1) Determine user's intent and tone (question, follow-up, greeting, or complaint).
2) Identify the exact HR topic or named entity (leave, benefits, attendance, employee name).
3) Consult the session summary (recent chat) and the provided {source_label} and use only those facts.
4) If the question is ambiguous, infer the most likely meaning using HR commonsense but do NOT invent facts.
5) Produce a concise, helpful, human-sounding answer (1–3 sentences) and offer a small next-step where helpful.

Mandatory rules:
- Always use the detected language: {lang_label}.
- Be warm and conversational — not robotic.
- Ground answers only in the provided context (session summary + {source_label} + employees.json).
- If information is missing, reply exactly: "⚠️ I couldn't find that in HR policies. Please contact HR for details."
- Do NOT reveal internal reasoning, do NOT ask for Gmail verification, and do NOT fabricate.
- If user asks "Who are you?" or similar, respond with a short friendly identity line, e.g.:
    - English: "I'm your HR assistant — I can help with leave, policies, and employee info. How can I help?"
    - Bangla: "আমি আপনার HR সহকারী — ছুটি, নীতি এবং কর্মচারী তথ্যয়ে সাহায্য করতে পারি। কিভাবে সাহায্য করি?"
    - Banglish: "Ami apnar HR assistant — chuti, policy ebong employee info e help korte pari. Kibhabe help korbo?"
- When quoting dates/numbers, be explicit (e.g., "18 days per year").

{lang_instruction}
{example_note}

Now produce only the final answer (no reasoning), concise and context-aware.
"""

# ---------------- Quick canned replies for identity/greetings ----------------
def is_greeting_or_identity_query(text: str) -> str:
    """Return 'identity' if identity question, 'greeting' if greeting, else ''."""
    t = text.lower().strip()
    # identity / name queries
    if re.search(r"\b(who (are|r) (you|u)|what is your name|whoami|who you are)\b", t):
        return "identity"
    # simple greetings
    if re.search(r"\b(hello|hi|hey|good morning|good afternoon|good evening|assalam|salam)\b", t):
        return "greeting"
    return ""

def canned_identity_reply(lang_label: str) -> str:
    if lang_label == "bn":
        return "আমি আপনার HR সহকারী — ছুটি, নীতি এবং কর্মচারী তথ্য বিষয়ে সহায়তা করতে পারি। কি জানতে চান?"
    elif lang_label == "banglish":
        return "Ami apnar HR assistant — chuti, policy ar employee info e help korte pari. Ki jante chan?"
    else:
        return "I'm your HR assistant — I can help with leave, policies, and employee info. How can I help?"

def canned_greeting_reply(lang_label: str) -> str:
    if lang_label == "bn":
        return "হ্যালো! কিভাবে সাহায্য করতে পারি?"
    elif lang_label == "banglish":
        return "Hello! Kibhabe help korte pari?"
    else:
        return "Hi! How can I help you today?"

# ---------------- Main public API: handle query ----------------
def handle_hr_query(user_input: str, session_id: str = None):
    user_input = (user_input or "").strip()
    if not user_input:
        return "Please enter a valid question."

    # --- language detection ---
    if contains_bangla(user_input):
        lang = "bn"
    elif is_probable_banglish(user_input):
        lang = "banglish"
    else:
        lang = "en"

    # --- quick canned handling for identity/greetings (avoid model uncertainty) ---
    qtype = is_greeting_or_identity_query(user_input)
    if qtype == "identity":
        # store in session and return
        append_session_message(session_id or "anon", "user", user_input)
        reply = canned_identity_reply(lang)
        append_session_message(session_id or "anon", "assistant", reply)
        return reply
    if qtype == "greeting":
        append_session_message(session_id or "anon", "user", user_input)
        reply = canned_greeting_reply(lang)
        append_session_message(session_id or "anon", "assistant", reply)
        return reply

    # --- employee lookup ---
    name = extract_person_name_simple(user_input)
    if name:
        emp = get_employee_by_name(name)
        if emp:
            reply = format_employee(emp)
            append_session_message(session_id or "anon", "user", user_input)
            append_session_message(session_id or "anon", "assistant", reply)
            return reply

    # --- knowledge retrieval ---
    context, score = search_knowledge_base(user_input)
    if not context:
        # no context: short fallback
        return "⚠️ I couldn't find that in HR policies. Please contact HR for details."

    # --- prepare session summary and combined context ---
    session_summary = summarize_session(session_id or "anon")
    combined_context = ""
    if session_summary:
        combined_context += f"Recent chat: {session_summary}\n\n"
    combined_context += f"Relevant HR context:\n{context}\n\n"

    # --- compose prompts and call model ---
    system_prompt = build_system_prompt(lang)
    user_prompt = f"{combined_context}User question: {user_input}\n\nAnswer concisely."

    reply = call_ollama_threaded(system_prompt, user_prompt)

    # store in session
    append_session_message(session_id or "anon", "user", user_input)
    append_session_message(session_id or "anon", "assistant", reply)

    return reply

# --- API alias ---
def ask_hr_bot(user_input, chat_history=None, session_id=None):
    # keep backward compat; session_id may be provided from api_server
    return handle_hr_query(user_input, session_id=session_id)

# --- Command-line debug ---
if __name__ == "__main__":
    sid = str(uuid4())
    print("HR backend running (debug). Session:", sid)
    while True:
        q = input("You: ").strip()
        if q.lower() in ("exit", "quit"):
            break
        print("Bot:", handle_hr_query(q, session_id=sid))
