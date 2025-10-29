import os
import re
import json
import threading
import requests
import numpy as np
import faiss
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from bangla_handler import process_mixed_input  # updated detector
from pdf_reader import load_pdf_text

# ==========================================================
# Initialization
# ==========================================================
load_dotenv()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
EMPLOYEE_JSON_PATH = os.getenv("EMPLOYEE_JSON_PATH", "data/employees.json")
INDEX_PATH = os.getenv("INDEX_PATH", "vectorstores/db_faiss/index.faiss")
TEXTS_PATH = os.getenv("TEXTS_PATH", "vectorstores/db_faiss/texts.npy")

# --- Load Employee Data ---
if os.path.exists(EMPLOYEE_JSON_PATH):
    try:
        with open(EMPLOYEE_JSON_PATH, "r", encoding="utf-8") as f:
            EMPLOYEES = json.load(f)
        print(f"✅ Loaded {len(EMPLOYEES)} employees")
    except Exception as e:
        EMPLOYEES = []
        print(f"⚠️ Could not load employees.json: {e}")
else:
    EMPLOYEES = []
    print(f"⚠️ employees.json not found at {EMPLOYEE_JSON_PATH}")

# --- Load FAISS Knowledge Base ---
try:
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    index = faiss.read_index(INDEX_PATH)
    texts = np.load(TEXTS_PATH, allow_pickle=True)
    print(f"✅ Knowledge base loaded ({len(texts)} entries)")
except Exception as e:
    index, texts = None, []
    print(f"⚠️ Could not load FAISS index: {e}")

# ==========================================================
# Employee Lookup
# ==========================================================
def get_employee_by_name(name: str):
    name = name.lower().strip()
    for emp in EMPLOYEES:
        if name in emp.get("name", "").lower():
            return emp
    return None

def format_employee(emp):
    if not emp:
        return "❌ Employee not found in HR records."
    return (
        f"👤 **{emp.get('name','N/A')}**\n"
        f"🏢 Position: {emp.get('position','N/A')}\n"
        f"📧 Email: {emp.get('email','N/A')}\n"
        f"🩸 Blood Group: {emp.get('blood_group','N/A')}\n"
        f"📋 Table: {emp.get('table','N/A')}"
    )

# ==========================================================
# Knowledge Base Search
# ==========================================================
def search_knowledge_base(query, top_k=4):
    if not index or len(texts) == 0:
        return None, 0.0
    q_emb = embedding_model.encode([query])
    faiss.normalize_L2(q_emb)
    D, I = index.search(q_emb, top_k)
    results = [(texts[i], D[0][idx]) for idx, i in enumerate(I[0]) if i < len(texts)]
    if not results:
        return None, 0.0
    context = "\n\n".join([r[0] for r in results])
    return context, results[0][1]

# ==========================================================
# Ollama Integration
# ==========================================================
def call_ollama(prompt, system_prompt="", temperature=0.6, max_tokens=250):
    """Send a single chat request to Ollama."""
    try:
        url = f"{OLLAMA_BASE_URL}/api/chat"
        payload = {
            "model": OLLAMA_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        r = requests.post(url, json=payload, timeout=90)
        r.raise_for_status()
        data = r.json()
        if "message" in data:
            return data["message"].get("content", "").strip()
        return data.get("response", "").strip()
    except Exception as e:
        return f"⚠️ Ollama error: {e}"

def call_ollama_threaded(prompt, system_prompt):
    """Threaded call for multi-user concurrency."""
    result_holder = []
    thread = threading.Thread(target=lambda: result_holder.append(call_ollama(prompt, system_prompt)))
    thread.start()
    thread.join()
    return result_holder[0] if result_holder else "⚠️ No response."

# ==========================================================
# Language-Aware System Prompt Builder
# ==========================================================
def build_system_prompt(lang_label: str) -> str:
    """Choose correct reply language (Bangla, Banglish, or English)."""
    if lang_label == "bn":
        lang_instruction = "Bangla (বাংলা ভাষায় উত্তর দিন)"
    elif lang_label == "banglish":
        lang_instruction = "Banglish (Roman Bangla, e.g., 'Tumi kemon aso?')"
    else:
        lang_instruction = "English"

    return f"""
You are HR Chatbot – a multilingual, context-aware HR assistant.
Before answering, think step by step (internally), but show only the final answer.
Always reply in {lang_instruction}.
Keep responses short (2–4 sentences), friendly, and human-like.
Never ask for Gmail verification.
If unsure, admit it politely and suggest contacting HR.
"""

# ==========================================================
# Main Query Handler
# ==========================================================
def handle_hr_query(user_input: str):
    """Detects user language, routes to employee or HR info."""
    detected = process_mixed_input(user_input)
    lang_label = detected.get("lang", "en")
    print(f"🌐 Detected language: {lang_label} | Query: {user_input}")

    # Employee check
    name_match = re.search(r"(?:who is|about|details of|employee)\s+([A-Za-z .-]+)", user_input, re.I)
    if name_match:
        emp = get_employee_by_name(name_match.group(1))
        if emp:
            return format_employee(emp)

    # HR policy or knowledge
    context, _ = search_knowledge_base(user_input)
    if not context:
        return "⚠️ Sorry, I couldn’t find that in HR policies."

    # Build reasoning-style prompt
    system_prompt = build_system_prompt(lang_label)
    full_prompt = f"Let's think step by step.\n\nContext:\n{context}\n\nUser: {user_input}"
    reply = call_ollama_threaded(full_prompt, system_prompt)
    return reply.strip()

# ==========================================================
# Compatibility Wrapper (for api_server)
# ==========================================================
def ask_hr_bot(user_input, chat_history=None, session_id=None):
    return handle_hr_query(user_input)

# ==========================================================
# Manual Run
# ==========================================================
if __name__ == "__main__":
    print("🧠 HR Chatbot Ready (Bangla + Banglish + English)")
    while True:
        msg = input("You: ").strip()
        if msg.lower() in ["exit", "quit"]:
            break
        print("Bot:", handle_hr_query(msg))
