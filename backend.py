# backend.py - Optimized HR Backend with Performance Improvements
import os
import re
import json
import requests
from dotenv import load_dotenv
from typing import List, Dict, Optional
from functools import lru_cache
import time

# --- NEW IMPORT: Use simplified prompt configuration ---
import simple_prompt_config 

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
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3:8b-instruct-q5_K_M")
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
        print(f"Loaded {len(EMPLOYEES)} employees from {EMPLOYEE_JSON_PATH}")
    except Exception as e:
        EMPLOYEES = []
        print(f"Failed to load employees.json: {e}")
else:
    EMPLOYEES = []
    print(f"employees.json not found at {EMPLOYEE_JSON_PATH}")

# --- FAISS / embeddings optional with lazy loading ---
index, texts, embedding_model = None, [], None

def load_faiss_index():
    """Lazy load FAISS index only when needed"""
    global index, texts, embedding_model
    
    if index is not None:
        return
    
    if not HAS_FAISS:
        print("FAISS/embeddings not available (optional)")
        return
        
    if not (os.path.exists(INDEX_PATH) and os.path.exists(TEXTS_PATH)):
        print(f"FAISS index not found at {INDEX_PATH}")
        return
    
    try:
        start_time = time.time()
        index = faiss.read_index(INDEX_PATH)
        texts = np.load(TEXTS_PATH, allow_pickle=True)
        embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        elapsed = time.time() - start_time
        print(f"FAISS knowledge base loaded ({len(texts)} entries) in {elapsed:.2f}s")
    except Exception as e:
        print(f"Could not load FAISS index: {e}")
        index, texts, embedding_model = None, [], None

# --- Utility helpers ---
def extract_person_name_simple(text: str):
    """Simple name extraction heuristic"""
    m = re.search(r"(?:who is|who's|who are|about|details of)\s+([A-Za-z .'-]{2,})", text, re.I)
    if m:
        return m.group(1).strip()
    caps = re.findall(r"\b([A-Z][a-z]{1,}\s(?:[A-Z][a-z]{1,}\s?)*)\b", text)
    if caps:
        return caps[0].strip()
    return None

# --- Intent Classification ---
def simple_intent_classifier(text: str) -> str:
    text_l = text.lower().strip()
    
    if any(phrase in text_l for phrase in [
        "how are you", "what's up", "how's it going",
        "who are you", "what can you do", "hello", "hi", "salam",
        "thank you", "thanks"
    ]):
        return "small_talk"
    
    if any(phrase in text_l for phrase in ["who is", "who's", "details of", "contact for"]):
        return "employee_lookup"
        
    # <--- SOLUTION: Added a specific intent for leave-related queries ---
    if any(word in text_l for word in ["leave", "leaves", "vacation", "sick", "casual", "maternity", "earned", "unpaid", "polices"]): # "polices" catch
        return "leave_policy"
        
    if any(word in text_l for word in ["salary", "paycheck", "bonus", "increment", "pf", "allowance", "benefits", "loss hour", "deduction"]):
        return "salary_benefits"
    if any(word in text_l for word in ["how to", "procedure", "process for", "resign", "apply for leave", "onboarding", "how can i"]):
        return "procedure_howto"
    if any(word in text_l for word in ["office time", "address", "washroom", "kitchen", "ac", "rules for", "dress code", "emergency", "facilities"]):
        return "office_logistics"

    return "policy_question"

def get_employee_by_name(name: str):
    if not name:
        return None
    name_l = name.lower().strip()
    for emp in EMPLOYEES:
        if name_l in emp.get("name", "").lower():
            return emp
    return None

def format_employee(emp):
    if not emp:
        return "Employee not found in HR records."
    return (
        f"Employee: {emp.get('name','N/A')}\n"
        f"Position: {emp.get('position','N/A')}\n"
        f"Email: {emp.get('email','N/A')}\n"
    )

# --- OPTIMIZED: Cached FAISS search ---
@lru_cache(maxsize=100)
def get_cached_embedding(query: str):
    if embedding_model is None:
        return None
    return embedding_model.encode([query])

def search_knowledge_base(query, top_k=2):
    if index is None:
        load_faiss_index()
    
    if index is None or embedding_model is None or len(texts) == 0:
        return None, 0.0
    
    try:
        q_emb = get_cached_embedding(query)
        if q_emb is None:
            return None, 0.0
            
        faiss.normalize_L2(q_emb)
        D, I = index.search(q_emb, top_k)
        
        results = []
        for rank, i in enumerate(I[0]):
            if i < len(texts):
                results.append((str(texts[i]), float(D[0][rank])))
        
        if not results:
            return None, 0.0
        
        ctx = results[0][0]
        score = results[0][1]
        return ctx, score
    except Exception as e:
        print(f"FAISS search error: {e}")
        return None, 0.0

# --- OPTIMIZED: Ollama call with better error handling ---
def call_ollama(payload_json, timeout=45):
    try:
        url = f"{OLLAMA_BASE_URL}/api/chat"
        r = requests.post(url, json=payload_json, timeout=timeout)
        
        if r.status_code == 404:
            url = f"{OLLAMA_BASE_URL}/api/generate"
            r = requests.post(url, json=payload_json, timeout=timeout)
        
        r.raise_for_status()
        return r.json()
    except requests.exceptions.Timeout:
        print("Ollama timeout - response took too long")
        return {"error": "Response generation timeout"}
    except requests.exceptions.ConnectionError:
        print("Ollama connection error - is the service running?")
        return {"error": "Cannot connect to AI service"}
    except Exception as e:
        print(f"Ollama call failed: {e}")
        return {"error": str(e)}

def call_ollama_synchronous(system_prompt: str, messages_history: List[Dict], temperature=0.3, max_tokens=200):
    """
    Optimized synchronous Ollama call with shorter responses
    """
    messages_payload = [{"role": "system", "content": system_prompt}] + messages_history 
    
    payload = {
        "model": OLLAMA_MODEL,
        "messages": messages_payload,
        "stream": False,
        "options": {
            "temperature": temperature, 
            "num_predict": max_tokens
        }
    }
    
    resp = call_ollama(payload)
    
    if isinstance(resp, dict) and "error" in resp:
        return "I'm having trouble generating a response. Please try again or contact HR at people@acmeai.tech"
    
    if isinstance(resp, dict) and "message" in resp and isinstance(resp["message"], dict):
        return resp["message"].get("content", "").strip()
    elif isinstance(resp, dict) and "response" in resp:
        return str(resp.get("response", "")).strip()
    else:
        return str(resp) if resp else "No response generated."

# --- Main Query Handler ---
def handle_hr_query(user_input: str, chat_history: Optional[List[Dict]] = None, session_id: str = None):
    start_time = time.time()
    chat_history = chat_history or []
    user_input_original = (user_input or "").strip() # Keep original for LLM
    user_input_for_search = user_input_original
    
    # 2) Intent Classification
    intent = simple_intent_classifier(user_input_original)
    print(f"Intent: {intent}")
    
    # <--- SOLUTION: Normalize the search query if leave intent is detected ---
    # This overrides the user's misspelling for the RAG step
    if intent == "leave_policy":
        print("Leave intent detected. Normalizing query for RAG.")
        user_input_for_search = "What is the company leave policy?"
    # <--- END FIX ---

    # 3) Fast path: Employee Lookup (bypasses LLM)
    name = extract_person_name_simple(user_input_original)
    if name:
        emp = get_employee_by_name(name)
        if emp:
            elapsed = time.time() - start_time
            print(f"Employee lookup completed in {elapsed:.2f}s")
            return format_employee(emp)
        elif intent == "employee_lookup":
            return "Employee not found in HR records."

    context, score = None, 0.0 # Initialize context as None
    
    # 4) Search knowledge base (BUT ONLY IF IT'S NOT SMALL TALK)
    if intent not in ["small_talk", "employee_lookup"]:
        # Use the (potentially normalized) search query
        context, score = search_knowledge_base(user_input_for_search)
    
        # 5) Handle missing context (only if RAG was attempted)
        if not context: 
            return "I couldn't find that in HR policies. Please contact HR at people@acmeai.tech for details."

    # 6) Build prompts and call LLM
    system_prompt = simple_prompt_config.get_system_prompt(context_type=intent)
    
    messages_for_llm = []
    
    if chat_history:
        messages_for_llm.extend(chat_history[-6:]) 
    
    # Use the *original* user input for the LLM
    final_user_content = user_input_original
    
    if context:
        # Clearer structure for RAG input
        rag_prefix = f"Knowledge Base Context:\n---\n{context[:500]}\n---\n" 
        # Give the LLM the original query + the context we found
        final_user_content = rag_prefix + user_input_original
    
    messages_for_llm.append({"role": "user", "content": final_user_content})

    # Call LLM
    reply = call_ollama_synchronous(system_prompt, messages_for_llm)
    
    elapsed = time.time() - start_time
    print(f"Total response time: {elapsed:.3f}s")
    
    return reply

# Backwards-compatibility alias
def ask_hr_bot(user_input, chat_history=None, session_id=None):
    # Pass the original user input to the handler
    return handle_hr_query(user_input, chat_history=chat_history, session_id=session_id)

# CLI Debug Mode
if __name__ == "__main__":
    print("HR backend running (debug). Type 'exit' to quit.")
    history = []
    
    while True:
        q = input("\nYou: ").strip()
        if q.lower() in ("exit", "quit"):
            break
        
        reply = ask_hr_bot(q, chat_history=history)
        print(f"Bot: {reply}")
        
        history.append({"role": "user", "content": q})
        history.append({"role": "assistant", "content": reply})
        
        history = history[-6:]
