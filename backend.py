# backend.py -- Clean backend using prompt_config module with Session Management
import os
import re
import json
import requests
from dotenv import load_dotenv
from typing import List, Dict, Optional
import threading # For thread-safe FAISS loading
import numpy as np # Must be imported for np.load
import random # Imported via prompt_config but good practice to have for completeness

# Import external prompt configuration
import prompt_config
from prompt_config import (
    get_predefined_response,
    check_intent_patterns,
    GREETING_PATTERNS,
    IDENTITY_PATTERNS,
    THANKS_PATTERNS,
    GOODBYE_PATTERNS,
    SMALL_TALK_PATTERNS 
)

# Optional imports (only used if FAISS index exists)
try:
    import faiss
    from sentence_transformers import SentenceTransformer
    HAS_FAISS = True
except Exception:
    HAS_FAISS = False

load_dotenv()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3:8b") 
EMPLOYEE_JSON_PATH = os.getenv("EMPLOYEE_JSON_PATH", "data/employees.json")
INDEX_PATH = os.getenv("INDEX_PATH", "vectorstores/db_faiss/index.faiss")
TEXTS_PATH = os.getenv("TEXTS_PATH", "vectorstores/db_faiss/texts.npy")

# === CRITICAL FIX: More Permissive RAG Threshold ===
MIN_RAG_SCORE = 0.3  # Very permissive - only reject obvious garbage
RELIABLE_SCORE = 0.35  # Minimum score for confident answers
# ===================================================

# Global variables for FAISS index and model (Lazy Loading setup)
_FAISS_INDEX = None
_FAISS_TEXTS = None
_EMBEDDING_MODEL = None
_FAISS_LOCK = threading.Lock() 

# ============================================================================
# LOAD EMPLOYEE DATA
# ============================================================================
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

# ============================================================================
# LAZY LOAD FAISS KNOWLEDGE BASE
# ============================================================================
def get_faiss_resources():
    """Loads FAISS index and model only on first call (Lazy Loading)."""
    global _FAISS_INDEX, _FAISS_TEXTS, _EMBEDDING_MODEL

    if _FAISS_INDEX is not None:
        return _FAISS_INDEX, _FAISS_TEXTS, _EMBEDDING_MODEL
    
    # Use a lock to ensure only one thread loads the resources
    with _FAISS_LOCK:
        if _FAISS_INDEX is not None: 
            return _FAISS_INDEX, _FAISS_TEXTS, _EMBEDDING_MODEL
            
        if HAS_FAISS and os.path.exists(INDEX_PATH) and os.path.exists(TEXTS_PATH):
            try:
                _FAISS_INDEX = faiss.read_index(INDEX_PATH)
                _FAISS_TEXTS = np.load(TEXTS_PATH, allow_pickle=True)
                _EMBEDDING_MODEL = SentenceTransformer("all-MiniLM-L6-v2") 
                print(f"✅ FAISS knowledge base lazy loaded ({len(_FAISS_TEXTS)} entries)")
            except Exception as e:
                print(f"⚠️ Could not load FAISS index: {e}")
                _FAISS_INDEX, _FAISS_TEXTS, _EMBEDDING_MODEL = None, [], None
        else:
            print("⚠️ FAISS/embeddings not available (optional)")

    return _FAISS_INDEX, _FAISS_TEXTS, _EMBEDDING_MODEL

# ============================================================================
# LANGUAGE DETECTION UTILITIES
# ============================================================================
def contains_bangla(text: str) -> bool:
    """Check if text contains Bangla Unicode characters"""
    return any("\u0980" <= ch <= "\u09FF" for ch in text)

def is_probable_banglish(text: str) -> bool:
    """Check if text is romanized Bangla (Banglish)"""
    tokens = re.findall(r"\b[a-zA-Z]+\b", text)
    if len(tokens) < 2:
        return False
    
    banglish_patterns = [
        r"\bkemon\b", r"\bbhalo\b", r"\bami\b", r"\btumi\b", r"\bache\b",
        r"\bhobe\b", r"\bkorbo\b", r"\bekhane\b", r"\btomar\b",
        r"\bamar\b", r"\bbujh\b", r"\bki\b", r"\bapni\b", r"\bapnar\b"
    ]
    
    text_l = text.lower()
    matches = sum(1 for p in banglish_patterns if re.search(p, text_l))
    return matches >= 2

def detect_language(text: str) -> str:
    """
    Detect language of user input.
    Returns: 'bn' for Bangla, 'banglish' for romanized Bangla, 'en' for English
    """
    if contains_bangla(text):
        return "bn"
    elif is_probable_banglish(text):
        return "banglish"
    else:
        return "en"

# ============================================================================
# INTENT CLASSIFICATION - FULLY FIXED VERSION
# ============================================================================
def classify_intent(text: str) -> str:
    """
    Classify user intent using pattern matching from prompt_config.
    CRITICAL FIX: Reordered logic to prevent "what is" misclassification.
    """
    text_l = text.lower().strip()
    
    # 1. Conversational Intents FIRST (Check BEFORE policy keywords)
    # This prevents "how are you" from triggering procedure_howto
    if check_intent_patterns(text, GREETING_PATTERNS):
        return "greeting"
    
    if check_intent_patterns(text, SMALL_TALK_PATTERNS):
        return "small_talk"
    
    if check_intent_patterns(text, IDENTITY_PATTERNS):
        return "identity"
    
    if check_intent_patterns(text, THANKS_PATTERNS):
        return "thanks"
    
    if check_intent_patterns(text, GOODBYE_PATTERNS):
        return "goodbye"

    # 2. Employee Lookup - VERY SPECIFIC patterns only
    # ✅ FIX: Added trailing spaces to prevent "what is" matching "who is"
    employee_trigger_phrases = [
        "who is ",      # Note the trailing space!
        "who's ",
        "who are ",
        "details of ",
        "contact for ",
        "email of ",
        "find employee",
        "employee named",
        "staff member",
        "about "  # Only when followed by a name
    ]
    
    # Check if query has employee-specific trigger
    has_employee_trigger = any(phrase in text_l + " " for phrase in employee_trigger_phrases)
    
    if has_employee_trigger:
        # Extra validation: must have a name-like pattern
        name_guess = extract_person_name(text)
        if name_guess:
            return "employee_lookup"
    
    # 3. Specific Policy/Category Keywords
    
    # Salary & Benefits
    if any(word in text_l for word in ["salary", "paycheck", "bonus", "increment", "pf", "provident fund", "allowance", "benefits", "loss hour", "deduction"]):
        return "salary_benefits"
        
    # Procedure / How-To (but not "how are you")
    if any(word in text_l for word in ["how to", "procedure", "process for", "resign", "apply for leave", "onboarding", "how can i", "how do i", "steps to"]):
        return "procedure_howto"
        
    # Office Logistics / Rules / General Policy Terms 
    if any(word in text_l for word in ["office time", "office hour", "address", "washroom", "kitchen", "ac", "rules for", "dress code", "emergency", "facilities", "support", "conference", "office space", "floor", "location"]):
        return "office_logistics"
    
    # 4. General Policy Question Indicators
    # ✅ FIX: These are now checked AFTER conversational intents
    policy_question_indicators = [
        "what is", "what are", "what's", 
        "when is", "when are", 
        "where is", "where are",
        "why", "why is", "why are",
        "tell me about", "explain", "describe",
        "company goal", "company policy", "company rule"
    ]
    
    if any(indicator in text_l for indicator in policy_question_indicators):
        return "policy_question"
    
    # 5. Final Fallback - Single name lookup (LOWEST priority)
    name_guess = extract_person_name(text)
    if name_guess and len(text_l.split()) <= 3 and len(name_guess) >= 3:
        if get_employee_by_name(name_guess):
            return "employee_lookup"
            
    # 6. Default to Policy Question
    return "policy_question"

# ============================================================================
# EMPLOYEE LOOKUP UTILITIES
# ============================================================================
def extract_person_name(text: str) -> Optional[str]:
    """Extract person name from query"""
    # Look for explicit phrase patterns with trailing context
    patterns = [
        r"(?:who is|who's|who are)\s+([A-Za-z .'-]{2,})",
        r"(?:about|details of|email of|contact for)\s+([A-Za-z .'-]{2,})",
        r"(?:find|search)\s+(?:employee|staff)?\s*(?:named)?\s+([A-Za-z .'-]{2,})"
    ]
    
    for pattern in patterns:
        m = re.search(pattern, text, re.I)
        if m:
            return m.group(1).strip()
    
    # Fallback: find capitalized sequences (English names)
    caps = re.findall(r"\b([A-Z][a-z]{1,}\s(?:[A-Z][a-z]{1,}\s?)*)\b", text)
    if caps:
        return max(caps, key=len).strip()
    
    # Last resort: single capitalized word (if query is very short)
    if len(text.split()) <= 3:
        single_cap = re.findall(r"\b([A-Z][a-z]{2,})\b", text)
        if single_cap:
            return single_cap[0]
    
    return None

def get_employee_by_name(name: str) -> Optional[Dict]:
    """Search employee in database by name"""
    if not name:
        return None
    
    name_l = name.lower().strip()
    for emp in EMPLOYEES:
        # Fuzzy match on employee name
        if name_l in emp.get("name", "").lower():
            return emp
    
    return None

def format_employee_info(emp: Dict) -> str:
    """Format employee information for display"""
    if not emp:
        return "❌ Employee not found in HR records."
    
    # NOTE: The public function is kept minimal for security (Name, Position, Email ONLY)
    return (
        f"👤 {emp.get('name', 'N/A')}\n"
        f"🏢 Position: {emp.get('position', 'N/A')}\n"
        f"📧 Email: {emp.get('email', 'N/A')}"
    )

# ============================================================================
# KNOWLEDGE BASE SEARCH (FAISS) - FULLY FIXED VERSION
# ============================================================================
def search_knowledge_base(query: str, top_k: int = 3) -> tuple:
    """
    Search FAISS knowledge base for relevant context.
    Returns: (context_text, relevance_score)
    
    ✅ FIX: More permissive threshold and better result handling
    """
    # Get resources via lazy loader
    index, texts, embedding_model = get_faiss_resources() 
    
    if index is None or embedding_model is None or len(texts) == 0:
        return None, 0.0
    
    # Encode query
    q_emb = embedding_model.encode([query])
    faiss.normalize_L2(q_emb)
    
    # Search
    D, I = index.search(q_emb, top_k)
    
    # Extract results
    results = []
    
    # ✅ FIX: Collect ALL results first, then filter
    for rank, i in enumerate(I[0]):
        score = float(D[0][rank])
        if i < len(texts) and score >= MIN_RAG_SCORE:  # Very permissive minimum
            results.append((str(texts[i]), score))
    
    if not results:
        print("📚 RAG: No results found above minimum threshold")
        return None, 0.0
    
    # Sort by score (highest first)
    results.sort(key=lambda x: x[1], reverse=True)
    
    # Combine context from top results (max 2 chunks)
    context_chunks = [r[0] for r in results[:2]]
    context = "\n\n".join(context_chunks)
    top_score = results[0][1]
    
    print(f"📚 RAG Search: Top score = {top_score:.3f}, Chunks used = {len(context_chunks)}")
    
    return context, top_score

# ============================================================================
# OLLAMA LLM CALLS
# ============================================================================
def call_ollama(payload_json: dict, timeout: int = 60) -> dict:
    """Make synchronous call to Ollama API"""
    try:
        url = f"{OLLAMA_BASE_URL}/api/chat"
        r = requests.post(url, json=payload_json, timeout=timeout) 
        
        # Fallback to /api/generate for older Ollama versions
        if r.status_code == 404:
            url = f"{OLLAMA_BASE_URL}/api/generate"
            r = requests.post(url, json=payload_json, timeout=timeout)
        
        r.raise_for_status()
        return r.json()
    
    except Exception as e:
        print(f"❌ Ollama call failed: {e}")
        return {"error": str(e)}

def call_ollama_with_history(
    system_prompt: str,
    messages_history: List[Dict],
    temperature: float = 0.3,
    max_tokens: int = 300
) -> str:
    """
    Call Ollama with full conversation history.
    """
    # Construct full message payload
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
    
    # Extract response
    if isinstance(resp, dict) and "message" in resp and isinstance(resp["message"], dict):
        return resp["message"].get("content", "").strip()
    elif isinstance(resp, dict) and "response" in resp:
        return str(resp.get("response", "")).strip()
    else:
        # Fallback for LLM failure/garbage
        return "⚠️ I couldn't get a coherent response from the AI model. Please try again or contact HR at people@acmeai.tech."

# ============================================================================
# MAIN QUERY HANDLER - UPDATED WITH SESSION AWARENESS
# ============================================================================
def handle_hr_query(
    user_input: str,
    chat_history: Optional[List[Dict]] = None,
    session_id: Optional[str] = None
) -> str:
    """
    Main function to handle HR queries with session awareness.
    ✅ FULLY FIXED: Proper intent classification and RAG handling
    """
    chat_history = chat_history or []
    user_input = (user_input or "").strip()
    
    # Log session info for debugging
    print(f"🆔 Session ID: {session_id}")
    print(f"💬 History length: {len(chat_history)}")
    
    # Validate input
    if not user_input:
        return "Please ask me something!"
    
    # STEP 1: Detect Language
    lang = detect_language(user_input)
    print(f"🌍 Detected language: {lang}")
    
    # STEP 2: Classify Intent
    intent = classify_intent(user_input)
    print(f"🎯 Detected intent: {intent}")
    
    # STEP 3: Handle with Predefined Responses (Fast Path)
    if intent in ["greeting", "identity", "thanks", "goodbye", "small_talk"]: 
        response = get_predefined_response(intent, lang)
        if response:
            print(f"⚡ Using predefined response for {intent}")
            return response
    
    # STEP 4: Handle Employee Lookup (Direct Path)
    if intent == "employee_lookup":
        name = extract_person_name(user_input)
        if name:
            emp = get_employee_by_name(name)
            if emp:
                print(f"👤 Found employee: {name}")
                return format_employee_info(emp)
            else:
                print(f"❌ Employee not found: {name}")
                return "❌ I couldn't find that employee in our records."
        # If intent is lookup but no name was extracted, proceed to RAG (Step 5)
    
    # STEP 5: Search Knowledge Base
    context, score = search_knowledge_base(user_input)
    print(f"📚 Knowledge base search score: {score:.3f}")
    
    # STEP 6: Validate Context Requirement
    requires_context = intent in [
        "policy_question",
        "salary_benefits",
        "procedure_howto",
        "office_logistics"
    ]
    
    # ✅ FIX: More nuanced context validation
    if requires_context:
        if not context:
            print("⚠️ No context found for policy question")
            return "⚠️ I couldn't find that in HR policies. Please contact HR at people@acmeai.tech for details."
        
        # ✅ FIX: Warn if score is low but still attempt answer
        if score < RELIABLE_SCORE:
            print(f"⚠️ Low confidence score: {score:.3f} (threshold: {RELIABLE_SCORE})")
            # Still proceed but with caveat in system prompt
    
    # STEP 7: Build Messages for LLM
    messages_for_llm = []
    current_user_message = user_input
    
    # Use chat history for context (provided by session management)
    if chat_history and len(chat_history) > 0:
        # Use the full chat history provided by the session
        messages_for_llm.extend(chat_history)
    
    # Prepend context to current message if available
    if context:
        rag_prefix = f"Context from HR Knowledge Base:\n{context}\n\n"
        current_user_message = rag_prefix + current_user_message
    
    # Add current message
    messages_for_llm.append({"role": "user", "content": current_user_message})
    
    # STEP 8: Get System Prompt from prompt_config
    system_prompt = prompt_config.get_system_prompt(language=lang, context_type=intent)
    
    # ✅ FIX: Add warning for low-confidence answers
    if context and score < RELIABLE_SCORE:
        system_prompt += f"\n\n⚠️ IMPORTANT: The retrieved context has a lower confidence score ({score:.2f}). If you're not confident in your answer, recommend contacting HR."
    
    # STEP 9: Call LLM
    print("🤖 Calling Ollama LLM...")
    reply = call_ollama_with_history(system_prompt, messages_for_llm)
    
    return reply

# ============================================================================
# BACKWARD COMPATIBILITY ALIAS
# ============================================================================
def ask_hr_bot(user_input, chat_history=None, session_id=None):
    """Alias for backward compatibility with api_server.py"""
    return handle_hr_query(user_input, chat_history=chat_history, session_id=session_id)

# ============================================================================
# COMMAND-LINE DEBUG MODE
# ============================================================================
if __name__ == "__main__":
    print("\n" + "="*70)
    print("HR CHATBOT - DEBUG MODE")
    print("="*70)
    print("\nTry these commands:")
    print("  - hello")
    print("  - who are you")
    print("  - what is the leave policy")
    print("  - what is the company goal for the next two years")
    print("  - how are you")
    print("  - who is Omar Faruk")
    print("  - exit (to quit)")
    print("="*70 + "\n")
    
    history = []
    session_id = "debug_session"
    
    while True:
        try:
            q = input("You: ").strip()
            
            if q.lower() in ("exit", "quit"):
                print("\n👋 Goodbye!\n")
                break
            
            if not q:
                continue
            
            history.append({"role": "user", "content": q})

            print()
            response = ask_hr_bot(q, chat_history=history, session_id=session_id)
            print(f"Bot: {response}\n")

            history.append({"role": "assistant", "content": response})
            history = history[-6:]  # Keep last 6 messages
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!\n")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}\n")
