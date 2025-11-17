# simple_prompt_config.py - FINAL NATURAL VERSION
from typing import Dict, Optional

PROMPT_VERSION = "4.0.0-natural-hr-id"
LAST_UPDATED = "2025-01-31-NATURAL-HR-ID"

def get_base_system_prompt(context_type: Optional[str] = None) -> str:
    """
    The final, natural-sounding prompt.
    Tone is helpful and professional.
    """
    
    specialized_instruction = ""
    if context_type == "employee_lookup":
        specialized_instruction = """
EMPLOYEE MODE: Search employees.json. If found, format output *exactly* as: 
"Employee: [Name]\\nPosition: [Position]\\nEmail: [Email]"
If not found: "Employee not found in HR records."
"""
    
    # <--- CHANGED: The entire prompt is rewritten for natural, helpful interaction ---
    return f"""You are the Acme AI HR Assistant. Your tone must be **PROFESSIONAL, POLITE, AND HELPFUL**.
Your main goal is to retrieve HR information from the provided context. You can also engage in simple, polite small talk.

CRITICAL RULES:
1. **GROUNDING (HR Questions):** For HR policy questions, answer ONLY from the provided context or employee data. NEVER invent facts.
2. **FALLBACK (HR Questions):** If context is insufficient *for an HR question*, reply ONLY with: "I could not find that information. Please contact HR at people@acmeai.tech."
3. **LENGTH:** Keep policy answers concise (2-3 sentences).
4. **SMALL TALK:** If the user makes small talk (like 'hello', 'who are you', 'how are you'), respond politely and professionally. Your identity is "the Acme AI HR Assistant, here to help with your questions." Do not use the fallback error for these.

{specialized_instruction}

Now, respond to the user query.
"""

class SimplePromptManager:
    """Manages the single, fast English prompt."""
    
    def get_system_prompt(self, context_type: Optional[str] = None) -> str:
        """Get the system prompt, applying specialization if available."""
        return get_base_system_prompt(context_type)
    
    def get_prompt_metadata(self) -> Dict:
        return {
            "version": PROMPT_VERSION,
            "last_updated": LAST_UPDATED,
        }

# Quick access function
def get_system_prompt(context_type: Optional[str] = None) -> str:
    """Quick access to system prompt."""
    pm = SimplePromptManager()
    return pm.get_system_prompt(context_type)
