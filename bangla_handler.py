import re

def contains_bangla(text: str) -> bool:
    """Detect true Bangla script."""
    return bool(re.search(r'[\u0980-\u09FF]', text))

def is_probable_banglish(text: str) -> bool:
    """Detect Romanized Bangla (Banglish) more reliably."""
    roman_bn_keywords = [
        "tumi", "ami", "valo", "bhalo", "kemon", "aso", "asi",
        "korbo", "korchho", "korchi", "hobe", "ache", "bolo",
        "amar", "tomar", "ekhane", "onek", "bujhte", "parbo"
    ]
    tokens = text.lower().split()
    matches = sum(1 for t in tokens if t in roman_bn_keywords)
    return matches >= 2  # need clear pattern

def process_mixed_input(user_input: str) -> dict:
    """
    Detect input language:
      - 'bn'  → Bangla (বাংলা)
      - 'banglish' → Roman Bangla
      - 'en' → English
    """
    if contains_bangla(user_input):
        lang = "bn"
    elif is_probable_banglish(user_input):
        lang = "banglish"
    else:
        lang = "en"
    return {"lang": lang}
