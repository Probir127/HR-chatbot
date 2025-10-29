import re
import json
from pdfminer.high_level import extract_text

PDF_PATH = "D:/tutorial/data/General HR Queries.pdf"
OUTPUT_PATH = "D:/tutorial/data/employees.json"

def clean_pdf_text(text):
    # Merge broken lines and remove double spaces
    text = text.replace("\n", " ")
    text = re.sub(r"\s{2,}", " ", text)

    # Fix split emails like 'acmeai@g mail.com'
    text = re.sub(r"(\w)@\s*(\w)", r"\1@\2", text)
    text = re.sub(r"(\w)\s*\.\s*(\w)", r"\1.\2", text)
    text = re.sub(r"\s+@\s+", "@", text)
    text = re.sub(r"\s+([.])", r"\1", text)

    return text


def looks_like_name(s):
    # Reject short or non-name fragments
    if len(s.split()) < 2:
        return False
    if any(word.lower() in s.lower() for word in ["email", "contact", "please", "hr", "department"]):
        return False
    return True


def extract_employees(text):
    employees = []

    # Pattern that captures: Name + Email + optional role
    pattern = re.compile(
        r"([A-Z][A-Za-z .'-]{2,})\s+([a-zA-Z0-9_.+-]+@[a-zA-Z0-9_.+-]+\.[a-zA-Z]+)(?:\s+([A-Za-z\s]+(?:Manager|Lead|Coordinator|Engineer|Officer|Support)))?",
        re.IGNORECASE
    )

    for match in pattern.findall(text):
        name, email, position = match
        name = name.strip().title()
        email = email.lower().replace(" ", "")
        position = position.strip() if position else ""

        if looks_like_name(name) and len(email) > 6:
            employees.append({
                "name": name,
                "email": email,
                "position": position
            })

    # Deduplicate by email
    unique = {e["email"]: e for e in employees}
    return list(unique.values())


def main():
    print("📄 Extracting employees from:", PDF_PATH)
    text = extract_text(PDF_PATH)
    cleaned = clean_pdf_text(text)
    employees = extract_employees(cleaned)

    print(f"✅ Found {len(employees)} valid employee records")
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(employees, f, indent=4, ensure_ascii=False)
    print(f"💾 Saved cleaned JSON to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
