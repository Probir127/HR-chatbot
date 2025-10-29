"""
pdf_reader.py
-------------
Reads the entire HR knowledge base PDF properly using PyMuPDF
and exposes a simple function: load_pdf_text()
You can import it inside backend.py or any other file.
"""

import fitz  # PyMuPDF
import os


def load_pdf_text(pdf_path: str) -> str:
    """
    Reads all text from a properly formatted PDF file (not plain text).
    Returns one combined text string containing all pages.
    """

    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"❌ PDF not found: {pdf_path}")

    try:
        doc = fitz.open(pdf_path)
        full_text = ""
        for page in doc:
            page_text = page.get_text("text")
            if page_text.strip():
                full_text += page_text + "\n"
        doc.close()
        print(f"✅ Successfully extracted text from {len(full_text.splitlines())} lines.")
        return full_text.strip()

    except Exception as e:
        print(f"⚠️ Error reading PDF: {e}")
        return ""


# Manual test (optional)
if __name__ == "__main__":
    pdf_path = "D:/tutorial/data/General HR Queries.pdf"
    text = load_pdf_text(pdf_path)
    print(f"\nSample (first 1000 chars):\n{text[:1000]}")
