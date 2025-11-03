"""
create_db.py - Build FAISS Vector Database from HR PDF
"""
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
import re

# Configuration
PDF_PATH = "./data/General HR Queries.pdf"
DB_FAISS_PATH = "./vectorstores/db_faiss"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

def read_pdf(pdf_path):
    """Extract text from PDF"""
    print(f"📄 Reading PDF: {pdf_path}")
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"❌ PDF not found: {pdf_path}")
    
    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text("text") + "\n"
    doc.close()
    
    print(f"✅ Extracted {len(full_text)} characters")
    return full_text

def smart_chunk_text(text):
    """Chunk text by Q&A pairs"""
    print("🧩 Chunking text...")
    
    chunks = []
    qa_sections = re.split(r'###Question###', text)
    
    for section in qa_sections:
        if not section.strip():
            continue
        
        if '###Answer###' in section:
            qa_pair = "###Question###" + section
            
            # Split long Q&A pairs
            if len(qa_pair) > CHUNK_SIZE:
                # Split by sentences
                sentences = re.split(r'(?<=[.!?])\s+', qa_pair)
                current = ""
                for sent in sentences:
                    if len(current) + len(sent) > CHUNK_SIZE and current:
                        chunks.append(current.strip())
                        current = sent
                    else:
                        current += " " + sent
                if current.strip():
                    chunks.append(current.strip())
            else:
                chunks.append(qa_pair.strip())
    
    print(f"✅ Created {len(chunks)} chunks")
    return chunks

def build_faiss_index(chunks, output_path):
    """Build and save FAISS index"""
    print("🔧 Building FAISS index...")
    
    # Remove duplicates
    unique_chunks = list(set(chunks))
    print(f"✅ After deduplication: {len(unique_chunks)} chunks")
    
    # Filter short chunks
    filtered = [c for c in unique_chunks if len(c.strip()) > 50]
    print(f"✅ After filtering: {len(filtered)} quality chunks")
    
    # Create embeddings
    print("🧮 Creating embeddings...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(filtered, show_progress_bar=True)
    
    # Normalize for cosine similarity
    faiss.normalize_L2(embeddings)
    
    # Build FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # Inner product = cosine similarity
    index.add(embeddings)
    
    # Save
    os.makedirs(output_path, exist_ok=True)
    faiss.write_index(index, f"{output_path}/index.faiss")
    np.save(f"{output_path}/texts.npy", np.array(filtered, dtype=object))
    
    print(f"✅ FAISS index saved to {output_path}")
    print(f"📊 Total entries: {len(filtered)}")

def main():
    print("\n" + "="*70)
    print("BUILDING HR KNOWLEDGE BASE")
    print("="*70 + "\n")
    
    # Read PDF
    text = read_pdf(PDF_PATH)
    
    # Chunk text
    chunks = smart_chunk_text(text)
    
    # Build index
    build_faiss_index(chunks, DB_FAISS_PATH)
    
    print("\n" + "="*70)
    print("✅ DATABASE BUILD COMPLETE")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
