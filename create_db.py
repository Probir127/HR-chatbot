import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
import numpy as np
import os
import re


# === PATH CONFIGURATION ===
PDF_PATH = "D:/tutorial/data/General HR Queries.pdf"
DB_FAISS_PATH = "D:/tutorial/vectorstores/db_faiss"


# === STEP 1: READ FULL PDF TEXT ===
def read_full_pdf(pdf_path):
    print("📄 Reading complete HR knowledge base...")
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"❌ PDF not found: {pdf_path}")

    doc = fitz.open(pdf_path)
    full_text = ""
    for page_num, page in enumerate(doc, start=1):
        page_text = page.get_text("text")
        if page_text.strip():
            full_text += page_text + "\n"
    doc.close()
    print(f"✅ Extracted {len(full_text.splitlines())} total lines.")
    return full_text


# === STEP 2: SMART CHUNKING - Q&A Based ===
def smart_chunk_by_qa(full_text):
    """
    Split by ###Question### markers to keep Q&A pairs together
    """
    print("🧩 Using smart Q&A-based chunking...")
    
    chunks = []
    
    # Split by ###Question### markers
    qa_sections = re.split(r'###Question###', full_text)
    
    current_chunk = ""
    chunk_size_limit = 1000  # Characters per chunk
    
    for section in qa_sections:
        if not section.strip():
            continue
        
        # Check if section has ###Answer###
        if '###Answer###' in section:
            # This is a complete Q&A pair
            qa_pair = "###Question###" + section
            
            # If adding this would exceed limit, save current chunk
            if len(current_chunk) + len(qa_pair) > chunk_size_limit and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = qa_pair
            else:
                current_chunk += "\n\n" + qa_pair if current_chunk else qa_pair
        else:
            # Non-Q&A content (like headers, employee tables)
            if current_chunk and len(current_chunk) > chunk_size_limit:
                chunks.append(current_chunk.strip())
                current_chunk = section
            else:
                current_chunk += "\n" + section if current_chunk else section
    
    # Add the last chunk
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    print(f"✅ Created {len(chunks)} Q&A-based chunks")
    return chunks


# === STEP 3: EXTRACT EMPLOYEE DATA SEPARATELY ===
def extract_employee_chunks(full_text):
    """
    Extract employee table data as separate chunks
    """
    print("👥 Extracting employee data...")
    
    employee_chunks = []
    
    # Find employee table sections
    lines = full_text.split('\n')
    in_employee_section = False
    current_employee_chunk = ""
    
    for i, line in enumerate(lines):
        # Detect start of employee section
        if 'Employee Name' in line and 'Email' in line:
            in_employee_section = True
            current_employee_chunk = "Employee Information:\n"
            continue
        
        # Detect end of employee section
        if in_employee_section and ('###Question###' in line or 'Job Description' in line):
            if current_employee_chunk.strip():
                employee_chunks.append(current_employee_chunk.strip())
            current_employee_chunk = ""
            in_employee_section = False
            continue
        
        # Collect employee data
        if in_employee_section:
            current_employee_chunk += line + "\n"
            
            # Also check if this line has an email (individual employee entry)
            if '@acmeai' in line or '@acmetechltd' in line:
                # Create a focused chunk for this employee
                # Look back and forward a few lines for context
                start_idx = max(0, i - 2)
                end_idx = min(len(lines), i + 3)
                employee_detail = '\n'.join(lines[start_idx:end_idx])
                
                if employee_detail.strip():
                    employee_chunks.append(f"Employee Detail:\n{employee_detail}")
    
    # Add any remaining employee data
    if current_employee_chunk.strip():
        employee_chunks.append(current_employee_chunk.strip())
    
    print(f"✅ Extracted {len(employee_chunks)} employee-specific chunks")
    return employee_chunks


# === STEP 4: FALLBACK SPLITTING FOR NON-Q&A CONTENT ===
def fallback_split(text):
    """Use recursive splitter for any remaining large text"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", "! ", "? ", " "]
    )
    return splitter.split_text(text)


# === STEP 5: BUILD FAISS DATABASE ===
def build_faiss_index(chunks, db_path):
    print(f"🔧 Creating embeddings and FAISS index from {len(chunks)} chunks...")
    
    # Remove duplicates
    unique_chunks = list(set(chunks))
    print(f"✅ After deduplication: {len(unique_chunks)} unique chunks")
    
    # Filter out empty or very short chunks
    filtered_chunks = [c for c in unique_chunks if len(c.strip()) > 50]
    print(f"✅ After filtering: {len(filtered_chunks)} quality chunks")
    
    embeddings = HuggingFaceBgeEmbeddings(model_name="all-MiniLM-L6-v2")
    db = FAISS.from_texts(filtered_chunks, embeddings)
    db.save_local(db_path)
    np.save(f"{db_path}/texts.npy", np.array(filtered_chunks, dtype=object))
    print(f"✅ FAISS database saved to {db_path}")
    print(f"📊 Total entries in database: {len(filtered_chunks)}")


# === MAIN EXECUTION ===
if __name__ == "__main__":
    try:
        print("\n" + "="*70)
        print("CREATING OPTIMIZED HR KNOWLEDGE BASE")
        print("="*70 + "\n")
        
        # Read PDF
        text = read_full_pdf(PDF_PATH)
        
        # Smart chunking by Q&A
        qa_chunks = smart_chunk_by_qa(text)
        
        # Extract employee data separately
        employee_chunks = extract_employee_chunks(text)
        
        # Combine all chunks
        all_chunks = qa_chunks + employee_chunks
        
        print(f"\n📊 Total chunks before FAISS: {len(all_chunks)}")
        print(f"   - Q&A chunks: {len(qa_chunks)}")
        print(f"   - Employee chunks: {len(employee_chunks)}")
        
        # Build FAISS
        build_faiss_index(all_chunks, DB_FAISS_PATH)
        
        print("\n" + "="*70)
        print("🎉 Done! The HR bot now has a properly chunked knowledge base.")
        print("="*70 + "\n")
        
        # Show sample chunks
        print("Sample chunks:")
        for i, chunk in enumerate(all_chunks[:3]):
            print(f"\n--- Chunk {i+1} (length: {len(chunk)}) ---")
            print(chunk[:300] + "..." if len(chunk) > 300 else chunk)
        
    except Exception as e:
        print(f"❌ Error building FAISS index: {e}")
        import traceback
        traceback.print_exc()