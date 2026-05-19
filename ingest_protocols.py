import pypdf
import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

def ingest_protocols_spatially():
    # 1. SETUP LOCAL EMBEDDINGS
    print("📡 Initializing Local Embedding Engine...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    pdf_path = r"C:\EMT_Agent\assets\protocols.pdf"
    all_docs = []

    if not os.path.exists(pdf_path):
        print(f"❌ ERROR: PDF not found at {pdf_path}")
        return

    # Use pure-Python reader to bypass Windows Application Control DLL blocks
    reader = pypdf.PdfReader(pdf_path)
    total_pages = len(reader.pages)
    
    print(f"📖 PDF loaded successfully. Processing {total_pages} pages...")

    # Spatial page-slicing loop with targeted exclusions
    for page_idx in range(total_pages):
        human_page_num = page_idx + 1
        
        # --- THE CUT LIST ---
        # 1. Skip State Formulary (Pages 392 to 404)
        if 392 <= human_page_num <= 404:
            continue
            
        # 2. Skip VAD Maintenance & Appendix (Pages 405 to 435)
        if 405 <= human_page_num <= 435:
            continue
        # --------------------

        page = reader.pages[page_idx]
        
        # Extract layout-preserved text blocks using pypdf's visitor/layout features
        # We look at the horizontal layout to split BLS (left) and ALS (right)
        bls_lines = []
        als_lines = []
        page_title = ""

        def visitor_body(text, cm, tm, font_dict, font_size):
            nonlocal page_title
            if not text.strip(): 
                return
            
            # tm[4] is the X coordinate (horizontal position on the page)
            # tm[5] is the Y coordinate (vertical position on the page)
            x_pos = tm[4]
            y_pos = tm[5]
            
            # Capture the header/title area
            if y_pos > 700:  
                page_title += text.strip() + " "
            # Left side of the page is BLS (Assuming standard 612pt width letter page, midpoint ~306)
            elif x_pos < 300:  
                bls_lines.append(text.strip())
            # Right side of the page is ALS
            else:  
                als_lines.append(text.strip())

        # Trigger the content extraction visitor
        page.extract_text(visitor_text=visitor_body)
        
        clean_title = page_title.strip() if page_title.strip() else f"Page {human_page_num}"
        bls_text = " ".join(bls_lines)
        als_text = " ".join(als_lines)

        if bls_text.strip():
            all_docs.append(Document(
                page_content=f"PROTOCOL: {clean_title}\n[SCOPE: BLS]\n{bls_text}", 
                metadata={"scope": "BLS", "page": human_page_num}
            ))
        if als_text.strip():
            all_docs.append(Document(
                page_content=f"PROTOCOL: {clean_title}\n[SCOPE: ALS]\n{als_text}", 
                metadata={"scope": "ALS", "page": human_page_num}
            ))

    # 2. BUILD THE DATABASE LOCALLY
    print(f"🧠 Building Local Vector Database for {len(all_docs)} chunks...")
    vector_db = FAISS.from_documents(all_docs, embeddings)
    vector_db.save_local("protocol_db")
    print(f"✅ SUCCESS: Spatial database created locally using pure Python extraction.")

if __name__ == "__main__":
    ingest_protocols_spatially()