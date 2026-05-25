import pypdf
import os
import re
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Matches the repeated page header burned into every page of the PWC PCM PDF.
# Stripping it before chunking reclaims ~40 tokens per chunk and reduces
# artificial similarity between all chunks in the vector store.
_HEADER_RE = re.compile(
    r'Prince William County Fire\s*[&]\s*Rescue System\s*'
    r'Patient Care Manual\s*Date:[^\n]*\n?',
    re.IGNORECASE,
)

def ingest_protocols():
    print("Initializing embedding engine...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    pdf_path = r"C:\EMT_Agent\assets\protocols.pdf"
    if not os.path.exists(pdf_path):
        print(f"ERROR: PDF not found at {pdf_path}")
        return

    reader = pypdf.PdfReader(pdf_path)
    total_pages = len(reader.pages)
    print(f"PDF loaded: {total_pages} pages")

    # all-MiniLM-L6-v2 has a hard 256-token limit (~1 000 chars).
    # chunk_size=800 leaves headroom for the PROTOCOL title prefix added below.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        separators=["\n\n", "\n", " ", ""],
    )

    all_docs = []
    skipped = 0

    for page_idx in range(total_pages):
        human_page = page_idx + 1

        # Pages 1-4: administrative change bulletin, not clinical protocol content.
        # Indexing them risks surfacing deprecated dose values listed alongside new ones.
        if 1 <= human_page <= 4:
            skipped += 1
            continue

        # Pages 392-435: State Formulary and VAD Maintenance appendix (user-specified exclusion).
        if 392 <= human_page <= 435:
            skipped += 1
            continue

        page = reader.pages[page_idx]
        raw = page.extract_text() or ""
        if not raw.strip():
            continue

        # Strip the repeated page header (appears up to twice per page in the extracted text)
        text = _HEADER_RE.sub("", raw).strip()
        if not text:
            continue

        # Use the first non-empty line as the document title prefix for context
        first_line = next((l.strip() for l in text.splitlines() if l.strip()), f"Page {human_page}")
        title = first_line[:120]

        chunks = splitter.split_text(text)
        for chunk_idx, chunk in enumerate(chunks):
            chunk = chunk.strip()
            if not chunk:
                continue
            all_docs.append(Document(
                page_content=f"PROTOCOL: {title}\n{chunk}",
                metadata={"page": human_page, "chunk": chunk_idx},
            ))

    print(f"Skipped {skipped} pages. Building database from {len(all_docs)} chunks...")
    vector_db = FAISS.from_documents(all_docs, embeddings)
    vector_db.save_local("protocol_db")
    print("SUCCESS: protocol_db rebuilt.")


if __name__ == "__main__":
    ingest_protocols()
