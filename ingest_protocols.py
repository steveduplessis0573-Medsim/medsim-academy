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

# The PWC PCM uses a two-column layout (BLS left, ALS right, ACP far-right).
# pypdf's extract_text() concatenates these column headers without spaces,
# producing tokens like "BLSALS" or "ALSACP" that the LLM cannot parse for
# scope. Replace them with a clear section divider before chunking.
def _fix_scope_headers(text: str) -> str:
    """Turn BLSALS / ALSACP / BLSALSACP into readable section markers.

    In PWC PCM pages, 'BLS' and 'ALS' are column headers printed side-by-side.
    extract_text() concatenates them without spaces. The divider marks where
    BLS-accessible content begins (both BLS and ALS may perform what follows).
    """
    text = re.sub(r'\bBLSALSACP\b', '\n[BLS + ALS + ACP PROVIDERS:]\n', text)
    text = re.sub(r'\bBLSALS\b',    '\n[BLS + ALS PROVIDERS:]\n', text)
    text = re.sub(r'\bALSACP\b',    '\n[ALS + ACP PROVIDERS:]\n', text)
    return text


# Roman numerals and bare integers are page/section numbers, not useful titles.
_BARE_NUMBER_RE = re.compile(r'^[IVXLCDM\d]+$', re.IGNORECASE)


def _extract_title(text: str, page_num: int) -> str:
    """Return the first meaningful line of page text, skipping bare numbers."""
    for line in text.splitlines():
        stripped = line.strip()
        if stripped and not _BARE_NUMBER_RE.fullmatch(stripped):
            return stripped[:120]
    return f"Page {page_num}"

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
        # Repair column-header concatenation (BLSALS → BLS SCOPE / ALS SCOPE markers)
        text = _fix_scope_headers(text)
        if not text:
            continue

        title = _extract_title(text, human_page)

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
