import fitz  # PyMuPDF
import os
import re
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# PWC PCM sidebar colors as (R, G, B) floats returned by PyMuPDF get_drawings().
# These were measured on p57 (Allergic Reaction) and are consistent across the manual.
_BLS_SIDEBAR = (0.144, 0.252, 0.559)   # dark blue strip → BLS-only content
_ALS_SIDEBAR = (1.0,   0.804, 0.011)   # yellow/gold strip → ALS-only content
_COLOR_TOL   = 0.12                     # per-channel tolerance

# Page geometry constants (points, PyMuPDF: y=0 at top)
_SIDEBAR_X_MAX = 55.0   # sidebar strip ends here; content begins to the right
_HEADER_Y_MAX  = 55.0   # green header bar occupies top ~55 pts
_FOOTER_Y_MIN  = 740.0  # green footer bar occupies bottom ~52 pts

_BARE_NUMBER_RE = re.compile(r'^[IVXLCDM\d]+$', re.IGNORECASE)


def _color_close(c1, c2):
    return all(abs(a - b) <= _COLOR_TOL for a, b in zip(c1[:3], c2[:3]))


def _get_scope_zones(page):
    """
    Scan left-margin drawing fills for BLS/ALS sidebar strips.
    Returns a sorted list of (y0, y1, scope_label) tuples.
    """
    zones = []
    for path in page.get_drawings():
        fill = path.get("fill")
        if not fill or len(fill) < 3:
            continue
        r = path["rect"]
        # Only the narrow left sidebar strip: x0 near 0, x1 around 50
        if r.x0 > 10 or r.x1 < 30 or r.x1 > _SIDEBAR_X_MAX:
            continue
        if _color_close(fill, _BLS_SIDEBAR):
            zones.append((r.y0, r.y1, "BLS"))
        elif _color_close(fill, _ALS_SIDEBAR):
            zones.append((r.y0, r.y1, "ALS"))
    return sorted(zones)


def _extract_page_text(page):
    """
    Extract page content text with BLS/ALS scope markers where color zones exist.
    Falls back to plain extraction (skipping header/footer bands) when no zones found.
    """
    pr = page.rect
    fallback_clip = fitz.Rect(_SIDEBAR_X_MAX, _HEADER_Y_MAX, pr.width, _FOOTER_Y_MIN)

    zones = _get_scope_zones(page)
    if not zones:
        return page.get_text("text", clip=fallback_clip).strip()

    parts = []
    for y0, y1, scope in zones:
        clip = fitz.Rect(_SIDEBAR_X_MAX, y0, pr.width, y1)
        text = page.get_text("text", clip=clip).strip()
        if text:
            parts.append(f"[{scope} SECTION:]\n{text}")

    if not parts:
        return page.get_text("text", clip=fallback_clip).strip()

    return "\n\n".join(parts)


def _extract_title(page, page_num):
    """
    Pull the protocol title from white text in the green header bar (y < 55 pts).
    Falls back to 'Page N' if no suitable text is found.
    """
    for block in page.get_text("dict")["blocks"]:
        if block.get("type") != 0:
            continue
        if block["bbox"][1] > _HEADER_Y_MAX:
            continue
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                text = span.get("text", "").strip()
                if not text or _BARE_NUMBER_RE.fullmatch(text):
                    continue
                color = span.get("color", 0)
                r = (color >> 16) & 0xFF
                g = (color >> 8) & 0xFF
                b = color & 0xFF
                if r > 200 and g > 200 and b > 200:   # white text = protocol title
                    return text[:120]
    return f"Page {page_num}"


def ingest_protocols():
    print("Initializing embedding engine...")
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"local_files_only": True},
    )

    pdf_path = r"C:\EMT_Agent\assets\protocols.pdf"
    if not os.path.exists(pdf_path):
        print(f"ERROR: PDF not found at {pdf_path}")
        return

    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    print(f"PDF loaded: {total_pages} pages")

    # all-MiniLM-L6-v2 has a 256-token ceiling (~1 000 chars).
    # chunk_size=800 leaves headroom for the PROTOCOL title prefix prepended below.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        separators=["\n\n", "\n", " ", ""],
    )

    all_docs = []
    skipped = 0

    for page_idx in range(total_pages):
        human_page = page_idx + 1

        # Pages 1-4: administrative change bulletin containing deprecated dose values.
        if 1 <= human_page <= 4:
            skipped += 1
            continue

        # Pages 392-435: State Formulary and VAD Maintenance appendix (user-specified exclusion).
        if 392 <= human_page <= 435:
            skipped += 1
            continue

        page = doc[page_idx]
        text = _extract_page_text(page)
        if not text:
            continue

        title = _extract_title(page, human_page)
        chunks = splitter.split_text(text)

        for chunk_idx, chunk in enumerate(chunks):
            chunk = chunk.strip()
            if not chunk:
                continue
            all_docs.append(Document(
                page_content=f"PROTOCOL: {title}\n{chunk}",
                metadata={"page": human_page, "chunk": chunk_idx},
            ))

    doc.close()
    print(f"Skipped {skipped} pages. Building database from {len(all_docs)} chunks...")
    vector_db = FAISS.from_documents(all_docs, embeddings)
    vector_db.save_local("protocol_db")
    print("SUCCESS: protocol_db rebuilt.")


if __name__ == "__main__":
    ingest_protocols()
