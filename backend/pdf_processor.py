import os
import fitz  # PyMuPDF
from config import PDF_ROOT, PDF_DIR, CHUNK_SIZE, CHUNK_OVERLAP


def extract_pages_from_pdf(pdf_path: str) -> list[tuple[int, str]]:
    """
    Extract text page-by-page from a PDF.
    Returns list of (page_number, text) tuples (1-indexed).
    Only returns pages that have actual text content.
    """
    doc = fitz.open(pdf_path)
    pages = []
    for page_num, page in enumerate(doc):
        text = page.get_text("text")
        if text.strip():
            pages.append((page_num + 1, text))
    doc.close()
    return pages


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract all text from a PDF (works with searchable/OCR PDFs)."""
    pages = extract_pages_from_pdf(pdf_path)
    return "\n".join(f"[Page {num}]\n{text}" for num, text in pages)


def _split_long_page(text: str, chunk_size: int, overlap: int) -> list[str]:
    """
    Sub-split a single page's text by word count when it is too long.
    Used only when a single page exceeds chunk_size words.
    """
    words = text.split()
    if len(words) <= chunk_size:
        return [text]
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start += chunk_size - overlap
    return chunks


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Split text into chunks, respecting page boundaries.
    Each [Page N] block becomes its own chunk(s).
    If a page is very long it is sub-split by word count.
    Kept for backward compatibility (used by upload path).
    """
    # Split on [Page N] markers that were inserted by extract_text_from_pdf
    import re
    page_blocks = re.split(r'\[Page \d+\]\n?', text)
    # First element before any [Page] tag is the metadata header (if any)
    header = page_blocks[0] if page_blocks else ""
    chunks = []
    for block in page_blocks[1:]:
        block = block.strip()
        if not block:
            continue
        sub = _split_long_page(block, chunk_size, overlap)
        for s in sub:
            # Keep the metadata header attached to every sub-chunk
            chunks.append((header.strip() + "\n" + s).strip() if header.strip() else s)
    # Fallback: if no [Page] markers found, use old word-count split
    if not chunks:
        words = text.split()
        start = 0
        while start < len(words):
            end = min(start + chunk_size, len(words))
            chunks.append(" ".join(words[start:end]))
            if end == len(words):
                break
            start += chunk_size - overlap
    return chunks


def get_metadata_from_path(pdf_path: str, root: str) -> dict:
    """
    Extract metadata from folder structure.
    Example path: .../REGULARIZATION/PUR/NS/NS.pdf
    Returns: { category, case_number, doc_type, filename, relative_path }
    """
    rel = os.path.relpath(pdf_path, root)   # e.g. REGULARIZATION\PUR\NS\NS.pdf
    parts = rel.replace("\\", "/").split("/")

    category    = parts[0] if len(parts) > 0 else "UNKNOWN"
    case_number = parts[1] if len(parts) > 1 else "UNKNOWN"
    doc_type    = parts[2] if len(parts) > 2 else "UNKNOWN"   # CS or NS
    filename    = parts[-1]

    return {
        "category":    category,
        "case_number": case_number,
        "doc_type":    doc_type,      # CS = Current Status, NS = Next Steps / Nakal
        "filename":    filename,
        "rel_path":    rel
    }


def process_pdf(pdf_path: str, root: str = None) -> list[dict]:
    """
    Process a single PDF → list of chunk dicts with rich metadata.
    Chunks by page: one chunk per page (sub-split if page is very long).
    Metadata header is prepended to EVERY chunk so retrieval always knows the case.
    """
    meta = get_metadata_from_path(pdf_path, root or PDF_ROOT)
    label = f"{meta['category']} / {meta['case_number']} / {meta['doc_type']}"
    print(f"  📄 Processing: {label}")

    pages = extract_pages_from_pdf(pdf_path)
    if not pages:
        print(f"  ⚠️  No text found in {label} (image-only PDF?)")
        return []

    # Metadata header — prepended to every chunk so every retrieval result is self-contained
    header = (
        f"[Category: {meta['category']}] "
        f"[Case: {meta['case_number']}] "
        f"[Type: {meta['doc_type']}]"
    )

    result = []
    chunk_index = 0
    for page_num, page_text in pages:
        page_text = page_text.strip()
        if not page_text:
            continue

        # Sub-split only if page exceeds CHUNK_SIZE words
        sub_chunks = _split_long_page(page_text, CHUNK_SIZE, CHUNK_OVERLAP)

        for sub in sub_chunks:
            full_chunk = f"{header}\n[Page {page_num}]\n{sub}"
            result.append({
                "text":        full_chunk,
                "filename":    f"{meta['category']}/{meta['case_number']}/{meta['doc_type']}/{meta['filename']}",
                "category":    meta["category"],
                "case_number": meta["case_number"],
                "doc_type":    meta["doc_type"],
                "chunk_index": chunk_index,
                "page_number": page_num,
            })
            chunk_index += 1

    print(f"  ✅ {label} → {len(result)} chunks from {len(pages)} pages")
    return result


def scan_pdf_root(root: str = PDF_ROOT) -> list[str]:
    """Recursively find all PDFs under root directory."""
    pdf_files = []
    if not os.path.exists(root):
        print(f"  ⚠️  PDF_ROOT not found: {root}")
        return []
    for dirpath, _, filenames in os.walk(root):
        for fname in filenames:
            if fname.lower().endswith(".pdf"):
                pdf_files.append(os.path.join(dirpath, fname))
    return sorted(pdf_files)


def load_all_pdfs() -> list[dict]:
    """Load and chunk PDFs from PDF_ROOT only (recursive). No uploads."""
    all_chunks = []
    print(f"\n  📂 Scanning: {PDF_ROOT}")
    root_pdfs = scan_pdf_root(PDF_ROOT)
    if root_pdfs:
        print(f"  📂 Found {len(root_pdfs)} PDF(s)")
        for pdf_path in root_pdfs:
            all_chunks.extend(process_pdf(pdf_path, root=PDF_ROOT))
    else:
        print("  📂 No PDFs found")
    return all_chunks
