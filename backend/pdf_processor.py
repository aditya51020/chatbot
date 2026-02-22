import os
import fitz  # PyMuPDF
from config import PDF_ROOT, PDF_DIR, CHUNK_SIZE, CHUNK_OVERLAP


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract all text from a PDF (works with searchable/OCR PDFs)."""
    doc = fitz.open(pdf_path)
    full_text = []
    for page_num, page in enumerate(doc):
        text = page.get_text("text")
        if text.strip():
            full_text.append(f"[Page {page_num + 1}]\n{text}")
    doc.close()
    return "\n".join(full_text)


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping word chunks."""
    words = text.split()
    chunks = []
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
    """
    meta = get_metadata_from_path(pdf_path, root or PDF_ROOT)
    label = f"{meta['category']} / {meta['case_number']} / {meta['doc_type']}"
    print(f"  📄 Processing: {label}")

    raw_text = extract_text_from_pdf(pdf_path)
    if not raw_text.strip():
        print(f"  ⚠️  No text found in {label} (image-only PDF?)")
        return []

    # Prepend metadata header so LLM knows context of each chunk
    header = (
        f"[Category: {meta['category']}] "
        f"[Case: {meta['case_number']}] "
        f"[Type: {meta['doc_type']}]\n"
    )
    chunks = chunk_text(header + raw_text)

    result = []
    for i, chunk in enumerate(chunks):
        result.append({
            "text":        chunk,
            "filename":    f"{meta['category']}/{meta['case_number']}/{meta['doc_type']}/{meta['filename']}",
            "category":    meta["category"],
            "case_number": meta["case_number"],
            "doc_type":    meta["doc_type"],
            "chunk_index": i
        })

    print(f"  ✅ {label} → {len(chunks)} chunks")
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
