import os
import re
import fitz
from config import PDF_ROOT, PDF_DIR, CHUNK_SIZE, CHUNK_OVERLAP


def extract_pages_from_pdf(pdf_path: str) -> list[tuple[int, str]]:
    doc = fitz.open(pdf_path)
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text("text")
        if text.strip():
            pages.append((i + 1, text))
    doc.close()
    return pages


def extract_text_from_pdf(pdf_path: str) -> str:
    pages = extract_pages_from_pdf(pdf_path)
    return "\n".join(f"[Page {n}]\n{t}" for n, t in pages)


def _is_heading(line: str) -> bool:
    line = line.strip()
    if not line or len(line) > 100:
        return False
    if line.isupper() and 4 <= len(line) <= 60:
        return True
    if line.endswith(':') and len(line) <= 80:
        return True
    patterns = [
        r'(?i)^name\s+of\s+work\b',
        r'(?i)^subject\s*[:\-]',
        r'(?i)^work\s+order\b',
        r'(?i)^notice\s+inviting\s+tender',
        r'(?i)^details?\s+of\b',
        r'(?i)^schedule\s+[a-z]\b',
        r'(?i)^clause\s+\d+',
        r'(?i)^section\s+\d+',
        r'(?i)^part\s+[ivxIVX\d]+\b',
        r'(?i)^(?:sr\.?\s*no|s\.?\s*no|क्रमांक)',
    ]
    return any(re.match(p, line) for p in patterns)


def _semantic_split(page_text: str, chunk_size: int, overlap: int) -> list[str]:
    lines = page_text.split('\n')
    segments = []
    current = []

    for line in lines:
        if _is_heading(line) and current:
            seg = '\n'.join(current).strip()
            if seg:
                segments.append(seg)
            current = [line]
        else:
            current.append(line)

    if current:
        seg = '\n'.join(current).strip()
        if seg:
            segments.append(seg)

    chunks = []
    for seg in segments:
        words = seg.split()
        if len(words) <= chunk_size:
            chunks.append(seg)
        else:
            start = 0
            while start < len(words):
                end = min(start + chunk_size, len(words))
                chunks.append(' '.join(words[start:end]))
                if end == len(words):
                    break
                start += chunk_size - overlap

    return chunks if chunks else [page_text.strip()]


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    page_blocks = re.split(r'\[Page \d+\]\n?', text)
    header = page_blocks[0] if page_blocks else ""
    chunks = []
    for block in page_blocks[1:]:
        block = block.strip()
        if not block:
            continue
        for sub in _semantic_split(block, chunk_size, overlap):
            chunks.append((header.strip() + "\n" + sub).strip() if header.strip() else sub)
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
    rel = os.path.relpath(pdf_path, root)
    parts = rel.replace("\\", "/").split("/")
    return {
        "category":    parts[0] if len(parts) > 0 else "UNKNOWN",
        "case_number": parts[1] if len(parts) > 1 else "UNKNOWN",
        "doc_type":    parts[2] if len(parts) > 2 else "UNKNOWN",
        "filename":    parts[-1],
        "rel_path":    rel
    }


def process_pdf(pdf_path: str, root: str = None) -> list[dict]:
    meta = get_metadata_from_path(pdf_path, root or PDF_ROOT)
    label = f"{meta['category']} / {meta['case_number']} / {meta['doc_type']}"
    print(f"  Processing: {label}")

    pages = extract_pages_from_pdf(pdf_path)
    if not pages:
        print(f"  No text found in {label} (image-only PDF?)")
        return []

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
        for sub in _semantic_split(page_text, CHUNK_SIZE, CHUNK_OVERLAP):
            if not sub.strip():
                continue
            result.append({
                "text":        f"{header}\n[Page {page_num}]\n{sub}",
                "filename":    f"{meta['category']}/{meta['case_number']}/{meta['doc_type']}/{meta['filename']}",
                "category":    meta["category"],
                "case_number": meta["case_number"],
                "doc_type":    meta["doc_type"],
                "chunk_index": chunk_index,
                "page_number": page_num,
            })
            chunk_index += 1

    print(f"  {label} -> {len(result)} chunks from {len(pages)} pages")
    return result


def scan_pdf_root(root: str = PDF_ROOT) -> list[str]:
    pdf_files = []
    if not os.path.exists(root):
        print(f"  PDF root not found: {root}")
        return []
    for dirpath, _, filenames in os.walk(root):
        for fname in filenames:
            if fname.lower().endswith(".pdf"):
                pdf_files.append(os.path.join(dirpath, fname))
    return sorted(pdf_files)


def load_all_pdfs() -> list[dict]:
    all_chunks = []
    print(f"\n  Scanning: {PDF_ROOT}")
    pdfs = scan_pdf_root(PDF_ROOT)
    if pdfs:
        print(f"  Found {len(pdfs)} PDF(s)")
        for path in pdfs:
            all_chunks.extend(process_pdf(path, root=PDF_ROOT))
    else:
        print("  No PDFs found")
    return all_chunks
