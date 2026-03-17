import os
import re
import fitz  # PyMuPDF

from config import CHUNK_SIZE, CHUNK_OVERLAP, PDF_ROOT


# ---------------------------------------------------------------------------
# Text Normalization
# ---------------------------------------------------------------------------

def normalize_text(text: str) -> str:
    """
    Clean extracted PDF text before indexing.
    - Strips pipe characters (table separators / OCR noise)
    - Collapses multiple whitespace into single space
    - Strips leading/trailing whitespace
    """
    text = text.replace("|", " ")
    text = re.sub(r" {2,}", " ", text)   # collapse multiple spaces only
    # Preserve newlines — they are paragraph/sentence boundaries used by chunker
    text = re.sub(r"\n{3,}", "\n\n", text)  # max 2 consecutive newlines
    return text.strip()


# ---------------------------------------------------------------------------
# Digital-only PDF extraction (no OCR)
# ---------------------------------------------------------------------------

def extract_pages_from_pdf(pdf_path: str) -> list[tuple[int, str]]:
    """
    Extract text from each page using PyMuPDF digital extraction only.
    Pages with no extractable text are skipped with a diagnostic log.
    Returns: list of (page_number, page_text) tuples (1-indexed).
    """
    pages = []
    try:
        doc = fitz.open(pdf_path)
        for i in range(len(doc)):
            raw = doc[i].get_text().strip()
            if not raw:
                print(f"  [pdf_processor] No digital text found on page {i + 1} – skipped.")
                continue
            clean = normalize_text(raw)
            if clean:
                pages.append((i + 1, clean))
        doc.close()
    except Exception as e:
        print(f"  [pdf_processor] Error reading {pdf_path}: {e}")
    return pages


def extract_text_from_pdf(pdf_path: str) -> str:
    """Convenience wrapper: returns a single string of all pages."""
    pages = extract_pages_from_pdf(pdf_path)
    return "\n\n".join(f"[Page {n}]\n{t}" for n, t in pages)


# ---------------------------------------------------------------------------
# Chunker — targets 350-450 chars, hard cap 600 chars
# ---------------------------------------------------------------------------

CHUNK_TARGET   = 400   # ideal chunk size in characters
CHUNK_MAX      = 600   # absolute hard cap
CHUNK_OVERLAP_CHARS = 80  # character overlap between consecutive chunks


def _split_into_sentences(text: str) -> list[str]:
    """
    Split text into sentence-like units using punctuation + newlines.
    Preserves the delimiter at the end of each sentence.
    """
    # Split on: sentence-ending punctuation, line breaks, bullets
    parts = re.split(r'(?<=[।\.!\?])\s+|\n+', text)
    return [p.strip() for p in parts if p.strip()]


def _chunk_by_sentences(text: str) -> list[str]:
    """
    Build chunks greedily from sentences.
    - Target: 350-450 chars
    - Hard cap: 600 chars
    - Overlap: last CHUNK_OVERLAP_CHARS chars of previous chunk prepended to next
    """
    sentences = _split_into_sentences(text)
    chunks = []
    current = ""

    for sent in sentences:
        # If a single sentence is already over cap, force-split it by words
        if len(sent) > CHUNK_MAX:
            # Flush current buffer first
            if current.strip():
                chunks.append(current.strip())
                current = ""
            # Word-level split
            words = sent.split()
            buf = ""
            for word in words:
                if len(buf) + len(word) + 1 > CHUNK_MAX and buf:
                    chunks.append(buf.strip())
                    # Carry overlap
                    buf = buf[-CHUNK_OVERLAP_CHARS:].strip() + " " + word
                else:
                    buf = (buf + " " + word).strip() if buf else word
            if buf.strip():
                current = buf.strip()
            continue

        candidate = (current + " " + sent).strip() if current else sent

        if len(candidate) <= CHUNK_TARGET:
            current = candidate
        elif len(candidate) <= CHUNK_MAX:
            # Within hard cap — accept but flush if adding more would exceed
            current = candidate
        else:
            # Flush current chunk
            if current.strip():
                chunks.append(current.strip())
                # Start next chunk with overlap tail
                overlap_tail = current[-CHUNK_OVERLAP_CHARS:].strip()
                current = (overlap_tail + " " + sent).strip() if overlap_tail else sent
            else:
                # current was empty, just use sentence
                current = sent

    if current.strip():
        chunks.append(current.strip())

    return chunks


# ---------------------------------------------------------------------------
# Heading / section detection (for smarter boundaries)
# ---------------------------------------------------------------------------

def _is_section_break(line: str) -> bool:
    """Returns True if this line looks like a major section heading."""
    line = line.strip()
    if not line or len(line) > 120:
        return False
    patterns = [
        r'(?i)^(?:notice\s+inviting\s+tender|nit)\b',
        r'(?i)^eligibility(\s+criteria)?\b',
        r'(?i)^terms\s+(and|&)\s+conditions\b',
        r'(?i)^schedule\s+[a-z]\b',
        r'(?i)^(?:sr\.?\s*no|s\.?\s*no|क्रमांक)\b',
        r'(?i)^(?:name\s+of\s+work|work\s+order)\b',
        r'^\d+[\.\)]\s+.{3,}',  # numbered list item
    ]
    return any(re.match(p, line) for p in patterns)


def chunk_page_text(page_text: str) -> list[str]:
    """
    Main chunker for a single page.
    Splits on section breaks first, then sentence-aware character chunking.
    """
    lines = page_text.split("\n")
    segments: list[str] = []
    current_seg_lines: list[str] = []

    for line in lines:
        if _is_section_break(line) and current_seg_lines:
            segments.append("\n".join(current_seg_lines).strip())
            current_seg_lines = [line]
        else:
            current_seg_lines.append(line)

    if current_seg_lines:
        segments.append("\n".join(current_seg_lines).strip())

    all_chunks: list[str] = []
    for seg in segments:
        if not seg:
            continue
        if len(seg) <= CHUNK_MAX:
            all_chunks.append(seg)
        else:
            all_chunks.extend(_chunk_by_sentences(seg))

    return [c for c in all_chunks if c.strip()]


# ---------------------------------------------------------------------------
# Metadata helpers
# ---------------------------------------------------------------------------

def get_metadata_from_path(pdf_path: str, root: str) -> dict:
    rel_path = os.path.relpath(pdf_path, root).replace("\\", "/")
    parts = rel_path.split("/")
    return {
        "rel_path":    rel_path,
        "category":    parts[0] if len(parts) > 1 else "General",
        "case_number": parts[1] if len(parts) > 2 else "Unknown",
        "doc_type":    "Land Record",
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def process_pdf(pdf_path: str, root: str = PDF_ROOT) -> list[dict]:
    """
    Full pipeline: extract → normalize → chunk → return chunk dicts.
    chunk["content"] contains ONLY clean page text — NO metadata tokens.
    Metadata lives exclusively in chunk["metadata"].
    """
    rel_path = os.path.relpath(pdf_path, root).replace("\\", "/")
    meta  = get_metadata_from_path(pdf_path, root)
    label = f"[{rel_path}]"
    print(f"  Processing: {label}")

    pages = extract_pages_from_pdf(pdf_path)
    if not pages:
        print(f"  {label} → no extractable text, skipped.")
        return []

    result: list[dict] = []
    chunk_lengths: list[int] = []
    chunk_index = 0

    for page_num, page_text in pages:
        chunks_for_page = chunk_page_text(page_text)
        for sub in chunks_for_page:
            sub = sub.strip()
            if not sub:
                continue
            result.append({
                "content": sub,          # ← PURE TEXT ONLY — no [Source:] / [Category:] / [Page N]
                "metadata": {
                    "filename":    rel_path,
                    "category":    meta["category"],
                    "case_number": meta["case_number"],
                    "doc_type":    meta["doc_type"],
                    "chunk_index": chunk_index,
                    "page_number": page_num,
                },
            })
            chunk_lengths.append(len(sub))
            chunk_index += 1

    if chunk_lengths:
        avg = sum(chunk_lengths) / len(chunk_lengths)
        print(f"  {label} → {len(result)} chunks")
        print(f"  [chunk_stats] total={len(result)} avg={avg:.0f} max={max(chunk_lengths)} min={min(chunk_lengths)} chars")
    else:
        print(f"  {label} → 0 chunks")

    return result


# ---------------------------------------------------------------------------
# Batch helpers
# ---------------------------------------------------------------------------

def scan_pdf_root(root: str = PDF_ROOT) -> list[str]:
    pdf_files = []
    if not os.path.exists(root):
        return []
    for dirpath, _, filenames in os.walk(root):
        for fname in filenames:
            if fname.lower().endswith(".pdf"):
                pdf_files.append(os.path.join(dirpath, fname))
    return sorted(pdf_files)


def load_all_pdfs() -> list[dict]:
    all_chunks: list[dict] = []
    print(f"\n  Scanning: {PDF_ROOT}")
    pdfs = scan_pdf_root(PDF_ROOT)
    if pdfs:
        print(f"  Found {len(pdfs)} PDF(s)")
        for path in pdfs:
            all_chunks.extend(process_pdf(path, root=PDF_ROOT))
    else:
        print("  No PDFs found in Sample_PDFs/")
    return all_chunks


import hashlib

def get_file_hash(filepath: str) -> str:
    hasher = hashlib.md5()
    with open(filepath, "rb") as f:
        hasher.update(f.read())
    return hasher.hexdigest()
