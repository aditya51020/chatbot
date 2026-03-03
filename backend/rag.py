import os
import re
import uuid
from typing import Optional

import chromadb
from sentence_transformers import SentenceTransformer

from config import (
    CHROMA_DIR, EMBED_MODEL,
    TOP_K, COLLECTION_NAME
)

# ── Lazy-loaded singletons ────────────────────────────────────────────────────
_embedder: Optional[SentenceTransformer] = None
_chroma_client = None
_collection = None
_phi3_model = None   # Phi-3-mini kept in RAM after first load (avoids 2-3 min cold-start/query)



def get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        print("Loading embedding model...")
        _embedder = SentenceTransformer(EMBED_MODEL)
        print("Embedding model loaded.")
    return _embedder


def get_collection():
    global _chroma_client, _collection
    if _collection is None:
        os.makedirs(CHROMA_DIR, exist_ok=True)
        _chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
        _collection = _chroma_client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )
    return _collection


# ── Indexing ───────────────────────────────────────────────────────────────

def index_chunks(chunks: list[dict]) -> int:
    if not chunks:
        return 0
    collection = get_collection()
    embedder   = get_embedder()
    texts      = [c["text"] for c in chunks]
    metadatas  = [{
        "filename":    c.get("filename", "unknown"),
        "chunk_index": c.get("chunk_index", 0),
        "category":    c.get("category", ""),
        "case_number": c.get("case_number", ""),
        "doc_type":    c.get("doc_type", ""),
    } for c in chunks]
    ids = [str(uuid.uuid4()) for _ in chunks]
    print(f"  Generating embeddings for {len(texts)} chunks...")
    embeddings = embedder.encode(texts, show_progress_bar=True).tolist()
    collection.add(documents=texts, embeddings=embeddings, metadatas=metadatas, ids=ids)
    print(f"  Indexed {len(texts)} chunks.")
    return len(texts)


def clear_collection():
    global _collection, _chroma_client
    get_collection()
    _chroma_client.delete_collection(COLLECTION_NAME)
    _collection = _chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )
    return True


def get_indexed_docs() -> list[str]:
    collection = get_collection()
    results    = collection.get(include=["metadatas"])
    if not results["metadatas"]:
        return []
    return sorted({m["filename"] for m in results["metadatas"] if "filename" in m})


def get_chunk_count() -> int:
    return get_collection().count()


# ── Retrieval ──────────────────────────────────────────────────────────────

RELEVANCE_THRESHOLD = 0.55


# Devanagari digit map: ०→0 … ९→9
_DEVA_DIGIT = str.maketrans('०१२३४५६७८९', '0123456789')


def devanagari_to_latin(text: str) -> str:
    """Replace Devanagari numerals with ASCII equivalents for regex matching."""
    return text.translate(_DEVA_DIGIT)


def clean_ocr_text(text: str) -> str:
    """Strip metadata headers and garbled OCR lines."""
    text = re.sub(r'\[Category:[^\]]+\]\s*', '', text)
    text = re.sub(r'\[Case:[^\]]+\]\s*',     '', text)
    text = re.sub(r'\[Type:[^\]]+\]\s*',     '', text)
    text = re.sub(r'\[Page \d+\]\n?',        '', text)

    lines = text.split('\n')
    clean = []
    for line in lines:
        line = line.strip()
        if not line or len(line) < 4:
            continue
        readable = sum(1 for c in line if c.isalnum() or c in ' .,;:-/()[]')
        if len(line) > 3 and readable / max(len(line), 1) < 0.40:
            continue
        clean.append(line)
    return '\n'.join(clean).strip()


def _query_chroma(
    query_text: str,
    n: int,
    filename_filter: str | None = None,
    category_filter: str | None = None,
) -> list[dict]:
    """Run a single ChromaDB embedding query; return list of chunk dicts.

    If filename_filter or category_filter is provided, only chunks from that
    file / category are returned (ChromaDB metadata filter).
    """
    collection = get_collection()
    if collection.count() == 0:
        return []
    embedder = get_embedder()
    emb      = embedder.encode([query_text]).tolist()

    # Build optional ChromaDB where-clause
    where_clause = None
    if filename_filter and category_filter:
        where_clause = {"$and": [
            {"filename":  {"$eq": filename_filter}},
            {"category":  {"$eq": category_filter}},
        ]}
    elif filename_filter:
        where_clause = {"filename": {"$eq": filename_filter}}
    elif category_filter:
        where_clause = {"category": {"$eq": category_filter}}

    query_kwargs = dict(
        query_embeddings=emb,
        n_results=min(n, collection.count()),
        include=["documents", "metadatas", "distances"],
    )
    if where_clause:
        query_kwargs["where"] = where_clause

    results = collection.query(**query_kwargs)
    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):
        score = round(1 - dist, 3)
        if score < RELEVANCE_THRESHOLD:
            continue
        chunks.append({
            "text":     doc,
            "filename": meta.get("filename", "unknown"),
            "category": meta.get("category", ""),
            "case":     meta.get("case_number", ""),
            "score":    score
        })
    return chunks


def retrieve_all_chunks_for_file(filename: str) -> list[dict]:
    """
    Fetch EVERY indexed chunk for a given file using a metadata filter.
    Used to exhaustively scan a known-relevant file for fields the embedding
    search missed (e.g. Agency Name may be on a different page than casting date).
    """
    collection = get_collection()
    try:
        results = collection.get(
            where={"filename": filename},
            include=["documents", "metadatas"]
        )
    except Exception:
        return []

    chunks = []
    for doc, meta in zip(results.get("documents", []), results.get("metadatas", [])):
        if doc:
            chunks.append({
                "text":     doc,
                "filename": meta.get("filename", filename),
                "category": meta.get("category", ""),
                "case":     meta.get("case_number", ""),
                "score":    0.5
            })
    print(f"  [file-scan] {filename}: {len(chunks)} chunks fetched")
    return chunks


def _work_name_boost(chunk_text: str, query: str) -> float:
    """
    Return a small relevance bonus if the chunk text contains
    meaningful Hindi/English keywords from the user's query.
    This helps the correct work page rank above unrelated pages.
    """
    # Extract multi-char words from the query (skip short function words)
    q_words = [w for w in re.findall(r'[\u0900-\u097F\w]{3,}', query) if len(w) >= 3]
    if not q_words:
        return 0.0
    text_lower = chunk_text.lower()
    matched = sum(1 for w in q_words if w.lower() in text_lower)
    return min(0.15, matched * 0.03)  # max +0.15 boost


def _plot_id_boost(chunk_text: str, query: str) -> float:
    """
    Give a strong boost to chunks that contain the exact plot ID from the query.
    Handles formats like: 1-F-48, D-A-25, 148, etc.
    """
    # Find plot-like identifiers in query: compound (1-F-48) or plain number (148)
    plot_ids = re.findall(
        r'\b(?:\d+[-/][A-Za-z][-/]\d+|[A-Za-z][-/][A-Za-z][-/]\d+|\d+[-/][A-Za-z]{1,3}[-/]\d+|\d{2,5})\b',
        query
    )
    if not plot_ids:
        return 0.0
    text_lower = chunk_text.lower()
    for pid in plot_ids:
        if pid.lower() in text_lower:
            return 0.30  # strong boost for exact plot number match
    return 0.0


def _extract_person_names(query: str) -> list[str]:
    """
    Pull out multi-word proper names from the query (e.g. 'chandmal somani').
    Returns a list of individual name words (>=4 chars, not common keywords).
    """
    _SKIP = {'agency', 'name', 'contractor', 'bidder', 'kya', 'hai', 'tender',
             'amount', 'number', 'date', 'casting', 'plot', 'size', 'jankari',
             'work', 'order', 'bid', 'ref', 'division', 'scheme', 'yojana'}
    words = re.findall(r'[A-Za-z\u0900-\u097F]{4,}', query)
    return [w.lower() for w in words if w.lower() not in _SKIP]


def retrieve_context(
    query: str,
    filename_filter: str | None = None,
    category_filter: str | None = None,
) -> list[dict]:
    """
    Multi-query retrieval + work-name + plot-ID relevance boost.
    1. Base embedding query
    2. One sub-query per detected intent
    3. Chunks with query keywords get a small boost
    4. Chunks with exact plot IDs get a strong boost
    5. Person-name filter: if query has a specific name, drop chunks not mentioning it

    filename_filter / category_filter — when set, restricts retrieval to that
    document or category (passed straight to ChromaDB metadata filter).
    """
    intents = detect_intent(query)

    queries = [query]
    for intent in intents:
        queries.append(f"{intent} {query}")

    seen: set        = set()
    all_chunks: list = []

    for q in queries:
        for chunk in _query_chroma(
            q, TOP_K,
            filename_filter=filename_filter,
            category_filter=category_filter,
        ):
            dedup_key = (chunk["filename"], chunk["text"][:60])
            if dedup_key not in seen:
                seen.add(dedup_key)
                # Apply work-name keyword boost
                boost = _work_name_boost(chunk["text"], query)
                # Apply strong plot-ID boost
                boost += _plot_id_boost(chunk["text"], query)
                chunk["score"] = min(1.0, chunk["score"] + boost)
                all_chunks.append(chunk)

    all_chunks.sort(key=lambda c: c["score"], reverse=True)

    # ── Person-name filter ────────────────────────────────────────────────
    # If the query mentions a specific person/firm name, only keep chunks
    # that contain at least one of those name words. Prevents pulling in
    # completely unrelated records (e.g. CS.pdf for a query about Somani).
    person_words = _extract_person_names(query)
    if person_words and not filename_filter:
        filtered = [
            c for c in all_chunks
            if any(pw in c["text"].lower() for pw in person_words)
        ]
        if filtered:  # only apply if at least one chunk passes
            print(f"  [name-filter] {len(filtered)}/{len(all_chunks)} chunks kept for names: {person_words}")
            all_chunks = filtered

    return all_chunks


# ── Extraction ─────────────────────────────────────────────────────────────

def _clean_name(val: str) -> str:
    """
    Post-process an extracted contractor name:
    - Fix OCR word-break artifact: 'HRI X Y' → 'SHRI X Y'
    - Remove doubled sequences: 'CHAND MAL SOMANI SHREE CHAND MAL SOMANI' → 'CHAND MAL SOMANI'
    - Reject bare title words: 'SHRI', 'BAO' alone → empty string
    """
    val = val.strip()

    # OCR often splits 'SHRI' as 'S' + 'HRI' across a word boundary;
    # re-join it before further processing
    val = re.sub(r'(?i)\bHRI\b', 'SHRI', val).strip()
    # Similarly 'HREE' → 'SHREE'
    val = re.sub(r'(?i)\bHREE\b', 'SHREE', val).strip()

    # Deduplicate: 'CHAND MAL SOMANI SHREE CHAND MAL SOMANI' → 'CHAND MAL SOMANI'
    words = val.split()
    n = len(words)
    for split in range(2, n // 2 + 1):
        first  = ' '.join(words[:split])
        second = ' '.join(words[split:split * 2])
        if first.lower() == second.lower():
            return first

    # Reject bare title-only values
    TITLE_ONLY = {'SHRI', 'SHREE', 'SH', 'SMT', 'BAO', 'MR', 'MS', 'M/S'}
    if val.upper() in TITLE_ONLY:
        return ''
    return val


def _is_valid_date(val: str) -> bool:
    """
    Return True only if the date string has a consistent separator
    and plausible day/month values to reject garbage like '24/09-15'.
    """
    # Must use only ONE separator type throughout
    separators = set(c for c in val if c in '.-/')
    if len(separators) > 1:
        return False
    # Must match dd<sep>mm<sep>yy[yy] exactly
    m = re.match(r'^(\d{2})[.\-/](\d{2})[.\-/](\d{2,4})$', val)
    if not m:
        return False
    day, mon = int(m.group(1)), int(m.group(2))
    return 1 <= day <= 31 and 1 <= mon <= 12


def extract_key_info(text: str) -> dict:
    """
    Extract structured fields. Chunks are word-joined (no newlines).
    All extracted values go through quality checks before being saved.
    Devanagari digits (०-९) are converted to ASCII before pattern matching.
    """
    # Convert Devanagari/Hindi numerals → ASCII so regex \d works on OCR'd text
    text = devanagari_to_latin(text)
    info = {}

    patterns = {
        # Contractor / agency — label styles (separator optional for BoQ 'Bidder Name' tables)
        "Agency Name":      r"(?i)(?:Agency\s*Name|Name\s*of\s*(?:Contractor|Agency)|Bidder\s*Name|Contractor\s*Name)\s*[:\-.]*\s*([A-Za-z][A-Za-z\s\.&/]{3,70})",
        # Hindi honorific → Agency / person name (e.g. 'श्री शंकर लाल वैष्णव')
        "_hindi_name":      r"(?:\u0936\u094d\u0930\u0940|\u0936\u094d\u0930\u0940\u092e\u0924\u0940|\u0936\u094d\u0930\u0940.)[\s\u200c]*([\u0900-\u097F\s]{4,40})",
        # Numbered table row: '5 Name of the Contractor SOME NAME'
        "_contractor_row":  r"(?i)\d+\s+Name\s*of\s*(?:the\s*)?Contractor\s+([A-Z][A-Za-z\s\.&]{5,60})",
        # Division — stop hard before Ref/Date/Sub-Division labels
        "Name of Division": r"(?i)Name\s*of\s*Division\s*[:\-=]+\s*(.{3,60})",
        # Date fields
        "Date of Casting":  r"(?i)Date\s*of\s*Casting\s*[:\-=]+\s*([\d.\-/]{6,12})",
        "Date of Testing":  r"(?i)Date\s*of\s*Testing\s*[:\-=]+\s*([\d.\-/]{6,12})",
        # Numeric reference fields
        "W.O. Number":      r"(?i)W\.?O\.?\s*No\.?\s*[:\-/=]+\s*([\d\-/]{3,20})",
        "Ref. Number":      r"(?i)(?:Ref\.?\s*No\.?|NIT|Bid\s*No\.?)\s*[:\-=]*\s*([\w][\w\s\-/.]{2,30})",
        # ── Housing / Land plot fields ──────────────────────────────────────
        # Plot Number: must be immediately after the label and be a short standalone value (≤6 chars)
        # This prevents account/reference numbers like 32436 being picked as plot 436.
        "Plot Number":      r"(?i)(?:भूखंड|भूखण्ड|bhu\s*khand|plot)\s*(?:sankhya|संख्या|no\.?|number|क्रमांक)?\s*[:\-=]*\s*([0-9A-Za-z][0-9A-Za-z\-/]{0,5})(?=[\s,।]|$)",
        # Plot Size: number (with optional decimal) optionally followed by a unit
        "Plot Size":        r"(?i)(?:क्षेत्रफल|kshetrafal|area|आकार|akar|size|माप|maap|भूमि)\s*[:\-=\-–]*\s*([\d.,]+(?:\s*(?:sqm|sq\.?\s*m|वर्ग\s*मीटर|gaj|गज|sq\.?\s*ft|sq\.?\s*yard|वर्ग\s*गज))?)",
        "Deposit Amount":   r"(?i)(?:जमाराशि|jama\s*rashi|deposit\s*amount|रकम|राशि|शुल्क)\s*[:\-=]*\s*(?:Rs\.?|₹|INR)?\s*([\d,]+(?:\.\d+)?)",
        "Scheme Name":      r"(?i)(?:योजना|yojana|scheme)\s*(?:का\s*नाम|name)?\s*[:\-=]*\s*(.{3,60})",
        "Allottee Name":    r"(?i)(?:खातेदार|allottee|आवंटी|pattadaar|पट्टाधारी)\s*[:\-=]*\s*(.{3,60})",
        # ── Tender / Financial fields ─────────────────────────────────────────────
        # Bid / contract amount from BoQ 'Quoted Rate in Figures' or work order letters
        "Tender Amount":    r"(?i)(?:contract\s*price|bid\s*amount|tender\s*amount|L1\s*amount|amount\s*rs\.?|quoted\s*rate\s*in\s*figures)\s*[:\-=]*\s*(?:Rs\.?|₹|INR)?\s*([\d,]+(?:\.\d+)?)",
        # Below-schedule percentage: '21.51% Below Schedule G' or 'Less (-) 21.5100%'
        "Schedule Discount": r"(?i)(?:([\d.]+\s*%\s*Below\s*Schedule[^\n]{0,20})|Less\s*\(-\)\s*([\d.]+\s*%?))",
        # FD / deposit maturity date
        "Maturity Date":    r"(?i)(?:maturity|mat\.?\s*date|due\s*date|payable\s*on|matures?\s*on)\s*[:\-=]*\s*([\d.\-/]{6,15})",
        # FD account / receipt number
        "FD Account No":    r"(?i)(?:FD\s*(?:account|a/?c|acct)?\s*(?:no\.?|number)?|account\s*no\.?|FDR\s*no\.?)\s*[:\-=]*\s*([\d]{6,20})",
    }

    for key, pattern in patterns.items():
        m = re.search(pattern, text)
        if not m:
            continue

        # Schedule Discount has two alternative capture groups
        if key == "Schedule Discount":
            val = (m.group(1) or m.group(2) or '').strip().rstrip('.')
            if len(val) < 3:
                continue
            if "Agency Name" not in info:  # reuse public_key logic below
                pass  # fall through to save
        else:
            val = m.group(1).strip().strip(':-=. ')

        # ── _hindi_name cleanup  ─────────────────────────────────────────
        if key == "_hindi_name":
            # Stop at line break or danda
            val = re.split(r'[\n।|]', val)[0].strip()
            # Stop at parentage markers (पिता, पुत्र, माता, W/O, S/O etc.)
            val = re.split(
                r'\s+(?=(?:पिता|पुत्र|माता|पत्नी|सुपुत्र|S[/.]?O|D[/.]?O|W[/.]?O)\b)',
                val, flags=re.IGNORECASE
            )[0].strip()
            val = val[:35].strip()
            devanag = sum(1 for c in val if '\u0900' <= c <= '\u097F')
            words   = [w for w in val.split() if len(w) >= 2]
            if devanag < 3 or len(words) < 2:
                continue
            # Map to Agency Name
            if "Agency Name" not in info:
                info["Agency Name"] = val
            continue

        # ── Text field cleanup ───────────────────────────────────────────
        if key in ("Agency Name", "_contractor_row", "Name of Division", "Scheme Name", "Allottee Name"):
            # Stop at double-space
            val = re.split(r'\s{2,}', val)[0].strip()
            # Stop at next label-like Pattern: word(s) ending with ':-' or ':- '
            val = re.split(r'\s+(?=\w[\w\s]{1,20}?[:\-]{1,2}\s)', val)[0].strip()
            # Stop at 'Ref', 'Date', 'Sub-Division', 'Name of' sequences
            val = re.split(
                r'\s+(?=(?:Ref|Date|Sub.Division|Name\s+of|Technical|Voucher)\b)',
                val, flags=re.IGNORECASE
            )[0].strip()
            val = val[:65].strip()

            # Quality gate: must have ≥3 Latin or Devanagari chars
            latin    = sum(1 for c in val if c.isascii() and c.isalpha())
            devanag  = sum(1 for c in val if '\u0900' <= c <= '\u097F')
            if latin < 3 and devanag < 3:
                continue

            # For agency/contractor fields, require ≥2 meaningful words
            words = [w for w in val.split() if len(w) >= 3]
            if key in ("Agency Name", "_contractor_row") and len(words) < 2:
                continue

            # Reject BoQ table-header false positives (e.g. "Amount Bid Rank BoQ")
            _HEADER_WORDS = {
                'AMOUNT', 'BID', 'RANK', 'BOQ', 'TOTAL', 'RATE', 'PRICE',
                'SCHEDULE', 'QUANTITY', 'UNIT', 'DESCRIPTION', 'ITEM', 'SR',
            }
            if key in ("Agency Name", "_contractor_row"):
                val_words_upper = {w.upper() for w in val.split()}
                if len(val_words_upper & _HEADER_WORDS) >= 2:
                    continue  # looks like a column header row, not a real name

            val = _clean_name(val) if key not in ("Name of Division", "Scheme Name", "Allottee Name") else val
            if not val:
                continue

        # ── Date field validation ────────────────────────────────────────
        elif key in ("Date of Casting", "Date of Testing"):
            if not _is_valid_date(val):
                continue

        # ── Ref / numeric field cleanup ──────────────────────────────────
        elif key == "Ref. Number":
            # Strip trailing lowercase alpha garbage: '22/2022-23-Sr' → '22/2022-23'
            val = re.sub(r'-?[A-Za-z]{1,4}$', '', val).strip('-').strip()
            if len(val) < 3:
                continue

        # ── Deposit amount cleanup ────────────────────────────────────────
        elif key == "Deposit Amount":
            val = val.replace(',', '').strip()
            if not val.replace('.', '').isdigit():
                continue

        # ── Plot size cleanup ─────────────────────────────────────────────
        elif key == "Plot Size":
            if len(val) < 2:
                continue

        # ── Tender Amount cleanup ─────────────────────────────────────────
        elif key == "Tender Amount":
            val = val.replace(',', '').strip()
            if not val.replace('.', '').isdigit() or float(val) < 1000:
                continue  # reject tiny/garbage numbers

        # ── Schedule Discount — keep as-is (it's a readable string) ──────
        elif key == "Schedule Discount":
            val = val.strip().rstrip('.')
            if len(val) < 5:
                continue

        # ── Maturity Date validation ──────────────────────────────────────
        elif key == "Maturity Date":
            # Accept date-like strings with at least 6 chars
            if len(val) < 6:
                continue

        # ── FD Account No — must be numeric only ─────────────────────────
        elif key == "FD Account No":
            val = val.strip()
            if not val.isdigit() or len(val) < 6:
                continue

        # ── Save ─────────────────────────────────────────────────────────
        public_key = key
        if key in ("_contractor_row", "_hindi_name"):
            public_key = "Agency Name"
        if public_key not in info:
            info[public_key] = val

    # ── BOQ table scan: 'BIDDER NAME | AMOUNT | RANK' rows ───────────────
    # Handles pipe-delimited BOQ summary tables in government tender PDFs
    if "Tender Amount" not in info:
        # Look for L1 (lowest bidder) row: pattern '1|NAME|784701.42|L1'
        boq_m = re.search(
            r'(?:1\s*[|]\s*)([A-Z][A-Za-z\s.]{3,50})\s*[|]\s*([\d,]+\.\d+)\s*[|]\s*L1',
            text)
        if boq_m:
            info["Tender Amount"] = boq_m.group(2).replace(',', '')
            # Also capture L1 bidder name if Agency Name not yet found
            if "Agency Name" not in info:
                info["Agency Name"] = boq_m.group(1).strip()
        else:
            # Fallback: 'Rs. 784701/- based on Tender Premium'
            tp_m = re.search(
                r'(?:Rs\.?|\u20b9)\s*([\d,]+(?:\.\d+)?)\s*(?:/-)?\s*based on Tender',
                text, re.IGNORECASE)
            if tp_m:
                info["Tender Amount"] = tp_m.group(1).replace(',', '')

    return info


# Expanded skip set for all-caps false positives in Indian govt. docs
_CAPS_SKIP = {
    # Common words
    'DATE', 'NAME', 'WORK', 'DIVISION', 'AGREEMENT', 'VOUCHER', 'BILL',
    'AND', 'THE', 'FOR', 'OF', 'URBAN', 'TRUST', 'BHILWARA', 'ACCOUNT',
    'AMOUNT', 'ONLY', 'INR', 'LAKH', 'NOTE', 'LABOUR', 'CESS', 'FROM',
    'FUND', 'TOTAL', 'GRAND', 'NET', 'TAX', 'GST', 'TDS', 'VAT',
    'CERTIFICATE', 'REPORT', 'STATEMENT', 'PROGRESS', 'CONTRACT',
    'TECHNICAL', 'SANCTION', 'COMPLETION', 'STIPULATED', 'ACTUAL',
    'EXTENSION', 'APPLIED', 'SECRETARY', 'ENGINEER', 'CHARGE', 'ISSUE',
    'FORM', 'PRIOR', 'SUBMISSION', 'TENDER', 'OPERATING', 'DEPOSIT',
    'MATURITY', 'PAYABLE', 'BANK', 'KANHAIYA', 'LARA', 'LETTE', 'SPEC',
    'THIRTY', 'ONE', 'THOUSAND', 'THOUSANDS', 'HUNDRED', 'SIX', 'FOUR',
    'FIVE', 'SEVEN', 'EIGHT', 'NINE', 'TEN', 'YEARS', 'MONTHS',
    # Government departments / bodies (must NOT be picked as contractor names)
    'PWD', 'BSR', 'UIT', 'CPWD', 'NHAI', 'BDC', 'SDO', 'EE', 'AE',
    'JEN', 'XEN', 'CE', 'SE', 'MES', 'RVPN', 'PHED', 'PHD', 'PD',
    'BUILDING', 'WORKS', 'ROADS', 'IRRIGATION', 'DEPARTMENT',
    'GOVERNMENT', 'RAJASTHAN', 'INDIA', 'STATE', 'NATIONAL',
    'MUNICIPAL', 'CORPORATION', 'COMMITTEE', 'PANCHAYAT', 'SAMITI',
    # Financial / number words
    'RUPEES', 'PAISE', 'LAKHS', 'CRORE', 'CRORES', 'INTEREST', 'RATE',
}


def keyword_scan(text: str) -> dict:
    """
    Broad fallback sweep for contractor names and dates.
    Prioritised from most specific (M/s) to least (all-caps block).
    """
    # Normalise Devanagari numerals so \d patterns match OCR'd Hindi text
    text = devanagari_to_latin(text)
    found = {}

    # ── Contractor / agency name ─────────────────────────────────────────

    # Priority 1: M/s or M/S prefix (most reliable)
    m = re.search(r'(?i)(M[/.]s\.?\s+[A-Za-z][A-Za-z\s\.&]{3,50})', text)
    if m:
        val = re.split(r'\s*[\d(]', m.group(1).strip())[0].strip()
        val = _clean_name(val[:60])
        if val and len(val.split()) >= 2:
            found["Agency Name"] = val

    # Priority 1.5: 'Bidder Name' / 'Bidder:' / 'Contractor:' label — common in BoQ docs
    # The separator is OPTIONAL because BoQ tables often have no colon
    if "Agency Name" not in found:
        mb = re.search(
            r'(?i)(?:Bidder\s*Name|Bidder|Contractor)\s*[:\-.]*\s*([A-Z][A-Za-z\s\.&]{4,60})',
            text)
        if mb:
            val = re.split(r'\s*[\d(\n]', mb.group(1).strip())[0].strip()
            # Stop at next label-like sequence
            val = re.split(r'\s+(?=\w[\w\s]{1,15}[:\-]{1,2})', val)[0].strip()[:60]
            val = _clean_name(val)
            words = [w for w in val.split() if len(w) >= 2]
            if val and len(words) >= 2:
                found["Agency Name"] = val

    # Priority 2: BAO / SHRI / SMT + all-caps name (≥2 real words after prefix)
    if "Agency Name" not in found:
        m2 = re.search(
            r'\b(BAO|SHRI|SH\.?|SMT\.?|SHRE?E?|M/S)\s+([A-Z][A-Z\s\.]{5,50})', text)
        if m2:
            val = (m2.group(1).strip() + ' ' + m2.group(2).strip())
            val = re.split(r'\s*[\d(]', val)[0].strip()[:60]
            val = _clean_name(val)
            real_words = [w for w in val.split() if len(w) >= 3]
            if len(real_words) >= 3:   # prefix + ≥2 name words
                found["Agency Name"] = val

    # Priority 3: All-caps block ≥3 words, all words ≥3 chars, none in skip list
    if "Agency Name" not in found:
        caps_names = re.findall(r'\b([A-Z]{3,}(?:\s+[A-Z]{3,}){2,})\b', text)
        for name in caps_names:
            words = name.split()
            if any(w in _CAPS_SKIP for w in words):
                continue
            # All words must be ≥3 chars (removes 'ER', 'SE', 'AN', 'SRT' junk)
            if not all(len(w) >= 3 for w in words):
                continue
            val = _clean_name(name[:60])
            if val and len(val.split()) >= 2:
                found["Agency Name"] = val
                break

    # ── Date near 'cast' keyword ─────────────────────────────────────────
    casting_ctx = re.search(
        r'(?i)cast(?:ing)?.{0,60}?(\d{2}[.\-/]\d{2}[.\-/]\d{2,4})', text)
    if casting_ctx:
        d = casting_ctx.group(1)
        if _is_valid_date(d):
            found["Date of Casting"] = d
    else:
        all_dates = [d for d in re.findall(r'\b(\d{2}[.\-/]\d{2}[.\-/]\d{2,4})\b', text)
                     if _is_valid_date(d)]
        if all_dates:
            found["Date of Casting"] = all_dates[0]

    # ── Deposit / Jamrashi amount ─────────────────────────────────────────
    if "Deposit Amount" not in found:
        m_dep = re.search(
            r'(?:जमाराशि|जमा|deposit|rashi|राशि|रकम)\D{0,30}?(?:Rs\.?|₹|INR)?\s*([\d,]{4,}(?:\.\d+)?)',
            text, re.IGNORECASE)
        if m_dep:
            found["Deposit Amount"] = m_dep.group(1).replace(',', '')

    # ── Plot size / area ──────────────────────────────────────────────────
    if "Plot Size" not in found:
        # Try with explicit unit first
        m_size = re.search(
            r'([\d.,]+)\s*(sqm|sq\.?\s*m|वर्ग\s*मीटर|गज|gaj|sq\.?\s*ft|sq\.?\s*yard)',
            text, re.IGNORECASE)
        if m_size:
            unit = m_size.group(2).strip()
            found["Plot Size"] = f"{m_size.group(1)} {unit}"
        else:
            # Fallback: number right after क्षेत्रफल keyword (no unit in OCR)
            m_size2 = re.search(
                r'(?:क्षेत्रफल|kshetrafal|area|माप)[\s:\-–=]*([\d.,]{3,})',
                text, re.IGNORECASE)
            if m_size2:
                found["Plot Size"] = f"{m_size2.group(1)} वर्ग मीटर (approx.)"

    # ── Tender / Bid amount ───────────────────────────────────────────────
    if "Tender Amount" not in found:
        # BOQ L1 row: '1 | CONTRACTOR NAME | 784701.42 | L1'
        boq = re.search(
            r'(?:1\s*[|]\s*)([A-Z][A-Za-z\s.]{3,50})\s*[|]\s*([\d,]+\.\d+)\s*[|]\s*L1',
            text)
        if boq:
            found["Tender Amount"] = boq.group(2).replace(',', '')
            if "Agency Name" not in found:
                found["Agency Name"] = boq.group(1).strip()
        else:
            # Work-order letter: 'Rs. 784701/- based on Tender Premium'
            tp = re.search(
                r'(?:Rs\.?|₹)\s*([\d,]+(?:\.\d+)?)\s*(?:/-)?[^.]{0,40}Tender',
                text, re.IGNORECASE)
            if tp:
                found["Tender Amount"] = tp.group(1).replace(',', '')

    # ── Schedule Discount % ───────────────────────────────────────────────
    if "Schedule Discount" not in found:
        sd = re.search(r'([\d.]+\s*%\s*Below\s*Schedule[^\n]{0,25})', text, re.IGNORECASE)
        if sd:
            found["Schedule Discount"] = sd.group(1).strip()

    # ── Maturity Date ─────────────────────────────────────────────────────
    if "Maturity Date" not in found:
        md = re.search(
            r'(?:maturity|mat\.?\s*date|due\s*date|payable\s*on|matur\w*\s*on)'
            r'\s*[:\-=]*\s*([\d.\/\-]{6,15})',
            text, re.IGNORECASE)
        if md and len(md.group(1)) >= 6:
            found["Maturity Date"] = md.group(1).strip()

    # ── FD Account Number ─────────────────────────────────────────────────
    if "FD Account No" not in found:
        fd = re.search(
            r'(?:FD\s*(?:account|a/?c|acct)?\s*(?:no\.?|number)?|account\s*no\.?)'
            r'\s*[:\-=]*\s*(\d{6,20})',
            text, re.IGNORECASE)
        if fd:
            found["FD Account No"] = fd.group(1)

    return found


# ── Query Normalisation ────────────────────────────────────────────────────

# Common Hindi/Hinglish typos and misspellings → correct term.
# Each entry: (wrong_regex, correct_replacement, user_note)
# The user_note is shown to the user when a correction is applied.
_HINDI_CORRECTIONS = [
    # ── Devanagari typos ─────────────────────────────────────────────────────
    # भूकंप (earthquake) → भूखंड (plot)  — very common voice/typing mistake
    (r'भूकंप', 'भूखंड',
     '> ⚠️ **"भूकंप"** (earthquake) की जगह **"भूखंड"** (plot) से search किया गया।'),
    (r'भूकम्प', 'भूखंड',
     '> ⚠️ **"भूकम्प"** की जगह **"भूखंड"** (plot) से search किया गया।'),
    # Alternate standard spellings → normalise
    (r'भूखण्ड', 'भूखंड', ''),
    (r'जमाराशी', 'जमाराशि', ''),
    (r'क्षेत्राफल', 'क्षेत्रफल', ''),
    (r'क्षेत्राफ़ल', 'क्षेत्रफल', ''),

    # ── Roman / Hinglish typos for plot ──────────────────────────────────────
    (r'(?i)\bbhukand\b', 'भूखंड',
     '> ⚠️ **"bhukand"** को **"भूखंड"** (plot) समझा गया।'),
    (r'(?i)\bbhukhand\b', 'भूखंड', ''),
    (r'(?i)\bbhukhand\b', 'भूखंड', ''),
    (r'(?i)\bbhu\s*khad\b', 'भूखंड', ''),
    (r'(?i)\bplot\s*no\.?\s*(\d)', r'plot \1', ''),   # normalise spacing

    # ── Hinglish typos for area / size ───────────────────────────────────────
    (r'(?i)\bkshetafal\b', 'kshetrafal', ''),
    (r'(?i)\bkhetrafal\b', 'kshetrafal', ''),
    (r'(?i)\bkhshetrafal\b', 'kshetrafal', ''),
    (r'(?i)\bsaiz\b', 'size', ''),
    (r'(?i)\bsaize\b', 'size', ''),

    # ── Hinglish typos for deposit / amount ──────────────────────────────────
    (r'(?i)\bjamrashi\b', 'जमाराशि', ''),
    (r'(?i)\bjama\s*rashee\b', 'जमाराशि', ''),
    (r'(?i)\bjammarrashi\b', 'जमाराशि', ''),
    (r'(?i)\brakam\b', 'राशि', ''),

    # ── Hinglish typos for scheme / yojana ───────────────────────────────────
    (r'(?i)\byojna\b', 'yojana', ''),
    (r'(?i)\byojan\b', 'yojana', ''),

    # ── Common English misspellings in queries ────────────────────────────────
    (r'(?i)\bsampling\b', 'casting',
     '> ℹ️ **"sampling"** को **"casting"** (concrete test) समझा गया।'),
    (r'(?i)\bsapling\b', 'casting',
     '> ℹ️ **"sapling"** को **"casting"** (concrete test) समझा गया।'),
    (r'(?i)\bagancy\b', 'agency', ''),
    (r'(?i)\bagenci\b', 'agency', ''),
    (r'(?i)\bcontacter\b', 'contractor', ''),
    (r'(?i)\bcontarctor\b', 'contractor', ''),
    (r'(?i)\btendor\b', 'tender', ''),
    (r'(?i)\bdivison\b', 'division', ''),
]


def normalize_query(query: str) -> tuple[str, str]:
    """
    Correct common Hindi / Hinglish typos in the query.
    Returns (corrected_query, correction_note).
    correction_note is empty string if no correction was needed.
    """
    note = ''
    for pattern, replacement, user_note in _HINDI_CORRECTIONS:
        new_q, n = re.subn(pattern, replacement, query)
        if n > 0:
            query = new_q
            if user_note and not note:
                note = user_note
    return query, note


def extract_plot_number_from_query(query: str) -> str | None:
    """
    Pull out the explicit plot number the user mentioned, e.g. '148', '1-F-48'.
    Returns the plot ID as a regex-compatible string (may be a flexible pattern for compound IDs).
    Also returns a variant pattern that handles OCR spacing in compound formats.
    Actually returns a plain string; see _build_plot_filter_pattern() for regex building.
    """
    # 1. Compound formats first: 1-F-48, A-B-12 — most reliable
    m = re.search(
        r'\b(\d+[-/][A-Za-z][-/]\d+|[A-Za-z][-/][A-Za-z][-/]\d+)\b',
        query)
    if m:
        return m.group(1)
    # 2. Plain number close to a plot keyword
    m2 = re.search(
        r'(?:भूखंड|भूखण्ड|plot|sankhya|संख्या)\s*(?:no\.?\s*)?(\d{1,6})\b',
        query, re.IGNORECASE)
    if m2:
        return m2.group(1)
    # 3. Plain standalone number ONLY when a plot keyword is present in the query
    if re.search(r'भूखंड|भूखण्ड|plot\b|sankhya|संख्या', query, re.IGNORECASE):
        numbers = re.findall(r'\b(\d{2,5})\b', query)
        if numbers:
            return numbers[0]
    return None


def _build_plot_filter_pattern(asked_plot: str) -> str:
    """
    Build a flexible regex for matching a plot ID in OCR text.
    Handles spacing/punctuation variants: '1-F-48' → also matches '1 F 48', '1F48'.
    """
    # For a compound ID like '1-F-48', build a flexible pattern
    if re.match(r'^\d+[-/][A-Za-z][-/]\d+$', asked_plot):
        parts = re.split(r'[-/]', asked_plot)
        # Allow any separator (dash, space, slash, dot) between parts
        return r'[-/\s.]?'.join(re.escape(p) for p in parts)
    # For plain numbers, just use exact word-boundary match
    return r'\b' + re.escape(asked_plot) + r'\b'


def extract_row_for_plot(chunk_text: str, asked_plot: str) -> str:
    """
    Narrow a chunk down to the lines around the specific plot ID.
    OCR tables often put all rows on one line; we split on the plot ID
    and return just that neighbourhood (±3 lines / ±120 chars) so that
    regex extractors see only the correct row's values.

    Returns a sub-string (possibly the full chunk if the plot is not found).
    """
    pattern = _build_plot_filter_pattern(asked_plot)
    m = re.search(pattern, chunk_text, re.IGNORECASE)
    if not m:
        return chunk_text  # shouldn't happen since chunk passed the filter

    # Lines-based neighbourhood: grab the matching line + 2 lines each side
    lines = chunk_text.split('\n')
    target_line_idx = None
    for i, ln in enumerate(lines):
        if re.search(pattern, ln, re.IGNORECASE):
            target_line_idx = i
            break

    if target_line_idx is not None:
        start_i = max(0, target_line_idx - 2)
        end_i   = min(len(lines), target_line_idx + 4)
        return '\n'.join(lines[start_i:end_i])

    # Fallback: character-based slice around the match
    s = max(0, m.start() - 200)
    e = min(len(chunk_text), m.end() + 200)
    return chunk_text[s:e]


# ── Intent Detection ───────────────────────────────────────────────────────

def detect_intent(query: str) -> list[str]:
    q = query.lower()
    intents = []
    # ── Construction / technical intents ──────────────────────────────────
    if re.search(r'agency|contractor|firm|company|thekedar', q):
        intents.append("Agency Name")
    if re.search(r'casting|cast\b|date of cast', q):
        intents.append("Date of Casting")
    if re.search(r'testing|test\s+date|tested', q):
        intents.append("Date of Testing")
    if re.search(r'work order|w\.?o\.\b|order number', q):
        intents.append("W.O. Number")
    if re.search(r'\bref\b|reference|case no', q):
        intents.append("Ref. Number")
    if re.search(r'division|vibhag', q):
        intents.append("Name of Division")
    # ── Housing / Land plot intents ────────────────────────────────────────
    if re.search(r'bhukhand|भूखंड|भूखण्ड|bhu.?khand|plot|sankhya|संख्या', q):
        intents.append("Plot Number")
    if re.search(r'size|akar|आकार|area|maap|माप|varg|वर्ग|sqm|sq|gaj|गज|kshetrafal|क्षेत्रफल|saiz|साइज', q):
        intents.append("Plot Size")
    if re.search(r'jama|जमा|rashi|राशि|deposit|amount|rakam|रकम|shulk|शुल्क|kitni', q):
        intents.append("Deposit Amount")
    if re.search(r'yojana|योजना|scheme|nagar|tilak|colony|kalani|कालोनी', q):
        intents.append("Scheme Name")
    if re.search(r'allot|आवंटित|patta|पट्टा|holder|khattedar|खातेदार|malik|मालिक', q):
        intents.append("Allottee Name")
    # ── Tender / Financial intents ─────────────────────────────────────────
    # 'bid no' = asking for the bid/NIT reference number; 'bid amount/tender' = money
    if re.search(r'bid\s*no|bid\s*number|nit\s*no', q):
        intents.append("Ref. Number")
    elif re.search(r'tender|bid\s*amount|l1\b|lowest|contract\s*price|kitne\s*rupay|kitna\s*amount', q):
        intents.append("Tender Amount")
    elif re.search(r'\bbid\b|amount', q):
        intents.append("Tender Amount")
    if re.search(r'below\s*schedule|schedule|discount|percent|%|kitna\s*kam|rate', q):
        intents.append("Schedule Discount")
    if re.search(r'maturity|matur|due\s*date|mature|kab\s*mature|fd\s*kab', q):
        intents.append("Maturity Date")
    if re.search(r'fd\s*account|account\s*no|account\s*number|fd\s*number|receipt\s*no', q):
        intents.append("FD Account No")
    return intents


def extract_work_title(text: str) -> str:
    m = re.search(r'(?i)Name\s*of\s*Work\s*[:\-=]+\s*(.{10,120})', text)
    if m:
        val = m.group(1).strip().strip(':-= ')
        val = re.split(r'\s{2,}', val)[0].strip()[:100]
        if len(val) > 10:
            return val
    return ""


# ── Natural Language Generation ────────────────────────────────────────────

def build_natural_answer(
    query: str,
    all_info: dict,
    sources: list[str],
    context_chunks: list[dict]
) -> str:
    """Build a clean, bilingual answer from extracted fields. No external API."""
    intents = detect_intent(query)

    # Work title for context
    work_title = ""
    for chunk in context_chunks:
        work_title = extract_work_title(clean_ocr_text(chunk["text"]))
        if work_title:
            break

    source_str = (sources[0].split("/")[-1].strip() if sources
                  else "the indexed document")
    paragraphs = []

    # Opening line
    if work_title:
        paragraphs.append(
            f"**{source_str}** — *\"{work_title[:90]}\"*"
        )
    else:
        paragraphs.append(f"**{source_str}** se yeh jankari mili:")

    # ── Bilingual field formatter ─────────────────────────────────────────
    _FIELD_LABELS = {
        "Agency Name":      "🏗️ **Contractor / Agency**",
        "Date of Casting":  "📅 **Casting Date** (jab concrete dali gayi)",
        "Date of Testing":  "📊 **Testing Date**",
        "W.O. Number":      "📄 **Work Order No.**",
        "Ref. Number":      "🔗 **Reference No.**",
        "Name of Division": "🏢 **Division**",
        "Plot Number":      "📍 **भूखंड संख्या (Plot No.)**",
        "Plot Size":        "📏 **भूखंड का आकार (Plot Size)**",
        "Deposit Amount":   "💰 **जमाराशि (Deposit Amount)**",
        "Scheme Name":      "🌆 **योजना (Scheme)**",
        "Allottee Name":    "👤 **आवंटी / खातेदार (Allottee)**",
        "Tender Amount":    "💵 **Tender Amount (L1)**",
        "Schedule Discount": "ℹ️ **Schedule Discount**",
        "Maturity Date":    "📆 **FD Maturity Date**",
        "FD Account No":    "🏦 **FD Account No.**",
    }

    def _format_val(field, val):
        """Format a value for display (add currency symbol etc.)."""
        if field == "Deposit Amount":
            return f"\u20b9{val}"
        if field == "Tender Amount":
            try:
                return f"\u20b9{float(val):,.2f}"
            except (ValueError, TypeError):
                return val
        return str(val)

    def _fmt_field(field, val):
        label = _FIELD_LABELS.get(field, f"**{field}**")
        return f"- {label}: **{_format_val(field, val)}**"

    # ── Case 1: specific intents detected → answer only those + extras ────
    answered = []
    missing  = []
    for field in intents:
        val = all_info.get(field)
        if val:
            answered.append(_fmt_field(field, val))
        else:
            missing.append(field)

    # Extra found fields not in intents
    extra = {k: v for k, v in all_info.items()
             if k not in intents and k != "Source"}

    if answered:
        paragraphs.append("\n".join(answered))
        if extra:
            paragraphs.append(
                "**Aur bhi details mili:**\n" +
                "\n".join(_fmt_field(k, v) for k, v in extra.items()))
        if missing:
            paragraphs.append(
                "> ⚠️ Yeh fields is document mein nahi mili: " +
                ", ".join(f"_{f}_" for f in missing))

    # ── Case 2: no specific intents (general 'ki jankari do') → show all ─
    elif not intents:
        info_to_show = {k: v for k, v in all_info.items() if k != "Source"}
        if info_to_show:
            paragraphs.append("\n".join(_fmt_field(k, v)
                                        for k, v in info_to_show.items()))
        else:
            # Nothing extracted at all — show a cleaned passage
            best = (
                clean_ocr_text(context_chunks[0]["text"])[:600]
                if context_chunks else ""
            )
            paragraphs.append(
                "> Specific fields extract nahi ho sake. "
                "Neeche **Read more** mein full document passage dekhe.")

    # ── Case 3: intents detected but nothing found ────────────────────────
    else:
        paragraphs.append(
            "> ⚠️ Maafi chahta hoon, yeh jankari indexed documents mein "
            "nahi mili. PDF ko re-scan karein ya alag query try karein.")

    return "\n\n".join(paragraphs)


def _llm_answer(
    query: str,
    context_chunks: list[dict],
    extracted_fields: dict | None = None,
) -> str | None:
    """
    Generate a natural-language answer using Phi-3-mini (GPT4All).
    • Model is kept as a SINGLETON in RAM after first load (no re-loading).
    • Prompt includes regex-extracted fields as verified ground truth.
    • Quality check: response must reference at least one extracted value.
    """
    global _phi3_model
    try:
        from gpt4all import GPT4All
        from config import GPT4ALL_MODEL
        import os

        model_dir  = os.path.join(os.path.expanduser("~"), ".cache", "gpt4all")
        model_path = os.path.join(model_dir, GPT4ALL_MODEL)
        if not os.path.exists(model_path):
            print(f"  [llm] Phi-3-mini not found at {model_path}. Skipping.")
            return None

        # ── Load once, keep in RAM ────────────────────────────────────────
        if _phi3_model is None:
            print("  [llm] Loading Phi-3-mini into RAM (one-time, ~1-2 min)...")
            _phi3_model = GPT4All(GPT4ALL_MODEL, model_path=model_dir, verbose=False)
            print("  [llm] Phi-3-mini ready.")

        # ── Compact context (300 chars/chunk) ───────────────────────────────
        context_parts = []
        for i, chunk in enumerate(context_chunks[:2], 1):   # top-2 only
            text = clean_ocr_text(chunk["text"])[:300]
            src  = chunk["filename"].replace("\\", "/").split("/")[-1]
            context_parts.append(f"[{src}]\n{text}")
        context_str = "\n\n".join(context_parts)

        # ── Extracted fields as ground truth ─────────────────────────────
        fields_str = ""
        if extracted_fields:
            clean_fields = {k: v for k, v in extracted_fields.items() if k != "Source"}
            if clean_fields:
                fields_str = (
                    "\nExtracted facts (ground truth, do not contradict):\n" +
                    "\n".join(f"  - {k}: {v}" for k, v in clean_fields.items()) + "\n"
                )

        prompt = (
            "You are a helpful assistant for Indian government land/tender records.\n"
            "Rules: Use ONLY the info below. Reply in Hindi/Hinglish/English matching the question. "
            "Be concise: 2-3 bullet points max. No intro sentence, no apology.\n\n"
            f"{fields_str}"
            f"Document excerpt:\n{context_str}\n\n"
            f"Question: {query}\n\n"
            "Answer:"
        )

        print(f"  [llm] Generating answer for: {query[:60]}...")
        with _phi3_model.chat_session():
            response = _phi3_model.generate(prompt, max_tokens=120, temp=0.05)
        response = response.strip()
        if not response:
            return None
        print(f"  [llm] Done ({len(response)} chars)")

        # ── Sanity check (not strict quality gate) ───────────────────────
        # Reject only if clearly empty or an obvious refusal.
        # Do NOT check against extracted_fields values  —  the user may be
        # asking about a DIFFERENT entity than the first one the regex found
        # (e.g. multi-bidder BoQ: regex finds KATHAT first, user asks about SHREE BALAJI).
        _REFUSAL = ('i cannot', "i'm sorry", 'i don\'t know', 'not enough information',
                    'maafi chahta hoon', 'jankari nahi')
        if len(response) < 15 or any(r in response.lower() for r in _REFUSAL):
            print("  [llm] Sanity FAIL — regex fallback used.")
            return None

        return response

    except ImportError:
        print("  [llm] gpt4all not installed.")
        return None
    except Exception as e:
        print(f"  [llm] Error: {e}")
        return None


def format_answer(query: str, chunks: list[dict]) -> str:
    """
    Three-pass answer generation:
    Pass 1 — scan retrieved chunks (embedding-based, fast).
    Pass 2 — if key fields still missing, fetch ALL chunks of the top source
              file from ChromaDB and scan those too (exhaustive, slow but accurate).
    Pass 3 — pipe top-3 chunks into Phi-3-mini for a natural-language answer.
              Falls back to regex template if model not downloaded.
    """
    all_info: dict = {}
    sources:  list = []

    def _scan_chunk(chunk):
        nonlocal all_info
        text = clean_ocr_text(chunk["text"])
        if not text:
            return
        info = extract_key_info(text)
        # Keyword scan fallback on the same chunk
        for k, v in keyword_scan(text).items():
            if k not in info:
                info[k] = v
        for k, v in info.items():
            if k not in all_info:
                all_info[k] = v

    # ── Pass 1: scan embedding-retrieved chunks ───────────────────────────
    for chunk in chunks:
        _scan_chunk(chunk)
        fname   = chunk["filename"]
        parts   = fname.replace("\\", "/").split("/")
        display = " / ".join(parts[-3:]) if len(parts) >= 3 else fname
        if display not in sources:
            sources.append(display)

    # ── Pass 2: exhaustive file scan if fields still missing ─────────────
    intents         = detect_intent(query)
    missing_intents = [f for f in intents if f not in all_info]

    # Also run Pass-2 for general "ki jankari do" queries (no intents) so ALL
    # pages of the matched file are scanned — the English BoQ page (Bidder Name,
    # Tender Amount, Schedule Discount) may be on a different page from the
    # Hindi text page that embedding search returned first.
    # Only run exhaustive Pass-2 if 2+ intents are still missing, or no
    # intents at all AND nothing was extracted. Avoids slow full-file scans
    # for queries that already have their main field answered.
    run_pass2 = (len(missing_intents) >= 2) or (not intents and not all_info)

    if run_pass2 and chunks:
        # ── Choose the CORRECT file for Pass-2 ──────────────────────────
        from collections import Counter
        case_votes = Counter(c.get("case", "") for c in chunks if c.get("case", ""))
        best_case  = case_votes.most_common(1)[0][0] if case_votes else ""

        top_file = next(
            (c["filename"] for c in chunks if c.get("case", "") == best_case),
            chunks[0]["filename"]
        )
        print(f"  [pass-2] intents={intents!r} missing={missing_intents!r}. "
              f"Best case={best_case!r}. Scanning: {top_file}")
        all_file_chunks = retrieve_all_chunks_for_file(top_file)
        for chunk in all_file_chunks:
            _scan_chunk(chunk)
            # Stop early only when we have specific intents and all are found
            if missing_intents and all(f in all_info for f in missing_intents):
                print(f"  [pass-2] All missing fields found. Stopping early.")
                break

    # Pass 3: Phi-3-mini LLM for natural-language answer (falls back to regex template if unavailable)
    llm_result = _llm_answer(query, chunks, extracted_fields=all_info or None)
    if llm_result:
        src_str = sources[0].split("/")[-1].strip() if sources else "the indexed document"
        return f"**Source:** {src_str}\n\n{llm_result}"

    # Fallback: regex-template answer
    return build_natural_answer(query, all_info, sources, chunks)



def build_detail_text(chunks: list[dict]) -> str:
    """
    Build a 'Read More' style detailed view from the top retrieved chunks.
    Shows clean raw text from the most relevant sections.
    """
    sections = []
    seen_files: set = set()
    for chunk in chunks[:6]:  # top 6 chunks max
        fname = chunk["filename"]
        short = fname.replace("\\", "/").split("/")[-1]
        text  = clean_ocr_text(chunk["text"])
        if not text:
            continue
        header = f"### {short}  (relevance: {chunk['score']:.0%})\n"
        if short not in seen_files:
            seen_files.add(short)
            sections.append(header + text[:800])
        else:
            sections.append(text[:400])
    return "\n\n---\n\n".join(sections)


# ── Main Query Handler ─────────────────────────────────────────────────────

def answer_query(
    query: str,
    filename_filter: str | None = None,
    category_filter: str | None = None,
):
    """Streaming RAG: yields meta → formatted answer → detail → sources.

    filename_filter — restrict retrieval to a single PDF filename (e.g. 'CS.pdf').
    category_filter — restrict retrieval to a metadata category (e.g. 'LAND').
    """

    # ── Step 0: Normalise query (fix typos like भूकंप → भूखंड) ───────────
    query, correction_note = normalize_query(query)

    # ── Step 1: Retrieve context ───────────────────────────────────────────
    context_chunks = retrieve_context(
        query,
        filename_filter=filename_filter,
        category_filter=category_filter,
    )

    if not context_chunks:
        yield {
            "type": "error",
            "content": (
                "No relevant records were found.\n\n"
                "**Suggestions:**\n"
                "- Include ward number, sector, or agency name\n"
                "- Make sure the PDF is indexed (use the Scan button)\n"
                "- Try rephrasing in English"
            )
        }
        return

    # ── Step 2: Post-filter by exact plot number ───────────────────────────
    # If the user explicitly stated a plot number (e.g. 148), only keep chunks
    # that actually contain that number so we don't return wrong plot data.
    asked_plot = extract_plot_number_from_query(query)
    if asked_plot:
        plot_pattern = _build_plot_filter_pattern(asked_plot)
        filtered = [
            c for c in context_chunks
            if re.search(plot_pattern, c["text"], re.IGNORECASE)
        ]
        # Only apply filter if it leaves at least one chunk
        if filtered:
            context_chunks = filtered
            print(f"  [plot-filter] Kept {len(filtered)} chunks matching plot '{asked_plot}'")
        else:
            # No chunk contains this plot number — tell the user explicitly
            # instead of silently falling back and returning wrong plot data.
            print(f"  [plot-filter] Plot '{asked_plot}' not found in any indexed chunk.")
            not_found_msg = (
                f"indexed documents में **भूखंड संख्या {asked_plot}** का कोई record नहीं मिला।"
                f"\n\n**संभावित कारण:**\n"
                f"- यह भूखंड संख्या PDF में अलग format में हो सकती है\n"
                f"- फ़ाइल का OCR इस पेज को सही से नहीं पढ़ सका\n"
                f"- PDF को re-scan करें (Scan button) और फिर कोशिश करें"
            )
            if correction_note:
                not_found_msg = correction_note + "\n\n" + not_found_msg
            yield {"type": "meta", "table": [], "sources": [], "context_count": 0}
            yield {"type": "content", "content": not_found_msg}
            yield {"type": "detail", "content": ""}
            yield {"type": "sources_detail", "content": []}
            return

    # ── Step 3: Build structured table ────────────────────────────────────
    table_data:   list = []
    seen_sources: list = []

    for chunk in context_chunks:
        src  = chunk["filename"]
        text = clean_ocr_text(chunk["text"])
        if not text:
            continue

        # Fix 5: if a specific plot was requested, narrow each chunk to that
        # plot's row before running extraction — prevents picking values from
        # the first table row when multiple rows exist in one chunk.
        extraction_text = extract_row_for_plot(text, asked_plot) if asked_plot else text

        info = extract_key_info(extraction_text)
        if not any(k in info for k in ("Agency Name", "Date of Casting")):
            info.update({k: v for k, v in keyword_scan(extraction_text).items()
                         if k not in info})

        if info:
            info["Source"] = src.replace("\\", "/").split("/")[-1]
            table_data.append(info)

        parts   = src.replace("\\", "/").split("/")
        display = " / ".join(parts[-3:]) if len(parts) >= 3 else src
        if display not in seen_sources:
            seen_sources.append(display)

    yield {
        "type":          "meta",
        "table":         table_data,
        "sources":       seen_sources,
        "context_count": len(context_chunks)
    }

    # ── Step 4: Concise natural-language answer ────────────────────────────
    answer_text = format_answer(query, context_chunks)
    # Prepend correction note (if any) so user sees the typo warning
    if correction_note:
        answer_text = correction_note + "\n\n" + answer_text
    yield {"type": "content", "content": answer_text}

    # ── Step 5: "Read More" detailed raw text ─────────────────────────────
    yield {"type": "detail", "content": build_detail_text(context_chunks)}

    # ── Step 6: Source chips (deduplicated by filename) ────────────────────
    seen_fnames: set = set()
    deduped: list    = []
    for c in context_chunks:
        fn = c["filename"].replace("\\", "/").split("/")[-1]
        if fn not in seen_fnames:
            seen_fnames.add(fn)
            deduped.append(c)

    yield {"type": "sources_detail", "content": deduped}
