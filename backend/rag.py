import os
import re
import uuid
from typing import Optional

import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder

from config import (
    CHROMA_DIR, EMBED_MODEL,
    TOP_K, COLLECTION_NAME, RERANK_TOP_N, RERANK_MODEL,
    LLM_ENABLED, RERANK_ENABLED,
)

# singletons — loaded once, reused for every query
_embedder: Optional[SentenceTransformer] = None
_chroma_client = None
_collection = None
_phi3_model = None
_cross_encoder = None
_bm25_index = None
_bm25_corpus = None


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
        try:
            _collection = _chroma_client.get_or_create_collection(
                name=COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"}
            )
        except Exception as e:
            if "dimension" in str(e).lower() or "InvalidDimension" in str(e):
                print(f"  [db] Dimension mismatch detected ({e}). Rebuilding collection...")
                try: 
                    _chroma_client.delete_collection(COLLECTION_NAME)
                except Exception:
                    pass
                _collection = _chroma_client.create_collection(
                    name=COLLECTION_NAME,
                    metadata={"hnsw:space": "cosine"}
                )
            else:
                raise
    return _collection


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
    print(f"  Indexed {len(texts)} chunks into Chroma.")
    _invalidate_bm25()

    # --- Also extract structured metadata into SQLite ---
    _index_metadata_sqlite(chunks)

    return len(texts)


def _index_metadata_sqlite(chunks: list[dict]) -> None:
    """Aggregate fields per PDF and upsert into SQLite metadata store."""
    from database import upsert_doc_metadata

    # Group chunks by filename
    by_file: dict[str, list[dict]] = {}
    for c in chunks:
        fn = c.get("filename", "unknown")
        by_file.setdefault(fn, []).append(c)

    for filename, file_chunks in by_file.items():
        all_fields: dict = {}
        total_chars = 0
        readable_chars = 0

        for c in file_chunks:
            raw = c["text"]
            cleaned, quality = clean_ocr_text_scored(raw)
            total_chars    += max(len(raw), 1)
            readable_chars += int(quality * len(raw))

            info = extract_key_info(cleaned)
            for k, v in info.items():
                if k not in all_fields:
                    all_fields[k] = v
            for k, v in keyword_scan(cleaned).items():
                if k not in all_fields:
                    all_fields[k] = v

        all_fields["category"]    = file_chunks[0].get("category", "")
        all_fields["case_number"] = file_chunks[0].get("case_number", "")
        ocr_q = readable_chars / max(total_chars, 1)

        if ocr_q < 0.5:
            print(f"  [metadata_db] LOW OCR quality ({ocr_q:.2f}) for {filename.split('/')[-1]}")

        upsert_doc_metadata(filename, all_fields, ocr_quality=ocr_q)

        from table_extractor import extract_bidders_from_pdf
        from database import upsert_tender_bidders
        from config import PDF_ROOT

        # filename is stored as a relative path (e.g. 'HOUSING/D-A-25/CS/CS.pdf')
        # pdfplumber needs the full absolute path
        abs_pdf_path = os.path.join(PDF_ROOT, filename)
        if not os.path.exists(abs_pdf_path):
            abs_pdf_path = filename  # fallback for uploaded PDFs stored at absolute paths

        bidders = extract_bidders_from_pdf(abs_pdf_path)
        upsert_tender_bidders(filename, bidders)

    print(f"  Structured metadata upserted for {len(by_file)} file(s).")


def clear_collection():
    global _collection, _chroma_client
    get_collection()
    _chroma_client.delete_collection(COLLECTION_NAME)
    _collection = _chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )
    _invalidate_bm25()
    return True


def get_indexed_docs() -> list[str]:
    collection = get_collection()
    filenames = set()
    offset = 0
    limit = 500
    while True:
        results = collection.get(include=["metadatas"], limit=limit, offset=offset)
        metas = results.get("metadatas")
        if not metas:
            break
        for m in metas:
            if m and "filename" in m:
                filenames.add(m["filename"])
        offset += limit
    
    return sorted(list(filenames))


def get_chunk_count() -> int:
    return get_collection().count()


RELEVANCE_THRESHOLD = 0.58

_DEVA_DIGIT = str.maketrans('०१२३४५६७८९', '0123456789')


def devanagari_to_latin(text: str) -> str:
    return text.translate(_DEVA_DIGIT)


def clean_ocr_text(text: str) -> str:
    cleaned, _ = clean_ocr_text_scored(text)
    return cleaned


def clean_ocr_text_scored(text: str) -> tuple[str, float]:
    """Returns (cleaned_text, ocr_quality_score 0-1)."""
    text = re.sub(r'\[Category:[^\]]+\]\s*', '', text)
    text = re.sub(r'\[Case:[^\]]+\]\s*',     '', text)
    text = re.sub(r'\[Type:[^\]]+\]\s*',     '', text)
    text = re.sub(r'\[Page \d+\]\n?',        '', text)
    lines = text.split('\n')
    clean = []
    total = 0
    readable_total = 0
    for line in lines:
        line = line.strip()
        if not line or len(line) < 4:
            continue
        readable = sum(1 for c in line if c.isalnum() or c in ' .,;:-/()[]')
        total    += len(line)
        readable_total += readable
        if readable / max(len(line), 1) < 0.40:
            continue
        clean.append(line)
    quality = readable_total / max(total, 1)
    return '\n'.join(clean).strip(), quality


def _build_where_clause(
    filename_filter: str | None,
    category_filter: str | None,
) -> dict | None:
    if filename_filter and category_filter:
        return {"$and": [
            {"filename": {"$eq": filename_filter}},
            {"category": {"$eq": category_filter}},
        ]}
    if filename_filter:
        return {"filename": {"$eq": filename_filter}}
    if category_filter:
        return {"category": {"$eq": category_filter}}
    return None


def _query_chroma_emb(
    emb: list,
    n: int,
    filename_filter: str | None = None,
    category_filter: str | None = None,
) -> list[dict]:
    """Query Chroma with a pre-computed embedding vector."""
    collection = get_collection()
    if collection.count() == 0:
        return []
    where_clause = _build_where_clause(filename_filter, category_filter)
    query_kwargs = dict(
        query_embeddings=[emb],
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


def _query_chroma(
    query_text: str,
    n: int,
    filename_filter: str | None = None,
    category_filter: str | None = None,
) -> list[dict]:
    """Encode a single query and search Chroma (legacy helper)."""
    emb = get_embedder().encode([query_text])[0].tolist()
    return _query_chroma_emb(emb, n, filename_filter, category_filter)


def retrieve_all_chunks_for_file(filename: str) -> list[dict]:
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
    q_words = [w for w in re.findall(r'[\u0900-\u097F\w]{3,}', query) if len(w) >= 3]
    if not q_words:
        return 0.0
    text_lower = chunk_text.lower()
    matched = sum(1 for w in q_words if w.lower() in text_lower)
    return min(0.15, matched * 0.03)


def _plot_id_boost(chunk_text: str, query: str) -> float:
    plot_ids = re.findall(
        r'\b(?:\d+[-/][A-Za-z][-/]\d+|[A-Za-z][-/][A-Za-z][-/]\d+|\d+[-/][A-Za-z]{1,3}[-/]\d+|\d{2,5})\b',
        query
    )
    if not plot_ids:
        return 0.0
    text_lower = chunk_text.lower()
    for pid in plot_ids:
        if pid.lower() in text_lower:
            return 0.30
    return 0.0


def _get_cross_encoder() -> CrossEncoder:
    global _cross_encoder
    if _cross_encoder is None:
        print("  Loading cross-encoder re-ranker...")
        _cross_encoder = CrossEncoder(RERANK_MODEL)
        print("  Cross-encoder ready.")
    return _cross_encoder


def _rerank_chunks(query: str, chunks: list[dict], top_n: int) -> list[dict]:
    if not chunks:
        return chunks
    if not RERANK_ENABLED:
        return chunks[:top_n]
    try:
        encoder = _get_cross_encoder()
        pairs  = [(query, c["text"][:512]) for c in chunks]
        scores = encoder.predict(pairs)
        ranked = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)
        
        CONFIDENCE_THRESHOLD = 0.05
        filtered = [(s, c) for s, c in ranked if s > CONFIDENCE_THRESHOLD]
        result = [c for _, c in filtered[:top_n]]
        
        if not result and ranked:
            print(f"  [rerank] ALL {len(chunks)} chunks blocked! Best score: {ranked[0][0]:.3f} < {CONFIDENCE_THRESHOLD}")
        else:
            print(f"  [rerank] {len(chunks)} -> {len(result)} chunks (Top score: {result[0]['score']:.3f} if exist)")
            
        return result
    except Exception as e:
        print(f"  [rerank] failed: {e}, using original order")
        return chunks[:top_n]


def _build_bm25_index():
    global _bm25_index, _bm25_corpus
    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        print("  rank-bm25 not installed — pip install rank-bm25")
        return

    collection = get_collection()
    if collection.count() == 0:
        return

    print("  Building BM25 index...")
    docs = []
    metas = []
    offset = 0
    limit = 500
    
    while True:
        results = collection.get(include=["documents", "metadatas"], limit=limit, offset=offset)
        batch_docs = results.get("documents", [])
        if not batch_docs:
            break
        docs.extend(batch_docs)
        metas.extend(results.get("metadatas", []))
        offset += limit

    corpus = []
    for doc, meta in zip(docs, metas):
        corpus.append({
            "text":     doc,
            "filename": meta.get("filename", "unknown"),
            "category": meta.get("category", ""),
            "case":     meta.get("case_number", ""),
            "score":    0.0,
        })

    tokenized = [c["text"].lower().split() for c in corpus]
    _bm25_index = BM25Okapi(tokenized)
    _bm25_corpus = corpus
    print(f"  BM25 index built: {len(corpus)} chunks.")


def _invalidate_bm25():
    global _bm25_index, _bm25_corpus
    _bm25_index = None
    _bm25_corpus = None


def _extract_person_names(query: str) -> list[str]:
    skip = {'agency', 'name', 'contractor', 'bidder', 'kya', 'hai', 'tender',
            'amount', 'number', 'date', 'casting', 'plot', 'size', 'jankari',
            'work', 'order', 'bid', 'ref', 'division', 'scheme', 'yojana'}
    words = re.findall(r'[A-Za-z\u0900-\u097F]{4,}', query)
    return [w.lower() for w in words if w.lower() not in skip]


def retrieve_context(
    query: str,
    filename_filter: str | None = None,
    category_filter: str | None = None,
) -> list[dict]:
    """Hybrid BM25 + vector search fused via Reciprocal Rank Fusion."""
    intents = detect_intent(query)
    queries = [query] + [f"{intent} {query}" for intent in intents]

    # --- batch encode ALL query variants in ONE call ---
    embedder = get_embedder()
    all_embeddings = embedder.encode(queries, batch_size=len(queries), show_progress_bar=False)

    # vector retrieval — one Chroma call per variant, no extra encode() calls
    seen_vector: set = set()
    vector_chunks: list[dict] = []
    for emb in all_embeddings:
        for chunk in _query_chroma_emb(emb.tolist(), TOP_K, filename_filter=filename_filter, category_filter=category_filter):
            key = (chunk["filename"], chunk["text"][:60])
            if key not in seen_vector:
                seen_vector.add(key)
                boost  = _work_name_boost(chunk["text"], query)
                boost += _plot_id_boost(chunk["text"], query)
                chunk["score"] = min(1.0, chunk["score"] + boost)
                vector_chunks.append(chunk)
    vector_chunks.sort(key=lambda c: c["score"], reverse=True)

    # BM25 retrieval with synonym expansion
    bm25_chunks: list[dict] = []
    if not filename_filter:
        if _bm25_index is None:
            _build_bm25_index()
        if _bm25_index is not None and _bm25_corpus is not None:
            tokens   = _expand_bm25_tokens(query)
            scores   = _bm25_index.get_scores(tokens)
            top_idxs = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:TOP_K]
            for idx in top_idxs:
                if scores[idx] > 0:
                    c = dict(_bm25_corpus[idx])
                    c["bm25_score"] = float(scores[idx])
                    bm25_chunks.append(c)

    # Dynamic Weigting based on Intent
    is_numeric_intent = any(i in [
        "W.O. Number", "Ref. Number", "Plot Number", "Deposit Amount", 
        "Tender Amount", "FD Account No", "Schedule Discount", 
        "Date of Casting", "Date of Testing", "Maturity Date"
    ] for i in intents)
    
    # Keyword-heavy queries need more BM25 dominance. Semantic needs more Vector.
    vector_weight = 0.3 if is_numeric_intent else 0.7
    bm25_weight   = 0.7 if is_numeric_intent else 0.3

    # Reciprocal Rank Fusion (k=60)
    rrf_scores: dict[tuple, float] = {}
    chunk_map:  dict[tuple, dict]  = {}
    K = 60

    for rank, c in enumerate(vector_chunks):
        key = (c["filename"], c["text"][:80])
        score = 1.0 / (K + rank + 1)
        rrf_scores[key] = rrf_scores.get(key, 0.0) + (score * vector_weight)
        chunk_map[key] = c

    for rank, c in enumerate(bm25_chunks):
        key = (c["filename"], c["text"][:80])
        score = 1.0 / (K + rank + 1)
        rrf_scores[key] = rrf_scores.get(key, 0.0) + (score * bm25_weight)
        if key not in chunk_map:
            chunk_map[key] = c

    fused = sorted(
        chunk_map.values(),
        key=lambda c: rrf_scores.get((c["filename"], c["text"][:80]), 0),
        reverse=True
    )[:TOP_K]

    # person-name filter — avoid pulling unrelated records when a name is in the query
    person_words = _extract_person_names(query)
    if person_words and not filename_filter:
        filtered = [c for c in fused if any(pw in c["text"].lower() for pw in person_words)]
        if filtered:
            print(f"  [name-filter] kept {len(filtered)}/{len(fused)} chunks")
            fused = filtered

    return fused


# ---------------------------------------------------------------------------
# Extraction helpers (used by index_chunks, format_answer, and answer_query)
# ---------------------------------------------------------------------------

def _clean_name(val: str) -> str:
    val = val.strip()
    val = re.sub(r'(?i)\bHRI\b', 'SHRI', val).strip()
    val = re.sub(r'(?i)\bHREE\b', 'SHREE', val).strip()
    words = val.split()
    n = len(words)
    for split in range(2, n // 2 + 1):
        first  = ' '.join(words[:split])
        second = ' '.join(words[split:split * 2])
        if first.lower() == second.lower():
            return first
    TITLE_ONLY = {'SHRI', 'SHREE', 'SH', 'SMT', 'BAO', 'MR', 'MS', 'M/S'}
    if val.upper() in TITLE_ONLY:
        return ''
    return val


def _is_valid_date(val: str) -> bool:
    separators = set(c for c in val if c in '.-/')
    if len(separators) > 1:
        return False
    m = re.match(r'^(\d{2})[.\-/](\d{2})[.\-/](\d{2,4})$', val)
    if not m:
        return False
    day, mon = int(m.group(1)), int(m.group(2))
    return 1 <= day <= 31 and 1 <= mon <= 12


def extract_key_info(text: str) -> dict:
    text = devanagari_to_latin(text)
    info = {}

    patterns = {
        "Agency Name":       r"(?i)(?:Agency\s*Name|Name\s*of\s*(?:Contractor|Agency)|Bidder\s*Name|Contractor\s*Name)\s*[:\-.]*\s*([A-Za-z][A-Za-z\s\.&/]{3,70})",
        "_hindi_name":       r"(?:\u0936\u094d\u0930\u0940|\u0936\u094d\u0930\u0940\u092e\u0924\u0940|\u0936\u094d\u0930\u0940.)[\s\u200c]*([\u0900-\u097F\s]{4,40})",
        "_contractor_row":   r"(?i)\d+\s+Name\s*of\s*(?:the\s*)?Contractor\s+([A-Z][A-Za-z\s\.&]{5,60})",
        "Name of Division":  r"(?i)Name\s*of\s*Division\s*[:\-=]+\s*(.{3,60})",
        "Date of Casting":   r"(?i)Date\s*of\s*Casting\s*[:\-=]+\s*([\d.\-/]{6,12})",
        "Date of Testing":   r"(?i)Date\s*of\s*Testing\s*[:\-=]+\s*([\d.\-/]{6,12})",
        "W.O. Number":       r"(?i)W\.?O\.?\s*No\.?\s*[:\-/=]+\s*([\d\-/]{3,20})",
        "Ref. Number":       r"(?i)(?:Ref\.?\s*No\.?|NIT|Bid\s*No\.?)\s*[:\-=]*\s*([\w][\w\s\-/.]{2,30})",
        "Plot Number":       r"(?i)(?:भूखंड|भूखण्ड|bhu\s*khand|plot)\s*(?:sankhya|संख्या|no\.?|number|क्रमांक)?\s*[:\-=]*\s*([0-9A-Za-z][0-9A-Za-z\-/]{0,5})(?=[\s,।]|$)",
        "Plot Size":         r"(?i)(?:क्षेत्रफल|kshetrafal|area|आकार|akar|size|माप|maap|भूमि)\s*[:\-=\-–]*\s*([\d.,]+(?:\s*(?:sqm|sq\.?\s*m|वर्ग\s*मीटर|gaj|गज|sq\.?\s*ft|sq\.?\s*yard|वर्ग\s*गज))?)",
        "Deposit Amount":    r"(?i)(?:जमाराशि|jama\s*rashi|deposit\s*amount|रकम|राशि|शुल्क)\s*[:\-=]*\s*(?:Rs\.?|₹|INR)?\s*([\d,]+(?:\.\d+)?)",
        "Scheme Name":       r"(?i)(?:योजना|yojana|scheme)\s*(?:का\s*नाम|name)?\s*[:\-=]*\s*(.{3,60})",
        "Allottee Name":     r"(?i)(?:पट्टेदार|allottee|आवंती|pattadaar|पट्टाधारी)\s*[:\-=]*\s*(.{3,60})",
        "Tender Amount":     r"(?i)(?:contract\s*price|bid\s*amount|tender\s*amount|L1\s*amount|amount\s*rs\.?|quoted\s*rate\s*in\s*figures)\s*[:\-=]*\s*(?:Rs\.?|₹|INR)?\s*([\d,]+(?:\.\d+)?)",
        "Schedule Discount": r"(?i)(?:([\d.]+\s*%\s*Below\s*Schedule[^\n]{0,20})|Less\s*\(-\)\s*([\d.]+\s*%?))",
        "Maturity Date":     r"(?i)(?:maturity|mat\.?\s*date|due\s*date|payable\s*on|matures?\s*on)\s*[:\-=]*\s*([\d.\-/]{6,15})",
        "FD Account No":     r"(?i)(?:FD\s*(?:account|a/?c|acct)?\s*(?:no\.?|number)?|account\s*no\.?|FDR\s*no\.?)\s*[:\-=]*\s*([\d]{6,20})",
    }

    for key, pattern in patterns.items():
        m = re.search(pattern, text)
        if not m:
            continue

        if key == "Schedule Discount":
            val = (m.group(1) or m.group(2) or '').strip().rstrip('.')
            if len(val) < 3:
                continue
        else:
            val = m.group(1).strip().strip(':-=. ')

        if key == "_hindi_name":
            val = re.split(r'[\n।|]', val)[0].strip()
            val = re.split(
                r'\s+(?=(?:पिता|पुत्र|माता|पत्नी|सुपुत्र|S[/.]?O|D[/.]?O|W[/.]?O)\b)',
                val, flags=re.IGNORECASE
            )[0].strip()
            val = val[:35].strip()
            devanag = sum(1 for c in val if '\u0900' <= c <= '\u097F')
            words = [w for w in val.split() if len(w) >= 2]
            if devanag < 3 or len(words) < 2:
                continue
            if "Agency Name" not in info:
                info["Agency Name"] = val
            continue

        if key in ("Agency Name", "_contractor_row", "Name of Division", "Scheme Name", "Allottee Name"):
            val = re.split(r'\s{2,}', val)[0].strip()
            val = re.split(r'\s+(?=\w[\w\s]{1,20}?[:\-]{1,2}\s)', val)[0].strip()
            val = re.split(
                r'\s+(?=(?:Ref|Date|Sub.Division|Name\s+of|Technical|Voucher)\b)',
                val, flags=re.IGNORECASE
            )[0].strip()
            val = val[:65].strip()

            latin   = sum(1 for c in val if c.isascii() and c.isalpha())
            devanag = sum(1 for c in val if '\u0900' <= c <= '\u097F')
            if latin < 3 and devanag < 3:
                continue

            words = [w for w in val.split() if len(w) >= 3]
            if key in ("Agency Name", "_contractor_row") and len(words) < 2:
                continue

            _HEADER_WORDS = {
                'AMOUNT', 'BID', 'RANK', 'BOQ', 'TOTAL', 'RATE', 'PRICE',
                'SCHEDULE', 'QUANTITY', 'UNIT', 'DESCRIPTION', 'ITEM', 'SR',
            }
            if key in ("Agency Name", "_contractor_row"):
                if len({w.upper() for w in val.split()} & _HEADER_WORDS) >= 2:
                    continue

            val = _clean_name(val) if key not in ("Name of Division", "Scheme Name", "Allottee Name") else val
            if not val:
                continue

        elif key in ("Date of Casting", "Date of Testing"):
            if not _is_valid_date(val):
                continue

        elif key == "Ref. Number":
            val = re.split(r'\s+(?=[a-z]{4,})', val)[0].strip()
            val = re.sub(r'-?[A-Za-z]{1,4}$', '', val).strip('-').strip()
            if len(val) < 3:
                continue

        elif key == "Deposit Amount":
            val = val.replace(',', '').strip()
            if not val.replace('.', '').isdigit():
                continue

        elif key == "Plot Size":
            if len(val) < 2:
                continue

        elif key == "Tender Amount":
            val = val.replace(',', '').strip()
            if not val.replace('.', '').isdigit() or float(val) < 1000:
                continue

        elif key == "Schedule Discount":
            val = val.strip().rstrip('.')
            if len(val) < 5:
                continue

        elif key == "Maturity Date":
            if len(val) < 6:
                continue

        elif key == "FD Account No":
            val = val.strip()
            if not val.isdigit() or len(val) < 6:
                continue

        public_key = "Agency Name" if key in ("_contractor_row", "_hindi_name") else key
        if public_key not in info:
            info[public_key] = val

    # BOQ table: look for L1 (lowest bidder) row
    if "Tender Amount" not in info:
        boq_m = re.search(
            r'(?:1\s*[|]\s*)([A-Z][A-Za-z\s.]{3,50})\s*[|]\s*([\d,]+\.\d+)\s*[|]\s*L1',
            text)
        if boq_m:
            info["Tender Amount"] = boq_m.group(2).replace(',', '')
            if "Agency Name" not in info:
                info["Agency Name"] = boq_m.group(1).strip()
        else:
            tp_m = re.search(
                r'(?:Rs\.?|\u20b9)\s*([\d,]+(?:\.\d+)?)\s*(?:/-)?[^.]{0,40}based on Tender',
                text, re.IGNORECASE)
            if tp_m:
                info["Tender Amount"] = tp_m.group(1).replace(',', '')

    return info


_CAPS_SKIP = {
    'DATE', 'NAME', 'WORK', 'DIVISION', 'AGREEMENT', 'VOUCHER', 'BILL',
    'AND', 'THE', 'FOR', 'OF', 'URBAN', 'TRUST', 'BHILWARA', 'ACCOUNT',
    'AMOUNT', 'ONLY', 'INR', 'LAKH', 'NOTE', 'LABOUR', 'CESS', 'FROM',
    'FUND', 'TOTAL', 'GRAND', 'NET', 'TAX', 'GST', 'TDS', 'VAT',
    'CERTIFICATE', 'REPORT', 'STATEMENT', 'PROGRESS', 'CONTRACT',
    'TECHNICAL', 'SANCTION', 'COMPLETION', 'STIPULATED', 'ACTUAL',
    'EXTENSION', 'APPLIED', 'SECRETARY', 'ENGINEER', 'CHARGE', 'ISSUE',
    'FORM', 'PRIOR', 'SUBMISSION', 'TENDER', 'OPERATING', 'DEPOSIT',
    'MATURITY', 'PAYABLE', 'BANK', 'THIRTY', 'ONE', 'THOUSAND', 'HUNDREDS',
    'FIVE', 'SEVEN', 'EIGHT', 'NINE', 'TEN', 'YEARS', 'MONTHS',
    'PWD', 'BSR', 'UIT', 'CPWD', 'NHAI', 'BDC', 'SDO', 'EE', 'AE',
    'JEN', 'XEN', 'CE', 'SE', 'MES', 'RVPN', 'PHED', 'PHD', 'PD',
    'BUILDING', 'WORKS', 'ROADS', 'IRRIGATION', 'DEPARTMENT',
    'GOVERNMENT', 'RAJASTHAN', 'INDIA', 'STATE', 'NATIONAL',
    'MUNICIPAL', 'CORPORATION', 'COMMITTEE', 'PANCHAYAT', 'SAMITI',
    'RUPEES', 'PAISE', 'LAKHS', 'CRORE', 'CRORES', 'INTEREST', 'RATE',
}


def keyword_scan(text: str) -> dict:
    """Fallback scan for contractor names and dates when regex patterns miss them."""
    text = devanagari_to_latin(text)
    found = {}

    m = re.search(r'(?i)(M[/.]s\.?\s+[A-Za-z][A-Za-z\s\.&]{3,50})', text)
    if m:
        val = re.split(r'\s*[\d(]', m.group(1).strip())[0].strip()
        val = _clean_name(val[:60])
        if val and len(val.split()) >= 2:
            found["Agency Name"] = val

    if "Agency Name" not in found:
        mb = re.search(
            r'(?i)(?:Bidder\s*Name|Bidder|Contractor)\s*[:\-.]*\s*([A-Z][A-Za-z\s\.&]{4,60})',
            text)
        if mb:
            val = re.split(r'\s*[\d(\n]', mb.group(1).strip())[0].strip()
            val = re.split(r'\s+(?=\w[\w\s]{1,15}[:\-]{1,2})', val)[0].strip()[:60]
            val = _clean_name(val)
            if val and len([w for w in val.split() if len(w) >= 2]) >= 2:
                found["Agency Name"] = val

    if "Agency Name" not in found:
        m2 = re.search(r'\b(BAO|SHRI|SH\.?|SMT\.?|SHRE?E?|M/S)\s+([A-Z][A-Z\s\.]{5,50})', text)
        if m2:
            val = (m2.group(1).strip() + ' ' + m2.group(2).strip())
            val = re.split(r'\s*[\d(]', val)[0].strip()[:60]
            val = _clean_name(val)
            if len([w for w in val.split() if len(w) >= 3]) >= 3:
                found["Agency Name"] = val

    if "Agency Name" not in found:
        for name in re.findall(r'\b([A-Z]{3,}(?:\s+[A-Z]{3,}){2,})\b', text):
            words = name.split()
            if any(w in _CAPS_SKIP for w in words):
                continue
            if not all(len(w) >= 3 for w in words):
                continue
            val = _clean_name(name[:60])
            if val and len(val.split()) >= 2:
                found["Agency Name"] = val
                break

    casting_ctx = re.search(r'(?i)cast(?:ing)?.{0,60}?(\d{2}[.\-/]\d{2}[.\-/]\d{2,4})', text)
    if casting_ctx:
        d = casting_ctx.group(1)
        if _is_valid_date(d):
            found["Date of Casting"] = d
    else:
        all_dates = [d for d in re.findall(r'\b(\d{2}[.\-/]\d{2}[.\-/]\d{2,4})\b', text)
                     if _is_valid_date(d)]
        if all_dates:
            found["Date of Casting"] = all_dates[0]

    if "Deposit Amount" not in found:
        m_dep = re.search(
            r'(?:जमाराशि|जमा|deposit|rashi|राशि|रकम)\D{0,30}?(?:Rs\.?|₹|INR)?\s*([\d,]{4,}(?:\.\d+)?)',
            text, re.IGNORECASE)
        if m_dep:
            found["Deposit Amount"] = m_dep.group(1).replace(',', '')

    if "Plot Size" not in found:
        m_size = re.search(
            r'([\d.,]+)\s*(sqm|sq\.?\s*m|वर्ग\s*मीटर|गज|gaj|sq\.?\s*ft|sq\.?\s*yard)',
            text, re.IGNORECASE)
        if m_size:
            found["Plot Size"] = f"{m_size.group(1)} {m_size.group(2).strip()}"
        else:
            m_size2 = re.search(
                r'(?:क्षेत्रफल|kshetrafal|area|माप)[:\s\-–=]*([\d.,]{3,})',
                text, re.IGNORECASE)
            if m_size2:
                found["Plot Size"] = f"{m_size2.group(1)} वर्ग मीटर (approx.)"

    if "Tender Amount" not in found:
        boq = re.search(
            r'(?:1\s*[|]\s*)([A-Z][A-Za-z\s.]{3,50})\s*[|]\s*([\d,]+\.\d+)\s*[|]\s*L1',
            text)
        if boq:
            found["Tender Amount"] = boq.group(2).replace(',', '')
            if "Agency Name" not in found:
                found["Agency Name"] = boq.group(1).strip()
        else:
            tp = re.search(
                r'(?:Rs\.?|₹)\s*([\d,]+(?:\.\d+)?)\s*(?:/-)?[^.]{0,40}Tender',
                text, re.IGNORECASE)
            if tp:
                found["Tender Amount"] = tp.group(1).replace(',', '')

    if "Schedule Discount" not in found:
        sd = re.search(r'([\d.]+\s*%\s*Below\s*Schedule[^\n]{0,25})', text, re.IGNORECASE)
        if sd:
            found["Schedule Discount"] = sd.group(1).strip()

    if "Maturity Date" not in found:
        md = re.search(
            r'(?:maturity|mat\.?\s*date|due\s*date|payable\s*on|matur\w*\s*on)'
            r'\s*[:\-=]*\s*([\d.\/\-]{6,15})',
            text, re.IGNORECASE)
        if md and len(md.group(1)) >= 6:
            found["Maturity Date"] = md.group(1).strip()

    if "FD Account No" not in found:
        fd = re.search(
            r'(?:FD\s*(?:account|a/?c|acct)?\s*(?:no\.?|number)?|account\s*no\.?)'
            r'\s*[:\-=]*\s*(\d{6,20})',
            text, re.IGNORECASE)
        if fd:
            found["FD Account No"] = fd.group(1)

    return found


_HINDI_CORRECTIONS = [
    (r'भूकंप', 'भूखंड', '> ⚠️ **"भूकंप"** (earthquake) की जगह **"भूखंड"** (plot) से search किया गया।'),
    (r'भूक्म्प', 'भूखंड', ''),
    (r'भूखण्ड', 'भूखंड', ''),
    (r'जमाराशी', 'जमाराशि', ''),
    (r'क्षेत्राफल', 'क्षेत्रफल', ''),
    (r'(?i)\bbhukand\b', 'भूखंड', '> ⚠️ **"bhukand"** को **"भूखंड"** (plot) समझा गया।'),
    (r'(?i)\bbhukhand\b', 'भूखंड', ''),
    (r'(?i)\bbhu\s*khad\b', 'भूखंड', ''),
    (r'(?i)\bplot\s*no\.?\s*(\d)', r'plot \1', ''),
    (r'(?i)\bkshetafal\b', 'kshetrafal', ''),
    (r'(?i)\bkhetrafal\b', 'kshetrafal', ''),
    (r'(?i)\bkhshetrafal\b', 'kshetrafal', ''),
    (r'(?i)\bsaiz\b', 'size', ''),
    (r'(?i)\bsaize\b', 'size', ''),
    (r'(?i)\bjamrashi\b', 'जमाराशि', ''),
    (r'(?i)\bjama\s*rashee\b', 'जमाराशि', ''),
    (r'(?i)\bjammarrashi\b', 'जमाराशि', ''),
    (r'(?i)\brakam\b', 'राशि', ''),
    (r'(?i)\byojna\b', 'yojana', ''),
    (r'(?i)\byojan\b', 'yojana', ''),
    (r'(?i)\bsampling\b', 'casting', '> ℹ️ **"sampling"** को **"casting"** (concrete test) समझा गया।'),
    (r'(?i)\bsapling\b', 'casting', ''),
    (r'(?i)\bagancy\b', 'agency', ''),
    (r'(?i)\bagenci\b', 'agency', ''),
    (r'(?i)\bcontacter\b', 'contractor', ''),
    (r'(?i)\bcontarctor\b', 'contractor', ''),
    (r'(?i)\btendor\b', 'tender', ''),
    (r'(?i)\bdivison\b', 'division', ''),
]


def normalize_query(query: str) -> tuple[str, str]:
    note = ''
    for pattern, replacement, user_note in _HINDI_CORRECTIONS:
        new_q, n = re.subn(pattern, replacement, query)
        if n > 0:
            query = new_q
            if user_note and not note:
                note = user_note
    return query, note


def extract_plot_number_from_query(query: str) -> str | None:
    m = re.search(r'\b(\d+[-/][A-Za-z][-/]\d+|[A-Za-z][-/][A-Za-z][-/]\d+)\b', query)
    if m:
        return m.group(1)
    m2 = re.search(
        r'(?:भूखंड|भूखण्ड|plot|sankhya|संख्या)\s*(?:no\.?\s*)?(\d{1,6})\b',
        query, re.IGNORECASE)
    if m2:
        return m2.group(1)
    if re.search(r'भूखंड|भूखण्ड|plot\b|sankhya|संख्या', query, re.IGNORECASE):
        numbers = re.findall(r'\b(\d{2,5})\b', query)
        if numbers:
            return numbers[0]
    return None


def _build_plot_filter_pattern(asked_plot: str) -> str:
    if re.match(r'^\d+[-/][A-Za-z][-/]\d+$', asked_plot):
        parts = re.split(r'[-/]', asked_plot)
        return r'[-/\s.]?'.join(re.escape(p) for p in parts)
    return r'\b' + re.escape(asked_plot) + r'\b'


def extract_row_for_plot(chunk_text: str, asked_plot: str) -> str:
    pattern = _build_plot_filter_pattern(asked_plot)
    m = re.search(pattern, chunk_text, re.IGNORECASE)
    if not m:
        return chunk_text
    lines = chunk_text.split('\n')
    for i, ln in enumerate(lines):
        if re.search(pattern, ln, re.IGNORECASE):
            return '\n'.join(lines[max(0, i - 2):min(len(lines), i + 4)])
    s = max(0, m.start() - 200)
    e = min(len(chunk_text), m.end() + 200)
    return chunk_text[s:e]


def detect_intent(query: str) -> list[str]:
    q = query.lower()
    intents = []
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
    if re.search(r'bhukhand|भूखंड|भूखण्ड|bhu.?khand|plot|sankhya|संख्या', q):
        intents.append("Plot Number")
    if re.search(r'size|akar|आकार|area|maap|माप|varg|वर्ग|sqm|sq|gaj|गज|kshetrafal|क्षेत्रफल|saiz|साइज', q):
        intents.append("Plot Size")
    if re.search(r'jama|जमा|rashi|राशि|deposit|amount|rakam|रकम|shulk|शुल्क|kitni', q):
        intents.append("Deposit Amount")
    if re.search(r'yojana|योजना|scheme|nagar|tilak|colony|kalani|कालोनी', q):
        intents.append("Scheme Name")
    if re.search(r'allot|आवंति|patta|पट्टा|holder|khattedar|पट्टेदार|malik|मालिक', q):
        intents.append("Allottee Name")
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
    # Bid existence — separate check used by SQLite fast path
    if re.search(r'bid\s*kiya|participated|participate|kiya\s*hai|ne\s*bid', q):
        intents.append("Bid Existence")
    # List bidders
    if re.search(r'sare\s*bidder|list.*bidder|bidder.*list|kaun.*bid|saare.*participant', q):
        intents.append("list_bidders")
    return intents


def extract_work_title(text: str) -> str:
    m = re.search(r'(?i)Name\s*of\s*Work\s*[:\-=]+\s*(.{10,120})', text)
    if m:
        val = m.group(1).strip().strip(':-= ')
        val = re.split(r'\s{2,}', val)[0].strip()[:100]
        if len(val) > 10:
            return val
    return ""


def _expand_bm25_tokens(query: str) -> list[str]:
    """Expand query tokens with synonyms for BM25 retrieval."""
    synonyms = {
        'agency': ['agency', 'contractor', 'bidder', 'firm'],
        'tender': ['tender', 'bid', 'nit', 'estimate'],
        'casting': ['casting', 'cast', 'concrete'],
        'plot': ['plot', 'bhukhand', 'bhu', 'khand'],
        'deposit': ['deposit', 'jama', 'rashi', 'amount'],
    }
    base_tokens = query.lower().split()
    expanded = list(base_tokens)
    for token in base_tokens:
        for key, syns in synonyms.items():
            if token in syns and token != key:
                expanded.extend(syns)
                break
    return list(dict.fromkeys(expanded))  # deduplicate while preserving order


def build_natural_answer(
    query: str,
    all_info: dict,
    sources: list[str],
    context_chunks: list[dict]
) -> str:
    intents = detect_intent(query)

    work_title = ""
    for chunk in context_chunks:
        work_title = extract_work_title(clean_ocr_text(chunk["text"]))
        if work_title:
            break

    source_str = sources[0].split("/")[-1].strip() if sources else "the indexed document"
    paragraphs = []

    if work_title:
        paragraphs.append(f"**{source_str}** — *\"{work_title[:90]}\"*")
    else:
        paragraphs.append(f"**{source_str}** se yeh jankari mili:")

    _FIELD_LABELS = {
        "Agency Name":       "🏗️ **Contractor / Agency**",
        "Date of Casting":   "📅 **Casting Date** (jab concrete dali gayi)",
        "Date of Testing":   "📈 **Testing Date**",
        "W.O. Number":       "📄 **Work Order No.**",
        "Ref. Number":       "🔗 **Reference No.**",
        "Name of Division":  "🏛️ **Division**",
        "Plot Number":       "📌 **भूखंड संख्या (Plot No.)**",
        "Plot Size":         "📐 **भूखंड का आकार (Plot Size)**",
        "Deposit Amount":    "💰 **जमाराशि (Deposit Amount)**",
        "Scheme Name":       "🌅 **योजना (Scheme)**",
        "Allottee Name":     "👤 **आवंती / पट्टेदार (Allottee)**",
        "Tender Amount":     "💵 **Tender Amount (L1)**",
        "Schedule Discount": "ℹ️ **Schedule Discount**",
        "Maturity Date":     "📅 **FD Maturity Date**",
        "FD Account No":     "🏦 **FD Account No.**",
    }

    def _format_val(field, val):
        if field == "Deposit Amount":
            return f"₹{val}"
        if field == "Tender Amount":
            try:
                return f"₹{float(val):,.2f}"
            except (ValueError, TypeError):
                return val
        return str(val)

    def _fmt_field(field, val):
        label = _FIELD_LABELS.get(field, f"**{field}**")
        return f"- {label}: **{_format_val(field, val)}**"

    answered = []
    missing  = []
    real_intents = [f for f in intents if f not in ("Bid Existence", "list_bidders")]
    for field in real_intents:
        val = all_info.get(field)
        if val:
            answered.append(_fmt_field(field, val))
        else:
            missing.append(field)

    extra = {k: v for k, v in all_info.items() if k not in real_intents and k != "Source"}

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
    elif not real_intents:
        # Prevent confident hallucination on unknown queries by NOT dumping random Regex matches
        paragraphs.append(
            "> ⚠️ Answer not found in document. (Direct extraction failed because the query intent was not recognized)."
        )
    else:
        paragraphs.append(
            "> ⚠️ Maafi chahta hoon, yeh jankari indexed documents mein "
            "nahi mili. PDF ko re-scan karein ya alag query try karein.")

    return "\n\n".join(paragraphs)


# ---------------------------------------------------------------------------
# Query Rewriter Agent
# ---------------------------------------------------------------------------

def _rewrite_query(query: str, intent: str) -> str:
    """Agentic step to rewrite user chat into an optimized RAG search query."""
    if intent == "list_bidders":
        # Extract the core project description from the query
        clean = re.sub(r'(?i)sare\s*bidder|ka\s*naam\s*do|batao|list|kaun|participation', '', query)
        clean = re.sub(r'[^a-zA-Z0-9\s]', '', clean).strip()
        if clean:
            return f"Find all bidder names and construction companies related to: {clean}"
        return "List of all participating bidders agencies contractors"
    return query

# ---------------------------------------------------------------------------
# Local LLM Fallback (Phi-3-mini-4k via gpt4all)
# ---------------------------------------------------------------------------
def _llm_answer(query: str, chunks: list[dict], extracted_fields: dict = None, force_bidder_mode: bool = False) -> str | None:
    global _phi3_model
    try:
        from gpt4all import GPT4All
        from config import GPT4ALL_MODEL
        import os

        model_dir  = os.path.join(os.path.expanduser("~"), ".cache", "gpt4all")
        model_path = os.path.join(model_dir, GPT4ALL_MODEL)
        if not os.path.exists(model_path):
            print(f"  [llm] Model not found at {model_path}. Skipping.")
            return None

        if _phi3_model is None:
            print("  [llm] Loading Phi-3-mini...")
            _phi3_model = GPT4All(GPT4ALL_MODEL, model_path=model_dir, verbose=False)
            print("  [llm] Ready.")

        # Context Deduplication
        seen_content = set()
        deduped_chunks = []
        for chunk in chunks:
            raw_text = clean_ocr_text(chunk["text"])
            norm = re.sub(r'\W+', '', raw_text.lower())
            if norm not in seen_content:
                seen_content.add(norm)
                deduped_chunks.append(chunk)

        context_parts = []
        LLM_CONTEXT_LIMIT = 8
        for i, chunk in enumerate(deduped_chunks[:LLM_CONTEXT_LIMIT], 1):
            text = clean_ocr_text(chunk["text"])[:400]
            src  = chunk["filename"].replace("\\", "/").split("/")[-1]
            context_parts.append(f"[Chunk {i} | Source: {src}]\n{text}")
        context_str = "\n\n".join(context_parts)

        if force_bidder_mode:
            prompt = (
                "<|system|>\n"
                "You are a strict data extraction assistant.\n"
                "1. Extract ONLY the bidder names from the provided text.\n"
                "2. If bidders exist, list them clearly.\n"
                "3. If the answer is not present in the context, say 'Answer not found in document'. Do not guess.\n"
                "4. Cite the Page/Source number.\n"
                "<|user|>\n"
                f"Document Context:\n{context_str}\n\n"
                f"Question:\n{query}\n"
                "<|assistant|>\n"
            )
        else:
            fields_str = ""
            if extracted_fields:
                clean_fields = {k: v for k, v in extracted_fields.items() if k != "Source"}
                if clean_fields:
                    fields_str = (
                        "\nVerified extracted facts (do NOT contradict these):\n" +
                        "\n".join(f"  - {k}: {v}" for k, v in clean_fields.items()) + "\n"
                    )
            prompt = (
                "You are a strict government records assistant.\n"
                "Rules:\n"
                "1. Answer ONLY using the provided Context. Do NOT guess or use outside knowledge.\n"
                "2. If the answer is not present, say: 'Answer not found in the document.'\n"
                "3. Reply in Hindi/English mix. Be direct and factual.\n"
                "4. For yes/no questions, start with 'Haan!' or 'Nahi.' clearly.\n"
                "5. Keep it concise: 2-3 lines max. Cite the source filename from the [Source: filename] label provided in the Context.\n"
                "6. CRITICAL: Do NOT invent, hallucinate, or use any filename that is NOT explicitly labeled in the Context below. If unsure of the filename, just use the information provided.\n"
                f"{fields_str}\n"
                f"Context:\n{context_str}\n\n"
                f"Question: {query}\n\n"
                "Answer:"
            )

        print(f"  [llm] Generating answer...")
        with _phi3_model.chat_session():
            response = _phi3_model.generate(prompt, max_tokens=120, temp=0.05)
        response = response.strip()
        if not response:
            return None
        print(f"  [llm] Done ({len(response)} chars)")

        _REFUSAL = ('i cannot', "i'm sorry", "i don't know", 'not enough information',
                    'maafi chahta hoon', 'jankari nahi', 'not found in document',
                    'yeh jankari documents mein nahi')
        if len(response) < 15 or any(r in response.lower() for r in _REFUSAL):
            print("  [llm] LLM refused or did not find answer: 'Answer not found in document'")
            return "⚠️ Answer not found in the document. I searched the relevant pages but could not verify this information."

        # Numeric Extraction Guard (regex + word-to-digit validation)
        # Prevents LLM numbering hallucinations by checking if digits exist in context.
        money_matches = re.findall(r'₹?\s*\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d{4,}', response)
        
        try:
            from rag_utils import words_to_digits_string
            word_numbers = words_to_digits_string(response)
        except ImportError:
            word_numbers = []
            
        all_targets = money_matches + word_numbers
        
        for mm in all_targets:
            num_val = re.sub(r'[^\d]', '', mm)
            if len(num_val) >= 2:
                context_digits = re.sub(r'[^\d]', '', context_str)
                if num_val not in context_digits:
                    print(f"  [llm] Hallucination guard triggered (Number {mm} not in context)")
                    return "Answer not found in the document."

        # Reject if LLM is hallucinating Q&A pairs from the context
        if re.search(r'(?i)\bQuestion\s*:', response) or response.count('?') > 2:
            print("  [llm] Response rejected (hallucinated Q&A), using regex fallback")
            return None

        return response

    except ImportError:
        print("  [llm] gpt4all not installed.")
        return None
    except Exception as e:
        print(f"  [llm] Error: {e}")
        return None


def format_answer(query: str, chunks: list[dict]) -> str:
    all_info: dict = {}
    sources:  list = []

    def _scan_chunk(chunk):
        nonlocal all_info
        text = clean_ocr_text(chunk["text"])
        if not text:
            return
        info = extract_key_info(text)
        for k, v in keyword_scan(text).items():
            if k not in info:
                info[k] = v
        for k, v in info.items():
            if k not in all_info:
                all_info[k] = v

    reranked_chunks = _rerank_chunks(query, chunks, top_n=RERANK_TOP_N)

    for chunk in reranked_chunks:
        _scan_chunk(chunk)
        fname   = chunk["filename"]
        parts   = fname.replace("\\", "/").split("/")
        display = " / ".join(parts[-3:]) if len(parts) >= 3 else fname
        if display not in sources:
            sources.append(display)

    intents = detect_intent(query)
    is_bid_existence = "Bid Existence" in intents

    real_intents    = [f for f in intents if f != "Bid Existence"]
    missing_intents = [f for f in real_intents if f not in all_info]
    
    # Removed Pass-2 brute-force scanning (e.g. 1269 chunks file-scan) to strictly adhere 
    # to Production RAG architecture: Hybrid search -> Top 20 -> Rerank Top 5 -> LLM Answer

    # --- Smart Fallback to LLM ---
    from config import LLM_ENABLED
    
    # Trigger LLM if: 
    # 1. We specifically looked for something but missed it completely
    # 2. Or if there were no structured intents detected at all (custom question)
    missing_intents_after_pass2 = [f for f in real_intents if f not in all_info]
    llm_triggered = False
    
    if LLM_ENABLED and (missing_intents_after_pass2 or not intents):
        print(f"  [fallback] Using LLM for deep parsing. Missing: {missing_intents_after_pass2}, Intents: {intents}")
        llm_resp = _llm_answer(query, reranked_chunks, extracted_fields=all_info)
        if llm_resp:
            llm_triggered = True
            # Overwrite natural answer to just return the LLM's response + sources
            return llm_resp + f"\n\n**(Sources: {', '.join(sources)})**"

    if not llm_triggered:
        return build_natural_answer(query, all_info, sources, reranked_chunks)


def build_detail_text(chunks: list[dict]) -> str:
    sections = []
    seen_files: set = set()
    for chunk in chunks[:6]:
        fname = chunk["filename"]
        short = fname.replace("\\", "/").split("/")[-1]
        text  = clean_ocr_text(chunk["text"])
        if not text:
            continue
        
        score_val = chunk.get('score', 0)
        if score_val > 0:
            header = f"### {short}  (relevance: {score_val:.0%})\n"
        else:
            header = f"### {short}\n"
            
        if short not in seen_files:
            seen_files.add(short)
            sections.append(header + text[:800])
        else:
            sections.append(text[:400])
    return "\n\n---\n\n".join(sections)


def answer_query(
    query: str,
    filename_filter: str | None = None,
    category_filter: str | None = None,
):
    """Main entry point — yields streaming response events for the frontend."""
    from database import (
        query_by_agency, query_by_plot, query_by_allottee,
        get_all_fields_for_file, write_audit,
    )

    query, correction_note = normalize_query(query)
    intents = detect_intent(query)

    # -----------------------------------------------------------------------
    # FAST PATH: SQLite structured metadata lookup
    # For known-entity queries (bid existence, plot, allottee, bidder lists)
    # try the DB first — no vector search, no reranking, deterministic, <40ms.
    # -----------------------------------------------------------------------
    if not filename_filter and not category_filter:
        db_rows: list[dict] = []
        is_list_bidders = "list_bidders" in intents

        if is_list_bidders:
            # Table extraction: we need to find the correct document if the user provides 
            # some context, otherwise we let the Hybrid Path retrieve the document, 
            # but since SQLite bidder lookup requires a filename, we will handle list_bidders 
            # in Hybrid Path if filename is unknown, or we can see if there is an agency name in query
            pass

        if "Bid Existence" in intents or "Agency Name" in intents:
            person_words = _extract_person_names(query)
            for pw in person_words:
                if len(pw) >= 4:
                    db_rows = query_by_agency(pw)
                    if db_rows:
                        break

        elif "Plot Number" in intents:
            asked_plot = extract_plot_number_from_query(query)
            if asked_plot:
                db_rows = query_by_plot(asked_plot)

        elif "Allottee Name" in intents:
            person_words = _extract_person_names(query)
            for pw in person_words:
                if len(pw) >= 4:
                    db_rows = query_by_allottee(pw)
                    if db_rows:
                        break

        if db_rows:
            row = db_rows[0]
            fname = row.get("filename", "")
            src   = fname.split("/")[-1] if fname else "metadata.db"
            ocr_q = row.get("ocr_quality", 1.0)

            # Build structured fields dict from SQLite row
            fields = {
                "Agency Name":       row.get("agency_name", ""),
                "Ref. Number":       row.get("ref_number", ""),
                "Tender Amount":     row.get("tender_amount", ""),
                "Schedule Discount": row.get("schedule_discount", ""),
                "Date of Casting":   row.get("date_casting", ""),
                "Date of Testing":   row.get("date_testing", ""),
                "W.O. Number":       row.get("wo_number", ""),
                "Name of Division":  row.get("division", ""),
                "Plot Number":       row.get("plot_number", ""),
                "Plot Size":         row.get("plot_size", ""),
                "Deposit Amount":    row.get("deposit_amount", ""),
                "Scheme Name":       row.get("scheme_name", ""),
                "Allottee Name":     row.get("allottee_name", ""),
                "Maturity Date":     row.get("maturity_date", ""),
                "FD Account No":     row.get("fd_account", ""),
            }
            fields = {k: v for k, v in fields.items() if v}

            ocr_warning = ""
            if ocr_q < 0.5:
                ocr_warning = (
                    "\n\n> ⚠️ **OCR quality low** ("
                    f"{ocr_q:.0%}) — some fields may be missing or incorrect."
                )

            table_data = [{**fields, "Source": src}]
            answer_text = build_natural_answer(query, fields, [fname], [])
            if correction_note:
                answer_text = correction_note + "\n\n" + answer_text
            answer_text += ocr_warning

            print(f"  [sqlite-fast] Answered from DB in <40ms ({src})")
            write_audit(query, intents=[i for i in intents], doc_ids=[fname],
                        chunks_returned=0, llm_used=False)

            yield {"type": "meta", "table": table_data, "sources": [src], "context_count": 0}
            yield {"type": "content", "content": answer_text}
            yield {"type": "detail",  "content": ""}
            yield {"type": "sources_detail", "content": []}
            return

    # -----------------------------------------------------------------------
    # STANDARD PATH: Hybrid retrieval — BM25 + vector + rerank + regex
    # -----------------------------------------------------------------------
    context_chunks = retrieve_context(
        query,
        filename_filter=filename_filter,
        category_filter=category_filter,
    )

    if not context_chunks:
        yield {
            "type":    "error",
            "content": (
                "No relevant records were found.\n\n"
                "**Suggestions:**\n"
                "- Include ward number, sector, or agency name\n"
                "- Make sure the PDF is indexed (use the Scan button)\n"
                "- Try rephrasing in English"
            )
        }
        return

    asked_plot = extract_plot_number_from_query(query)
    if asked_plot:
        plot_pattern = _build_plot_filter_pattern(asked_plot)
        filtered     = [c for c in context_chunks if re.search(plot_pattern, c["text"], re.IGNORECASE)]
        if filtered:
            context_chunks = filtered
            print(f"  [plot-filter] {len(filtered)} chunks matched")
        else:
            not_found_msg = (
                f"indexed documents में **भूखंड संख्या {asked_plot}** का कोई record नहीं मिला।"
                f"\n\n**संभावित कारण:**\n"
                f"- यह भूखंड संख्या PDF में अलग format में हो सकती है\n"
                f"- फ़ाइल का OCR इस पेज को सही से नहीं पढ़ सका\n"
                f"- PDF को re-scan करें (Scan button) और फिर कोशिश करें"
            )
            if correction_note:
                not_found_msg = correction_note + "\n\n" + not_found_msg
            yield {"type": "meta", "table": [], "sources": [], "context_count": 0}
            yield {"type": "content", "content": not_found_msg}
            yield {"type": "detail",  "content": ""}
            yield {"type": "sources_detail", "content": []}
            return

    # Handle fast SQLite Table fetching if intent is list_bidders
    if "list_bidders" in intents and context_chunks:
        from database import query_bidders_for_tender
        top_file = context_chunks[0]["filename"]
        bidders_db = query_bidders_for_tender(top_file)
        
        short_src = top_file.split('/')[-1]
        
        if bidders_db:
            ans = f"**{short_src}** mein following bidders ne participate kiya tha:\n\n"
            for idx, b in enumerate(bidders_db, 1):
                amt = f" (₹{b['bid_amount']})" if b.get('bid_amount') else ""
                rnk = f" - **{b['rank']}**" if b.get('rank') else ""
                ans += f"{idx}️⃣ {b['bidder_name']}{amt}{rnk}\n"
                
            yield {"type": "meta", "table": [{"Source": short_src}], "sources": [short_src], "context_count": 0}
            yield {"type": "content", "content": ans}
            yield {"type": "detail", "content": ""}
            yield {"type": "sources_detail", "content": []}
            return
        else:
            print(f"  [sqlite-fast] No bidders found in SQLite for {short_src}, falling back to Agentic Hybrid Retrieval...")
            # Agentic rewriting applies here for better context pulling from the vector DB
            query = _rewrite_query(query, "list_bidders")

    table_data:   list = []
    seen_sources: list = []

    for chunk in context_chunks:
        src  = chunk["filename"]
        text = clean_ocr_text(chunk["text"])
        if not text:
            continue
        extraction_text = extract_row_for_plot(text, asked_plot) if asked_plot else text
        info = extract_key_info(extraction_text)
        if not any(k in info for k in ("Agency Name", "Date of Casting")):
            info.update({k: v for k, v in keyword_scan(extraction_text).items() if k not in info})
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
        "context_count": len(context_chunks),
    }

    # Answer Forcing Prompt (Agent Validation layer)
    if "list_bidders" in intents:
        from config import GPT4ALL_MODEL  # Even if LLM_ENABLED=False globally, we force it for fallback
        print("  [RAG] Booting LLM for Table Fallback Bidder Extraction...")
        llm_resp = _llm_answer(query, context_chunks, force_bidder_mode=True)
        if llm_resp:
            answer_text = llm_resp + f"\n\n**(Agentic Retrieval Sources: {', '.join(seen_sources)})**"
        else:
            answer_text = format_answer(query, context_chunks)
    else:
        answer_text = format_answer(query, context_chunks)
            
    # OCR quality warning for hybrid path (mirrors SQLite fast-path behavior)
    ocr_scores = [c.get("ocr_quality", 1.0) for c in context_chunks if "ocr_quality" in c]
    if ocr_scores:
        avg_ocr = sum(ocr_scores) / len(ocr_scores)
        if avg_ocr < 0.5:
            answer_text += (
                "\n\n> ⚠️ **OCR quality low** ("
                f"{avg_ocr:.0%}) — some fields may be missing or incorrect."
            )

    if correction_note:
        answer_text = correction_note + "\n\n" + answer_text
    yield {"type": "content", "content": answer_text}

    yield {"type": "detail", "content": build_detail_text(context_chunks)}

    seen_fnames: set = set()
    deduped:    list = []
    for c in context_chunks:
        fn = c["filename"].replace("\\", "/").split("/")[-1]
        if fn not in seen_fnames:
            seen_fnames.add(fn)
            deduped.append(c)

    # Audit log — hash only, no raw text
    doc_ids = [c["filename"] for c in context_chunks]
    write_audit(query, intents=[i for i in intents], doc_ids=doc_ids,
                chunks_returned=len(context_chunks), llm_used=False)

    yield {"type": "sources_detail", "content": deduped}
