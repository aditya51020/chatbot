import os
import re
import uuid
from typing import Optional

import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder

from config import (
    CHROMA_DIR, EMBED_MODEL,
    TOP_K, COLLECTION_NAME, RERANK_TOP_N
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
        _collection = _chroma_client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )
    return _collection


def index_chunks(chunks: list[dict]) -> int:
    if not chunks:
        return 0
    collection = get_collection()
    embedder = get_embedder()
    texts = [c["text"] for c in chunks]
    metadatas = [{
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
    _invalidate_bm25()
    return len(texts)


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
    results = collection.get(include=["metadatas"])
    if not results["metadatas"]:
        return []
    return sorted({m["filename"] for m in results["metadatas"] if "filename" in m})


def get_chunk_count() -> int:
    return get_collection().count()


RELEVANCE_THRESHOLD = 0.55

_DEVA_DIGIT = str.maketrans('०१२३४५६७८९', '0123456789')


def devanagari_to_latin(text: str) -> str:
    return text.translate(_DEVA_DIGIT)


def clean_ocr_text(text: str) -> str:
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
    collection = get_collection()
    if collection.count() == 0:
        return []
    embedder = get_embedder()
    emb = embedder.encode([query_text]).tolist()

    where_clause = None
    if filename_filter and category_filter:
        where_clause = {"$and": [
            {"filename": {"$eq": filename_filter}},
            {"category": {"$eq": category_filter}},
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
        _cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        print("  Cross-encoder ready.")
    return _cross_encoder


def _rerank_chunks(query: str, chunks: list[dict], top_n: int) -> list[dict]:
    if not chunks:
        return chunks
    try:
        encoder = _get_cross_encoder()
        pairs = [(query, c["text"][:512]) for c in chunks]
        scores = encoder.predict(pairs)
        ranked = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)
        result = [c for _, c in ranked[:top_n]]
        print(f"  [rerank] {len(chunks)} -> {len(result)} chunks")
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
    results = collection.get(include=["documents", "metadatas"])
    docs  = results.get("documents", [])
    metas = results.get("metadatas", [])

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

    # vector retrieval
    seen_vector: set = set()
    vector_chunks: list[dict] = []
    for q in queries:
        for chunk in _query_chroma(q, TOP_K, filename_filter=filename_filter, category_filter=category_filter):
            key = (chunk["filename"], chunk["text"][:60])
            if key not in seen_vector:
                seen_vector.add(key)
                boost  = _work_name_boost(chunk["text"], query)
                boost += _plot_id_boost(chunk["text"], query)
                chunk["score"] = min(1.0, chunk["score"] + boost)
                vector_chunks.append(chunk)
    vector_chunks.sort(key=lambda c: c["score"], reverse=True)

    # BM25 retrieval
    bm25_chunks: list[dict] = []
    if not filename_filter:
        if _bm25_index is None:
            _build_bm25_index()
        if _bm25_index is not None and _bm25_corpus is not None:
            tokens   = query.lower().split()
            scores   = _bm25_index.get_scores(tokens)
            top_idxs = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:TOP_K]
            for idx in top_idxs:
                if scores[idx] > 0:
                    c = dict(_bm25_corpus[idx])
                    c["bm25_score"] = float(scores[idx])
                    bm25_chunks.append(c)

    # Reciprocal Rank Fusion (k=60)
    rrf_scores: dict[tuple, float] = {}
    chunk_map:  dict[tuple, dict]  = {}
    K = 60

    for rank, c in enumerate(vector_chunks):
        key = (c["filename"], c["text"][:80])
        rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (K + rank + 1)
        chunk_map[key] = c

    for rank, c in enumerate(bm25_chunks):
        key = (c["filename"], c["text"][:80])
        rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (K + rank + 1)
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
            print(f"  [name-filter] {len(filtered)}/{len(fused)} chunks for names: {person_words}")
            fused = filtered

    return fused


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
        "Allottee Name":     r"(?i)(?:खातेदार|allottee|आवंटी|pattadaar|पट्टाधारी)\s*[:\-=]*\s*(.{3,60})",
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
            # stop at lowercase words (e.g. 'panchayat samiti'), keep uppercase codes (e.g. 'UITBH_295273')
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
    'MATURITY', 'PAYABLE', 'BANK', 'KANHAIYA', 'LARA', 'LETTE', 'SPEC',
    'THIRTY', 'ONE', 'THOUSAND', 'THOUSANDS', 'HUNDRED', 'SIX', 'FOUR',
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

    # M/s prefix — most reliable signal for contractor name
    m = re.search(r'(?i)(M[/.]s\.?\s+[A-Za-z][A-Za-z\s\.&]{3,50})', text)
    if m:
        val = re.split(r'\s*[\d(]', m.group(1).strip())[0].strip()
        val = _clean_name(val[:60])
        if val and len(val.split()) >= 2:
            found["Agency Name"] = val

    # 'Bidder Name' / 'Contractor:' label
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

    # BAO / SHRI / SMT + all-caps name
    if "Agency Name" not in found:
        m2 = re.search(r'\b(BAO|SHRI|SH\.?|SMT\.?|SHRE?E?|M/S)\s+([A-Z][A-Z\s\.]{5,50})', text)
        if m2:
            val = (m2.group(1).strip() + ' ' + m2.group(2).strip())
            val = re.split(r'\s*[\d(]', val)[0].strip()[:60]
            val = _clean_name(val)
            if len([w for w in val.split() if len(w) >= 3]) >= 3:
                found["Agency Name"] = val

    # all-caps block of 3+ meaningful words
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

    # date near 'cast' keyword
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
                r'(?:क्षेत्रफल|kshetrafal|area|माप)[\s:\-–=]*([\d.,]{3,})',
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
    (r'भूकम्प', 'भूखंड', '> ⚠️ **"भूकम्प"** की जगह **"भूखंड"** (plot) से search किया गया।'),
    (r'भूखण्ड', 'भूखंड', ''),
    (r'जमाराशी', 'जमाराशि', ''),
    (r'क्षेत्राफल', 'क्षेत्रफल', ''),
    (r'क्षेत्राफ़ल', 'क्षेत्रफल', ''),
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
    (r'(?i)\bsapling\b', 'casting', '> ℹ️ **"sapling"** को **"casting"** (concrete test) समझा गया।'),
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
    if re.search(r'allot|आवंटित|patta|पट्टा|holder|khattedar|खातेदार|malik|मालिक', q):
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
    return intents


def extract_work_title(text: str) -> str:
    m = re.search(r'(?i)Name\s*of\s*Work\s*[:\-=]+\s*(.{10,120})', text)
    if m:
        val = m.group(1).strip().strip(':-= ')
        val = re.split(r'\s{2,}', val)[0].strip()[:100]
        if len(val) > 10:
            return val
    return ""


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
        "Date of Testing":   "📊 **Testing Date**",
        "W.O. Number":       "📄 **Work Order No.**",
        "Ref. Number":       "🔗 **Reference No.**",
        "Name of Division":  "🏢 **Division**",
        "Plot Number":       "📍 **भूखंड संख्या (Plot No.)**",
        "Plot Size":         "📏 **भूखंड का आकार (Plot Size)**",
        "Deposit Amount":    "💰 **जमाराशि (Deposit Amount)**",
        "Scheme Name":       "🌆 **योजना (Scheme)**",
        "Allottee Name":     "👤 **आवंटी / खातेदार (Allottee)**",
        "Tender Amount":     "💵 **Tender Amount (L1)**",
        "Schedule Discount": "ℹ️ **Schedule Discount**",
        "Maturity Date":     "📆 **FD Maturity Date**",
        "FD Account No":     "🏦 **FD Account No.**",
    }

    def _format_val(field, val):
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

    answered = []
    missing  = []
    for field in intents:
        val = all_info.get(field)
        if val:
            answered.append(_fmt_field(field, val))
        else:
            missing.append(field)

    extra = {k: v for k, v in all_info.items() if k not in intents and k != "Source"}

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
    elif not intents:
        info_to_show = {k: v for k, v in all_info.items() if k != "Source"}
        if info_to_show:
            paragraphs.append("\n".join(_fmt_field(k, v) for k, v in info_to_show.items()))
        else:
            paragraphs.append(
                "> Specific fields extract nahi ho sake. "
                "Neeche **Read more** mein full document passage dekhe.")
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

        context_parts = []
        for i, chunk in enumerate(context_chunks[:RERANK_TOP_N], 1):
            text = clean_ocr_text(chunk["text"])[:400]
            src  = chunk["filename"].replace("\\", "/").split("/")[-1]
            context_parts.append(f"[Chunk {i} | Source: {src}]\n{text}")
        context_str = "\n\n".join(context_parts)

        fields_str = ""
        if extracted_fields:
            clean_fields = {k: v for k, v in extracted_fields.items() if k != "Source"}
            if clean_fields:
                fields_str = (
                    "\nVerified extracted facts (do NOT contradict these):\n" +
                    "\n".join(f"  - {k}: {v}" for k, v in clean_fields.items()) + "\n"
                )

        prompt = (
            "You are a STRICT QA system for Indian government land and tender records.\n"
            "Rules:\n"
            "1. Answer ONLY from the provided context. Do NOT guess or use outside knowledge.\n"
            "2. If the answer is not in the context, say exactly: Not found in documents.\n"
            "3. Be concise: 2-3 bullet points max. No intro sentence.\n"
            "4. Cite the source file name for each fact.\n"
            "5. Match the language of the question (Hindi/Hinglish/English).\n"
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
                    'maafi chahta hoon', 'jankari nahi')
        if len(response) < 15 or any(r in response.lower() for r in _REFUSAL):
            print("  [llm] Response rejected, using regex fallback")
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

    intents         = detect_intent(query)
    missing_intents = [f for f in intents if f not in all_info]
    run_pass2 = (len(missing_intents) >= 2) or (not intents and not all_info)

    if run_pass2 and chunks:
        from collections import Counter
        case_votes = Counter(c.get("case", "") for c in chunks if c.get("case", ""))
        best_case  = case_votes.most_common(1)[0][0] if case_votes else ""
        top_file = next(
            (c["filename"] for c in chunks if c.get("case", "") == best_case),
            chunks[0]["filename"]
        )
        print(f"  [pass-2] missing={missing_intents!r}. Scanning: {top_file}")
        for chunk in retrieve_all_chunks_for_file(top_file):
            _scan_chunk(chunk)
            if missing_intents and all(f in all_info for f in missing_intents):
                print("  [pass-2] All fields found.")
                break

    llm_result = _llm_answer(query, reranked_chunks, extracted_fields=all_info or None)
    if llm_result:
        src_str = sources[0].split("/")[-1].strip() if sources else "the indexed document"
        return f"**Source:** {src_str}\n\n{llm_result}"

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
        header = f"### {short}  (relevance: {chunk['score']:.0%})\n"
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
    query, correction_note = normalize_query(query)

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

    asked_plot = extract_plot_number_from_query(query)
    if asked_plot:
        plot_pattern = _build_plot_filter_pattern(asked_plot)
        filtered = [c for c in context_chunks if re.search(plot_pattern, c["text"], re.IGNORECASE)]
        if filtered:
            context_chunks = filtered
            print(f"  [plot-filter] {len(filtered)} chunks for plot '{asked_plot}'")
        else:
            print(f"  [plot-filter] Plot '{asked_plot}' not found in any chunk.")
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
        "context_count": len(context_chunks)
    }

    answer_text = format_answer(query, context_chunks)
    if correction_note:
        answer_text = correction_note + "\n\n" + answer_text
    yield {"type": "content", "content": answer_text}

    yield {"type": "detail", "content": build_detail_text(context_chunks)}

    seen_fnames: set = set()
    deduped: list    = []
    for c in context_chunks:
        fn = c["filename"].replace("\\", "/").split("/")[-1]
        if fn not in seen_fnames:
            seen_fnames.add(fn)
            deduped.append(c)

    yield {"type": "sources_detail", "content": deduped}
