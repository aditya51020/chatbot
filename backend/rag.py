import os
import re
import uuid
from typing import Optional

import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder

from config import (
    CHROMA_DIR, EMBED_MODEL,
    TOP_K, COLLECTION_NAME, RERANK_TOP_N, RERANK_MODEL,
    LLM_ENABLED, RERANK_ENABLED, OFFLINE_MODE,
)

# singletons — loaded once, reused for every query
_embedder: Optional[SentenceTransformer] = None
_chroma_client = None
_collection = None
_llm_model = None
_cross_encoder = None
_bm25_index = None
_bm25_corpus = None


def clean_for_llm(text: str) -> str:
    """
    Remove PDF extraction garbage before text is sent to the LLM.
    Keeps: Devanagari, Latin letters, digits, common punctuation, currency symbols.
    Strips: stray unicode blocks, control chars, excessive punctuation noise.
    """
    # Keep Devanagari (\u0900-\u097F), Latin (a-zA-Z), digits, spaces,
    # common punctuation, and currency symbols
    text = re.sub(r'[^\u0900-\u097F\u0966-\u096Fa-zA-Z0-9\s₹.,;:/()\'"\-–\n]', ' ', text)
    # Collapse runs of spaces (but keep newlines for structure)
    text = re.sub(r'[ \t]{2,}', ' ', text)
    # Remove lines that are nothing but punctuation/spaces
    lines = [ln.strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if len(re.sub(r'[\W\d]', '', ln)) > 1]
    return '\n'.join(lines).strip()


def get_embedder() -> Optional[SentenceTransformer]:
    global _embedder
    if _embedder is None:
        print(f"[EMBEDDER] Loading model: {EMBED_MODEL}, OFFLINE_MODE={OFFLINE_MODE}")
        try:
            # When offline mode is enabled, the HuggingFace/transformers libs will
            # refuse network access and only load from the local cache.
            _embedder = SentenceTransformer(
                EMBED_MODEL,
                device="cpu",
                local_files_only=OFFLINE_MODE,
            )
            dim = _embedder.get_sentence_embedding_dimension()
            print(f"[EMBEDDER] Successfully loaded: {EMBED_MODEL}")
            print(f"[EMBEDDER] Embedding dimension: {dim}")
        except Exception as e:
            print(f"[ERROR] Failed to load embedding model: {e}")
            print("        * If you have no internet, set OFFLINE_MODE=1 and ensure the model is cached locally.")
            print("        * If you do have internet, verify connectivity or choose a smaller locally installed model.")
            _embedder = None
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

def index_chunks(chunks: list[dict]):
    if not chunks:
        print("!!! No chunks found to index !!!")
        return 0
    
    collection = get_collection()
    embedder = get_embedder()

    if embedder is None:
        print("[ERROR] Cannot index chunks because embedding model isn't available.")
        print("        Ensure you have internet access or set OFFLINE_MODE=1 with a cached model.")
        return 0

    ids = [str(uuid.uuid4()) for _ in chunks]

    # FIX: Change 'text' to 'content' to match pdf_processor.py
    texts = [c["content"] for c in chunks]

    metadatas = [c.get("metadata", {}) for c in chunks]

    print(f"[INDEX] Encoding {len(texts)} chunks with embedding model...")
    # normalize_embeddings=True is REQUIRED for bge-m3 — without it cosine scores
    # collapse to ~0.05 instead of the expected 0.7–0.95 range
    embeddings = embedder.encode(
        texts,
        batch_size=16,
        show_progress_bar=True,
        normalize_embeddings=True,   # ← critical for correct cosine similarity
    ).tolist()
    print(f"[INDEX] Generated {len(embeddings)} embeddings")
    print(f"[INDEX] Sample embedding dimension: {len(embeddings[0])}")

    print(f"[INDEX] Storing {len(ids)} chunks in ChromaDB (batched)...")
    BATCH_SIZE = 200
    total_added = 0
    for i in range(0, len(ids), BATCH_SIZE):
        batch_end = min(i + BATCH_SIZE, len(ids))
        collection.add(
            ids=ids[i:batch_end],
            embeddings=embeddings[i:batch_end],
            documents=texts[i:batch_end],
            metadatas=metadatas[i:batch_end],
        )
        total_added += batch_end - i
        print(f"[INDEX]   Stored batch {i // BATCH_SIZE + 1}: {total_added}/{len(ids)} chunks")
    print(f"[INDEX] Successfully indexed {total_added} chunks")
    print(f"ChromaDB: Successfully saved {total_added} chunks.")
    return total_added


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
    
    # Aggressively clean OCR garbage out to improve embedding search space
    text = re.sub(r'[^\u0900-\u097F0-9A-Za-z₹.,:\-\n]', ' ', text)
    # Collapse runaway spaces, but preserve newlines
    text = re.sub(r'[ \t]{2,}', ' ', text)

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
    embedder = get_embedder()
    if embedder is None:
        print("[WARN] _query_chroma called but no embedding model is available.")
        return []
    emb = embedder.encode([query_text])[0].tolist()
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
        # Defensive strip: remove any residual metadata tokens from old index entries
        # clean_ocr_text() strips [Source:...], [Category:...], [Page N] patterns
        clean_doc, _ = clean_ocr_text_scored(doc)
        corpus.append({
            "text":     clean_doc if clean_doc else doc,  # fallback to raw if cleaning empties it
            "filename": meta.get("filename", "unknown"),
            "category": meta.get("category", ""),
            "case":     meta.get("case_number", ""),
            "score":    0.0,
        })

    tokenized = [c["text"].lower().split() for c in corpus]
    _bm25_index = BM25Okapi(tokenized)
    _bm25_corpus = corpus
    print(f"  BM25 index built: {len(corpus)} chunks.")

    # Diagnostic: print the first chunk to confirm index is clean
    if corpus:
        preview = corpus[0]["text"][:200].replace("\n", " ")
        print(f"  [BM25 SANITY] First chunk preview: {preview}...")


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

def extract_structured_data_from_chunks(chunks: list[dict], filename: str) -> dict:
    """
    Analyzes OCR text chunks to populate PostgreSQL columns.
    Matches Requirement 5 (Area) and Requirement 2 (Loan).
    """
    full_text = " ".join([c['text'] for c in chunks])
    
    # Simple regex/keyword logic to find data in the Hindi/English mix
    data = {
        "filename": filename,
        "plot_number": extract_plot_number_from_query(full_text), # reuse your existing helper
        "has_active_loan": "Yes" if any(x in full_text for x in ["ऋण", "loan", "bank"]) else "No",
        "has_stay_order": "Yes" if "स्टे" in full_text or "stay" in full_text.lower() else "No",
    }
    
    # Logic for Requirement 5 (Hectare/Area)
    area_match = re.search(r"(\d+\.\d+)\s*(hectare|हेक्टेयर|वर्गगज)", full_text)
    if area_match:
        data["land_area_recorded"] = float(area_match.group(1))
        
    return data

def retrieve_context(
    query: str,
    filename_filter: str | None = None,
    category_filter: str | None = None,
) -> list[dict]:
    """Hybrid BM25 + vector search fused via Reciprocal Rank Fusion."""
    intents = detect_intent(query)
    queries = [query] + [f"{intent} {query}" for intent in intents]

    # --- vector retrieval (optional, requires embedding model) ---
    vector_chunks: list[dict] = []
    embedder = get_embedder()
    if embedder is not None:
        print(f"[VECTOR] Starting vector search with {len(queries)} query variants")
        try:
            all_embeddings = embedder.encode(
                queries,
                batch_size=len(queries),
                show_progress_bar=False,
                normalize_embeddings=True,   # ← must match how index embeddings were built
            )

            # vector retrieval — one Chroma call per variant, no extra encode() calls
            seen_vector: set = set()
            for idx, emb in enumerate(all_embeddings):
                results = _query_chroma_emb(emb.tolist(), TOP_K, filename_filter=filename_filter, category_filter=category_filter)
                for chunk in results:
                    key = (chunk["filename"], chunk["text"][:60])
                    if key not in seen_vector:
                        seen_vector.add(key)
                        boost  = _work_name_boost(chunk["text"], query)
                        boost += _plot_id_boost(chunk["text"], query)
                        chunk["score"] = min(1.0, chunk["score"] + boost)
                        vector_chunks.append(chunk)
            vector_chunks.sort(key=lambda c: c["score"], reverse=True)
            # ── capture raw cosine top score NOW, before RRF loop overwrites chunk["score"]
            _raw_vector_top_score = vector_chunks[0]["score"] if vector_chunks else 0.0
            print(f"[VECTOR] Retrieved {len(vector_chunks)} chunks from vector search")
            if vector_chunks:
                print(f"[VECTOR] Top score: {_raw_vector_top_score:.3f}")
        except Exception as e:
            print(f"[WARN] Embedding model failed during query encoding: {e}")
            print("       Falling back to BM25-only retrieval.")
    else:
        print("[WARN] No embedding model available — using BM25-only retrieval.")
    _raw_vector_top_score: float = vector_chunks[0]["score"] if vector_chunks else 0.0

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

    # RRF (Reciprocal Rank Fusion)
    # Rank-based fusion: immune to score distribution shifts
    # Uses only rank position, not score magnitude
    K = 60  # Standard RRF constant
    
    rrf_scores: dict[tuple, float] = {}
    chunk_map:  dict[tuple, dict]  = {}

    # Add vector results by rank (rank position, not score value)
    for rank, c in enumerate(vector_chunks, 1):
        key = (c["filename"], c["text"][:80])
        score = 1.0 / (K + rank)
        rrf_scores[key] = rrf_scores.get(key, 0.0) + score
        chunk_map[key] = c

    # Add BM25 results by rank
    for rank, c in enumerate(bm25_chunks, 1):
        key = (c["filename"], c["text"][:80])
        score = 1.0 / (K + rank)
        rrf_scores[key] = rrf_scores.get(key, 0.0) + score
        if key not in chunk_map:
            chunk_map[key] = c

    # Sort by combined RRF scores
    fused = sorted(
        chunk_map.values(),
        key=lambda c: rrf_scores.get((c["filename"], c["text"][:80]), 0),
        reverse=True
    )[:TOP_K]
    
    # Attach RRF fusion scores to chunks for display
    for chunk in fused:
        key = (chunk["filename"], chunk["text"][:80])
        chunk["score"] = rrf_scores.get(key, 0)

    # person-name filter — avoid pulling unrelated records when a name is in the query
    person_words = _extract_person_names(query)
    if person_words and not filename_filter:
        filtered = [c for c in fused if any(pw in c["text"].lower() for pw in person_words)]
        if filtered:
            print(f"  [name-filter] kept {len(filtered)}/{len(fused)} chunks")
            fused = filtered

    # ── Retrieval diagnostics ─────────────────────────────────────────────
    # IMPORTANT: _raw_vector_top_score is captured BEFORE the RRF loop below
    # mutates chunk["score"] in-place. Reading vector_chunks[0]["score"] here
    # would give the RRF score (~0.046), not the cosine similarity (~0.930).
    vector_hits    = len(vector_chunks)
    raw_vector_top = _raw_vector_top_score                 # cosine sim 0–1
    rrf_top        = fused[0]["score"] if fused else 0.0   # RRF score max ~0.033
    print(f"\n  [retrieval] vector_hits={vector_hits} raw_vector_top={raw_vector_top:.3f} "
          f"rrf_top={rrf_top:.4f} total_fused={len(fused)}")
    for i, chunk in enumerate(fused[:3]):
        rrf_s   = chunk.get("score", 0)
        preview = chunk["text"][:120].replace("\n", " ")
        fname   = chunk.get("filename", "unknown").split("/")[-1]
        print(f"    Chunk {i+1} (rrf={rrf_s:.4f}, file={fname}): {preview}...")

    # ── Quality filter using RAW VECTOR score (not RRF) ──────────────────
    # RRF scores are inherently tiny (1/(60+rank) ≈ 0.016 for rank 1).
    # Comparing 0.049 against 0.15 will always fail. Use cosine similarity instead.
    # 0.50 = "at least somewhat semantically related"
    # 3 chunks minimum to give LLM enough context
    MIN_VECTOR_SCORE = 0.50
    MIN_CHUNK_COUNT  = 3
    if raw_vector_top < MIN_VECTOR_SCORE or len(fused) < MIN_CHUNK_COUNT:
        print(f"  [retrieval] LOW CONFIDENCE — raw_vector_top={raw_vector_top:.3f} "
              f"chunks={len(fused)} → returning empty")
        return []   # caller returns 'जानकारी उपलब्ध नहीं है'

    # Return top 6 to LLM
    return fused[:6]


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
        "Plot Number":       r"(?i)(?:भूखंड|भूखण्ड|bhu\s*khand|plot|पट्टी)\s*(?:sankhya|संख्या|no\.?|number|क्रमांक)?\s*[:\-.=]*\s*([0-9A-Za-z][0-9A-Za-z\-/]{0,8}?)(?=[\s,।/\n]|$)",
        "Plot Size":         r"(?i)(?:क्षेत्रफल|kshetrafal|area|आकार|akar|size|माप|maap|साईज)\s*(?:का\s*)?[:\-=\/.]*\s*([0-9.,]+)(?:\s*(?:sqm|sq\.?\s*m|वर्ग\s*मीटर|वर्ग|gaj|गज|sq\.?\s*ft|sq\.?\s*yard|bigha|बिघा|acre|एकड़))?",
        "Deposit Amount":    r"(?i)(?:अमानत\s*राशि|वाणिज्यिक|जमाराशि|jama\s*rashi|deposit\s*amount|रकम|राशि|शुल्क)\s*[:\-=]*\s*(?:Rs\.?|₹|INR|रु\.?|रू\.?)?\s*([\d,]+(?:\.\d+)?)",
        "Scheme Name":       r"(?i)(?:योजना|yojana|scheme)\s*(?:का\s*नाम|name)?\s*[:\-.=]*\s*([^,\n\|]{2,60}?)(?=\n|,|\||\d|$)",
        "Second Party":      r"(?i)(?:दूसरा\s*पक्ष|second\s*party|party\s*2|दूसरा|पक्ष)[\s:.\-=]*([A-Za-z\s\u0900-\u097F]{2,60}?)(?=\n|,|$)",
        "Owner":             r"(?i)(?:मालिक|malik|owner|धारक|darak|पट्टेदार|pattadar|स्वामी|malik|malik)\s*[:\-.=]*\s*([A-Za-z\s\u0900-\u097F]{2,60}?)(?=\n|,|$)",
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

    # Additional extraction for land record specific fields
    if "Second Party" not in found:
        sp = re.search(
            r"(?i)(?:दूसरा\s*पक्ष|second\s*party|party\s*2)[\s:.\-=]*([A-Za-z\s\u0900-\u097F]{2,60}?)(?=\n|,|$)",
            text, re.IGNORECASE)
        if sp:
            val = sp.group(1).strip()
            if len(val) >= 2 and len([w for w in val.split() if w]):
                found["Second Party"] = val[:60]

    if "Owner" not in found:
        owner = re.search(
            r"(?i)(?:मालिक|malik|owner|स्वामी|swami)[\s:.\-=]*([A-Za-z\s\u0900-\u097F]{2,60}?)(?=\n|,|$)",
            text, re.IGNORECASE)
        if owner:
            val = owner.group(1).strip()
            if len(val) >= 2:
                found["Owner"] = val[:60]

    if "Plot Size" not in found:
        # Improved plot size extraction handling units better
        m_size = re.search(
            r'(?i)([\d.,]+)\s*(?:वर्ग\s*)?(?:sqm|sq\.?\s*m|मीटर|gaj|गज|sq\.?\s*ft|sq\.?\s*yard|bigha|बिघा|सेंट|cent)',
            text)
        if m_size:
            found["Plot Size"] = m_size.group(1)

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


# Hinglish → Hindi token mapping for query normalization
_HINGLISH_MAP = [
    # Filler words to remove
    (r'\bkitni\s+hai\b',   ''),
    (r'\bkya\s+hai\b',     ''),
    (r'\bkitna\s+hai\b',   ''),
    (r'\bkitne\s+hai\b',   ''),
    (r'\bplease\b',        ''),
    (r'\bbatao\b',         ''),
    (r'\bbata\s+do\b',     ''),
    # Hinglish → Hindi
    (r'\bki\b',            'की'),
    (r'\bka\b',            'का'),
    (r'\bke\b',            'के'),
    (r'\bhai\b',           'है'),
    (r'\bkitni\b',         'कितनी'),
    (r'\bkitna\b',         'कितना'),
    (r'\bkya\b',           'क्या'),
    (r'\bmein\b',          'में'),
    (r'\byojana\b',        'योजना'),
    (r'\byojna\b',         'योजना'),
    (r'\bnilami\b',        'नीलामी'),
    (r'\bamanat\b',        'अमानत'),
    (r'\brashi\b',         'राशि'),
    (r'\bbhukhand\b',      'भूखंड'),
    (r'\bbhukand\b',       'भूखंड'),
    (r'\baawaasiya\b',     'आवासीय'),
]


def normalize_query(query: str) -> tuple[str, str]:
    """
    1. Apply Hindi spelling corrections.
    2. Convert Hinglish tokens to Hindi.
    3. Strip filler words (kitni hai, kya hai, please...).
    Returns (normalized_query, user_note).
    """
    note = ''
    # Step 1: spelling corrections
    for pattern, replacement, user_note in _HINDI_CORRECTIONS:
        new_q, n = re.subn(pattern, replacement, query)
        if n > 0:
            query = new_q
            if user_note and not note:
                note = user_note
    # Step 2: Hinglish → Hindi + filler strip
    for pattern, replacement in _HINGLISH_MAP:
        query = re.sub(pattern, replacement, query, flags=re.IGNORECASE)
    # Collapse extra whitespace
    query = re.sub(r'\s+', ' ', query).strip()
    return query, note


def generate_query_variants(query: str) -> list[str]:
    """
    Generate up to 4 search query variants from the normalized query.
    Used to improve multi-perspective retrieval coverage.
    """
    variants = [query]

    # Keyword-only variant (extract important words)
    keywords = re.findall(
        r'[\u0900-\u097F][\u0900-\u097F\s]{3,}[\u0900-\u097F]|\b\w{5,}\b',
        query
    )
    if keywords:
        variants.append(' '.join(keywords[:6]))

    # English-only variant
    en_only = re.sub(r'[\u0900-\u097F]', '', query).strip()
    if en_only and en_only != query:
        variants.append(en_only)

    # Amount-focused reformulation for deposit/money queries
    if any(w in query for w in ['राशि', 'अमानत', 'deposit', 'रकम', 'amount']):
        variants.append(f"अमानत राशि नीलामी योजना")

    return list(dict.fromkeys(v.strip() for v in variants if v.strip()))[:4]


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
    if re.search(r'jama|जमा|rashi|राशि|deposit|amount|rakam|रकम|shulk|शुल्क|kitni|amanat|अमानत|biana|बयाना', q):
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
        "Date of Casting":   "📅 **Casting Date**",
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
        paragraphs.append(
            "> ⚠️ Answer not found in document. (Query intent not recognized)."
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
    """Rewrite user chat into an optimized RAG search query."""
    if intent == "list_bidders":
        clean = re.sub(r'(?i)sare\s*bidder|ka\s*naam\s*do|batao|list|kaun|participation', '', query)
        clean = re.sub(r'[^a-zA-Z0-9\s]', '', clean).strip()
        if clean:
            return f"Find all bidder names and construction companies related to: {clean}"
        return "List of all participating bidders agencies contractors"
    return query


# ---------------------------------------------------------------------------
# Local LLM Fallback (Mistral-7B-Instruct via llama-cpp-python)
# ---------------------------------------------------------------------------

def _llm_answer(query: str, chunks: list[dict], extracted_fields: dict = None,
                force_bidder_mode: bool = False, custom_prompt: str = None) -> str | None:
    try:
        import ollama
        from config import OLLAMA_MODEL, LLM_ENABLED

        if not LLM_ENABLED:
            return None

        # ── Deduplicate chunks ────────────────────────────────────────────
        seen_content: set = set()
        deduped_chunks: list = []
        for chunk in chunks:
            text_val = chunk.get("text") or chunk.get("content") or ""
            raw_text = clean_ocr_text(text_val)
            norm = re.sub(r'\W+', '', raw_text.lower())
            if norm not in seen_content:
                seen_content.add(norm)
                deduped_chunks.append(chunk)

        # ── Build context — cleaned before sending to LLM ─────────────────
        context_parts = []
        for i, chunk in enumerate(deduped_chunks[:6], 1):
            text_val = chunk.get("text") or chunk.get("content") or ""
            # clean_for_llm strips PDF garbage (stray chars, broken unicode)
            text = clean_for_llm(clean_ocr_text(text_val))[:600]
            src  = chunk.get("filename", "unknown").replace("\\", "/").split("/")[-1]
            meta = chunk.get("metadata") or {}
            page = meta.get("page_number", "?") if isinstance(meta, dict) else "?"
            context_parts.append(f"[DOCUMENT EXTRACT {i}]\nSource: {src} | Page: {page}\n{text}\n")
        context_str = "\n\n".join(context_parts)

        # ── Build prompt ───────────────────────────────────────────────────
        if custom_prompt:
            prompt = custom_prompt
        elif force_bidder_mode:
            prompt = (
                "You are a government document assistant.\n"
                "Extract ALL bidder/contractor/agency names from the context below.\n"
                "List each name on a separate line.\n"
                "If NO names are found, write: कोई बोलीदाता नहीं मिला\n\n"
                f"Context:\n{context_str}\n\n"
                f"Question: {query}\n"
                "Answer:"
            )
        else:
            fields_str = ""
            if extracted_fields:
                clean_fields = {k: v for k, v in extracted_fields.items() if k != "Source"}
                if clean_fields:
                    fields_str = (
                        "Already extracted facts:\n" +
                        "\n".join(f"  {k}: {v}" for k, v in clean_fields.items()) + "\n\n"
                    )
            
            prompt = (
                "You are a document question-answering assistant. Answer in Hindi.\n\n"
                "Follow these rules strictly:\n"
                "1. Answer ONLY using the information provided in the CONTEXT.\n"
                "2. Do NOT use outside knowledge.\n"
                "3. If the answer is not clearly present in the context, say:\n"
                "   जानकारी उपलब्ध नहीं है\n"
                "4. If numbers or dates appear in the context, copy them exactly.\n"
                "5. Do NOT guess or infer missing data.\n"
                "6. Be concise and factual (1-2 lines).\n\n"
                "---\n\n"
                "CONTEXT:\n"
                f"{fields_str}"
                f"{context_str}\n\n"
                "---\n\n"
                "QUESTION:\n"
                f"{query}\n\n"
                "---\n\n"
                "INSTRUCTIONS:\n"
                "* Extract the relevant information from the context.\n"
                "* If multiple values exist, return the most relevant one.\n"
                "* If the answer cannot be found, say the document does not contain it.\n\n"
                "ANSWER:"
            )

        print(f"  [llm] Generating answer via Ollama ({OLLAMA_MODEL})...")
        try:
            output = ollama.chat(
                model=OLLAMA_MODEL,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                options={
                    "temperature": 0.05,
                    "num_ctx": 4096
                }
            )
            response = output.get("message", {}).get("content", "").strip()
            if not response:
                return None
            print(f"  [llm] Done ({len(response)} chars)")
        except Exception as gen_err:
            print(f"  [llm] Generation failed: {gen_err}")
            return None

        _REFUSAL = ('i cannot', "i'm sorry", "i don't know", 'not enough information',
                    'maafi chahta hoon', 'jankari nahi', 'not found in document',
                    'yeh jankari documents mein nahi')
        if len(response) < 15 or any(r in response.lower() for r in _REFUSAL):
            print("  [llm] LLM refused — no clear answer found.")
            return "⚠️ Answer not found in the document."

        # Numeric hallucination guard (Relaxed: must have AT LEAST ONE grounded number if numbers are generated)
        money_matches = re.findall(r'₹?\s*\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d{4,}', response)
        try:
            from rag_utils import words_to_digits_string
            word_numbers = words_to_digits_string(response)
        except ImportError:
            word_numbers = []

        all_numbers = money_matches + word_numbers
        if all_numbers:
            context_digits_str = re.sub(r'[^\d]', '', context_str)
            found_grounded = False
            for mm in all_numbers:
                num_val = re.sub(r'[^\d]', '', mm)
                if len(num_val) >= 2 and num_val in context_digits_str:
                    found_grounded = True
                    break
            
            if not found_grounded:
                # None of the significant numbers generated by the LLM were in the context
                print(f"  [llm] Hallucination guard: None of generated numbers {all_numbers} in context")
                return "Answer not found in the document."

        if re.search(r'(?i)\bQuestion\s*:', response) or response.count('?') > 2:
            print("  [llm] Rejected: hallucinated Q&A pairs")
            return None

        return response

    except ImportError as ie:
        print(f"  [llm] llama-cpp-python not installed: {ie}")
        return None
    except Exception as e:
        print(f"  [llm] Unexpected error: {e}")
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
    """Refined entry point for Land Record Chatbot with Proactive Questioning."""
    from database import (
        query_by_plot, query_by_allottee, get_all_fields_for_file, write_audit
    )

    # 1. NORMALIZE & DETECT HINGLISH INTENT
    # This handles "plot number xyz is approved or not?" or "village name me kitne plot hai?"
    query, correction_note = normalize_query(query)
    intents = detect_intent(query)
    asked_plot = extract_plot_number_from_query(query)

    # -----------------------------------------------------------------------
    # PROACTIVE LOGIC: Requirement 1 (Self-Questioning)
    # -----------------------------------------------------------------------
    # If query is too short or lacks specific keywords, suggest critical land checks
    critical_keywords = ["stay", "loan", "rc", "mismatch", "recovery", "acquisition"]
    if len(query.split()) < 4 and not any(k in query.lower() for k in critical_keywords):
        proactive_msg = (
            f"Mujhe Plot/Khasra **{asked_plot if asked_plot else ''}** ki file mil gayi hai. \n\n"
            "Kya aap is property ke baare mein ye jaanna chahte hain:\n"
            "1. “Kya is khasra/plotno par kisi bhi prakaar ka stay order currently active hai?”\n"
            "2. “Is property ke khilaf kisi bhi prakar ki revenue recovery (RC) pending hai?”\n"
            "3. “Is land par abhi tak koi government acquisition notice issue hua hai?”"
        )
        yield {"type": "content", "content": proactive_msg}
        return

    # -----------------------------------------------------------------------
    # FAST PATH: SQL Structured Metadata Lookup
    # -----------------------------------------------------------------------
    db_rows = []
    if asked_plot:
        db_rows = query_by_plot(asked_plot)
    
    if db_rows:
        row = db_rows[0]
        # LOGIC FOR REQUIREMENT 5: Area Mismatch Detection
        # Compare DB value vs what user thinks (if mentioned in query)
        user_area_match = re.search(r"(\d+(\.\d+)?) hectare", query.lower())
        if user_area_match and row.get("land_area_recorded"):
            user_val = float(user_area_match.group(1))
            db_val = float(row["land_area_recorded"])
            if abs(user_val - db_val) > 0.01:
                correction_note = (
                    f"⚠️ **Data Mismatch Detected:** Aapne {user_val} hectare pucha hai, "
                    f"lekin sarkari record (PDF) mein ye **{db_val} hectare** dikh raha hai."
                )

        # Build answer from SQL
        fields = {k: v for k, v in row.items() if v and k not in ['id', 'file_hash']}
        table_data = [{**fields, "Source": row['filename'].split("/")[-1]}]
        answer_text = build_natural_answer(query, fields, [row['filename']], [])
        
        if correction_note:
            answer_text = correction_note + "\n\n" + answer_text

        yield {"type": "meta", "table": table_data, "sources": [row['filename']], "context_count": 0}
        yield {"type": "content", "content": answer_text}
        return

    # -----------------------------------------------------------------------
    # HYBRID PATH: For complex/historical queries (Requirement 1, 4, 8)
    # -----------------------------------------------------------------------
    context_chunks = retrieve_context(query, filename_filter, category_filter)

    if not context_chunks:
        yield {"type": "error", "content": "Records nahi mile. Kripya sahi se jaankaari likhein."}
        return

    # -----------------------------------------------------------------------
    # AGENTIC LLM LAYER: For Requirement 10 (Legal Risk) & Requirement 4 (SC/ST)
    # -----------------------------------------------------------------------
    # We use the LLM to 'reason' over the OCR text extracted from NS.pdf/CS.pdf
    # Aggressive truncation for Phi-3's 2048 token limit (Hindi chars eat more tokens)
    formatted_context = ""
    for i, c in enumerate(context_chunks[:3], 1):
        text_snippet = (c.get("text") or c.get("content") or "")[:300].strip()
        formatted_context += f"[Doc {i}: {c.get('filename', 'Unknown')}]\n{text_snippet}\n\n"

    llm_prompt = f"""Context:
{formatted_context}
User Query: {query}
Role: Expert Land Records & Legal Document Analyzer.
Instructions:
1. Analyze OCR text. If info not found, say "Mujhe ye jaankari nahi mili."
2. Check for Legal Risk (Stay, Court, Vivad).
3. Explain local terms (Khasra, Mutation).
4. Check SC/ST ownership transfer restrictions.
5. Answer in Hinglish.
"""
    
    # Try LLM first (if enabled), fall back to regex extraction
    answer_text = _llm_answer(query, context_chunks, custom_prompt=llm_prompt)
    
    # Fallback to regex extraction if LLM is disabled or failed
    if answer_text is None:
        all_info = {}
        sources = []
        for chunk in context_chunks[:5]:  # Process top 5 chunks
            text = clean_ocr_text(chunk.get("text", ""))
            if text:
                info = extract_key_info(text)
                for k, v in keyword_scan(text).items():
                    if k not in info:
                        info[k] = v
                for k, v in info.items():
                    if k not in all_info:
                        all_info[k] = v
            
            fname = chunk["filename"]
            parts = fname.replace("\\", "/").split("/")
            display = " / ".join(parts[-3:]) if len(parts) >= 3 else fname
            if display not in sources:
                sources.append(display)
        
        answer_text = build_natural_answer(query, all_info, sources, context_chunks)

    yield {
        "type": "meta",
        "sources": list(set([c["filename"].split("/")[-1] for c in context_chunks])),
        "context_count": len(context_chunks),
    }
    
    yield {"type": "content", "content": answer_text}
