import os
import uuid
from typing import Optional

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from gpt4all import GPT4All

from config import (
    CHROMA_DIR, EMBED_MODEL, GPT4ALL_MODEL,
    TOP_K, COLLECTION_NAME
)

# ── Lazy-loaded singletons ─────────────────────────────────────────────────
_embedder: Optional[SentenceTransformer] = None
_llm: Optional[GPT4All] = None
_chroma_client = None
_collection = None


def get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        print("🔄 Loading embedding model (first run may take a moment)...")
        _embedder = SentenceTransformer(EMBED_MODEL)
        print("✅ Embedding model loaded")
    return _embedder


def get_llm() -> GPT4All:
    global _llm
    if _llm is None:
        print(f"🔄 Loading LLM: {GPT4ALL_MODEL} (first run downloads ~2GB)...")
        # Increase context window to 4096 to handle larger documents
        _llm = GPT4All(GPT4ALL_MODEL, allow_download=True, n_ctx=4096)
        print("✅ LLM loaded")
    return _llm


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
    """Add text chunks to ChromaDB. Returns number of chunks added."""
    if not chunks:
        return 0

    collection = get_collection()
    embedder   = get_embedder()

    texts     = [c["text"] for c in chunks]
    # Store all available metadata fields
    metadatas = []
    for c in chunks:
        metadatas.append({
            "filename":    c.get("filename", "unknown"),
            "chunk_index": c.get("chunk_index", 0),
            "category":    c.get("category", ""),
            "case_number": c.get("case_number", ""),
            "doc_type":    c.get("doc_type", ""),
        })
    ids = [str(uuid.uuid4()) for _ in chunks]

    print(f"  🔢 Generating embeddings for {len(texts)} chunks...")
    embeddings = embedder.encode(texts, show_progress_bar=True).tolist()

    collection.add(documents=texts, embeddings=embeddings, metadatas=metadatas, ids=ids)
    print(f"  ✅ Indexed {len(texts)} chunks into ChromaDB")
    return len(texts)


def clear_collection():
    """Delete and recreate the ChromaDB collection."""
    global _collection, _chroma_client
    collection = get_collection()
    _chroma_client.delete_collection(COLLECTION_NAME)
    _collection = _chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )
    return True


def get_indexed_docs() -> list[str]:
    """Return list of unique filenames already indexed."""
    collection = get_collection()
    results = collection.get(include=["metadatas"])
    if not results["metadatas"]:
        return []
    filenames = list({m["filename"] for m in results["metadatas"] if "filename" in m})
    return sorted(filenames)


def get_chunk_count() -> int:
    return get_collection().count()


# Minimum cosine similarity to consider a chunk relevant
RELEVANCE_THRESHOLD = 0.40   # lowered from 0.50 to improve recall


def clean_ocr_text(text: str) -> str:
    """Remove garbled OCR artifacts and junk lines from extracted PDF text."""
    import re
    # Remove metadata headers injected during indexing
    text = re.sub(r'\[Category:[^\]]+\]\s*', '', text)
    text = re.sub(r'\[Case:[^\]]+\]\s*', '', text)
    text = re.sub(r'\[Type:[^\]]+\]\s*', '', text)
    text = re.sub(r'\[Page \d+\]\n?', '', text)

    lines = text.split('\n')
    clean = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # Skip lines that are mostly symbols/garbage
        # Count readable chars (letters, digits, common punctuation)
        readable = sum(1 for c in line if c.isalnum() or c in ' .,;:-/()[]')
        if len(line) > 3 and readable / max(len(line), 1) < 0.45:
            continue  # garbled line — skip it
        # Skip very short meaningless lines
        if len(line) < 4:
            continue
        clean.append(line)

    return '\n'.join(clean).strip()


def retrieve_context(query: str) -> list[dict]:
    """Retrieve TOP_K most relevant chunks for a query, filtered by relevance score."""
    collection = get_collection()
    if collection.count() == 0:
        return []

    embedder = get_embedder()
    query_embedding = embedder.encode([query]).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=min(TOP_K, collection.count()),
        include=["documents", "metadatas", "distances"]
    )

    chunks = []
    seen_files = set()   # deduplicate — 1 best chunk per source file
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):
        score = round(1 - dist, 3)
        if score < RELEVANCE_THRESHOLD:
            continue
        fname = meta.get("filename", "unknown")
        if fname in seen_files:
            continue   # already have a chunk from this file
        seen_files.add(fname)
        chunks.append({
            "text":     doc,
            "filename": fname,
            "category": meta.get("category", ""),
            "case":     meta.get("case_number", ""),
            "score":    score
        })
        if len(chunks) >= 3:   # max 3 unique sources
            break
    return chunks


def extract_key_info(text: str) -> dict:
    """Attempt to extract common land record and lab report fields for table display."""
    import re
    info = {}
    
    # Common Patterns (Hindi/English/Hybrid)
    patterns = {
        "Case / Ref No":   r"(?i)(?:Case|Ref|S\. No\.|W\.O\.)[:\-\.\s]+([A-Z0-9\-_/\.]+)",
        "Agency Name":     r"(?i)(?:Agency|Contractor|Contracting Name)[:\-\s]+([A-Z\s\.&/]{5,})",
        "Name of Work":    r"(?i)(?:Name of Work|Works|Site)[:\-\s]+([A-Z0-9\s\.]{10,})",
        "Location / Plot": r"(?i)(?:Plot|Sector|Ward|Location)[:\-\s]+([A-Z0-9\-_/ ]+)",
        "Owner / Name":    r"(?i)(?:Owner|Shri|Smti|Patient)[:\-\s]+([A-Z\s\.]{5,})",
        "Date of Casting": r"(?i)(?:Date of Casting|Casting Date)[:\-\s]+(\d{2}[/\-\.]\d{2}[/\-\.]\d{2,4})",
        "Test Date":       r"(?i)(?:Date|Dated|Testing Date)[:\-\s]+(\d{2}[/\-\.]\d{2}[/\-\.]\d{2,4})",
    }
    
    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        if match:
            val = match.group(1).strip()
            # Basic cleanup of punctuation at the end
            val = re.sub(r'[;\:\-\s]+$', '', val)
            info[key] = val
            
    return info


def answer_query(query: str):
    """Reliable Factual RAG: Yields structured data and verbatim source snippets."""
    context_chunks = retrieve_context(query)

    if not context_chunks:
        yield {
            "type": "error",
            "content": "Is sawaal se related koi information indexed documents mein nahi mili. Kripya case number ya property ID ke saath dobara poochein."
        }
        return

    # 1. Extract structured table data
    table_data = []
    seen_sources = []
    
    for chunk in context_chunks:
        src = chunk["filename"]
        text = clean_ocr_text(chunk["text"])
        if not text: continue
        
        info = extract_key_info(text)
        if info:
            info["Source"] = src.split('/')[-1] if '/' in src else src
            table_data.append(info)

        parts = src.replace('\\', '/').split('/')
        display = ' / '.join(parts[-3:]) if len(parts) >= 3 else src
        if display not in seen_sources:
            seen_sources.append(display)

    # Yield meta data
    yield {
        "type": "meta",
        "table": table_data,
        "sources": seen_sources,
        "context_count": len(context_chunks)
    }

    # 2. Yield verbatim snippets (Factual & Fast)
    intro = "### 🔍 Verified Extraction\nMaine niche di gayi sahi jaankari extract ki hai:\n\n"
    yield {"type": "content", "content": intro}
    
    for i, chunk in enumerate(context_chunks):
        display = chunk["filename"].split('/')[-1]
        snippet = clean_ocr_text(chunk["text"])[:800]
        content = f"**{i+1}. {display}** (Relevance: {chunk['score']:.0%})\n{snippet}...\n\n---\n"
        yield {"type": "content", "content": content}

    # Final yield source details for preview in UI
    yield {"type": "sources_detail", "content": context_chunks}

