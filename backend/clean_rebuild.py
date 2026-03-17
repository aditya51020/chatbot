"""
clean_rebuild.py
================
Run this ONCE after code fixes to:
  1. Delete old (polluted) indexes
  2. Re-ingest all PDFs with clean pipeline
  3. Print chunk diagnostics and BM25 sanity check

Usage:
    cd backend
    python clean_rebuild.py
"""

import os
import sys
import shutil

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Step 1: Delete all stale indexes ──────────────────────────────────────────
print("=" * 60)
print("STEP 1: Deleting stale indexes")
print("=" * 60)

chroma_dir = os.path.join(BASE_DIR, "chroma_db")
bm25_file  = os.path.join(BASE_DIR, "bm25_index.pkl")
emb_cache  = os.path.join(BASE_DIR, "embeddings_cache.pkl")

for path, label in [(chroma_dir, "chroma_db/"), (bm25_file, "bm25_index.pkl"), (emb_cache, "embeddings_cache.pkl")]:
    if os.path.isdir(path):
        shutil.rmtree(path)
        print(f"  ✓ Deleted directory: {label}")
    elif os.path.isfile(path):
        os.remove(path)
        print(f"  ✓ Deleted file: {label}")
    else:
        print(f"  – Not found (OK): {label}")

print()

# ── Step 2: Re-ingest all PDFs ────────────────────────────────────────────────
print("=" * 60)
print("STEP 2: Loading and indexing PDFs")
print("=" * 60)

try:
    from pdf_processor import load_all_pdfs
    from rag import index_chunks, clear_collection

    chunks = load_all_pdfs()
    
    if not chunks:
        print("\nERROR: No chunks produced. Check that Sample_PDFs directory has PDFs.")
        sys.exit(1)

    print(f"\n  Total chunks produced: {len(chunks)}")
    
    # Diagnostic: show sample chunk content (what goes into BM25/vector index)
    print("\n" + "=" * 60)
    print("STEP 3: Chunk diagnostics")
    print("=" * 60)
    
    lengths = [len(c["content"]) for c in chunks]
    avg_len = sum(lengths) / max(len(lengths), 1)
    print(f"  Chunk count : {len(chunks)}")
    print(f"  Avg length  : {avg_len:.0f} chars")
    print(f"  Max length  : {max(lengths)} chars")
    print(f"  Min length  : {min(lengths)} chars")
    
    print(f"\n  --- First chunk content (SHOULD NOT contain [Source:] or [Category:]) ---")
    first = chunks[0]["content"]
    print(f"  {first[:400]}")
    print(f"  ---")
    
    # Check for metadata pollution
    pollution_found = False
    for c in chunks[:20]:
        if "[Source:" in c["content"] or "[Category:" in c["content"] or "[Case:" in c["content"]:
            print(f"\n  ❌ POLLUTION DETECTED in chunk: {c['content'][:120]}")
            pollution_found = True
            break
    
    if not pollution_found:
        print(f"\n  ✓ No metadata pollution found in first 20 chunks — pipeline is CLEAN")
    
    print()

    # ── Step 3: Index into ChromaDB ───────────────────────────────────────────
    print("=" * 60)
    print("STEP 4: Indexing into ChromaDB + BM25")
    print("=" * 60)
    
    n = index_chunks(chunks)
    print(f"\n  ✓ Indexed {n} chunks into ChromaDB")

except Exception as e:
    import traceback
    print(f"\nFATAL ERROR during rebuild: {e}")
    traceback.print_exc()
    sys.exit(1)

print()
print("=" * 60)
print("REBUILD COMPLETE")
print("  The BM25 index will be built on first query.")
print("  Run the backend and test your query now.")
print("=" * 60)
