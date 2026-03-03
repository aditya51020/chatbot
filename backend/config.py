import os

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))

# Root folder where all land PDFs are stored (recursive scan)
PDF_ROOT    = os.path.join(BASE_DIR, "Sample_PDFs", "Sample_PDFs")

# Upload folder (for manual uploads via UI)
PDF_DIR     = os.path.join(BASE_DIR, "pdfs")

CHROMA_DIR  = os.path.join(BASE_DIR, "chroma_db")

# ── Embedding model (local, downloaded once via sentence-transformers) ─────
EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"  # ~420MB, understands Hindi natively

# ── GPT4All model (downloaded automatically first run) ────────────────────
GPT4ALL_MODEL = "Phi-3-mini-4k-instruct.Q4_0.gguf"

# ── RAG settings ──────────────────────────────────────────────────────────
CHUNK_SIZE    = 200   # reduced: tighter chunks = more focused retrieval per page
CHUNK_OVERLAP = 40    # overlap preserves context between adjacent chunks
TOP_K         = 6     # tighter: fewer but more relevant chunks per query

# ── ChromaDB collection name ───────────────────────────────────────────────
COLLECTION_NAME = "land_docs"
