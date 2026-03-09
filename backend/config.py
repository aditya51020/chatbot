import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PDF_ROOT  = os.path.join(BASE_DIR, "Sample_PDFs")
PDF_DIR   = os.path.join(BASE_DIR, "pdfs")
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")

EMBED_MODEL   = "BAAI/bge-m3"
GPT4ALL_MODEL = "Phi-3-mini-4k-instruct.Q4_0.gguf"

CHUNK_SIZE    = 400
CHUNK_OVERLAP = 100

TOP_K         = 50
RERANK_TOP_N  = 20
RERANK_MODEL  = "BAAI/bge-reranker-large"

COLLECTION_NAME = "land_docs"

# --- On-prem / security flags ------------------------------------------------

# Enable or disable the local LLM Fallback (Phi-3-mini fallback for unstructured queries)
# Set to False to ensure strict deterministic extraction and avoid latency.
LLM_ENABLED = True

# Set False to skip cross-encoder reranking (faster, less accurate).
RERANK_ENABLED = True

# PostgreSQL structured metadata DB — deterministic fast path for structured queries.
DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://postgres:12345678@localhost/land_db")

# SQLite structured metadata DB (Legacy, to be removed)
SQLITE_DB_PATH = os.path.join(BASE_DIR, "metadata.db")

# Append-only audit trail (query hash + doc_ids, NO raw text).
AUDIT_LOG_PATH = os.path.join(BASE_DIR, "audit.jsonl")

# When True: sets TRANSFORMERS_OFFLINE=1 so sentence-transformers
# never phone home. Requires models already downloaded locally.
OFFLINE_MODE = False

# Apply offline + telemetry-disable env vars immediately on import
if OFFLINE_MODE:
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
