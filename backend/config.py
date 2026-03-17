import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PDF_ROOT  = os.path.join(BASE_DIR, "Sample_PDFs")
# PDF_ROOT = r"D:\ChatBot\Sample_PDFs"
PDF_DIR   = os.path.join(BASE_DIR, "pdfs")
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")

EMBED_MODEL    = "BAAI/bge-m3"
OLLAMA_MODEL   = "mistral"

CHUNK_SIZE    = 400  # Reduced from 900 — smaller chunks = better BM25 scores & focused vector retrieval
CHUNK_OVERLAP = 80   # Reduced proportionally

TOP_K         = 50
RERANK_TOP_N  = 20
RERANK_MODEL  = "BAAI/bge-reranker-large"

COLLECTION_NAME = "land_docs"

# --- On-prem / security flags ------------------------------------------------

# Enable or disable the local LLM Fallback (Phi-3-mini fallback for unstructured queries)
# Set to False to ensure strict deterministic extraction and avoid latency.
# On first run or CPU-only systems, set to False to avoid CUDA/GPU errors
LLM_ENABLED = True  # Enabled for Mistral via Ollama

# Timeout (seconds) for LLM model loading. If exceeds this, skips LLM.
LLM_LOAD_TIMEOUT = 10

# Suppress CUDA/GPU warnings on CPU-only systems
SUPPRESS_CUDA_WARNINGS = True

# Set False to skip cross-encoder reranking (faster, less accurate).
RERANK_ENABLED = True

# PostgreSQL structured metadata DB — deterministic fast path for structured queries.
DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://postgres:12345678@localhost/land_db")

# SQLite structured metadata DB (Legacy, to be removed)
SQLITE_DB_PATH = os.path.join(BASE_DIR, "metadata.db")

# Append-only audit trail (query hash + doc_ids, NO raw text).
AUDIT_LOG_PATH = os.path.join(BASE_DIR, "audit.jsonl")

# When True: sets TRANSFORMERS_OFFLINE=1 so sentence-transformers never phone home.
# Requires models already downloaded locally (or pre-cached in ~/.cache/huggingface).
# You can enable it by setting the env var OFFLINE_MODE=1 (or TRUE/YES).
OFFLINE_MODE = os.environ.get("OFFLINE_MODE", "False").strip().lower() in ("1", "true", "yes")

# Apply offline + telemetry-disable env vars immediately on import
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

if OFFLINE_MODE:
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

# Suppress CUDA/GPU warnings if not available
if SUPPRESS_CUDA_WARNINGS:
    import warnings
    warnings.filterwarnings('ignore', category=UserWarning)
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Force CPU mode if CUDA not available

# --- New Land Record Specific Configs ---
OCR_ENABLED = True
OCR_LANG = 'hi'  # Support for Hindi and English

# For handling Requirement 1 (Clarification questions)
CONFIDENCE_THRESHOLD = 0.40

# Keywords for Hinglish intent detection
LEGAL_KEYWORDS = ["stay", "court", "recovery", "rc", "vivad", "kanooni"]
LOAN_KEYWORDS = ["loan", "bank", "girvi", "mortgage", "rin"]
AREA_KEYWORDS = ["hectare", "bigha", "area", "rakba", "mismatch"]
