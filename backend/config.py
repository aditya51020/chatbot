import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PDF_ROOT  = os.path.join(BASE_DIR, "Sample_PDFs", "Sample_PDFs")
PDF_DIR   = os.path.join(BASE_DIR, "pdfs")
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")

EMBED_MODEL   = "paraphrase-multilingual-MiniLM-L12-v2"
GPT4ALL_MODEL = "Phi-3-mini-4k-instruct.Q4_0.gguf"

CHUNK_SIZE    = 400
CHUNK_OVERLAP = 80

TOP_K         = 20
RERANK_TOP_N  = 5

COLLECTION_NAME = "land_docs"
