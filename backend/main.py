import os
import json
import shutil
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from config import PDF_DIR, PDF_ROOT, RERANK_ENABLED
from pdf_processor import load_all_pdfs, process_pdf
from rag import (
    index_chunks, answer_query, get_indexed_docs, get_chunk_count,
    clear_collection, get_embedder, _build_bm25_index, _get_cross_encoder,
)
from database import init_db  # ensure DB + schema initialized on startup


def _background_init():
    print("\nLand Chatbot starting up...")
    init_db()           # init PostgreSQL schema
    get_embedder()      # warm embedding model
    chunks = load_all_pdfs()
    if chunks:
        index_chunks(chunks)
    else:
        print("  No PDFs found – upload them from the UI")

    # Pre-build BM25 index so the first query has no cold-start delay
    _build_bm25_index()

    # Pre-load cross-encoder re-ranker model if enabled
    if RERANK_ENABLED:
        _get_cross_encoder()

    print("Ready!\n")


@asynccontextmanager
async def lifespan(app: FastAPI):
    t = threading.Thread(target=_background_init, daemon=True)
    t.start()
    yield


app = FastAPI(
    title="Land Info Chatbot API",
    description="Local RAG chatbot for land document queries",
    version="2.0.0",
    lifespan=lifespan,
    # Disable auto-generated docs in production if needed
    # docs_url=None, redoc_url=None,
)

app.mount("/docs/samples", StaticFiles(directory=PDF_ROOT), name="samples")
if os.path.exists(PDF_DIR):
    app.mount("/docs/uploads", StaticFiles(directory=PDF_DIR), name="uploads")

# CORS — restrict in production; localhost only during dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "http://127.0.0.1:5173"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Content-Type", "Accept"],
)


@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response: Response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"]         = "DENY"
    response.headers["X-XSS-Protection"]        = "1; mode=block"
    response.headers["Referrer-Policy"]          = "no-referrer"
    return response


class ChatRequest(BaseModel):
    query: str
    filename_filter: str | None = None
    category_filter: str | None = None


class ChatResponse(BaseModel):
    answer: str
    sources: list[str]
    context_used: int


@app.get("/")
def root():
    return {"status": "running", "message": "Land Chatbot API v2"}


@app.get("/status")
def status():
    from collections import defaultdict
    docs  = get_indexed_docs()
    count = get_chunk_count()
    categories: dict[str, list] = defaultdict(list)
    for d in docs:
        parts = d.split("/")
        cat = parts[0] if parts else "OTHER"
        categories[cat].append(d)
    return {
        "indexed_documents": docs,
        "total_chunks":      count,
        "categories":        dict(categories),
        "pdf_root":          PDF_ROOT,
        "ready":             count > 0,
    }


@app.post("/scan")
def rescan():
    clear_collection()
    from database import clear_metadata
    clear_metadata()
    chunks = load_all_pdfs()
    if not chunks:
        return {"message": "No PDFs found to index", "chunks_indexed": 0}
    indexed = index_chunks(chunks)
    return {
        "message":        f"Scanned and indexed {indexed} chunks from {PDF_ROOT}",
        "chunks_indexed": indexed,
    }


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    os.makedirs(PDF_DIR, exist_ok=True)
    save_path = os.path.join(PDF_DIR, file.filename)
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    chunks = process_pdf(save_path)
    if not chunks:
        raise HTTPException(
            status_code=422,
            detail="Could not extract text from PDF. Is it a searchable/OCR PDF?"
        )
    indexed = index_chunks(chunks)
    return {
        "message":        f"'{file.filename}' indexed successfully",
        "filename":       file.filename,
        "chunks_indexed": indexed,
    }


@app.post("/chat")
async def chat(request: ChatRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    def event_generator():
        for chunk in answer_query(
            request.query,
            filename_filter=request.filename_filter,
            category_filter=request.category_filter,
        ):
            yield json.dumps(chunk) + "\n"

    return StreamingResponse(event_generator(), media_type="application/x-ndjson")


@app.delete("/reset")
def reset():
    clear_collection()
    from database import clear_metadata
    clear_metadata()
    return {"message": "All indexed documents cleared"}
