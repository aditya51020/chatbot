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
    """
    Runs on startup in a background thread.
    1. Initialises DB schema.
    2. Warms up the embedding model.
    3. If ChromaDB is empty, auto-ingests all PDFs from Sample_PDFs/.
       This means after any clean rebuild or fresh install the system is
       ready immediately without requiring a manual /scan call.
    """
    print("\nLand Chatbot starting up...", flush=True)
    init_db()

    try:
        get_embedder()
    except Exception as e:
        print(f"[WARN] Embedding model init failed: {e}", flush=True)

    # --- Auto-ingest if vector DB is empty ---
    try:
        chunk_count = get_chunk_count()
        if chunk_count == 0:
            print("[STARTUP] ChromaDB is empty — auto-ingesting PDFs from Sample_PDFs/...", flush=True)
            chunks = load_all_pdfs()
            if chunks:
                indexed = index_chunks(chunks)
                print(f"[STARTUP] Auto-indexed {indexed} chunks. System ready.", flush=True)
            else:
                print("[STARTUP] No PDFs found in Sample_PDFs/. Add PDFs and call /scan.", flush=True)
        else:
            print(f"[STARTUP] ChromaDB has {chunk_count} chunks. Skipping auto-ingest.", flush=True)
    except Exception as e:
        print(f"[STARTUP] Auto-ingest failed: {e}", flush=True)

    print("Backend Ready!\n", flush=True)


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
)

app.mount("/docs/samples", StaticFiles(directory=PDF_ROOT), name="samples")
if os.path.exists(PDF_DIR):
    app.mount("/docs/uploads", StaticFiles(directory=PDF_DIR), name="uploads")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    os.makedirs(PDF_DIR, exist_ok=True)
    save_path = os.path.join(PDF_DIR, file.filename)
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    chunks = process_pdf(save_path)
    if not chunks:
        raise HTTPException(status_code=422, detail="OCR failed or no text found in PDF")

    indexed_count = index_chunks(chunks)
    return {"message": f"'{file.filename}' uploaded and indexed.", "chunks": indexed_count}


@app.post("/scan")
async def smart_scan():
    """
    Re-scans Sample_PDFs/ and indexes any files not already in ChromaDB.
    Uses chunk count per filename in ChromaDB (not the Postgres table) to
    determine what's new — so it works even after a clean rebuild.
    """
    from pdf_processor import scan_pdf_root
    import os

    print("\n--- /scan TRIGGERED ---", flush=True)

    # Get already-indexed filenames from ChromaDB (reliable after rebuild)
    already_indexed = set(get_indexed_docs())  # returns relative paths like 'Plot_X/CS/CS.pdf'

    files_on_disk = scan_pdf_root(PDF_ROOT)
    from config import PDF_ROOT as _PDF_ROOT

    new_files = []
    for f in files_on_disk:
        rel = os.path.relpath(f, _PDF_ROOT).replace("\\", "/")
        if rel not in already_indexed:
            new_files.append(f)

    if not new_files:
        print("Result: All PDFs already indexed.", flush=True)
        return {"message": "All PDFs already indexed. Use /reset then /scan to force re-index."}

    success_count = 0
    total_chunks  = 0
    for pdf_path in new_files:
        print(f"  Scanning: {os.path.basename(pdf_path)}...", flush=True)
        chunks = process_pdf(pdf_path)
        if chunks:
            n = index_chunks(chunks)
            total_chunks  += n
            success_count += 1

    print(f"Scan complete: {success_count} files, {total_chunks} chunks.", flush=True)
    return {"message": f"Indexed {success_count} new files ({total_chunks} chunks)."}


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
    return {"message": "All indexed documents cleared. Call /scan to re-index."}
