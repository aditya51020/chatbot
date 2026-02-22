import os
import shutil
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from config import PDF_DIR, PDF_ROOT
from pdf_processor import load_all_pdfs, process_pdf, scan_pdf_root
from rag import index_chunks, answer_query, get_indexed_docs, get_chunk_count, clear_collection, get_embedder


# ── Startup: pre-load models + index existing PDFs ────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("\n🚀 Land Chatbot starting up...")
    print("🔄 Pre-loading embedding model (this may take ~30s first time)...")
    get_embedder()          # warm up NOW so uploads are instant later
    print("📂 Checking for existing PDFs to index...")
    chunks = load_all_pdfs()
    if chunks:
        index_chunks(chunks)
    else:
        print("   (No PDFs found – upload them from the UI)")
    print("✅ Ready!\n")
    yield


from fastapi.staticfiles import StaticFiles

app = FastAPI(
    title="Land Info Chatbot API",
    description="Local RAG chatbot for land document queries",
    version="1.0.0",
    lifespan=lifespan
)

# Serves PDFs for preview/download
app.mount("/docs/samples", StaticFiles(directory=PDF_ROOT), name="samples")
if os.path.exists(PDF_DIR):
    app.mount("/docs/uploads", StaticFiles(directory=PDF_DIR), name="uploads")

# Allow all localhost origins (Vite sometimes uses 5173, 5174, etc.)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Models ─────────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    query: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[str]
    context_used: int


# ── Routes ─────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "running", "message": "Land Chatbot API"}


@app.get("/status")
def status():
    """Returns indexing status and list of indexed documents."""
    docs  = get_indexed_docs()
    count = get_chunk_count()
    # Group by category for richer UI
    from collections import defaultdict
    categories: dict[str, list] = defaultdict(list)
    for d in docs:
        parts = d.split("/")
        cat = parts[0] if parts else "OTHER"
        categories[cat].append(d)
    return {
        "indexed_documents": docs,
        "total_chunks": count,
        "categories": dict(categories),
        "pdf_root": PDF_ROOT,
        "ready": count > 0
    }


@app.post("/scan")
def rescan():
    """Re-scan PDF_ROOT and index all PDFs (clears old index first)."""
    clear_collection()
    chunks = load_all_pdfs()
    if not chunks:
        return {"message": "No PDFs found to index", "chunks_indexed": 0}
    indexed = index_chunks(chunks)
    return {
        "message": f"✅ Scanned and indexed {indexed} chunks from {PDF_ROOT}",
        "chunks_indexed": indexed
    }


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """Upload a PDF, save to pdfs/ folder, and index it."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    os.makedirs(PDF_DIR, exist_ok=True)
    save_path = os.path.join(PDF_DIR, file.filename)
    
    # Save file
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    
    # Process and index
    chunks = process_pdf(save_path)
    if not chunks:
        raise HTTPException(status_code=422, detail="Could not extract text from PDF. Is it a searchable/OCR PDF?")
    
    indexed = index_chunks(chunks)
    
    return {
        "message": f"✅ '{file.filename}' indexed successfully",
        "filename": file.filename,
        "chunks_indexed": indexed
    }


import json
from fastapi.responses import StreamingResponse

@app.post("/chat")
async def chat(request: ChatRequest):
    """Ask a question about the land documents with real-time streaming."""
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    def event_generator():
        for chunk in answer_query(request.query):
            # Yield as JSON strings separated by newlines (NDJSON style)
            yield json.dumps(chunk) + "\n"

    return StreamingResponse(event_generator(), media_type="application/x-ndjson")


@app.delete("/reset")
def reset():
    """Clear all indexed documents from ChromaDB."""
    clear_collection()
    return {"message": "✅ All indexed documents cleared"}
