import os
from celery import Celery
from config import REDIS_URL
from pdf_processor import process_pdf
from rag import index_chunks

app = Celery(
    'land_chatbot',
    broker=REDIS_URL,
    backend=REDIS_URL
)

app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    worker_concurrency=2, # Keep parallel chunks processing safe for RAM
)

@app.task(bind=True, max_retries=3)
def process_document_task(self, save_path: str):
    """
    Background task to process a PDF, chunk it, extract metadata,
    and save embeddings to ChromaDB + Postgres.
    """
    try:
        print(f"[Worker] Starting processing for {save_path}")
        chunks = process_pdf(save_path)
        if not chunks:
            print(f"[Worker] No text extracted from {save_path}")
            return {"status": "failed", "error": "No text extracted"}
            
        indexed_count = index_chunks(chunks)
        print(f"[Worker] Successfully indexed {indexed_count} chunks for {save_path}")
        return {"status": "success", "indexed": indexed_count}
    except Exception as exc:
        print(f"[Worker] Error processing {save_path}: {exc}")
        self.retry(exc=exc, countdown=10)
