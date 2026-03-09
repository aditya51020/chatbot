import chromadb
import os

from config import CHROMA_DIR, COLLECTION_NAME

def check():
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collections = client.list_collections()
    print(f"Collections in {CHROMA_DIR}:")
    for c in collections:
        print(f" - {c.name} (Count: {c.count()})")
        
    print(f"\nTarget collection: {COLLECTION_NAME}")
    try:
        coll = client.get_collection(COLLECTION_NAME)
        print(f"Collection '{COLLECTION_NAME}' exists with {coll.count()} items.")
    except Exception as e:
        print(f"Collection '{COLLECTION_NAME}' does not exist or error: {e}")

if __name__ == "__main__":
    check()
