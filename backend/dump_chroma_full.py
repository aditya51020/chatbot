from rag import get_collection

def dump_chroma():
    collection = get_collection()
    print(f"Collection count: {collection.count()}")
    results = collection.get(include=["documents", "metadatas"])
    
    docs = results.get("documents", [])
    metas = results.get("metadatas", [])
    
    for i, (doc, meta) in enumerate(zip(docs, metas)):
        if "बाबूलाल" in doc or "Dastaveda" in doc or "Dastaveda" in str(meta):
            print(f"--- MATCH FOUND AT INDEX {i} ---")
            print(f"Metadata: {meta}")
            print(f"Document: {doc[:500]}...")
            print("----------------------------")

if __name__ == "__main__":
    dump_chroma()
