from rag import get_collection

def expose():
    collection = get_collection()
    res = collection.get(include=["documents", "metadatas"])
    docs = res.get("documents", [])
    metas = res.get("metadatas", [])
    
    for d, m in zip(docs, metas):
        if "बाबूलाल" in d:
            print("--- CHUNK FOUND ---")
            print(f"Metadata: {m}")
            print(f"Text:\n{d}")
            print("-------------------")

if __name__ == "__main__":
    expose()
