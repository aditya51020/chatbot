from rag import get_collection

def audit():
    collection = get_collection()
    print(f"Total count: {collection.count()}")
    results = collection.get(include=["metadatas"])
    metas = results.get("metadatas", [])
    
    files = {}
    for m in metas:
        fn = m.get("filename", "unknown")
        files[fn] = files.get(fn, 0) + 1
        
    print("Files found in ChromaDB:")
    for f, count in files.items():
        print(f" - {f}: {count} chunks")

if __name__ == "__main__":
    audit()
