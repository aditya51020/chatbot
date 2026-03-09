from rag import get_collection

def verify():
    collection = get_collection()
    res = collection.get(include=["documents"])
    docs = res.get("documents", [])
    found = any("बाबूलाल" in d for d in docs)
    print("RESULT: GHOST DATA FOUND" if found else "RESULT: DATABASE IS CLEAN")
    print(f"Total chunks: {len(docs)}")

if __name__ == "__main__":
    verify()
