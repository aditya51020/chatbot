from rag import get_collection

def dump_samples():
    collection = get_collection()
    res = collection.get(where={"filename": "CS.pdf"}, limit=10, include=["documents"])
    docs = res.get("documents", [])
    
    with open("chunk_samples.txt", "w", encoding="utf-8") as f:
        for i, d in enumerate(docs):
            f.write(f"--- CHUNK {i} ---\n")
            f.write(d)
            f.write("\n\n")

if __name__ == "__main__":
    dump_samples()
