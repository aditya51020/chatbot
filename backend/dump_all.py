from rag import get_collection

def dump_all():
    collection = get_collection()
    res = collection.get(include=["documents"])
    docs = res.get("documents", [])
    
    with open("all_chunks.txt", "w", encoding="utf-8") as f:
        for i, d in enumerate(docs):
            f.write(f"CHUNK {i}:\n{d}\n\n")

if __name__ == "__main__":
    dump_all()
