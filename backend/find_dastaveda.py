from rag import get_collection

def find_dastaveda():
    collection = get_collection()
    res = collection.get(include=["documents", "metadatas"])
    docs = res.get("documents", [])
    metas = res.get("metadatas", [])
    
    for d, m in zip(docs, metas):
        if "Dastaveda" in d or "SecondParty" in d:
            print("--- CHUNK FOUND ---")
            print(f"Metadata: {m}")
            print(f"Text snippet: {d[:200]}")
            print("-------------------")

if __name__ == "__main__":
    find_dastaveda()
