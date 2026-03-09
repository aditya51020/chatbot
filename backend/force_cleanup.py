import chromadb
from rag import clear_collection, COLLECTION_NAME
from database import clear_metadata

def force_clear():
    print("Clearing SQL metadata...")
    clear_metadata()
    print("SQL metadata cleared.")
    
    print(f"Clearing Chroma collection '{COLLECTION_NAME}'...")
    try:
        clear_collection()
        print("Chroma collection cleared and recreated.")
    except Exception as e:
        print(f"Error clearing Chroma: {e}")

if __name__ == "__main__":
    force_clear()
