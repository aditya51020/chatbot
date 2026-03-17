import os
import sys

def main():
    print("\n" + "="*70)
    print("🤖 Land Chatbot - Model Download Manager")
    print("="*70 + "\n")
    
    try:
        from sentence_transformers import SentenceTransformer
        from sentence_transformers.cross_encoder import CrossEncoder
    except ImportError as e:
        print("❌ Error: sentence-transformers is not installed.")
        print(f"   Details: {e}")
        print("\n📦 Installing required packages...")
        os.system(f"{sys.executable} -m pip install --upgrade sentence-transformers")
        return

    base_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(base_dir, "local_models")
    os.makedirs(models_dir, exist_ok=True)
    
    # Also download Phi-3 model for GPT4All
    gpt4all_dir = os.path.join(os.path.expanduser("~"), ".cache", "gpt4all")
    os.makedirs(gpt4all_dir, exist_ok=True)
    
    embed_path = os.path.join(models_dir, "bge-m3")
    rerank_path = os.path.join(models_dir, "bge-reranker-large")
    phi3_path = os.path.join(gpt4all_dir, "Phi-3-mini-4k-instruct.Q4_0.gguf")
    
    print(f"📂 Models directory: {models_dir}\n")
    
    # Download embedding model
    print("⏳ Downloading BAAI/bge-m3 (Embedding Model)...")
    if not os.path.exists(embed_path):
        try:
            print(f"   → Saving to {embed_path}...")
            model = SentenceTransformer("BAAI/bge-m3")
            model.save(embed_path)
            print("   ✓ Successfully saved BAAI/bge-m3 locally.")
        except Exception as e:
            print(f"   ❌ Failed to download BAAI/bge-m3: {e}")
    else:
        print("   ✓ BAAI/bge-m3 already exists locally.")
        
    # Download re-ranker model
    print("\n⏳ Downloading BAAI/bge-reranker-large (Re-ranker Model)...")
    if not os.path.exists(rerank_path):
        try:
            print(f"   → Saving to {rerank_path}...")
            reranker = CrossEncoder("BAAI/bge-reranker-large")
            reranker.save(rerank_path)
            print("   ✓ Successfully saved BAAI/bge-reranker-large locally.")
        except Exception as e:
            print(f"   ❌ Failed to download BAAI/bge-reranker-large: {e}")
    else:
        print("   ✓ BAAI/bge-reranker-large already exists locally.")
    
    # Download Phi-3 model
    print("\n⏳ Downloading Phi-3-mini-4k (LLM Model) ~2.2GB...")
    if not os.path.exists(phi3_path):
        try:
            print(f"   → Saving to {phi3_path}...")
            print("   ⚠️  This may take 5-15 minutes on a slow connection...")
            from gpt4all import GPT4All
            model = GPT4All("Phi-3-mini-4k-instruct.Q4_0.gguf", model_path=gpt4all_dir, verbose=False)
            print("   ✓ Successfully saved Phi-3-mini-4k locally.")
        except Exception as e:
            print(f"   ⚠️  Phi-3-mini download skipped (optional):")
            print(f"       {e}")
            print(f"       You can still use the chatbot with embeddings only.")
    else:
        print("   ✓ Phi-3-mini-4k already exists locally.")
        
    print("\n" + "="*70)
    print("✅ Model download complete!")
    print("="*70)
    print("\n📝 Next steps:")
    print("   1. Run: cd backend && uvicorn main:app --reload --port 8000")
    print("   2. In another terminal: cd frontend && npm run dev")
    print("   3. Open http://localhost:5173 in your browser\n")

if __name__ == "__main__":
    # Temporarily unset offline mode for the download script to work
    os.environ["TRANSFORMERS_OFFLINE"] = "0"
    os.environ["HF_DATASETS_OFFLINE"] = "0"
    os.environ["HF_HUB_OFFLINE"] = "0"
    main()
