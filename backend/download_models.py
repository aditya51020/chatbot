import os
import sys

def main():
    try:
        from sentence_transformers import SentenceTransformer
        from sentence_transformers.cross_encoder import CrossEncoder
    except ImportError as e:
        print("Error: sentence-transformers is not installed.", e)
        sys.exit(1)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(base_dir, "local_models")
    os.makedirs(models_dir, exist_ok=True)
    
    embed_path = os.path.join(models_dir, "bge-m3")
    rerank_path = os.path.join(models_dir, "bge-reranker-large")
    
    print("--- Downloading BAAI/bge-m3 ---")
    if not os.path.exists(embed_path):
        print(f"Downloading model to {embed_path}...")
        model = SentenceTransformer("BAAI/bge-m3")
        model.save(embed_path)
        print("Successfully saved BAAI/bge-m3 locally.")
    else:
        print("BAAI/bge-m3 already exists locally.")
        
    print("\n--- Downloading BAAI/bge-reranker-large ---")
    if not os.path.exists(rerank_path):
        print(f"Downloading model to {rerank_path}...")
        reranker = CrossEncoder("BAAI/bge-reranker-large")
        reranker.save(rerank_path)
        print("Successfully saved BAAI/bge-reranker-large locally.")
    else:
        print("BAAI/bge-reranker-large already exists locally.")
        
    print("\nAll models have been downloaded and saved locally. You can now use local paths.")

if __name__ == "__main__":
    # Temporarily unset offline mode for the download script to work
    os.environ["TRANSFORMERS_OFFLINE"] = "0"
    os.environ["HF_DATASETS_OFFLINE"] = "0"
    os.environ["HF_HUB_OFFLINE"] = "0"
    main()
