from rag import retrieve_context

chunks = retrieve_context("Stamp duty payable कितनी है?", filename_filter="CS.pdf")
print("--- CHUNKS FOUND ---")
for i, c in enumerate(chunks):
    print(f"CHUNK {i}:")
    print(c["text"])
    print("------------------")
