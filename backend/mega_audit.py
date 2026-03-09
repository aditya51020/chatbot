import os
import fitz

root = r"c:\Users\adity\OneDrive\Desktop\chatbot\land-chatbot"
targets = ["बाबूलाल", "Dastaveda", "MAMTA"]

print(f"Scanning from root: {root}")
for dirpath, _, filenames in os.walk(root):
    for f in filenames:
        if f.lower().endswith(".pdf"):
            path = os.path.join(dirpath, f)
            print(f"\nFILE: {path} ({os.path.getsize(path)} bytes)")
            try:
                doc = fitz.open(path)
                text = " ".join(p.get_text() for p in doc)
                for t in targets:
                    if t.lower() in text.lower():
                        print(f"  [FOUND] {t}")
                    else:
                        print(f"  [NOT FOUND] {t}")
            except Exception as e:
                print(f"  [ERROR] {e}")

print("\nScan complete.")
