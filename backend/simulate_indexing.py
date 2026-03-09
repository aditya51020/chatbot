import os
import fitz
from config import PDF_ROOT

def simulate():
    print(f"Scanning PDF_ROOT: {PDF_ROOT}")
    files = os.listdir(PDF_ROOT)
    print(f"Files found: {files}")
    
    for f in files:
        if f.lower().endswith(".pdf"):
            path = os.path.join(PDF_ROOT, f)
            print(f"Processing {f}...")
            try:
                doc = fitz.open(path)
                text = ""
                for page in doc:
                    text += page.get_text()
                
                print(f" - Text length: {len(text)}")
                if "बाबूलाल" in text:
                    print(f" - !!! FOUND 'बाबूलाल' in {f} !!!")
                else:
                    print(f" - 'बाबूलाल' NOT FOUND in {f}")
                    
                if "Dastaveda" in text:
                    print(f" - !!! FOUND 'Dastaveda' in {f} !!!")
            except Exception as e:
                print(f" - Error reading {f}: {e}")

if __name__ == "__main__":
    simulate()
