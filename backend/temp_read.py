import pdfplumber

try:
    with pdfplumber.open('Sample_PDFs/CS.pdf') as pdf:
        text = ''
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                text += t + '\n'
        print(text)
except Exception as e:
    print("Error:", e)
