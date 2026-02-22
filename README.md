# 🏛️ Land Info Chatbot

Ek fully **local AI chatbot** jo aapki land-record PDFs se sawaloon ka jawab deta hai.
**Koi internet nahi, koi third-party API nahi** — sab kuch aapke apne computer pe.

---

## ⚡ Quick Start

### Step 1 – Backend (Python)
```powershell
cd land-chatbot\backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

### Step 2 – Frontend (React)
```powershell
# Naye terminal mein:
cd land-chatbot\frontend
npm install
npm run dev
```

### Step 3 – Browser open karein
```
http://localhost:5173
```

---

## 📂 PDF Files Kaise Add Karein?

**Option A – UI se (recommended):**
- Browser mein jaiye → Left panel → PDF drag-and-drop karein

**Option B – Directly folder mein:**
- `backend/pdfs/` folder mein apni PDF files paste karein
- Backend restart karein — automatically index ho jaayegi

---

## 🔧 Model Download (First Run)

Pehli baar run karne par **Phi-3 Mini model (~2.2GB)** automatically download hoga.
Internet connection chahiye sirf pehli baar. Uske baad sab kuch offline.

Model change karna ho to `backend/config.py` mein `GPT4ALL_MODEL` edit karein.

---

## 📁 Folder Structure

```
land-chatbot/
├── backend/
│   ├── main.py           # FastAPI server
│   ├── rag.py            # RAG pipeline
│   ├── pdf_processor.py  # PDF → text chunks
│   ├── config.py         # Settings
│   ├── requirements.txt  # Python packages
│   ├── pdfs/             # ← Apni PDFs yahan rakhein
│   └── chroma_db/        # Vector DB (auto-created)
└── frontend/
    ├── src/
    │   ├── App.jsx
    │   └── App.css
    └── package.json
```

---

## 🛡️ Security

- Sab data **local disk** pe store hota hai (`backend/chroma_db/`)
- Koi external server ya API call nahi hoti
- WiFi band karke bhi kaam karta hai (model download ke baad)
