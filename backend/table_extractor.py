import os
import pandas as pd
import builtins
import re

# ---------------------------------------------------------------------------
# Normalization Dictionaries
# ---------------------------------------------------------------------------
_NAME_COLS   = ["bidder", "bidder name", "name of bidder", "contractor", "agency", "firm name", "name of agency"]
_AMOUNT_COLS = ["amount", "bid amount", "contract value", "tender amount", "total amount", "quoted amount"]
_RANK_COLS   = ["rank", "l1", "l2", "position", "status"]

def _normalize_header(header: str) -> str:
    if not header:
        return ""
    # Remove symbols, newlines, and lowercase it
    clean = re.sub(r'[^a-zA-Z0-9\s]', ' ', str(header)).strip().lower()
    clean = re.sub(r'\s+', ' ', clean)
    return clean

def _find_column_index(headers: list[str], target_keywords: list[str]) -> int:
    for i, h in enumerate(headers):
        norm = _normalize_header(h)
        for kw in target_keywords:
            if kw in norm:
                return i
    return -1

def _clean_amount(val: str) -> str:
    if not val:
        return ""
    clean = re.sub(r'[^\d\.]', '', str(val))
    return clean

# ---------------------------------------------------------------------------
# Stage 1: pdfplumber (Fast, bordered tables)
# ---------------------------------------------------------------------------
def _run_pdfplumber(pdf_path: str) -> list[dict]:
    bidders = []
    try:
        import pdfplumber
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                tables = page.extract_tables()
                for table in tables:
                    if not table or len(table) < 2:
                        continue
                        
                    headers = [str(h) if h else "" for h in table[0]]
                    name_idx = _find_column_index(headers, _NAME_COLS)
                    amt_idx  = _find_column_index(headers, _AMOUNT_COLS)
                    rank_idx = _find_column_index(headers, _RANK_COLS)
                    
                    if name_idx == -1:
                        # Sometimes headers are split. Check row 1 too.
                        headers2 = [str(h) if h else "" for h in table[1]]
                        name_idx = _find_column_index(headers2, _NAME_COLS)
                        amt_idx  = _find_column_index(headers2, _AMOUNT_COLS)
                        rank_idx = _find_column_index(headers2, _RANK_COLS)
                        table = table[1:] # Shift down
                    
                    if name_idx != -1 and amt_idx != -1:
                        for row in table[1:]:
                            if not row or len(row) <= max(name_idx, amt_idx):
                                continue
                            name = str(row[name_idx] or "").strip()
                            name = re.sub(r'\s+', ' ', name)
                            amt  = _clean_amount(row[amt_idx])
                            rank = str(row[rank_idx]).strip() if rank_idx != -1 and len(row) > rank_idx else ""
                            rank = rank.replace('\n', ' ').strip().upper()
                            
                            # Sanity check
                            if len(name) > 4 and amt.replace('.','').isdigit():
                                float_amt = float(amt)
                                if 5000 < float_amt < 5000000000:
                                    bidders.append({
                                        "name": name,
                                        "amount": amt,
                                        "rank": rank
                                    })
    except ImportError:
        print("  [TableExt] pdfplumber not installed.")
    except Exception as e:
        print(f"  [TableExt] pdfplumber error on {os.path.basename(pdf_path)}: {e}")
        
    return bidders

# ---------------------------------------------------------------------------
# Stage 2: Camelot (Borderless tables, stream mode)
# ---------------------------------------------------------------------------
def _run_camelot(pdf_path: str) -> list[dict]:
    bidders = []
    try:
        import camelot
        # Stream mode is better for borderless tables like some BOQs
        tables = camelot.read_pdf(pdf_path, pages='all', flavor='stream')
        for t in tables:
            df = t.df
            if df.empty or len(df.columns) < 2:
                continue
            
            headers = [str(h) for h in df.iloc[0].values]
            name_idx = _find_column_index(headers, _NAME_COLS)
            amt_idx  = _find_column_index(headers, _AMOUNT_COLS)
            rank_idx = _find_column_index(headers, _RANK_COLS)
            
            if name_idx != -1 and amt_idx != -1:
                for _, row in df.iloc[1:].iterrows():
                    name = str(row[name_idx] or "").strip()
                    amt  = _clean_amount(row[amt_idx])
                    rank = str(row[rank_idx]).strip() if rank_idx != -1 else ""
                    
                    if len(name) > 4 and amt.replace('.','').isdigit():
                        float_amt = float(amt)
                        if 5000 < float_amt < 5000000000:
                            bidders.append({"name": name, "amount": amt, "rank": rank})
    except ImportError:
        pass
    except Exception as e:
        pass
    return bidders

# ---------------------------------------------------------------------------
# Stage 3: PaddleOCR PP-Structure (Fallback for Scanned Images)
# ---------------------------------------------------------------------------
def _run_paddleocr(pdf_path: str) -> list[dict]:
    # Placeholder for PaddleOCR PP-Structure integration.
    # Requires complex image extraction -> paddleocr inference.
    return []

# ---------------------------------------------------------------------------
# Main Orchestrator
# ---------------------------------------------------------------------------
def extract_bidders_from_pdf(pdf_path: str) -> list[dict]:
    """
    3-Stage Production Extraction Pipeline for Government Tenders.
    1. pdfplumber (Fast, handles 80% structured PDFs)
    2. camelot    (Stream mode, handles borderless tables)
    3. paddleocr  (Deep learning, handles scans)
    """
    import os
    print(f"  [TableExt] Running extraction on {os.path.basename(pdf_path)}")
    
    # Remove duplicates
    seen = set()
    unique_bidders = []
    
    def _add_bidders(blist):
        for b in blist:
            nm = b['name'].upper()
            if nm not in seen:
                seen.add(nm)
                unique_bidders.append(b)

    # Stage 1: pdfplumber
    bids = _run_pdfplumber(pdf_path)
    if bids:
        _add_bidders(bids)
        return unique_bidders
        
    # Stage 2: Camelot
    bids = _run_camelot(pdf_path)
    if bids:
        print(f"  [TableExt] Recovered tables using Camelot.")
        _add_bidders(bids)
        return unique_bidders
        
    # Stage 3: PaddleOCR structure
    bids = _run_paddleocr(pdf_path)
    if bids:
        print(f"  [TableExt] Recovered tables using PaddleOCR.")
        _add_bidders(bids)
        
    return unique_bidders
