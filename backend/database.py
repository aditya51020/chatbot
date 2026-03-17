import os
import re
import json
import hashlib
from datetime import datetime, timezone
from sqlalchemy import create_engine, text, Column, Integer, String, Float, DateTime, UniqueConstraint
from sqlalchemy.orm import declarative_base, sessionmaker
from config import DATABASE_URL, AUDIT_LOG_PATH

# Synchronous PostgreSQL Engine with connection pooling
engine = create_engine(
    DATABASE_URL,
    pool_size=20,
    max_overflow=10,
    pool_timeout=30,
    pool_recycle=1800
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# --- SQLAlchemy Models --------------------------------------------------------

class DocMetadata(Base):
    __tablename__ = "doc_metadata"
    
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, unique=True, nullable=False, index=True)
    file_hash = Column(String, index=True) # Add this to detect content changes

    # --- Existing fields (Keep for compatibility) ---
    category = Column(String, default='')
    case_number = Column(String, default='', index=True)
    plot_number = Column(String, default='', index=True)
    scheme_name = Column(String, default='') # e.g., "Chittorgarh Road Yojana"
    allottee_name = Column(String, default='')
    ocr_quality = Column(Float, default=1.0)
    indexed_at = Column(DateTime(timezone=True))
    # --- NEW FIELDS FOR YOUR 10 REQUIREMENTS ---
    
    # Requirement 5: Area Mismatch detection
    land_area_recorded = Column(Float, default=0.0) # From the official table
    land_area_unit = Column(String, default='sq yards') # or hectare
    
    # Requirement 2: Loan/Bank Info
    has_active_loan = Column(String, default='No') 
    bank_name = Column(String, default='')
    loan_expiry_date = Column(String, default='')
    
    # Requirement 1 & 6: Legal/Stay Orders
    has_stay_order = Column(String, default='No')
    court_case_id = Column(String, default='')
    next_hearing_date = Column(String, default='')

    agency_name = Column(String, default='') 
    division = Column(String, default='')
    
    # Requirement 4: SC/ST Restrictions
    is_sc_st_restricted = Column(String, default='No')
    restriction_section = Column(String, default='') # e.g., "Section 175"
    
    # Requirement 1: Revenue Recovery (RC)
    rc_pending_amount = Column(Float, default=0.0)
    
    # Requirement 9: GIS/Hazard Info (Extracted from text)
    is_flood_zone = Column(String, default='Unknown')

    file_hash = Column(String, default='', index=True)
class TenderBidder(Base):
    __tablename__ = "tender_bidders"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, nullable=False, index=True)
    bidder_name = Column(String, nullable=False)
    bid_amount = Column(String, default='')
    rank = Column(String, default='')
    
    __table_args__ = (UniqueConstraint('filename', 'bidder_name', name='_filename_bidder_uc'),)

class TenderMetadata(Base):
    __tablename__ = "tender_metadata"
    
    tender_id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, unique=True, nullable=False, index=True)
    tender_title = Column(String, default='')
    department = Column(String, default='')
    location = Column(String, default='')
    nit_number = Column(String, default='')
    estimated_value = Column(String, default='')
    date = Column(String, default='')

class AuditLog(Base):
    __tablename__ = "audit_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    q_hash = Column(String, index=True)
    intents = Column(String)  # JSON stored as string
    doc_ids = Column(String)  # JSON stored as string
    chunks_returned = Column(Integer)
    llm_used = Column(String) # Boolean string true/false

# --- Initialization ----------------------------------------------------------

def init_db():
    # 1. Create tables based on the class above
    Base.metadata.create_all(bind=engine)
    
    # 2. Create the index inside a try block
    with engine.connect() as conn:
        try:
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS doc_metadata_fts_idx ON doc_metadata 
                USING GIN (to_tsvector('english', 
                    coalesce(agency_name, '') || ' ' || 
                    coalesce(allottee_name, '') || ' ' || 
                    coalesce(scheme_name, '') || ' ' || 
                    coalesce(division, '')
                ));
            """))
            conn.commit()
            print("PostgreSQL: Table and Search Index ready.")
        except Exception as e:
            print(f"Note: Search index exists or skipped: {e}")
            
# --- Ingestion helpers -------------------------------------------------------

def _norm(val: str) -> str:
    """Normalize a name for prefix-index lookups."""
    return re.sub(r'\s+', ' ', val.lower().strip())

def upsert_doc_metadata(filename: str, fields: dict, ocr_quality: float = 1.0, file_hash: str = "") -> None:
    """Insert or replace one row. Called during indexing per PDF."""
    now = datetime.now(timezone.utc)
    agency_norm = _norm(fields.get("Agency Name", ""))
    allottee_norm = _norm(fields.get("Allottee Name", ""))
    
    with SessionLocal() as db:
        doc = db.query(DocMetadata).filter(DocMetadata.filename == filename).first()
        if not doc:
            doc = DocMetadata(filename=filename)
            db.add(doc)
            
        doc.category = fields.get("category", "")
        doc.case_number = fields.get("case_number", "")
        doc.agency_name = fields.get("Agency Name", "")
        doc.agency_name_norm = agency_norm
        doc.ref_number = fields.get("Ref. Number", "")
        doc.tender_amount = fields.get("Tender Amount", "")
        doc.schedule_discount = fields.get("Schedule Discount", "")
        doc.date_casting = fields.get("Date of Casting", "")
        doc.date_testing = fields.get("Date of Testing", "")
        doc.wo_number = fields.get("W.O. Number", "")
        doc.division = fields.get("Name of Division", "")
        doc.plot_number = fields.get("Plot Number", "")
        doc.plot_size = fields.get("Plot Size", "")
        doc.deposit_amount = fields.get("Deposit Amount", "")
        doc.scheme_name = fields.get("Scheme Name", "")
        doc.allottee_name = fields.get("Allottee Name", "")
        doc.allottee_norm = allottee_norm
        doc.maturity_date = fields.get("Maturity Date", "")
        doc.fd_account = fields.get("FD Account No", "")
        doc.ocr_quality = ocr_quality
        if file_hash:
            doc.file_hash = file_hash
        doc.indexed_at = now
        
        db.commit()

def upsert_tender_bidders(filename: str, bidders: list[dict]) -> None:
    with SessionLocal() as db:
        db.query(TenderBidder).filter(TenderBidder.filename == filename).delete()
        if bidders:
            for b in bidders:
                bidder = TenderBidder(
                    filename=filename,
                    bidder_name=b.get("name", ""),
                    bid_amount=b.get("amount", ""),
                    rank=b.get("rank", "")
                )
                db.add(bidder)
        db.commit()

def upsert_document_metadata(data: dict):
    """Saves or updates the structured land records in PostgreSQL."""
    with SessionLocal() as db:
        existing = db.query(DocMetadata).filter(DocMetadata.filename == data['filename']).first()
        if existing:
            for key, value in data.items():
                setattr(existing, key, value)
        else:
            new_meta = DocMetadata(**data)
            db.add(new_meta)
        db.commit() 

def save_doc_metadata(filename: str, category: str, **kwargs):
    with SessionLocal() as db:
        # Check if file already exists
        doc = db.query(DocMetadata).filter(DocMetadata.filename == filename).first()
        if not doc:
            doc = DocMetadata(filename=filename, category=category)
            db.add(doc)
        
        # Dynamically update any extra fields (area, plot_number, etc.)
        for key, value in kwargs.items():
            if hasattr(doc, key):
                setattr(doc, key, value)
        
        db.commit()

def clear_metadata() -> None:
    with SessionLocal() as db:
        db.query(DocMetadata).delete()
        db.query(TenderBidder).delete()
        db.commit()

# --- Query helpers -----------------------------------------------------------

def _row_to_dict(row) -> dict:
    if not row: return None
    return {c.name: getattr(row, c.name) for c in row.__table__.columns}

def query_by_agency(name_fragment: str, limit: int = 5) -> list[dict]:
    frag = name_fragment.strip()
    norm_frag = _norm(frag)
    
    with SessionLocal() as db:
        # Try full text search first
        query = text("""
            SELECT * FROM doc_metadata 
            WHERE to_tsvector('english', coalesce(agency_name, '') || ' ' || coalesce(scheme_name, '') || ' ' || coalesce(division, '')) @@ plainto_tsquery('english', :frag)
            LIMIT :limit
        """)
        rows = db.execute(query, {"frag": frag, "limit": limit}).mappings().all()
        if rows:
            return [dict(r) for r in rows]
            
        # Fallback to prefix
        docs = db.query(DocMetadata).filter(DocMetadata.agency_name_norm.like(f"{norm_frag}%")).limit(limit).all()
        return [_row_to_dict(d) for d in docs]

def query_by_plot(plot_no: str, limit: int = 5) -> list[dict]:
    with SessionLocal() as db:
        docs = db.query(DocMetadata).filter(DocMetadata.plot_number == plot_no.strip()).limit(limit).all()
        if not docs:
            clean = re.sub(r'[\s\-/]', '', plot_no)
            query = text("""
                SELECT * FROM doc_metadata 
                WHERE REPLACE(REPLACE(REPLACE(plot_number,'-',''),'/',''),' ','') = :clean
                LIMIT :limit
            """)
            rows = db.execute(query, {"clean": clean, "limit": limit}).mappings().all()
            return [dict(r) for r in rows]
        return [_row_to_dict(d) for d in docs]

def query_by_allottee(name_fragment: str, limit: int = 5) -> list[dict]:
    frag = name_fragment.strip()
    norm_frag = _norm(frag)
    
    with SessionLocal() as db:
        query = text("""
            SELECT * FROM doc_metadata 
            WHERE to_tsvector('english', coalesce(allottee_name, '')) @@ plainto_tsquery('english', :frag)
            LIMIT :limit
        """)
        rows = db.execute(query, {"frag": frag, "limit": limit}).mappings().all()
        if rows:
            return [dict(r) for r in rows]
            
        docs = db.query(DocMetadata).filter(DocMetadata.allottee_norm.like(f"{norm_frag}%")).limit(limit).all()
        return [_row_to_dict(d) for d in docs]

def query_bidders_for_tender(filename: str) -> list[dict]:
    with SessionLocal() as db:
        bidders = db.query(TenderBidder).filter(TenderBidder.filename == filename).all()
        return [_row_to_dict(b) for b in bidders]

def get_all_fields_for_file(filename: str) -> dict | None:
    with SessionLocal() as db:
        doc = db.query(DocMetadata).filter(DocMetadata.filename == filename).first()
        return _row_to_dict(doc)
    
def get_all_indexed_filenames() -> list[str]:
    """Helper for main.py to identify new files on disk."""
    with SessionLocal() as db:
        # Returns a list of strings like ['NS.pdf', 'CS.pdf']
        results = db.query(DocMetadata.filename).all()
        return [r.filename for r in results]

# --- Audit logging -----------------------------------------------------------

def write_audit(query: str, intents: list, doc_ids: list, chunks_returned: int, llm_used: bool) -> None:
    # We now also save to PostgreSQL audit_logs table instead of just JSONL
    q_hash = hashlib.sha256(query.encode()).hexdigest()[:16]
    
    with SessionLocal() as db:
        log = AuditLog(
            q_hash=q_hash,
            intents=json.dumps(intents),
            doc_ids=json.dumps([d.split("/")[-1] for d in doc_ids]),
            chunks_returned=chunks_returned,
            llm_used=str(llm_used).lower()
        )
        db.add(log)
        db.commit()
    
    # Keep legacy jsonl for backup
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "q_hash": q_hash,
        "intents": intents,
        "doc_ids": [d.split("/")[-1] for d in doc_ids],
        "chunks_returned": chunks_returned,
        "llm_used": llm_used,
    }
    line = json.dumps(entry, ensure_ascii=False)
    try:
        with open(AUDIT_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception as e:
        print(f"  [audit] file write failed: {e}")
