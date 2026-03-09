from sqlalchemy import text
from database import engine

def dump():
    with engine.connect() as conn:
        print("--- TABLE: doc_metadata ---")
        res = conn.execute(text("SELECT id, filename, agency_name, plot_number, scheme_name FROM doc_metadata"))
        for row in res.mappings().all():
            print(dict(row))
            
        print("\n--- TABLE: tender_bidders ---")
        res = conn.execute(text("SELECT id, filename, bidder_name FROM tender_bidders"))
        for row in res.mappings().all():
            print(dict(row))

if __name__ == "__main__":
    dump()
