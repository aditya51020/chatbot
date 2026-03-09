from sqlalchemy import text
from database import engine

def dump_full():
    with engine.connect() as conn:
        print("--- FULL TABLE: doc_metadata ---")
        res = conn.execute(text("SELECT * FROM doc_metadata"))
        rows = res.mappings().all()
        for i, row in enumerate(rows):
            print(f"ROW {i}: {dict(row)}")
        print(f"Total rows: {len(rows)}")

if __name__ == "__main__":
    dump_full()
