import sqlite3
from pathlib import Path

db_path = Path(__file__).resolve().parent.parent / "data" / "sf_cache.sqlite"
if not db_path.exists():
    print("Database does not exist yet.")
else:
    conn = sqlite3.connect(str(db_path))
    count = conn.execute("SELECT count(*) FROM sf_analysis_cache").fetchone()[0]
    size_mb = db_path.stat().st_size / (1024 * 1024)
    print(f"Records: {count}, Size: {size_mb:.2f} MB")
    conn.close()
