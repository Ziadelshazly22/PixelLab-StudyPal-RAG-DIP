"""Inspect ChromaDB SQLite schema to diagnose _type KeyError - v2."""
import sqlite3
import json
import os

db_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'chroma_db', 'chroma.sqlite3'))
print(f"DB path: {db_path}")

conn = sqlite3.connect(db_path)
cur = conn.cursor()

# Dump all table schemas
cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = [r[0] for r in cur.fetchall()]
print(f"\nTABLES: {tables}")

# Show schema of key tables
for tbl in ['collections', 'segments', 'collection_metadata', 'segment_metadata']:
    cur.execute(f"PRAGMA table_info({tbl})")
    cols = [(r[1], r[2]) for r in cur.fetchall()]
    print(f"\n{tbl} columns: {cols}")
    cur.execute(f"SELECT * FROM {tbl} LIMIT 10")
    for r in cur.fetchall():
        print(f"  {r}")

conn.close()
print("\nDone.")
