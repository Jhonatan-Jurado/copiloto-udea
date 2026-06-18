"""One-shot DB bootstrap: app schema + LangGraph checkpointer tables. Idempotent.

Run from the `backend/` directory:

    python scripts/setup_db.py

It (1) executes sql/01_schema.sql to create the `vector`/`pgcrypto` extensions,
the `documents` and `semantic_cache` tables and their indexes, then (2) calls
PostgresSaver.setup() to create the LangGraph checkpointer tables. Safe to re-run.
"""
import sys
from pathlib import Path

# Make `app` importable when invoked as `python scripts/setup_db.py` (the script
# dir, not backend/, lands on sys.path[0] otherwise).
BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

import psycopg
from psycopg.rows import dict_row
from langgraph.checkpoint.postgres import PostgresSaver

from app.config import settings  # exposes settings.database_url

SQL_FILE = BACKEND_DIR / "sql" / "01_schema.sql"


def run_schema() -> None:
    sql = SQL_FILE.read_text(encoding="utf-8")
    with psycopg.connect(settings.database_url, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(sql)
    print("✓ app schema (documents, semantic_cache, indexes) created")


def run_checkpointer_setup() -> None:
    # PostgresSaver needs autocommit + dict_row connections.
    try:
        with PostgresSaver.from_conn_string(settings.database_url) as checkpointer:
            checkpointer.setup()
    except Exception as exc:
        # Fallback: if from_conn_string does not yield connections with the
        # required settings in the installed version, open one explicitly.
        # Surface the original error so a real failure (auth/SSL) isn't masked.
        print(f"  primary checkpointer setup failed ({exc!r}); retrying with explicit connection")
        with psycopg.connect(
            settings.database_url, autocommit=True, row_factory=dict_row
        ) as conn:
            PostgresSaver(conn).setup()
    print("✓ checkpointer tables created")


if __name__ == "__main__":
    run_schema()
    run_checkpointer_setup()
    print("Done.")
