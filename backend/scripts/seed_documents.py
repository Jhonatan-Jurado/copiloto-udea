"""DEV-ONLY seed for the `documents` table — NOT the ingestion pipeline.

Throwaway helper so the team can test POST /chat before loading the full corpus.
It reads the processed .txt under preprocessing/data/processed/, chunks them
(shared logic in app/chunking.py), embeds each chunk with the SAME model the backend
uses (text-embedding-3-small, 1536 dims), and inserts rows into `documents`.

For the FULL corpus use `ingest_documents.py` instead. This seed tags every row
with the single sample's metadata and marks rows `metadata->>'seed' = 'true'`.

Re-runnable & atomic: it deletes its own previous rows and re-inserts inside one
transaction; it never touches real ingestion rows.

Run from the `backend/` directory (after setup_db.py):

    python scripts/seed_documents.py
"""
import sys
from pathlib import Path

# Make `app` importable when invoked as `python scripts/seed_documents.py`.
BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from app.chunking import chunk_document

# NOTE: app.db / app.azure_clients are imported inside main(), not at module
# level, so importing this module does not open the DB pool or build Azure
# clients — and the "Nothing to seed" guidance can run before setup_db.py.

REPO_ROOT = BACKEND_DIR.parent
PROCESSED_DIR = REPO_ROOT / "preprocessing" / "data" / "processed"

# Citation metadata for this dev corpus. The sample document is Acuerdo Superior
# 425 de 2014, cited by its real title rather than mislabeled as the Reglamento.
DEFAULT_SOURCE = "Acuerdo Superior 425 de 2014"
DEFAULT_NIVEL = "pregrado"
DEFAULT_URL = "https://normativa.udea.edu.co/Documentos/Consultar"


def collect_rows() -> list[dict]:
    txt_files = sorted(PROCESSED_DIR.rglob("*.txt"))
    if not txt_files:
        print(f"! No .txt files under {PROCESSED_DIR} — run the OCR pipeline first.")
        return []
    rows: list[dict] = []
    for path in txt_files:
        for articulo, pagina, content in chunk_document(path.read_text(encoding="utf-8")):
            rows.append({
                "content": content,
                "source": DEFAULT_SOURCE,
                "nivel": DEFAULT_NIVEL,
                "articulo": articulo,
                "pagina": pagina,
                "url": DEFAULT_URL,
                "metadata": {"seed": "true", "origen": path.name},
            })
    return rows


def main() -> None:
    rows = collect_rows()
    if not rows:
        print("Nothing to seed.")
        return

    # Imported here (not at module top) so opening the DB pool / building Azure
    # clients only happens when we actually have rows to insert.
    from psycopg.types.json import Jsonb
    from app.azure_clients import make_embeddings
    from app.db import pool

    print(f"Embedding {len(rows)} chunk(s) with text-embedding-3-small ...")
    embeddings = make_embeddings()
    vectors = embeddings.embed_documents([r["content"] for r in rows])
    if len(vectors) != len(rows):
        raise RuntimeError(
            f"embedding count mismatch: {len(vectors)} vectors for {len(rows)} chunks"
        )

    insert_sql = """
        INSERT INTO documents (content, embedding, source, nivel, articulo, pagina, url, metadata)
        VALUES (%(content)s, %(embedding)s::vector, %(source)s, %(nivel)s, %(articulo)s,
                %(pagina)s, %(url)s, %(metadata)s)
    """
    # Atomic delete+insert (explicit transaction over the autocommit pool).
    with pool.connection() as conn:
        with conn.transaction():
            deleted = conn.execute(
                "DELETE FROM documents WHERE metadata->>'seed' = 'true'"
            ).rowcount
            if deleted:
                print(f"  removed {deleted} previous seed row(s)")
            for r, vec in zip(rows, vectors):
                conn.execute(insert_sql, {
                    "content": r["content"],
                    "embedding": vec,
                    "source": r["source"],
                    "nivel": r["nivel"],
                    "articulo": r["articulo"],
                    "pagina": r["pagina"],
                    "url": r["url"],
                    "metadata": Jsonb(r["metadata"]),
                })
    print(f"✓ seeded {len(rows)} dev chunk(s) into documents (clearly marked seed=true)")


if __name__ == "__main__":
    main()
