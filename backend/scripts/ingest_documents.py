"""Full-corpus ingestion for the `documents` table.

Walks the OCR-processed .txt under preprocessing/data/processed/, derives citation
metadata per document (source from the filename, nivel from the folder, a generic
url), chunks each file, embeds every chunk with Azure text-embedding-3-small
(1536 dims), and inserts the rows into `documents`.

Shared logic lives in the app package (app.chunking, app.ingestion); this script
adds the corpus-level walk, the per-document report, and the --reset semantics.

Prerequisite: run the OCR pipeline first so every PDF has a .txt
(`python preprocessing/src/ocr/pipeline.py` from the repo root).

Run from the `backend/` directory:

    python scripts/ingest_documents.py --dry-run     # parse+chunk report, no embed/insert
    python scripts/ingest_documents.py --reset       # DELETE all rows, then load the full corpus
    python scripts/ingest_documents.py               # idempotent: replace only prior corpus rows

Uses backend/.env for Azure credentials (same as the API).
"""
import argparse
import sys
from collections import defaultdict
from pathlib import Path

# Make `app` importable when invoked as `python scripts/ingest_documents.py`.
BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from app.chunking import chunk_document
from app.ingestion import (
    DEFAULT_URL,
    EMBED_BATCH,
    INSERT_SQL,
    embed_all,
    nivel_from_path,
    parse_source_title,
)

# app.azure_clients is imported inside main(), so --dry-run never builds clients.

REPO_ROOT = BACKEND_DIR.parent
PROCESSED_DIR = REPO_ROOT / "preprocessing" / "data" / "processed"

# documento_17917413 is the scanned duplicate of acuerdo_superior_425 (which we
# ingest from the cleaner native PDF). Skip it to avoid double citations.
SKIP_STEMS = {"documento_17917413"}


def collect_rows() -> list[dict]:
    txt_files = sorted(PROCESSED_DIR.rglob("*.txt"))
    if not txt_files:
        print(f"! No .txt under {PROCESSED_DIR} — run the OCR pipeline first.")
        return []
    rows: list[dict] = []
    skipped = 0
    for path in txt_files:
        if path.stem in SKIP_STEMS:
            skipped += 1
            continue
        raw = path.read_text(encoding="utf-8")
        source = parse_source_title(path.stem, raw[:2000])
        nivel = nivel_from_path(path)
        for articulo, pagina, content in chunk_document(raw):
            rows.append({
                "content": content,
                "source": source,
                "nivel": nivel,
                "articulo": articulo,
                "pagina": pagina,
                "url": DEFAULT_URL,
                "metadata": {"origen": path.name, "ingest": "corpus"},
            })
    if skipped:
        print(f"  (skipped {skipped} file(s): {', '.join(sorted(SKIP_STEMS))})")
    return rows


def report(rows: list[dict]) -> None:
    by_doc: dict[str, dict] = defaultdict(lambda: {"source": "", "nivel": "", "chunks": 0, "art": 0})
    for r in rows:
        d = by_doc[r["metadata"]["origen"]]
        d["source"], d["nivel"] = r["source"], r["nivel"]
        d["chunks"] += 1
        d["art"] += 1 if r["articulo"] else 0
    print(f"\n{len(rows)} chunk(s) from {len(by_doc)} document(s):")
    for origen in sorted(by_doc):
        d = by_doc[origen]
        print(f"  [{d['nivel']:9}] {d['source'][:42]:42} {d['chunks']:4d} chunks, "
              f"{d['art']:3d} con artículo   ({origen})")
    cov = sum(1 for r in rows if r["articulo"]) / len(rows) if rows else 0.0
    print(f"cobertura de artículo: {cov:.0%}")


def write_rows(rows: list[dict], vectors: list[list[float]], reset: bool) -> None:
    from psycopg.types.json import Jsonb
    from app.db import pool
    with pool.connection() as conn:
        with conn.transaction():                    # atomic: delete + insert
            if reset:
                n = conn.execute("DELETE FROM documents").rowcount
                print(f"  --reset: removed ALL {n} existing row(s)")
            else:
                n = conn.execute(
                    "DELETE FROM documents WHERE metadata->>'ingest' = 'corpus'"
                ).rowcount
                if n:
                    print(f"  removed {n} previous corpus row(s)")
            for r, vec in zip(rows, vectors):
                conn.execute(INSERT_SQL, {
                    "content": r["content"],
                    "embedding": vec,
                    "source": r["source"],
                    "nivel": r["nivel"],
                    "articulo": r["articulo"],
                    "pagina": r["pagina"],
                    "url": r["url"],
                    "metadata": Jsonb(r["metadata"]),
                })
    print(f"✓ ingested {len(rows)} chunk(s) into documents (metadata.ingest='corpus')")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest the full UdeA corpus into documents.")
    parser.add_argument("--reset", action="store_true",
                        help="DELETE all rows in documents before inserting (atomic).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Parse+chunk and print a report; do not embed or insert.")
    args = parser.parse_args()

    rows = collect_rows()
    if not rows:
        print("Nothing to ingest.")
        return
    report(rows)
    if args.dry_run:
        print("--dry-run: no embeddings, no inserts.")
        return

    from app.azure_clients import make_embeddings
    print(f"\nEmbedding {len(rows)} chunk(s) in batches of {EMBED_BATCH} ...")
    vectors = embed_all(make_embeddings(), [r["content"] for r in rows])
    if len(vectors) != len(rows):
        raise RuntimeError(f"embedding count mismatch: {len(vectors)} != {len(rows)}")
    write_rows(rows, vectors, args.reset)


if __name__ == "__main__":
    main()
