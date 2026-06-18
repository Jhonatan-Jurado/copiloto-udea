"""DEV-ONLY seed for the `documents` table — NOT the ingestion pipeline.

Throwaway helper so the team can test POST /chat before the real ingestion
pipeline (owned by a teammate) is ready. It reads the processed .txt files the
OCR pipeline produced under preprocessing/data/processed/, splits each by page
and then by article heading, embeds every chunk with the SAME model the backend
uses (text-embedding-3-small, 1536 dims), and inserts rows into `documents`.

Article attribution: the text is split at each `ARTÍCULO N:` heading and the
body that follows (even across a page break) is tagged with that article, so a
chunk's `articulo` matches its content. Text before the first heading (the
considerandos / preamble) is tagged articulo=NULL.

Re-runnable & atomic: it deletes its own previous rows (metadata->>'seed' ==
'true') and re-inserts inside a single transaction, so it never duplicates and
never leaves a half-replaced corpus. Real ingestion rows are untouched.

Run from the `backend/` directory (after setup_db.py):

    python scripts/seed_documents.py
"""
import re
import sys
from pathlib import Path

# Make `app` importable when invoked as `python scripts/seed_documents.py`.
BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

# NOTE: app.db / app.azure_clients are imported inside main(), not at module
# level, so importing this module (e.g. for tests) does not open the DB pool or
# construct Azure clients — and the friendly "Nothing to seed" guidance can run
# even before setup_db.py has provisioned the database.

# Where the OCR pipeline writes its cleaned text (repo_root/preprocessing/...).
REPO_ROOT = BACKEND_DIR.parent
PROCESSED_DIR = REPO_ROOT / "preprocessing" / "data" / "processed"

# Citation metadata for this dev corpus. The sample document is Acuerdo Superior
# 425 de 2014, which MODIFIES articles of the undergraduate regulations — so we
# cite it by its real title rather than mislabeling it as the Reglamento itself.
DEFAULT_SOURCE = "Acuerdo Superior 425 de 2014"
DEFAULT_NIVEL = "pregrado"
DEFAULT_URL = "https://normativa.udea.edu.co/Documentos/Consultar"

# Chunking knobs (dev-only; the real ingestion owns its own strategy).
TARGET_CHARS = 600          # sub-split long segments to ~this size
MIN_CHUNK_CHARS = 60        # drop tiny noise fragments

PAGE_RE = re.compile(r"---\s*P[ÁA]GINA\s+(\d+)\s*---", re.IGNORECASE)
HEADING_RE = re.compile(r"^\s*ART[ÍI]CULO\s+(\d+)\s*:", re.IGNORECASE)
FOOTER_RE = re.compile(
    r"(Ciudad Universitaria|Comutador|Apartado:|Nit:|Medell[ií]n,|http)", re.IGNORECASE
)


def _strip_header(text: str) -> str:
    """Drop the FUENTE/PAGINAS/... header above the '====' separator line."""
    parts = re.split(r"^={5,}\s*$", text, maxsplit=1, flags=re.MULTILINE)
    return parts[1] if len(parts) == 2 else text


def _pages(body: str) -> list[tuple[int, str]]:
    """Split a document body into (page_number, page_text) pairs."""
    segments = PAGE_RE.split(body)
    # segments = [pre, num, text, num, text, ...]
    out: list[tuple[int, str]] = []
    for i in range(1, len(segments), 2):
        out.append((int(segments[i]), segments[i + 1]))
    if not out:  # no page markers — treat whole body as page 1
        out = [(1, body)]
    return out


def _segments(pages: list[tuple[int, str]]) -> list[tuple[str | None, int, str]]:
    """Split into one segment per article heading, carrying body across pages.

    Returns (articulo, pagina, text). Footer/boilerplate lines are dropped
    line-by-line (no brittle length gate). `pagina` is the page on which the
    article's heading appears; the preamble keeps the document's first page.
    """
    current_art: str | None = None
    start_page = pages[0][0] if pages else 1
    buf: list[str] = []
    segments: list[tuple[str | None, int, str]] = []

    def flush() -> None:
        text = "\n".join(buf).strip()
        if text:
            segments.append((current_art, start_page, text))

    for pagina, page_text in pages:
        for raw in page_text.splitlines():
            line = raw.strip()
            if not line:
                buf.append("")          # keep paragraph breaks for _split_text
                continue
            if FOOTER_RE.search(line):
                continue                # drop address/phone/URL boilerplate
            m = HEADING_RE.match(line)
            if m:
                flush()
                buf = []
                current_art = f"Artículo {m.group(1)}"
                start_page = pagina
            buf.append(line)
    flush()
    return segments


def _split_text(text: str) -> list[str]:
    """Group paragraphs into ~TARGET_CHARS chunks (a short article stays whole)."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    buf = ""
    for para in paragraphs:
        buf = f"{buf}\n\n{para}".strip() if buf else para
        if len(buf) >= TARGET_CHARS:
            chunks.append(buf)
            buf = ""
    if buf:
        chunks.append(buf)
    return [c for c in chunks if len(c) >= MIN_CHUNK_CHARS]


def collect_rows() -> list[dict]:
    txt_files = sorted(PROCESSED_DIR.rglob("*.txt"))
    if not txt_files:
        print(f"! No .txt files under {PROCESSED_DIR} — run the OCR pipeline first.")
        return []
    rows: list[dict] = []
    for path in txt_files:
        body = _strip_header(path.read_text(encoding="utf-8"))
        for articulo, pagina, seg_text in _segments(_pages(body)):
            for chunk in _split_text(seg_text):
                rows.append({
                    "content": chunk,
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
    # Atomic delete+insert (explicit transaction over the autocommit pool) so a
    # re-run never leaves the table with the old seed deleted and only a partial
    # new seed inserted.
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
