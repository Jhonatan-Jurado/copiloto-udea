"""Reusable ingestion helpers shared by the bulk CLI (scripts/ingest_documents.py)
and the upload endpoint (app/main.py).

Deriving metadata from filenames, batch-embedding, and inserting `documents` rows
with the explicit `%(emb)s::vector` cast. Kept free of top-level `app.db` /
`app.azure_clients` imports so importing this module never opens the DB pool or
builds Azure clients (the CLI's --dry-run / seed paths rely on that); `pool` is
imported lazily inside the insert helper, and the embeddings client is passed in.
"""
import re
import time

DEFAULT_URL = "https://normativa.udea.edu.co/Documentos/Consultar"

EMBED_BATCH = 128          # chunks per embed_documents call
MAX_RETRIES = 6
BASE_DELAY = 2.0           # seconds, exponential backoff

# ── filename -> source title ────────────────────────────────────────────────
_TIPO = {
    "acuerdo_superior":  "Acuerdo Superior",
    "acuerdo_academico": "Acuerdo Académico",
    "resolucion":        "Resolución",
}
# <tipo>_<number>_<optional alpha suffixes>
_NUM_RE = re.compile(r"^(?P<tipo>[a-z_]+?)_(?P<num>\d+)(?P<rest>(?:_[a-z]+)*)$")

_DOCHEAD_RE = re.compile(
    r"\b(ACUERDO\s+SUPERIOR|ACUERDO\s+ACAD[ÉE]MICO|RESOLUCI[ÓO]N)\s+(\d+)", re.IGNORECASE
)
_YEAR_RE = re.compile(r"\bde\s+(\d{4})\b", re.IGNORECASE)


def _title_from_content(stem: str, first_lines: str) -> str:
    m = _DOCHEAD_RE.search(first_lines)
    if not m:
        return f"Documento {stem.split('_')[-1]}"
    kind = re.sub(r"\s+", " ", m.group(1)).title()
    num = str(int(m.group(2)))
    y = _YEAR_RE.search(first_lines)
    return f"{kind} {num}" + (f" de {y.group(1)}" if y else "")


def parse_source_title(stem: str, first_lines: str = "") -> str:
    stem = stem.strip().lower()
    m = _NUM_RE.match(stem)
    label = _TIPO.get(m.group("tipo")) if m else None
    if not m or label is None:
        # opaque stem (e.g. documento_NNN): derive from content, else humanize.
        title = _title_from_content(stem, first_lines)
        return title if title.startswith(("Acuerdo", "Resolución")) else stem.replace("_", " ").title()

    num = str(int(m.group("num")))           # 019 -> "19"
    rest = m.group("rest") or ""
    tags: list[str] = []
    if "_viejo" in rest:
        tags.append("versión anterior")
    if "_nuevo" in rest:
        tags.append("versión nueva")
    if "_con_concordancias" in rest:
        tags.append("con concordancias")
    if "_sin_concordancias" in rest:
        tags.append("sin concordancias")
    title = f"{label} {num}"
    if tags:
        title += " (" + ", ".join(tags) + ")"
    return title


def nivel_from_path(path) -> str:
    """Derive nivel from a raw/processed path's folder components."""
    parts = {p.lower() for p in path.parts}
    if "reglamento_estudiantil_posgrado" in parts:
        return "posgrado"
    # pregrado folder, or the lone reglamento/ file -> pregrado.
    return "pregrado"


# ── embedding ────────────────────────────────────────────────────────────────
def _embed_with_retry(embeddings, batch: list[str]) -> list[list[float]]:
    for attempt in range(MAX_RETRIES):
        try:
            return embeddings.embed_documents(batch)
        except Exception as e:                      # rate limit / transient 5xx
            if attempt == MAX_RETRIES - 1:
                raise
            delay = BASE_DELAY * (2 ** attempt)
            print(f"  ! embed retry {attempt + 1}/{MAX_RETRIES}: {e} (sleep {delay:.0f}s)")
            time.sleep(delay)


def embed_all(embeddings, texts: list[str]) -> list[list[float]]:
    vectors: list[list[float]] = []
    total = len(texts)
    for start in range(0, total, EMBED_BATCH):
        vectors.extend(_embed_with_retry(embeddings, texts[start : start + EMBED_BATCH]))
        if total > EMBED_BATCH:
            print(f"  embedded {min(start + EMBED_BATCH, total)}/{total} chunk(s)")
    return vectors


# ── insert ───────────────────────────────────────────────────────────────────
INSERT_SQL = """
    INSERT INTO documents (content, embedding, source, nivel, articulo, pagina, url, metadata)
    VALUES (%(content)s, %(embedding)s::vector, %(source)s, %(nivel)s, %(articulo)s,
            %(pagina)s, %(url)s, %(metadata)s)
"""


def replace_document_rows(origen: str, rows: list[dict], vectors: list[list[float]]) -> int:
    """Idempotent per-file upsert: delete prior rows for this `origen`, then insert,
    in ONE transaction over the shared autocommit pool. Binds `::vector`. Returns
    the number of rows inserted. `rows` items carry content/source/nivel/articulo/
    pagina/url/metadata; `metadata` must already be a dict.
    """
    from psycopg.types.json import Jsonb
    from app.db import pool
    with pool.connection() as conn:
        with conn.transaction():
            conn.execute(
                "DELETE FROM documents WHERE metadata->>'origen' = %(origen)s",
                {"origen": origen},
            )
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
    return len(rows)
