# SPEC 01 — Database schema & setup

Postgres + pgvector on Supabase. One database, three logical groups of tables:
1. `documents` — RAG corpus (created here, **populated by a teammate's ingestion pipeline**).
2. `semantic_cache` — the semantic cache (created and used by this backend).
3. LangGraph checkpointer tables — conversation memory (created by `PostgresSaver.setup()`, **not** hand-written DDL).

Deliverables for Claude Code: `sql/01_schema.sql` and `scripts/setup_db.py`.

---

## 1. `sql/01_schema.sql`

Generate this file verbatim (adjust only if a comment tells you to):

```sql
-- ============================================================
-- UdeA FAQ backend — schema
-- Embedding model: text-embedding-3-small  ->  VECTOR(1536)
-- ============================================================

-- Extensions ---------------------------------------------------
CREATE EXTENSION IF NOT EXISTS vector;     -- pgvector
CREATE EXTENSION IF NOT EXISTS pgcrypto;   -- gen_random_uuid()

-- ------------------------------------------------------------
-- RAG corpus.
-- NOTE: created here so backend & ingestion agree on the contract,
-- but POPULATED by the ingestion pipeline (a teammate). Do not
-- insert rows from this backend.
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS documents (
    id          BIGSERIAL PRIMARY KEY,
    content     TEXT        NOT NULL,            -- the chunk text
    embedding   VECTOR(1536) NOT NULL,           -- text-embedding-3-small
    source      TEXT        NOT NULL,            -- e.g. 'Reglamento Estudiantil de Pregrado'
    nivel       TEXT,                            -- 'pregrado' | 'postgrado' | NULL
    articulo    TEXT,                            -- e.g. 'Artículo 45'
    pagina      INTEGER,                         -- page number, if known
    url         TEXT,                            -- source URL at normativa.udea.edu.co
    metadata    JSONB       NOT NULL DEFAULT '{}'::jsonb,  -- any extra fields
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Cosine ANN index (1536 dims <= 2000, so HNSW on `vector` works directly).
CREATE INDEX IF NOT EXISTS documents_embedding_hnsw
    ON documents USING hnsw (embedding vector_cosine_ops);

-- Optional helper index if you later filter retrieval by level.
CREATE INDEX IF NOT EXISTS documents_nivel_idx ON documents (nivel);

-- ------------------------------------------------------------
-- Semantic cache.
-- One row per indexed (novel) query. `response` holds the full
-- structured answer so cache hits can return citations too.
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS semantic_cache (
    id              UUID         PRIMARY KEY DEFAULT gen_random_uuid(),
    query_text      TEXT         NOT NULL,        -- the user query, verbatim
    query_embedding VECTOR(1536) NOT NULL,        -- embedding of query_text
    response        JSONB        NOT NULL,        -- {"answer": "...", "citations": [...]}
    embedding_model TEXT         NOT NULL DEFAULT 'text-embedding-3-small',
    created_at      TIMESTAMPTZ  NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS semantic_cache_embedding_hnsw
    ON semantic_cache USING hnsw (query_embedding vector_cosine_ops);
```

### Notes
- **Cosine distance:** query with the `<=>` operator. `similarity = 1 - (embedding <=> query)`.
- **HNSW on empty tables** is instant; the ingestion job inserts later. No reindex needed.
- **Vectors via psycopg:** the backend registers `pgvector.psycopg.register_vector` on each
  connection so embeddings can be passed as plain Python `list[float]` (see SPEC_02 §db). No `::vector`
  string casting is required in SQL when the value is bound as a vector param.
- **Dimensions:** if the embeddings deployment is ever changed, `VECTOR(1536)` and both HNSW indexes
  must be updated and the tables re-embedded. Keep `embedding_model` accurate.

---

## 2. Conversation-memory tables (checkpointer)

Do **not** hand-write the checkpointer DDL — its exact shape is owned by
`langgraph-checkpoint-postgres` and varies by version. Instead, call `.setup()` once. It creates
(idempotently) the tables LangGraph needs (e.g. `checkpoints`, `checkpoint_blobs`,
`checkpoint_writes`, `checkpoint_migrations`).

The bootstrap script below both runs `01_schema.sql` and calls `.setup()`.

---

## 3. `scripts/setup_db.py`

Generate a script equivalent to this (adapt imports/paths as needed):

```python
"""One-shot DB bootstrap: app schema + LangGraph checkpointer tables. Idempotent."""
from pathlib import Path

import psycopg
from psycopg.rows import dict_row
from langgraph.checkpoint.postgres import PostgresSaver

from app.config import settings  # exposes settings.database_url

SQL_FILE = Path(__file__).resolve().parent.parent / "sql" / "01_schema.sql"


def run_schema() -> None:
    sql = SQL_FILE.read_text(encoding="utf-8")
    with psycopg.connect(settings.database_url, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(sql)
    print("✓ app schema (documents, semantic_cache, indexes) created")


def run_checkpointer_setup() -> None:
    # PostgresSaver needs autocommit + dict_row connections.
    with PostgresSaver.from_conn_string(settings.database_url) as checkpointer:
        checkpointer.setup()
    print("✓ checkpointer tables created")


if __name__ == "__main__":
    run_schema()
    run_checkpointer_setup()
    print("Done.")
```

> If `PostgresSaver.from_conn_string` does not yield connections with the required settings in your
> installed version, open the connection explicitly with
> `psycopg.connect(settings.database_url, autocommit=True, row_factory=dict_row)` and pass it to
> `PostgresSaver(conn)` before calling `.setup()`.

### Run
```bash
python scripts/setup_db.py
```

---

## 4. (Manual) seed rows for local testing

So the team can test `POST /chat` before ingestion is ready, provide a tiny optional seed in the
README (NOT in `01_schema.sql`). Embeddings must be generated with the SAME model; the simplest is a
3–4 line Python snippet that embeds a couple of fake regulation chunks with `AzureOpenAIEmbeddings`
and inserts them into `documents`. Keep it clearly marked as throwaway test data.
