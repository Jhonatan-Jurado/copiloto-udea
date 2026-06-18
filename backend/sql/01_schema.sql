-- ============================================================
-- UdeA FAQ backend — schema
-- Embedding model: text-embedding-3-small  ->  VECTOR(1536)
-- Idempotent: safe to re-run (CREATE ... IF NOT EXISTS).
-- NOTE: LangGraph checkpointer tables are NOT created here; they
--       are created by PostgresSaver.setup() in scripts/setup_db.py.
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
