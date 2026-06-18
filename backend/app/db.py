"""Shared Postgres connection pool (SPEC_02 §2).

A single `psycopg_pool.ConnectionPool` is used by the cache, the retrieval
tool, AND the LangGraph checkpointer. `register_vector` runs on every new
connection so `VECTOR` columns parse back into Python. NOTE: this pgvector
version does not auto-adapt a plain `list[float]` on the way *in*, so the
queries bind embeddings with an explicit `%(emb)s::vector` cast (see cache.py /
retrieval.py). `autocommit=True` + `dict_row` are also what `PostgresSaver`
expects — that's why the pool is shared.
"""
from psycopg_pool import ConnectionPool
from psycopg.rows import dict_row
from pgvector.psycopg import register_vector

from app.config import settings

pool = ConnectionPool(
    conninfo=settings.database_url,
    min_size=1,
    max_size=10,
    kwargs={"autocommit": True, "row_factory": dict_row},
    configure=register_vector,   # registers the pgvector type adapter per connection
    open=True,
)
