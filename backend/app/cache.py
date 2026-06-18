"""Semantic cache over `semantic_cache` (SPEC_02 §4).

Two operations: `search` (top-k by cosine, we only act on top-1) and `upsert`
(insert a novel entry on a cache miss). Embeddings are passed as `list[float]`;
the pgvector adapter registered in app/db.py handles binding.
"""
from dataclasses import dataclass
from typing import Optional
import json

from psycopg.types.json import Jsonb

from app.db import pool


@dataclass
class CacheMatch:
    id: str
    query_text: str
    response: dict          # {"answer": str, "citations": list}
    similarity: float       # 0..1


def search(embedding: list[float], top_k: int) -> Optional[CacheMatch]:
    """Return the single best match (top-1), or None if the cache is empty."""
    sql = """
        SELECT id, query_text, response,
               (query_embedding <=> %(emb)s::vector) AS distance
        FROM semantic_cache
        ORDER BY query_embedding <=> %(emb)s::vector
        LIMIT %(k)s
    """
    with pool.connection() as conn:
        rows = conn.execute(sql, {"emb": embedding, "k": top_k}).fetchall()
    if not rows:
        return None
    best = rows[0]
    resp = best["response"]
    if isinstance(resp, str):       # psycopg may return JSONB as str depending on setup
        resp = json.loads(resp)
    return CacheMatch(
        id=str(best["id"]),
        query_text=best["query_text"],
        response=resp,
        # Clamp: pgvector cosine distance is in [0, 2], so similarity is in
        # [-1, 1]; the API contract (SPEC_00 §3) documents score as 0..1.
        similarity=max(0.0, min(1.0, 1.0 - float(best["distance"]))),
    )


def upsert(query_text: str, embedding: list[float], response: dict) -> None:
    """Index a novel (miss) query and its structured response."""
    sql = """
        INSERT INTO semantic_cache (query_text, query_embedding, response)
        VALUES (%(q)s, %(emb)s::vector, %(resp)s)
    """
    with pool.connection() as conn:
        conn.execute(sql, {"q": query_text, "emb": embedding, "resp": Jsonb(response)})
