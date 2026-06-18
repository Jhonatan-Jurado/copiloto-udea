"""The RAG retrieval tool (SPEC_02 §5).

A single LangChain tool the agent calls (up to MAX_TOOL_CALLS times). It embeds
its argument, runs a cosine vector search over `documents`, and returns the
chunks as a JSON string that includes citation metadata. The backend later
parses the same JSON shape to build the response's `citations` array, so
citations are grounded in real rows — not the model's prose.
"""
import json

from langchain_core.tools import tool

from app.db import pool
from app.azure_clients import make_embeddings
from app.config import settings

_embeddings = make_embeddings()   # reuse one client


def _search_documents(query: str, top_k: int) -> list[dict]:
    emb = _embeddings.embed_query(query)
    sql = """
        SELECT id, content, source, nivel, articulo, pagina, url,
               (embedding <=> %(emb)s::vector) AS distance
        FROM documents
        ORDER BY embedding <=> %(emb)s::vector
        LIMIT %(k)s
    """
    with pool.connection() as conn:
        rows = conn.execute(sql, {"emb": emb, "k": top_k}).fetchall()
    out = []
    for r in rows:
        out.append({
            "source": r["source"],
            "nivel": r.get("nivel"),
            "articulo": r.get("articulo"),
            "pagina": r.get("pagina"),
            "url": r.get("url"),
            "snippet": r["content"],
            "similarity": round(max(0.0, min(1.0, 1.0 - float(r["distance"]))), 4),
        })
    return out


@tool
def buscar_reglamento(consulta: str) -> str:
    """Busca en el reglamento estudiantil de la Universidad de Antioquia (pregrado y postgrado).
    Devuelve los fragmentos más relevantes con su fuente, artículo, página y URL.
    Úsala siempre antes de responder y reformula la consulta si los resultados no son suficientes.

    Args:
        consulta: la pregunta o términos a buscar, en español.
    """
    results = _search_documents(consulta, settings.rag_top_k)
    if not results:
        return json.dumps({"resultados": [], "nota": "Sin coincidencias en el reglamento."},
                          ensure_ascii=False)
    return json.dumps({"resultados": results}, ensure_ascii=False)
