# SPEC 02 — Backend implementation

FastAPI + LangChain 1.0 + Azure OpenAI + pgvector. Implement the modules under `app/` exactly as
described. Snippets below are **reference implementations** for the tricky parts — adapt names freely,
but preserve the behavior, the API contract (SPEC_00 §3), and the flow (SPEC_00 §4).

General rules:
- Sync code throughout (FastAPI sync handlers). Keep it simple.
- All tunables come from `app/config.py` (env). No magic numbers in logic.
- The app responds in **Spanish**; code, comments and identifiers are in English.

---

## 1. `app/config.py` — settings

Use `pydantic-settings`. Load `.env`. Expose a singleton `settings`.

```python
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Azure OpenAI
    azure_openai_endpoint: str
    azure_openai_api_key: str
    azure_openai_api_version: str = "2025-04-01-preview"
    azure_openai_chat_deployment: str = "gpt-5-nano"
    azure_openai_embeddings_deployment: str = "text-embedding-3-small"

    # Model behavior
    reasoning_effort: str = "minimal"   # minimal | low | medium | high
    embedding_dim: int = 1536

    # Database
    database_url: str

    # Semantic cache (single threshold: serve == index)
    semantic_cache_threshold: float = 0.85
    semantic_cache_top_k: int = 5

    # Agentic RAG
    rag_top_k: int = 5
    max_tool_calls: int = 10

    # App
    app_host: str = "0.0.0.0"
    app_port: int = 8000


settings = Settings()
```

---

## 2. `app/db.py` — connection pool

A single shared `psycopg_pool.ConnectionPool`, used by the cache, the retrieval tool, **and** the
checkpointer. Register pgvector on every connection so embeddings bind as vectors.

```python
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
```
- `autocommit=True` + `dict_row` are also what `PostgresSaver` expects — that's why we share this pool.
- `register_vector` lets you pass `list[float]` directly as a `%s` / named param of vector type.

---

## 3. `app/azure_clients.py` — model + embeddings factories

**gpt-5-nano is a reasoning model** (SPEC_00 §6.1): no `temperature`, no `max_tokens`; pass
`reasoning_effort`.

```python
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from app.config import settings


def make_chat_model() -> AzureChatOpenAI:
    # NOTE: do NOT pass temperature / max_tokens — gpt-5-nano (reasoning model) rejects them.
    return AzureChatOpenAI(
        azure_endpoint=settings.azure_openai_endpoint,
        api_key=settings.azure_openai_api_key,
        api_version=settings.azure_openai_api_version,
        azure_deployment=settings.azure_openai_chat_deployment,
        reasoning_effort=settings.reasoning_effort,   # minimal -> fast & cheap, sequential tool calls
    )


def make_embeddings() -> AzureOpenAIEmbeddings:
    return AzureOpenAIEmbeddings(
        azure_endpoint=settings.azure_openai_endpoint,
        api_key=settings.azure_openai_api_key,
        api_version=settings.azure_openai_api_version,
        azure_deployment=settings.azure_openai_embeddings_deployment,
        # text-embedding-3-small defaults to 1536 dims; do not override.
    )
```
> If your installed `langchain-openai` does not accept `reasoning_effort` as a direct kwarg, pass it
> via `model_kwargs={"reasoning_effort": settings.reasoning_effort}` instead. If even that 400s on a
> given API version, set `REASONING_EFFORT=low` and/or bump `AZURE_OPENAI_API_VERSION`.

Create the embeddings client once at startup and reuse it (it's used by both the cache and the tool).

---

## 4. `app/cache.py` — semantic cache

Two operations over `semantic_cache`: `search` (top-k by cosine) and `upsert` (insert a novel entry).
Embeddings are passed as `list[float]` (pgvector adapter handles binding).

```python
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
               (query_embedding <=> %(emb)s) AS distance
        FROM semantic_cache
        ORDER BY query_embedding <=> %(emb)s
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
        similarity=1.0 - float(best["distance"]),
    )


def upsert(query_text: str, embedding: list[float], response: dict) -> None:
    sql = """
        INSERT INTO semantic_cache (query_text, query_embedding, response)
        VALUES (%(q)s, %(emb)s, %(resp)s)
    """
    with pool.connection() as conn:
        conn.execute(sql, {"q": query_text, "emb": embedding, "resp": Jsonb(response)})
```
- The explicit `ORDER BY query_embedding <=> %(emb)s ... LIMIT k` guarantees the HNSW index is used.
- `top_k` is read from `settings.semantic_cache_top_k`; we only act on top-1 for the threshold check,
  but returning a few keeps it easy to inspect/log later.

---

## 5. `app/retrieval.py` — the RAG tool

A single LangChain tool that the agent calls (up to `MAX_TOOL_CALLS` times). It embeds its argument,
does a vector search over `documents`, and returns the chunks **as a JSON string** that includes
citation metadata. The same JSON shape is later parsed by the backend to build the response's
`citations` array (so citations are grounded in real rows, not the model's prose).

```python
import json
from langchain_core.tools import tool

from app.db import pool
from app.azure_clients import make_embeddings

_embeddings = make_embeddings()   # reuse one client


def _search_documents(query: str, top_k: int) -> list[dict]:
    emb = _embeddings.embed_query(query)
    sql = """
        SELECT id, content, source, nivel, articulo, pagina, url,
               (embedding <=> %(emb)s) AS distance
        FROM documents
        ORDER BY embedding <=> %(emb)s
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
            "similarity": round(1.0 - float(r["distance"]), 4),
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
    from app.config import settings
    results = _search_documents(consulta, settings.rag_top_k)
    if not results:
        return json.dumps({"resultados": [], "nota": "Sin coincidencias en el reglamento."},
                          ensure_ascii=False)
    return json.dumps({"resultados": results}, ensure_ascii=False)
```
- The docstring is the tool description the model reads — keep it instructive (it nudges the agent to
  always retrieve and to reformulate).
- `RAG_TOP_K` chunks per call.

---

## 6. `app/agent.py` — the agent

Build a LangChain 1.0 agent with `create_agent`, the tool, the `PostgresSaver` checkpointer, and a
Spanish, strongly-grounded system prompt.

```python
from langgraph.checkpoint.postgres import PostgresSaver

from langchain.agents import create_agent
from app.azure_clients import make_chat_model
from app.retrieval import buscar_reglamento
from app.db import pool

SYSTEM_PROMPT = """Eres un asistente experto en el reglamento estudiantil de pregrado y postgrado \
de la Universidad de Antioquia. Tu único objetivo es responder preguntas administrativas y \
normativas de la comunidad universitaria.

Reglas:
- Responde SIEMPRE en español, de forma clara, precisa y con tono institucional.
- Basa tu respuesta EXCLUSIVAMENTE en la información recuperada con la herramienta `buscar_reglamento`.
  Usa la herramienta al menos una vez antes de responder; nunca respondas de memoria.
- Puedes llamar la herramienta varias veces (reformulando la consulta) hasta un máximo de 10 veces.
- Cita las fuentes dentro de tu respuesta (p. ej. "según el Artículo 45 del Reglamento de Pregrado").
- Si tras buscar no encuentras la información, dilo explícitamente: "No encontré esta información en \
  el reglamento" y sugiere consultar la dependencia correspondiente. No inventes artículos ni datos.
"""

# Checkpointer shares the app pool (autocommit + dict_row already configured in app/db.py).
checkpointer = PostgresSaver(pool)

agent = create_agent(
    model=make_chat_model(),
    tools=[buscar_reglamento],
    system_prompt=SYSTEM_PROMPT,
    checkpointer=checkpointer,
)
```
> Run `scripts/setup_db.py` once before serving so the checkpointer tables exist (it calls
> `PostgresSaver(...).setup()`). Do not call `.setup()` on every request.

---

## 7. `app/metrics.py` — tiny metrics

Keep it minimal: process-level counters + one structured log line per request.

```python
import logging
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("udea-faq")


@dataclass
class Counters:
    total: int = 0
    cache_hits: int = 0
    agent_calls: int = 0


counters = Counters()


def record(source: str, score: float, latency_ms: int) -> None:
    counters.total += 1
    if source == "cache":
        counters.cache_hits += 1
    else:
        counters.agent_calls += 1
    logger.info("source=%s score=%.4f latency_ms=%d", source, score, latency_ms)


def snapshot() -> dict:
    hit_rate = counters.cache_hits / counters.total if counters.total else 0.0
    return {
        "total": counters.total,
        "cache_hits": counters.cache_hits,
        "agent_calls": counters.agent_calls,
        "hit_rate": round(hit_rate, 4),
    }
```

---

## 8. `app/main.py` — FastAPI app & orchestration

Wires everything per SPEC_00 §4. Includes: embed → cache search → (hit: `update_state` + return) /
(miss: agent.invoke → parse → index → return). Catches `GraphRecursionError`.

```python
import time
import uuid

from fastapi import FastAPI
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langgraph.errors import GraphRecursionError

from app.config import settings
from app.azure_clients import make_embeddings
from app.agent import agent
from app import cache, metrics

app = FastAPI(title="UdeA FAQ backend")
_embeddings = make_embeddings()


class ChatRequest(BaseModel):
    query: str
    session_id: str | None = None


def _parse_agent_result(result: dict) -> tuple[str, list[dict]]:
    """Final answer = last AIMessage content. Citations = sources from all tool outputs."""
    messages = result["messages"]
    answer = ""
    for m in reversed(messages):
        if isinstance(m, AIMessage) and m.content:
            answer = m.content if isinstance(m.content, str) else str(m.content)
            break

    citations: list[dict] = []
    seen = set()
    import json
    for m in messages:
        if isinstance(m, ToolMessage) and m.name == "buscar_reglamento":
            try:
                payload = json.loads(m.content)
            except Exception:
                continue
            for r in payload.get("resultados", []):
                key = (r.get("source"), r.get("articulo"), r.get("pagina"))
                if key in seen:
                    continue
                seen.add(key)
                citations.append({
                    "source": r.get("source"),
                    "nivel": r.get("nivel"),
                    "articulo": r.get("articulo"),
                    "pagina": r.get("pagina"),
                    "url": r.get("url"),
                    "snippet": r.get("snippet"),
                })
    return answer, citations


@app.get("/health")
def health() -> dict:
    from app.db import pool
    with pool.connection() as conn:
        conn.execute("SELECT 1")
    return {"status": "ok"}


@app.get("/stats")
def stats() -> dict:
    return metrics.snapshot()


@app.post("/chat")
def chat(req: ChatRequest) -> dict:
    t0 = time.perf_counter()
    session_id = req.session_id or str(uuid.uuid4())
    config = {
        "configurable": {"thread_id": session_id},
        "recursion_limit": 2 * settings.max_tool_calls + 1,
    }

    # 1) embed
    emb = _embeddings.embed_query(req.query)

    # 2) cache search
    best = cache.search(emb, settings.semantic_cache_top_k)
    score = best.similarity if best else 0.0

    # 3) cache hit
    if best and best.similarity >= settings.semantic_cache_threshold:
        answer = best.response.get("answer", "")
        citations = best.response.get("citations", [])
        # keep multi-turn memory coherent without calling the model
        agent.update_state(config, {"messages": [HumanMessage(req.query), AIMessage(answer)]})
        latency = int((time.perf_counter() - t0) * 1000)
        metrics.record("cache", score, latency)
        return {"answer": answer, "citations": citations, "source": "cache",
                "score": round(score, 4), "session_id": session_id, "latency_ms": latency}

    # 4) cache miss -> agent
    try:
        result = agent.invoke({"messages": [HumanMessage(req.query)]}, config=config)
        answer, citations = _parse_agent_result(result)
    except GraphRecursionError:
        answer = ("No pude completar la búsqueda en el reglamento en esta ocasión. "
                  "Por favor reformula tu pregunta o consulta la dependencia correspondiente.")
        citations = []

    # index (single threshold: this was a miss, so it is novel)
    if answer:
        cache.upsert(req.query, emb, {"answer": answer, "citations": citations})

    latency = int((time.perf_counter() - t0) * 1000)
    metrics.record("agent", score, latency)
    return {"answer": answer, "citations": citations, "source": "agent",
            "score": round(score, 4), "session_id": session_id, "latency_ms": latency}
```

> `from langchain_core.messages import ...` is stable; in LangChain 1.0 `from langchain.messages import ...`
> also works. `GraphRecursionError` lives in `langgraph.errors`.

---

## 9. README.md (generate a short one)

Include: prerequisites; `pip install -r requirements.txt`; copy `.env.example` → `.env` and fill it;
the Supabase **Session-pooler** connection-string note (SPEC_00 §6.2); `python scripts/setup_db.py`;
`uvicorn app.main:app --host 0.0.0.0 --port 8000`; a `curl` example for `POST /chat`; and the optional
throwaway seed snippet for `documents`.

---

## 10. Acceptance criteria
Same as SPEC_00 §10. In particular, verify:
- No `temperature`/`max_tokens` reaches `gpt-5-nano`.
- `source` flips from `agent` to `cache` when the same question is reworded.
- `citations` is non-empty and grounded in `documents` rows.
- Same `session_id` → the agent uses prior context.
- Tool calls capped at `MAX_TOOL_CALLS`; `GraphRecursionError` handled gracefully.
