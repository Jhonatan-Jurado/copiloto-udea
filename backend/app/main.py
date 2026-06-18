"""FastAPI app & orchestration (SPEC_02 §8, flow per SPEC_00 §4).

POST /chat: embed -> cache search -> (hit: update_state + return cached) /
(miss: agent.invoke -> parse final answer + grounded citations -> index ->
return). GraphRecursionError is caught and turned into a graceful Spanish
message (acceptance #6).
"""
import json
import logging
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

logger = logging.getLogger("udea-faq")

app = FastAPI(title="UdeA FAQ backend")
_embeddings = make_embeddings()

RECURSION_FALLBACK = (
    "No pude completar la búsqueda en el reglamento en esta ocasión. "
    "Por favor reformula tu pregunta o consulta la dependencia correspondiente."
)
EMPTY_FALLBACK = (
    "No encontré esta información en el reglamento. "
    "Te sugiero consultar la dependencia correspondiente."
)


class ChatRequest(BaseModel):
    query: str
    session_id: str | None = None


def _content_to_text(content) -> str:
    """Flatten a LangChain message `content` to plain text.

    Reasoning models (gpt-5-nano) on LangChain 1.0 may return `content` as a
    list of content blocks; concatenate their text rather than str(list), which
    would leak a Python repr into the answer (and the cache).
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
        return "".join(parts).strip()
    return "" if content is None else str(content)


def _current_turn(messages: list) -> list:
    """Return only the messages produced by the latest turn.

    With a PostgresSaver checkpointer, agent.invoke returns the FULL thread
    history. Scope answer/citation extraction to the messages after the last
    HumanMessage (the query we just sent) so we don't harvest citations from
    earlier, unrelated questions in the same session.
    """
    last_human = -1
    for i, m in enumerate(messages):
        if isinstance(m, HumanMessage):
            last_human = i
    return messages[last_human + 1:] if last_human >= 0 else messages


def _parse_agent_result(result: dict) -> tuple[str, list[dict]]:
    """Final answer = last AIMessage of this turn. Citations = this turn's tool outputs."""
    messages = _current_turn(result["messages"])

    answer = ""
    for m in reversed(messages):
        if isinstance(m, AIMessage):
            text = _content_to_text(m.content)
            if text:
                answer = text
                break

    citations: list[dict] = []
    seen = set()
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
        # Keep multi-turn memory coherent without calling the model. Never let a
        # checkpointer hiccup sink an answer we already have in hand.
        try:
            agent.update_state(config, {"messages": [HumanMessage(req.query), AIMessage(answer)]})
        except Exception:
            logger.warning("update_state failed on cache hit (session=%s)", session_id, exc_info=True)
        latency = int((time.perf_counter() - t0) * 1000)
        metrics.record("cache", score, latency)
        return {"answer": answer, "citations": citations, "source": "cache",
                "score": round(score, 4), "session_id": session_id, "latency_ms": latency}

    # 4) cache miss -> agent
    recursion_failed = False
    try:
        result = agent.invoke({"messages": [HumanMessage(req.query)]}, config=config)
        answer, citations = _parse_agent_result(result)
    except GraphRecursionError:
        answer, citations, recursion_failed = RECURSION_FALLBACK, [], True

    # Index ONLY genuine answers (single threshold: a miss is novel). Never cache
    # the recursion fallback or an empty answer, or we'd permanently serve a
    # transient failure to every semantically-similar future question.
    if answer and not recursion_failed:
        cache.upsert(req.query, emb, {"answer": answer, "citations": citations})
    elif not answer:
        answer, citations = EMPTY_FALLBACK, []

    latency = int((time.perf_counter() - t0) * 1000)
    metrics.record("agent", score, latency)
    return {"answer": answer, "citations": citations, "source": "agent",
            "score": round(score, 4), "session_id": session_id, "latency_ms": latency}
