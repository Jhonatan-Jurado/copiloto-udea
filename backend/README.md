# UdeA Agentic FAQ — Backend

FastAPI backend that answers questions (in Spanish) about Universidad de Antioquia's
student regulations. Two layers:

1. **Semantic cache** (pgvector, single threshold) — repeated/similar questions are
   served instantly with ~zero generation tokens.
2. **Agentic RAG** (LangChain 1.0 `create_agent`) — on a cache miss, an agent calls a
   single retrieval tool over a `documents` table up to `MAX_TOOL_CALLS` times, answers
   **with grounded citations**, and writes the answer back to the cache.

Conversation memory is a LangGraph `PostgresSaver` checkpointer. See `specs/` for the
full design (SPEC_00 overview, SPEC_01 schema, SPEC_02 modules).

> Run every command below **from the `backend/` directory** so the `app` package is
> importable and `.env` is found.

---

## Prerequisites

- Python **3.12**
- A **PostgreSQL + pgvector** database — **Supabase** recommended.
- **Azure OpenAI** with two deployments: a chat model (`gpt-5-nano`) and an embeddings
  model (`text-embedding-3-small`).

---

## 1. Install

```bash
cd backend
python -m venv .venv
# Windows:  .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate
pip install -r requirements.txt
```

After the first successful install, pin exact versions for the rest of the hackathon:

```bash
pip freeze > requirements.lock.txt
```

## 2. Configure

```bash
cp .env.example .env   # Windows: copy .env.example .env
```

Fill in `.env`:

- **Azure**: endpoint, API key, API version, and the two deployment names.
- **`DATABASE_URL`**: use the Supabase **Session-mode pooler** URI
  (Project Settings → Database → Connection string), on **port 5432**, and always
  include `?sslmode=require`. Session mode is required because the `PostgresSaver`
  checkpointer relies on prepared statements. Avoid the Transaction pooler (port 6543):
  it breaks those prepared statements. If you truly have no choice, disable prepared
  statements via psycopg's `prepare_threshold=None` (a client kwarg in the pool's
  `kwargs`, **not** a URI query parameter — and note `prepare_threshold=0` does *not*
  disable them, it eager-prepares). Session mode is strongly preferred. (See SPEC_00 §6.2.)

> `gpt-5-nano` is a reasoning model: the backend never sends `temperature`/`max_tokens`.
> Tune cost/latency with `REASONING_EFFORT` (`minimal` → `low`/`medium`/`high`) — no code
> change. (SPEC_00 §6.1.)

## 3. Create the schema + checkpointer tables (idempotent)

```bash
python scripts/setup_db.py
```

Creates the `vector`/`pgcrypto` extensions, the `documents` and `semantic_cache` tables
with HNSW cosine indexes, and the LangGraph checkpointer tables. Safe to re-run.

## 4. (Dev only) Seed `documents` so you can test `/chat`

The real `documents` corpus is populated by a separate ingestion pipeline. Until that's
ready, seed throwaway rows from the OCR-processed text so `/chat` returns grounded
citations:

```bash
python scripts/seed_documents.py
```

This reads `../preprocessing/data/processed/**/*.txt`, chunks + embeds them with
`text-embedding-3-small`, and inserts rows marked `metadata->>'seed' = 'true'`
(re-runnable; it clears its own previous seed rows and never touches real ingestion data).

> Alternatively, hand-seed a couple of fake chunks — embeddings **must** use the same
> model (`text-embedding-3-small`, 1536 dims) as the cache, or retrieval breaks.

## 5. Run

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

---

## API

### `GET /health`
```bash
curl http://localhost:8000/health
# {"status":"ok"}
```

### `POST /chat`
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "¿Cuántos créditos como máximo puedo matricular en un semestre?"}'
```

Response:
```json
{
  "answer": "Texto de la respuesta en español...",
  "citations": [
    {
      "source": "Reglamento Estudiantil de Pregrado",
      "nivel": "pregrado",
      "articulo": "Artículo 130",
      "pagina": 3,
      "url": "https://normativa.udea.edu.co/...",
      "snippet": "fragmento textual recuperado..."
    }
  ],
  "source": "agent",
  "score": 0.0,
  "session_id": "generated-or-supplied-id",
  "latency_ms": 1234
}
```

- `session_id` is optional; if omitted, the server generates one (uuid4) and returns it.
  Reuse it on follow-up questions to get conversation memory.
- `source` is `"cache"` or `"agent"`; `score` is the top-1 cache similarity (0..1) for
  this query — logged on every request so you can tune `SEMANTIC_CACHE_THRESHOLD` from
  real data.

### `GET /stats`
```bash
curl http://localhost:8000/stats
# {"total": 5, "cache_hits": 2, "agent_calls": 3, "hit_rate": 0.4}
```
In-process counters; reset on restart.

---

## Quick acceptance walkthrough

1. `python scripts/setup_db.py` → tables created (re-run: no errors).
2. `uvicorn app.main:app` → `GET /health` is `ok`.
3. First `POST /chat` → `source="agent"`, non-empty `citations`.
4. Re-ask the same question reworded → `source="cache"`, `score >= 0.85`.
5. Follow-up with the same `session_id` → answer uses prior context.
6. `GET /stats` → reflects hits vs agent calls.
