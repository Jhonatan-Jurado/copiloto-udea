# SPEC 00 — Overview & Architecture

**Project:** Agentic FAQ backend for Universidad de Antioquia student regulations
**Event:** Hackathon "Desafío Agéntico 2026" (SoftServe & UdeA) — 24h MVP
**This spec set:** `SPEC_00_overview.md` (this file), `SPEC_01_db_schema.md`, `SPEC_02_backend.md`

> Read all three specs before writing code. SPEC_00 = the big picture + global rules. SPEC_01 = SQL & DB setup. SPEC_02 = module-by-module implementation.

---

## 1. What we are building

A **FastAPI backend** that answers natural-language questions (in Spanish) about UdeA's
undergraduate and graduate student regulations ("reglamento estudiantil de pregrado y postgrado").
Public source corpus: <https://normativa.udea.edu.co/Documentos/Consultar>.

Two layers, in order:

1. **Semantic cache** — repeated/similar questions are answered instantly from a `pgvector`
   table, with (almost) zero generation tokens. Based on the dual-threshold / ports pattern from
   <https://medium.com/@juanjo.barrientos/semantic-caching-for-llms-answering-common-requests-with-low-latency-and-almost-zero-tokens-e8512d77c271>,
   simplified to a **single threshold** for this MVP.
2. **Agentic RAG agent** — on a cache **miss**, a LangChain 1.0 agent reasons over the question and
   calls a single retrieval tool against a `pgvector` document store **up to 10 times**, then answers
   **with citations** to the regulation documents it used. The new answer is written back to the cache.

### Explicitly OUT of scope for this spec (other teammates own these)
- **Frontend** (React / Chainlit). We only expose the HTTP API.
- **Ingestion pipeline** that *populates* the `documents` table (download PDFs, OCR/extract, chunk,
  embed, insert). We **define and create** the `documents` table so both sides align (see SPEC_01),
  but we do **NOT** implement ingestion. Assume a teammate fills `documents` with the same embedding
  model and dimension specified here.

---

## 2. Fixed technical decisions

| Area | Decision |
|---|---|
| Language / runtime | Python **3.12**, **FastAPI** (sync handlers — simplest for the demo) |
| Agent framework | **LangChain 1.0** `langchain.agents.create_agent` (built on **LangGraph 1.0**) |
| Chat model | Azure OpenAI **`gpt-5-nano`** (a reasoning model — see §6) |
| Embeddings | Azure OpenAI **`text-embedding-3-small`** → **1536 dims** |
| Vector DB | **PostgreSQL + pgvector** on **Supabase** (one DB, several tables) |
| Conversation memory | LangGraph **`PostgresSaver`** checkpointer (Postgres-backed) |
| Cache thresholds | **Single threshold** (`serve == index`), value tuned empirically via env |
| Retrieval | **Vector-only** (cosine), single tool, top-k configurable |
| Streaming | **Out** for the MVP (plain JSON response) |
| Architecture | **Simple, flat modules** under `app/` — no heavy abstractions/ports ceremony |

---

## 3. HTTP API contract

### `POST /chat`
Request body:
```json
{
  "query": "¿Cuántos créditos como máximo puedo matricular en un semestre?",
  "session_id": "optional-client-supplied-id"
}
```
- `query` (str, required): the user's question, verbatim.
- `session_id` (str, optional): conversation id. Maps 1:1 to the LangGraph `thread_id`
  (conversation memory). If omitted, the server generates one (uuid4) and returns it.

Response body:
```json
{
  "answer": "Texto de la respuesta en español...",
  "citations": [
    {
      "source": "Reglamento Estudiantil de Pregrado",
      "nivel": "pregrado",
      "articulo": "Artículo 45",
      "pagina": 12,
      "url": "https://normativa.udea.edu.co/...",
      "snippet": "fragmento textual recuperado..."
    }
  ],
  "source": "cache",          // "cache" | "agent"
  "score": 0.93,               // top-1 cache similarity for this query (0..1)
  "session_id": "the-thread-id",
  "latency_ms": 84
}
```

### `GET /health`
Returns `{"status": "ok"}` after verifying a DB connection can be acquired.

### `GET /stats`
Tiny in-process metrics (reset on restart): `{"total": N, "cache_hits": H, "agent_calls": M, "hit_rate": H/N}`.

---

## 4. End-to-end request flow (`POST /chat`)

```
1. Embed `query` with text-embedding-3-small  → 1536-dim vector.
2. Cache search: top_k nearest rows in `semantic_cache` by cosine distance.
      best = top-1 (if any);  similarity = 1 - cosine_distance(best).
3. CACHE HIT  (best exists AND similarity >= THRESHOLD):
      - Write the turn into the conversation thread WITHOUT calling the model:
            agent.update_state(config, {"messages": [HumanMessage(query), AIMessage(answer)]})
        (keeps multi-turn memory coherent even though we skipped the agent)
      - Return cached {answer, citations} with source="cache", score=similarity.
4. CACHE MISS:
      - Run the agent: agent.invoke({"messages":[HumanMessage(query)]}, config)
            config = {"configurable": {"thread_id": session_id}, "recursion_limit": 2*MAX_TOOL_CALLS + 1}
        The agent (PostgresSaver) auto-records the turn into the thread.
      - Parse the final answer + collect citations from the tool outputs (see SPEC_02 §retrieval/agent).
      - INDEX (single threshold): since this was a miss (similarity < THRESHOLD), upsert into
        `semantic_cache` the {query_text, query_embedding, response={answer,citations}}.
      - Return {answer, citations} with source="agent", score=similarity.
5. Always log one metrics line (source, score, latency_ms) and update the in-process counters.
```

> **Why the cache is consulted on every turn (even mid-conversation):** a *complete, standalone*
> question (the kind that gets cache hits) does not need conversation context, so serving it from
> cache is safe. A *context-dependent follow-up* (e.g. "¿y para postgrado?") embeds to something
> ambiguous, won't clear the threshold, and falls through to the agent — which has the full thread
> history via the checkpointer. The `update_state` call in step 3 keeps the thread complete.

---

## 5. Key design decisions (rationale)

- **Single similarity threshold (`serve == index`).** Simplest correct behavior from the article:
  if `similarity >= THRESHOLD` → serve from cache; otherwise generate **and** index. Tune the value
  empirically — we log `score` (top-1 similarity) on every request so you can pick the threshold from
  real data. Default `SEMANTIC_CACHE_THRESHOLD=0.85`; raise it to be more conservative.
- **Cache stores the full structured response** (`{answer, citations}` as JSONB), so cache hits also
  return citations — important for the "respuesta fundamentada" requirement of the challenge.
- **Cache + memory coherence** via `agent.update_state(...)` on hits (see flow §4.3).
- **Citations come from the DB, not from the model's prose.** The retrieval tool returns the chunks'
  metadata; the backend collects the *actually retrieved* sources as `citations`. The model is
  instructed to cite inline too, but the structured `citations` array is grounded in real rows.
- **No TTL / no versioning** (regulations don't change during the hackathon). We only store
  `created_at` and `embedding_model` for hygiene.

---

## 6. Critical gotchas — READ BEFORE CODING

### 6.1 `gpt-5-nano` is a reasoning model
- It **rejects** `temperature`, `top_p`, `presence_penalty`, `frequency_penalty`, and `max_tokens`.
  Do **NOT** pass any of them to `AzureChatOpenAI`. (If you need a token cap, use
  `max_completion_tokens` via `model_kwargs`, but prefer leaving it unset.)
- Control cost/latency with **`reasoning_effort`** (`minimal` | `low` | `medium` | `high`).
  Default to **`minimal`** (env `REASONING_EFFORT`): a FAQ task doesn't need deep reasoning, and
  `minimal` keeps reasoning-token usage and latency low. If answer quality is weak, bump to `low`/
  `medium` via env — **no code change**.
- With `reasoning_effort=minimal` the model emits **sequential single tool calls** (no parallel tool
  calls) — exactly what our RAG loop wants.

### 6.2 Supabase connection string
- Use the **Session-mode pooler** connection URI from the Supabase dashboard
  (Project Settings → Database → Connection string). Session mode is safe for prepared statements,
  which `PostgresSaver` relies on, and is IPv4-compatible.
- Avoid the **Transaction** pooler (port 6543) for the checkpointer; if you must use it, disable
  prepared statements (psycopg `prepare_threshold=None`) — but Session mode is strongly preferred.
- Always include `sslmode=require`.

### 6.3 pgvector dimensions & index
- `text-embedding-3-small` returns **1536** dims → use `VECTOR(1536)`. This is **≤ 2000**, so a
  standard **HNSW** index with `vector_cosine_ops` works directly (no `halfvec` needed). Do not
  switch to `text-embedding-3-large` (3072) without revisiting the index strategy.
- The `documents` table **must** use the **same** embedding model and **1536** dims as the cache, or
  cross-table comparisons and retrieval break. Confirm this with the teammate doing ingestion.

---

## 7. Repository structure

```
udea-faq-backend/
├── .env.example
├── requirements.txt
├── README.md
├── sql/
│   └── 01_schema.sql            # extensions + documents + semantic_cache + indexes
├── scripts/
│   └── setup_db.py              # runs 01_schema.sql AND PostgresSaver.setup()
└── app/
    ├── __init__.py
    ├── config.py                # env loading (pydantic-settings)
    ├── db.py                    # psycopg ConnectionPool (+ pgvector register)
    ├── azure_clients.py         # AzureChatOpenAI + AzureOpenAIEmbeddings factories
    ├── cache.py                 # SemanticCache: search() + upsert()
    ├── retrieval.py             # buscar_reglamento @tool (vector search over documents)
    ├── agent.py                 # create_agent(...) + PostgresSaver checkpointer
    ├── metrics.py               # tiny counters + structured logging
    └── main.py                  # FastAPI app: POST /chat, GET /stats, GET /health
```

---

## 8. Dependencies (`requirements.txt`)

```
fastapi
uvicorn[standard]
pydantic
pydantic-settings
python-dotenv
langchain>=1.0
langchain-openai
langgraph>=1.0
langgraph-checkpoint-postgres
psycopg[binary]
psycopg-pool
pgvector
```
> Let pip resolve provider versions compatible with `langchain>=1.0`. After the first successful
> install, run `pip freeze > requirements.lock.txt` to pin exact versions for the rest of the hackathon.

---

## 9. `.env.example` (Claude Code: generate exactly this file)

```dotenv
# --- Azure OpenAI (Azure AI Foundry) ---
AZURE_OPENAI_ENDPOINT=https://<your-resource>.openai.azure.com/
AZURE_OPENAI_API_KEY=<your-azure-openai-key>
AZURE_OPENAI_API_VERSION=2025-04-01-preview
AZURE_OPENAI_CHAT_DEPLOYMENT=gpt-5-nano
AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT=text-embedding-3-small

# --- Model behavior ---
REASONING_EFFORT=minimal          # minimal | low | medium | high
EMBEDDING_DIM=1536

# --- Database (Supabase Postgres, Session-mode pooler URI) ---
DATABASE_URL=postgresql://postgres.<ref>:<password>@<host>:5432/postgres?sslmode=require

# --- Semantic cache (single threshold: serve == index) ---
SEMANTIC_CACHE_THRESHOLD=0.85
SEMANTIC_CACHE_TOP_K=5

# --- Agentic RAG ---
RAG_TOP_K=5
MAX_TOOL_CALLS=10                 # max retrieval calls before forcing an answer

# --- App ---
APP_HOST=0.0.0.0
APP_PORT=8000
```

---

## 10. Acceptance criteria (definition of done)

1. `python scripts/setup_db.py` creates the extensions, `documents`, `semantic_cache`, **and** the
   LangGraph checkpointer tables, idempotently.
2. `uvicorn app.main:app` starts; `GET /health` returns `ok`.
3. With a manually seeded `documents` table, `POST /chat` returns a grounded Spanish answer **with a
   non-empty `citations` array** and `source="agent"` on the first ask.
4. Asking a **semantically equivalent** question (different wording) returns `source="cache"` with the
   same answer and `score >= THRESHOLD`.
5. A follow-up question in the **same `session_id`** shows the agent using prior context (memory works).
6. The agent calls the retrieval tool **at most `MAX_TOOL_CALLS` times**; a `GraphRecursionError` is
   caught and turned into a graceful Spanish message rather than a 500.
7. No `temperature`/`max_tokens` is sent to `gpt-5-nano` (no Azure 400 "unsupported parameter").
8. `GET /stats` reflects hits vs agent calls.

---

## 11. (Optional) Rubric alignment — useful for the pitch
- **Innovación / Viabilidad técnica:** semantic cache + Agentic RAG is an original, working combination.
- **Escalabilidad:** clean module boundaries; the cache is a swappable layer; pgvector scales with HNSW.
- **Ética / Sostenibilidad:** the cache cuts generation tokens (and therefore energy/cost) — the
  `/stats` hit-rate is a concrete, demoable sustainability metric.
- **Adecuación al reto:** grounded, cited answers over the official UdeA regulations corpus.
