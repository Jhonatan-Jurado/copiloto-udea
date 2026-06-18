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

## 4. Populate `documents`

Both loaders read `../preprocessing/data/processed/**/*.txt`, split each file into
article-based chunks (shared logic in `app/chunking.py`), embed them with
`text-embedding-3-small` (1536 dims — the **same** model as the cache, or retrieval breaks),
and insert into `documents`.

### 4a. Full corpus (recommended) — `ingest_documents.py`

First produce the `.txt` for every PDF by running the OCR pipeline (see "Corpus OCR" below),
then ingest:

```bash
python scripts/ingest_documents.py --dry-run     # parse+chunk report (no embed/insert)
python scripts/ingest_documents.py --reset       # DELETE all rows, then load the full corpus
python scripts/ingest_documents.py               # idempotent: replaces only prior corpus rows
```

It derives `source` from each filename (`acuerdo_superior_425` → "Acuerdo Superior 425"),
`nivel` from the folder (`pregrado`/`posgrado`), a generic `url`, and tags rows
`metadata->>'ingest' = 'corpus'`. The scanned duplicate `documento_17917413` is skipped.

### 4b. Dev sample only — `seed_documents.py`

Quick throwaway seed (only the single sample doc, rows tagged `metadata->>'seed' = 'true'`,
clears its own previous rows; never touches corpus rows):

```bash
python scripts/seed_documents.py
```

### Corpus OCR (prerequisite for 4a)

The OCR pipeline lives in `preprocessing/` and reads `TESSERACT_PATH` / `POPPLER_PATH` from
the **repo-root `.env`** (copy from the root `.env.example`). One-time setup:

1. Install **Tesseract** (UB Mannheim) with the **Spanish (`spa`)** language pack, and
   **Poppler** (oschwartz10612). Verify `tesseract --list-langs` includes `spa`.
2. In the root `.env`, set `TESSERACT_PATH` (path to `tesseract.exe`) and `POPPLER_PATH`
   (Poppler's `bin` folder).
3. Install the OCR deps in the env that runs the pipeline: from the repo root,
   `pip install -r requirements.txt` (the backend venv does not have PyMuPDF/pdf2image/pytesseract).

Then, from the **repo root**:

```bash
python preprocessing/src/ocr/pipeline.py
```

This writes `preprocessing/data/processed/**/*.txt` mirroring the `raw/` folder structure
(scanned PDFs use OCR; native PDFs use PyMuPDF).

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

### `POST /documents` — upload PDFs (OCR + ingest)

Upload one or more PDFs; each is OCR'd in-process, chunked, embedded, and inserted into
`documents` so the RAG agent can cite it. `multipart/form-data`:

```bash
curl -X POST http://localhost:8000/documents \
  -F "files=@acuerdo_superior_999.pdf" \
  -F "files=@otro_acuerdo.pdf" \
  -F "nivel=pregrado"
```

- `files` (one or more): the PDFs. `source` is derived from each filename
  (`acuerdo_superior_999` → "Acuerdo Superior 999"), so name them correctly.
- `nivel` (optional): `pregrado` | `posgrado`, applied to all files in the request; `NULL`
  if omitted.
- Re-uploading the same filename **replaces** its rows (idempotent; matched by
  `metadata->>'origen'`). Rows are tagged `metadata->>'ingest' = 'upload'`.

Response — one result per file:
```json
{
  "results": [
    {"filename": "acuerdo_superior_999.pdf", "source": "Acuerdo Superior 999",
     "nivel": "pregrado", "chunks_inserted": 12, "status": "ok", "error": null}
  ]
}
```
`status` ∈ `ok` | `rejected` (not a .pdf / empty / >25 MB) | `empty` (no extractable text) |
`ocr_error` (OCR deps/binaries missing) | `error`.

> **Requires the OCR stack** (same as "Corpus OCR" above): the backend env needs
> `pymupdf`/`pdf2image`/`pytesseract` (in `requirements.txt`), and **scanned** PDFs also need
> Tesseract (+ `spa`) and Poppler configured in `preprocessing/.env`. Native (text) PDFs work
> with PyMuPDF alone.

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
