# Copiloto UdeA — NormatIA

Copiloto agéntico para la comunidad de la Universidad de Antioquia. Responde preguntas
(en español) sobre el **reglamento estudiantil** de pregrado y posgrado, con respuestas
**fundamentadas y citadas**.

Arquitectura en dos capas: una **caché semántica** (pgvector) que responde preguntas repetidas
casi sin costo, y un **agente RAG** (LangChain) que, en un fallo de caché, busca en los documentos
de la normativa y responde citando la fuente. El agente también tiene una búsqueda web acotada a
`udea.edu.co` solo para temas de coyuntura (eventos, noticias, fechas).

## Componentes

| Carpeta | Qué es | Stack |
|---|---|---|
| [`backend/`](backend/) | API (caché semántica + agente RAG). Endpoints `/chat`, `/health`, `/stats` y `/documents` (sube PDFs → OCR → carga a la BD). | Python 3.12, FastAPI, LangChain, pgvector |
| [`frontend/`](frontend/) | Interfaz de chat **NormatIA UdeA** (consume `/chat`). | Chainlit |
| [`preprocessing/`](preprocessing/) | Pipeline de OCR: PDF → texto limpio (`.txt`). | Python, Poppler, Tesseract |
| [`normativa-scrapping/`](normativa-scrapping/) | Scraper de normativa.udea.edu.co: arma la lista de PDFs y los sube al backend. | Node 18+ |

## Requisitos

- **Python 3.12** y **Node 18+**
- **PostgreSQL + pgvector** (recomendado: Supabase)
- **Azure OpenAI** con dos deployments: chat (`gpt-5-nano`) y embeddings (`text-embedding-3-small`)
- Para OCR de PDFs escaneados: **Tesseract** (con idioma `spa`) y **Poppler**

Hay dos `.env`: uno en `backend/` (Azure + `DATABASE_URL`) y uno en `preprocessing/` (rutas
`TESSERACT_PATH` / `POPPLER_PATH`). Copia cada `.env.example` a `.env` y complétalos.

## Cómo iniciar cada parte

### 1. Backend (API)
```bash
cd backend
python -m venv venv && venv\Scripts\activate      # macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
copy .env.example .env                             # y completa Azure + DATABASE_URL
python scripts/setup_db.py                         # crea tablas + checkpointer (idempotente)
uvicorn app.main:app --port 8001
```

### 2. Frontend (chat)
```bash
cd frontend
python -m venv venv && venv\Scripts\activate
pip install -r requirements.txt
chainlit run app.py                                # abre en http://localhost:8000, llama al backend en :8001
```
(El frontend usa `API_URL`, por defecto `http://localhost:8001`.)

### 3. Preprocessing (OCR local, opcional)
Convierte PDFs de `preprocessing/data/raw/` a texto en `preprocessing/data/processed/`.
```bash
pip install -r requirements.txt                    # deps de OCR (desde la raíz)
# completa preprocessing/.env con TESSERACT_PATH y POPPLER_PATH
python preprocessing/src/ocr/pipeline.py           # ejecutar desde la raíz del repo
```

### 4. Scraper (normativa-scrapping)
```bash
cd normativa-scrapping
node urls.js                                       # scrapea normativa.udea.edu.co -> urls.txt
node ocr.js                                        # descarga los PDFs y los sube a POST /documents
```
Variables útiles para `ocr.js`: `API_URL` (default `http://localhost:8000/documents`; ajústalo a `:8001`),
`NIVEL=pregrado|posgrado`, `CONCURRENCY`, `MAX_MB`. Reanuda solo (`.uploaded.txt`).

## Poblar la base de documentos

Dos caminos (el backend debe estar corriendo para el camino A):

- **A — En vivo (recomendado):** `node urls.js` → `node ocr.js`. Cada PDF se descarga y se sube al
  endpoint `/documents`, que hace OCR + chunking + embedding + inserción automáticamente.
- **B — Local/batch:** deja PDFs en `preprocessing/data/raw/`, corre el pipeline de OCR (paso 3) y
  luego `cd backend && python scripts/ingest_documents.py --reset`.

> Más detalle del backend (endpoints, esquema, scripts) en [`backend/README.md`](backend/README.md).
