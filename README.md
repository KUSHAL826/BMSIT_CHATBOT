# BMSIT College AI Agent

A retrieval-augmented assistant for **B.M.S. Institute of Technology and Management**
(BMSIT&M), Bengaluru. It crawls the college website once a day, keeps a permanent
copy of every page it has seen, and answers questions only from that verified
content, with citations.

Flask + Supabase (PostgreSQL + pgvector) / FAISS + Gemini API.

---

## What makes this different: delta ingestion

Embedding is the expensive, rate-limited part of a RAG system. Re-embedding 3,000
pages every night is wasteful and will exhaust any API quota. This project embeds
**only what actually changed**.

Every tracked page has an entry in `data/ingestion_registry.json`:

| Field | Purpose |
|---|---|
| `content_hash` | SHA-256 of the cleaned page text: the change signal |
| `last_scraped` / `last_embedded` | when it was last fetched and last embedded |
| `chunk_ids` | the exact vector ids this page owns |
| `token_count` | tokens embedded, for quota and cost visibility |
| `embed_provider` | which embedding space its vectors live in |
| `revision` | how many times it has been re-embedded |

The daily run then does this per page:

```
hash matches registry   ->  skip entirely. No chunking, no embedding, no cost.
hash differs / is new   ->  delete exactly the vectors listed in chunk_ids,
                            re-chunk, embed, insert, overwrite the registry entry.
page not reached        ->  leave its content and vectors untouched.
```

**Nothing is ever deleted implicitly.** A partial crawl, a failed fetch or a page
disappearing from the website does not remove knowledge. Only an admin pressing
delete does that.

### Surviving the first bulk load

The initial run has to embed everything, which is where quota runs out. Two
mechanisms handle it:

1. **Key pool.** Set `GEMINI_API_KEYS=key1,key2,key3`. Requests rotate across the
   keys; any key that reports a quota error is parked for an hour and the run
   continues on the others.
2. **Sliced commits.** Pages are embedded and written in slices
   (`EMBED_COMMIT_SLICE_CHUNKS`, default 500 chunks). If every key runs dry
   mid-run, the run stops, keeps everything committed so far, and reports
   `partial`. The remaining pages were never recorded in the registry, so the
   next run picks them up automatically. Re-run until it reports `success`.

Set `EMBED_MAX_CHUNKS_PER_RUN` to deliberately spread the first load over several
days.

---

## Other design decisions worth knowing

**Vectors from different models are never mixed.** Each vector records the space
that produced it (`gemini:gemini-embedding-001` or `local:hash-v1`). A query is
only compared against vectors from its own space. Comparing a Gemini vector with
a hash-fallback vector produces meaningless similarity scores, which is a
silent-wrong-answer bug rather than a crash.

**Embed first, swap second.** Every index mutation generates new vectors before
removing old ones, so an interrupted or failing run can never leave the knowledge
base empty.

**Token-based chunking.** Chunks target 250-300 tokens on sentence and paragraph
boundaries, with overlap, because model limits and cost are token-based. Page
title and URL travel with each chunk as a header rather than being pushed through
the splitter.

**Site chrome is stripped.** Lines appearing on more than ~22% of crawled pages
(mega-menus, tickers, footers) are removed before chunking. Without this, most of
the index is navigation text and the assistant answers from menu labels.

**Atomic writes everywhere.** All JSON and index writes go to a temp file and are
renamed into place, so a crash cannot corrupt the store.

---

## Layout

```
run.py                          server entry point
app/
  config.py                     all settings, environment driven
  routes/
    admin_routes.py             dashboard + ingestion/registry APIs
    chat_routes.py              /chatbot, /api/chat, /health
  services/
    scraper.py                  crawler, boilerplate stripping, change detection
    chunker.py                  token-aware 250-300 token chunking
    ingestion_registry.py       hashes, timestamps, chunk ids  <- delta state
    ingestion_pipeline.py       the delta workflow (website + documents)
    embedding_service.py        key pool, batching, asyncio concurrency
    rag_service.py              FAISS spaces, hybrid retrieval, atomic updates
    knowledge_store.py          permanent raw text, one file per source
    gemini_service.py           guardrails, prompting, answer generation
    document_parser.py          PDF / DOCX / DOC / CSV extraction
    scheduler_service.py        daily cron trigger
    storage.py, fsutil.py       history, settings, atomic IO
    migration.py                upgrades pre-existing data on startup
data/                           runtime state (gitignored)
  index/                        chunks.json, embeddings.npy, faiss.index
  knowledge/                    permanent text, one JSON per source
  ingestion_registry.json       delta registry
  uploads/                      uploaded documents
static/, templates/             chatbot and admin front end
tests/                          unit and end-to-end tests
```

---

## Setup

```cmd
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

Edit `.env` and add your keys and Supabase credentials:

```env
# Gemini API keys
GEMINI_API_KEY=your_key_for_chat
GEMINI_API_KEYS=key1,key2,key3

# Supabase Vector Store (pgvector)
SUPABASE_URL=https://your-project-id.supabase.co
SUPABASE_KEY=your_supabase_anon_or_service_role_key
SUPABASE_TABLE=vectors
```

Get Gemini keys from [Google AI Studio](https://aistudio.google.com/apikey).

---

## Setting up Supabase Vector Store (PostgreSQL + pgvector)

### Difference between `.env` and Environment Variables:
- **`.env` file**: A local text file in the root directory used during local development.
- **Process Environment Variables**: Variables set in your operating system or cloud host (e.g. Render Dashboard -> Environment Variables). Environment variables take priority over `.env`.

### Step-by-Step Database Setup:

1. **Create a Supabase Project**:
   Sign in to [Supabase](https://supabase.com) and create a new project.

2. **Run the Database Setup Script**:
   Go to your Supabase Dashboard -> **SQL Editor** -> **New Query**, paste the following SQL script, and click **Run**:

   ```sql
   -- 1. Enable pgvector extension
   CREATE EXTENSION IF NOT EXISTS vector;

   -- 2. Create vectors table
   CREATE TABLE IF NOT EXISTS vectors (
       id TEXT PRIMARY KEY,
       source_id TEXT,
       source_name TEXT,
       source_type TEXT,
       item_key TEXT,
       text TEXT NOT NULL,
       tokens INT DEFAULT 0,
       context_header TEXT,
       metadata JSONB DEFAULT '{}'::jsonb,
       embed_provider TEXT,
       content_hash TEXT,
       embedding vector(768),
       created_at TIMESTAMPTZ DEFAULT NOW(),
       updated_at TIMESTAMPTZ DEFAULT NOW()
   );

   -- 3. Create index for fast vector similarity search
   CREATE INDEX IF NOT EXISTS vectors_embedding_idx ON vectors 
   USING ivfflat (embedding vector_ip_ops) WITH (lists = 100);
   ```

3. **Get Your Credentials**:
   In your Supabase Dashboard, go to **Project Settings** -> **API**:
   - Copy **Project URL** -> `SUPABASE_URL`
   - Copy **anon / public** or **service_role** key -> `SUPABASE_KEY`

4. **Configure `.env`**:
   Add `SUPABASE_URL` and `SUPABASE_KEY` to `.env` (or set them as environment variables on host platforms like Render).

5. **Run it**:

```cmd
.venv\Scripts\python.exe run.py
```

| Page | URL |
|---|---|
| Chatbot | http://127.0.0.1:5000/chatbot |
| Admin dashboard | http://127.0.0.1:5000/admin |
| Health JSON | http://127.0.0.1:5000/health |

First load: open the admin dashboard and press **Scrape Now**. Watch the status
line; if it finishes `partial`, press it again after quota resets.

> `python` may point at a different interpreter than your virtualenv. Use
> `.venv\Scripts\python.exe` to be certain, or activate the venv first.

---

## Useful endpoints

| Endpoint | What it gives you |
|---|---|
| `GET /health` | chunk, vector and source counts |
| `GET /api/admin/status` | RAG, scheduler, registry and store stats |
| `GET /api/admin/embeddings` | per-key usability, quota hits, batch limits |
| `GET /api/admin/knowledge` | every stored source and its chunk count |
| `GET /api/admin/registry/<source_id>` | per-page hash, vector ids, tokens |
| `POST /api/admin/scrape` | trigger a delta run now |
| `POST /api/admin/reembed` | upgrade fallback vectors once quota returns |
| `DELETE /api/admin/source/<id>` | remove one source: vectors, text, registry |

---

## Tests

```cmd
.venv\Scripts\python.exe -m unittest tests.test_bmsit_system
.venv\Scripts\python.exe -m unittest tests.test_e2e_workflow
```

The end-to-end test performs a real crawl and real embedding calls, so it is slow
and consumes quota. `partial` is a valid outcome there.

---

## Deployment notes

This is a **stateful, single-instance** application: the FAISS index lives in
process memory and all state is on local disk. Run **one** worker and give it a
persistent volume.

```bash
# Linux
gunicorn run:app --bind 127.0.0.1:5000 --workers 1 --threads 8 --timeout 300
```

```cmd
REM Windows
waitress-serve --host=127.0.0.1 --port=5000 --threads=8 run:app
```

The single worker is deliberate: multiple workers would each run their own
scheduler and hold diverging copies of the index. The long timeout matters
because a manual scrape runs inside the request.

**Before exposing this publicly:** `/admin` and every `/api/admin/*` route have
no authentication. `HOST` defaults to `127.0.0.1` for that reason. Put the app
behind a reverse proxy with TLS and add authentication to the admin blueprint
before binding to `0.0.0.0`.

To link it from the college site, point an "Ask BMSIT AI" button at
`https://your-host/chatbot`. That page has no admin controls.

---

Built for BMSIT&M. Powered by Google Gemini and FAISS.
