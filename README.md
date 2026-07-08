# Enterprise Knowledge Assistant (RAG)

A local, source-grounded question-answering system. Ask a plain-English question and
get an answer generated only from your own documents, streamed back token-by-token,
with weak/irrelevant retrievals filtered out before they ever reach the LLM.

## Architecture

**Indexing (offline, run once or whenever documents change)**
`data/documents/*` → LangChain loaders → `RecursiveCharacterTextSplitter`
(800-token chunks, 120-token overlap) → Nomic Embed Text (via Ollama) → ChromaDB

**Querying (per request)**
question → embed → Chroma similarity search (top-k, filtered by a similarity
threshold) → grounded prompt (`app/rag/prompts.py`) → LLM → tokens streamed to the
client over SSE. Conversation turns are cached in Redis per `session_id` so
follow-up questions carry context, entirely separate from the document index.

The LLM and embedding model are each swappable via env var (`LLM_PROVIDER`,
`EMBEDDING_PROVIDER`) without touching code:

| | local dev (default) | free cloud deploy |
|---|---|---|
| LLM | `ollama` — Llama 3 8B via Ollama | `groq` — Groq-hosted Llama 3 |
| Embeddings | `ollama` — Nomic Embed Text via Ollama | `fastembed` — local ONNX model, no torch, no API |
| Session store | standard Redis (`REDIS_URL`) | Upstash Redis REST API |

See [Deploying to Render](#deploying-to-render-free) below for why local dev and
the free cloud path use different providers.

```
app/
  config.py            settings (env-driven)
  models.py             request/response schemas
  vectorstore.py        shared Chroma + embeddings singletons (Ollama or fastembed)
  rag/
    ingestion.py         load -> chunk -> embed -> store
    retrieval.py          similarity search + threshold filtering
    prompts.py            the grounding prompt
    chain.py               LCEL chain: prompt | llm, streamed (Ollama or Groq)
  session/
    redis_store.py         per-session chat history (Redis or Upstash REST)
  main.py                 FastAPI app: /query, /ingest/*, /health; auto re-seeds
                           the index on startup if it's empty (ephemeral disks)
scripts/ingest.py        CLI to bulk-index data/documents/
ui/streamlit_app.py      minimal chat UI over /query (see "Web UI" below)
render.yaml              Render Blueprint for the free-tier deployment path
.github/workflows/       keep-alive cron to ping /health and prevent spin-down
```

## Prerequisites

- [Ollama](https://ollama.com) installed and running, with the models pulled:
  ```
  ollama pull llama3:8b
  ollama pull nomic-embed-text
  ```
- Docker + Docker Compose (for the containerized path), or Python 3.11+ and a local
  Redis instance for running the app directly.

## Run with Docker Compose (recommended)

This starts the FastAPI app, Redis, and an Ollama container together.

```bash
docker compose up --build -d
docker compose exec ollama ollama pull llama3:8b
docker compose exec ollama ollama pull nomic-embed-text
```

Then index the sample documents (runs the ingestion script inside the app container):

```bash
docker compose exec app python scripts/ingest.py
```

## Run locally (no Docker)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # defaults already point at localhost

redis-server &          # or use an existing Redis instance
python scripts/ingest.py
uvicorn app.main:app --reload
```

## Using the API

**Ask a question** (`POST /query`, server-sent events response):

```bash
curl -N -X POST localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "How many days of PTO do full-time employees get?"}'
```

The stream emits `session` (so you can pass it back on the next call for
follow-ups), `sources` (which chunks were retrieved and their similarity scores),
a `token` event per generated token, and a final `done` event.

**Add a document at runtime:**

```bash
# Raw text
curl -X POST localhost:8000/ingest/text \
  -H "Content-Type: application/json" \
  -d '{"text": "...", "source_name": "policy-update.txt"}'

# File upload (.txt, .md, .pdf)
curl -X POST localhost:8000/ingest/file -F "file=@./some-doc.pdf"
```

**Bulk re-index** everything in `data/documents/`:

```bash
python scripts/ingest.py
```

## Web UI

`ui/streamlit_app.py` is a minimal chat UI over the same `/query` endpoint —
streams tokens as they arrive and shows which sources were cited. It's a plain
web app (not a desktop toolkit) specifically so it can be hosted for free
alongside the API.

```bash
pip install -r requirements-ui.txt
# FASTAPI_SERVER in .env points it at your backend (local or deployed)
streamlit run ui/streamlit_app.py
```

**Deploying it for free:** push this repo to GitHub, then on
[Streamlit Community Cloud](https://streamlit.io/cloud) create a new app
pointing at `ui/streamlit_app.py`, with `requirements-ui.txt` as the
dependency file and `FASTAPI_SERVER` set to your Render URL in the app's
secrets. No server to manage, no cold-start workaround needed — Streamlit
Community Cloud's free tier is meant exactly for this.

## Deploying to Render (free)

Ollama + Llama 3 8B needs several GB of RAM just to load the model — that doesn't
fit Render's free web service tier (512MB). Render's free tier also has no
persistent disk, so anything written to `chroma_data/` is lost on every cold
start. The free deployment path works around both:

- **LLM** → [Groq](https://console.groq.com) hosts Llama 3 for free with a
  generous rate limit; `ChatGroq` is a drop-in swap behind the same LangChain
  interface (`LLM_PROVIDER=groq`).
- **Embeddings** → [`fastembed`](https://github.com/qdrant/fastembed) runs a
  small ONNX model in-process (~130MB, no torch, no external API), comfortably
  within 512MB (`EMBEDDING_PROVIDER=fastembed`).
- **Chroma persistence** → instead of a paid Disk, `app/main.py`'s startup
  handler re-ingests `data/documents/` automatically whenever the vector store
  is empty, so every cold start self-heals instead of serving an empty index.
- **Session store** → [Upstash](https://upstash.com) has a real free serverless
  Redis tier, reachable over REST (no persistent TCP connection required, which
  suits a service that may cold-start). `app/session/redis_store.py`
  auto-selects it when `UPSTASH_REDIS_REST_URL`/`UPSTASH_REDIS_REST_TOKEN` are
  set, and falls back to plain Redis (`REDIS_URL`) otherwise — local dev is
  unaffected.

**Steps:**

1. Get a free Groq API key at [console.groq.com](https://console.groq.com).
2. Create a free Redis database at [upstash.com](https://upstash.com) and copy
   its REST URL and token (the "REST API" section of the database dashboard,
   not the TCP connection string).
3. Push this repo to GitHub.
4. In the Render dashboard: **New → Blueprint**, point it at the repo. Render
   reads [`render.yaml`](render.yaml) and provisions the web service, prompting
   you for the three `sync: false` values: `GROQ_API_KEY`,
   `UPSTASH_REDIS_REST_URL`, and `UPSTASH_REDIS_REST_TOKEN`.
5. Deploy. First boot will auto-ingest `data/documents/` (check the Render logs
   for "auto-ingesting").

**Keeping it awake:** Render's free tier spins the service down after ~15
minutes of no traffic, and the next request pays a slow cold start. A scheduled
GitHub Actions workflow ([`.github/workflows/keep-alive.yml`](.github/workflows/keep-alive.yml))
pings `/health` every 10 minutes to keep it warm. To enable it: in your GitHub
repo, go to **Settings → Secrets and variables → Actions**, add a secret named
`RENDER_APP_URL` set to your deployed URL (e.g. `https://enterprise-rag.onrender.com`).
Note GitHub's free cron scheduler doesn't guarantee exact timing under load, so
treat this as best-effort rather than a hard uptime guarantee.

## How hallucination is limited

1. The prompt (`app/rag/prompts.py`) explicitly instructs the model to answer only
   from the provided context and to say "I don't know" otherwise.
2. Retrieval itself is filtered: chunks scoring below `SIMILARITY_SCORE_THRESHOLD`
   (default `0.5`, cosine similarity) are dropped before the prompt is built, so a
   question unrelated to the corpus yields no context rather than a tenuous match.
3. `temperature=0` on the LLM favors deterministic, extractive answers over creative
   completion.

## Validating answer quality

This is a prototype; validation is qualitative. With the sample documents indexed, try:

- "How many days of PTO do employees get?" → should cite `employee_handbook.txt`
  and say 15 days/year.
- "What's the on-call escalation time during business hours?" → should cite
  `engineering_onboarding.txt` and say 15 minutes.
- "What's our parental leave policy for same-sex couples?" → tests whether the
  model over-generalizes beyond what the primary/secondary caregiver language says.
- "What's the company's stock ticker symbol?" → out-of-corpus; should trigger the
  "I don't know" refusal rather than a hallucinated answer.

A labeled eval set with ground-truth answers (e.g. via `ragas` or ranking retrieved
chunks against known-good sources) is a natural next step if this grows beyond a
prototype.
