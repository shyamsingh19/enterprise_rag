# Enterprise Knowledge Assistant (RAG)

A source-grounded question-answering system built to demonstrate a complete,
production-shaped Retrieval-Augmented Generation pipeline. Ask a plain-English
question and get an answer generated only from a document base, streamed back
token-by-token, with weak or irrelevant retrievals filtered out before they
ever reach the LLM.

**Live:**
- Web UI: https://enterprise-rag-1-0xnz.onrender.com
- Backend API: https://enterprise-rag-xgfh.onrender.com ([interactive docs](https://enterprise-rag-xgfh.onrender.com/docs))

## Architecture

**Indexing (offline, run once or whenever documents change)**
`data/documents/*` → LangChain loaders → `RecursiveCharacterTextSplitter`
(800-token chunks, 120-token overlap) → embedding model → ChromaDB

**Querying (per request)**
question → embed → Chroma similarity search (top-k, filtered by a similarity
threshold) → grounded prompt (`app/rag/prompts.py`) → LLM → tokens streamed to the
client over SSE. Conversation turns are cached in Redis per `session_id` so
follow-up questions carry context, entirely separate from the document index.

The LLM and embedding model are each swappable via env var (`LLM_PROVIDER`,
`EMBEDDING_PROVIDER`) without touching code — local dev runs fully offline
against Ollama; the deployed version runs Groq (hosted Llama 3) and `fastembed`
(local ONNX embeddings) instead, since Ollama needs more RAM than a free cloud
instance provides. See [DEVELOPMENT.md](DEVELOPMENT.md) for why and how.

| | local dev (default) | cloud deploy |
|---|---|---|
| LLM | `ollama` — Llama 3 8B via Ollama | `groq` — Groq-hosted Llama 3 |
| Embeddings | `ollama` — Nomic Embed Text via Ollama | `fastembed` — local ONNX model, no torch, no API |
| Session store | standard Redis (`REDIS_URL`) | Upstash Redis REST API |

```
app/
  config.py            settings (env-driven)
  models.py             request/response schemas
  vectorstore.py        shared Chroma + embeddings singletons (Ollama or fastembed)
  rag/
    ingestion.py         load -> chunk -> embed -> store; list/delete indexed sources
    retrieval.py          similarity search + threshold filtering
    prompts.py            the grounding prompt
    chain.py               LCEL chain: prompt | llm, streamed (Ollama or Groq)
  session/
    redis_store.py         per-session chat history (Redis or Upstash REST)
  main.py                 FastAPI app: /query, /ingest/*, /documents/*, /sessions/*,
                           /health; auto re-seeds the index on startup if it's empty
scripts/ingest.py        CLI to bulk-index data/documents/
ui/streamlit_app.py      chat UI over /query, with document management sidebar
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

The stream emits `session` (pass it back on the next call for follow-ups),
`sources` (which chunks were retrieved and their similarity scores), a `token`
event per generated token, and a final `done` event.

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

**List / remove indexed documents:**

```bash
curl localhost:8000/documents
curl -X DELETE localhost:8000/documents/policy-update.txt
```

**Fetch a session's conversation history** (what's stored in Redis):

```bash
curl localhost:8000/sessions/<session_id>
```

## Web UI

`ui/streamlit_app.py` is a chat UI over the same `/query` endpoint — streams
tokens as they arrive, shows a "thinking" indicator while the first token is
in flight, shows cited sources under each answer, and has a sidebar for
adding/removing indexed documents. `session_id` is kept in the URL so a page
refresh reattaches to the same Redis-backed conversation instead of starting
over. It's a plain web app (not a desktop toolkit) specifically so it can be
hosted for free alongside the API.

```bash
pip install -r requirements-ui.txt
# FASTAPI_SERVER in .env points it at your backend (local or deployed)
streamlit run ui/streamlit_app.py
```

## How hallucination is limited

1. The prompt (`app/rag/prompts.py`) explicitly instructs the model to answer only
   from the provided context and to say "I don't know" otherwise.
2. Retrieval itself is filtered: chunks scoring below `SIMILARITY_SCORE_THRESHOLD`
   (default `0.5`, cosine similarity) are dropped before the prompt is built, so a
   question unrelated to the corpus yields no context rather than a tenuous match.
3. `temperature=0` on the LLM favors deterministic, extractive answers over creative
   completion.

**Known limitation:** retrieval only considers the current question's raw
text, not conversation history, so a pronoun-heavy follow-up ("what does *he*
know?") can fail to retrieve anything even when the antecedent is obvious from
context. The standard fix is query rewriting/condensation (rewrite the
follow-up into a standalone question using chat history before retrieving,
e.g. LangChain's `create_history_aware_retriever` pattern) — not yet
implemented here.

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

## Deploying

The live links above run on Render's free tier, which needs a few workarounds
(no GPU/RAM for self-hosted Ollama, no persistent disk, free-tier spin-down).
See [DEVELOPMENT.md](DEVELOPMENT.md) for the full deployment guide.
