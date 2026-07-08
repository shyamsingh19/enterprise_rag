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
threshold) → grounded prompt (`app/rag/prompts.py`) → `ChatOllama` (Llama 3 8B) →
tokens streamed to the client over SSE. Conversation turns are cached in Redis per
`session_id` so follow-up questions carry context, entirely separate from the
document index.

```
app/
  config.py            settings (env-driven)
  models.py             request/response schemas
  vectorstore.py        shared Chroma + embeddings singletons
  rag/
    ingestion.py         load -> chunk -> embed -> store
    retrieval.py          similarity search + threshold filtering
    prompts.py            the grounding prompt
    chain.py               LCEL chain: prompt | llm, streamed
  session/
    redis_store.py         per-session chat history
  main.py                 FastAPI app: /query, /ingest/*, /health
scripts/ingest.py        CLI to bulk-index data/documents/
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
