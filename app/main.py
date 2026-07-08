"""FastAPI entrypoint: exposes /query (streaming, grounded Q&A) and /ingest."""
import json
import logging
import tempfile
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse

from app.config import settings
from app.models import (
    DeleteResponse,
    HistoryResponse,
    HistoryTurn,
    IngestResponse,
    IngestTextRequest,
    QueryRequest,
)
from app.rag.chain import stream_answer
from app.rag.ingestion import (
    delete_source,
    index_directory,
    index_file,
    index_text,
    is_index_empty,
    list_sources,
)
from app.rag.retrieval import format_context, retrieve
from app.session.redis_store import append_turn, get_history

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Deployments with ephemeral disks (e.g. Render's free tier) lose the Chroma
    # index on every cold start. Re-seed it from the bundled corpus so the app is
    # never left with an empty knowledge base.
    auto_ingest_dir = Path(settings.auto_ingest_dir)
    if is_index_empty() and auto_ingest_dir.is_dir():
        logger.info("Vector store is empty; auto-ingesting %s", auto_ingest_dir)
        index_directory(auto_ingest_dir)
    yield


app = FastAPI(title="Enterprise Knowledge Assistant", lifespan=lifespan)


@app.get("/health", response_class=HTMLResponse)
async def health() -> str:
    """Human-readable status page -- also what uptime/cron pingers hit to keep
    a free-tier deployment (e.g. Render) from spinning down."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    return f"""<!doctype html>
<html><head><title>Enterprise Knowledge Assistant -- status</title>
<style>
  body {{ font-family: system-ui, sans-serif; display: flex; align-items: center;
          justify-content: center; height: 100vh; margin: 0; background: #0e1117; color: #e6e6e6; }}
  .card {{ text-align: center; }}
  .dot {{ color: #3fb950; font-size: 1.4rem; }}
  .ts {{ color: #8b949e; font-size: 0.85rem; margin-top: 0.5rem; }}
</style></head>
<body><div class="card">
  <div><span class="dot">&#9679;</span> Enterprise Knowledge Assistant is up</div>
  <div class="ts">{now}</div>
</div></body></html>"""


@app.post("/ingest/text", response_model=IngestResponse)
async def ingest_text(request: IngestTextRequest) -> IngestResponse:
    chunks_indexed = index_text(request.text, request.source_name)
    return IngestResponse(source_name=request.source_name, chunks_indexed=chunks_indexed)


@app.post("/ingest/file", response_model=IngestResponse)
async def ingest_file(file: UploadFile) -> IngestResponse:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in {".txt", ".md", ".pdf"}:
        raise HTTPException(400, f"Unsupported file type: {suffix or 'unknown'}")

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = Path(tmp.name)

    try:
        chunks_indexed = index_file(tmp_path, source_name=file.filename or tmp_path.name)
    finally:
        tmp_path.unlink(missing_ok=True)

    return IngestResponse(source_name=file.filename or tmp_path.name, chunks_indexed=chunks_indexed)


@app.get("/documents", response_model=list[str])
async def list_documents() -> list[str]:
    return list_sources()


@app.delete("/documents/{source_name}", response_model=DeleteResponse)
async def delete_document(source_name: str) -> DeleteResponse:
    chunks_deleted = delete_source(source_name)
    if chunks_deleted == 0:
        raise HTTPException(404, f"No indexed chunks found for source: {source_name}")
    return DeleteResponse(source_name=source_name, chunks_deleted=chunks_deleted)


@app.get("/sessions/{session_id}", response_model=HistoryResponse)
async def get_session_history(session_id: str) -> HistoryResponse:
    turns = await get_history(session_id)
    return HistoryResponse(
        session_id=session_id, turns=[HistoryTurn(role=role, content=content) for role, content in turns]
    )


def _sse(event: str, data: str) -> str:
    return f"event: {event}\ndata: {data}\n\n"


async def _generate_response(request: QueryRequest) -> AsyncIterator[str]:
    session_id = request.session_id or str(uuid.uuid4())
    yield _sse("session", session_id)

    chat_history = await get_history(session_id)

    retrieved = await retrieve(request.question)
    sources = [
        {"source": doc.metadata.get("source", "unknown"), "score": round(score, 3)}
        for doc, score in retrieved
    ]
    yield _sse("sources", json.dumps(sources))

    context = format_context(retrieved)
    answer_parts: list[str] = []
    async for token in stream_answer(request.question, context, chat_history):
        answer_parts.append(token)
        yield _sse("token", json.dumps(token))

    full_answer = "".join(answer_parts)
    await append_turn(session_id, "human", request.question)
    await append_turn(session_id, "ai", full_answer)

    yield _sse("done", "")


@app.post("/query")
async def query(request: QueryRequest) -> StreamingResponse:
    return StreamingResponse(_generate_response(request), media_type="text/event-stream")
