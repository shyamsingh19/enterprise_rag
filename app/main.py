"""FastAPI entrypoint: exposes /query (streaming, grounded Q&A) and /ingest."""
import json
import tempfile
import uuid
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from app.models import IngestResponse, IngestTextRequest, QueryRequest
from app.rag.chain import stream_answer
from app.rag.ingestion import index_file, index_text
from app.rag.retrieval import format_context, retrieve
from app.session.redis_store import append_turn, get_history

app = FastAPI(title="Enterprise Knowledge Assistant")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


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
