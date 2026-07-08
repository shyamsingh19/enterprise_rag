"""Pydantic request/response schemas for the API."""
from typing import Optional

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="User's question in plain English")
    session_id: Optional[str] = Field(
        default=None, description="Conversation session ID for follow-up context"
    )


class SourceChunk(BaseModel):
    content: str
    source: str
    score: float


class IngestTextRequest(BaseModel):
    text: str = Field(..., min_length=1)
    source_name: str = Field(..., description="Identifier stored as source metadata, e.g. a filename")


class IngestResponse(BaseModel):
    source_name: str
    chunks_indexed: int
