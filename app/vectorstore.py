"""Shared Chroma vector store instance, embedded via Nomic Embed Text (served by Ollama)."""
from functools import lru_cache

from chromadb.config import Settings as ChromaSettings
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

from app.config import settings


@lru_cache
def get_embeddings() -> OllamaEmbeddings:
    return OllamaEmbeddings(
        model=settings.embedding_model,
        base_url=settings.ollama_base_url,
    )


@lru_cache
def get_vectorstore() -> Chroma:
    return Chroma(
        collection_name=settings.chroma_collection_name,
        embedding_function=get_embeddings(),
        persist_directory=settings.chroma_persist_dir,
        # Cosine similarity gives scores in [0, 1], which is what the retrieval
        # threshold in app/rag/retrieval.py assumes.
        collection_metadata={"hnsw:space": "cosine"},
        client_settings=ChromaSettings(anonymized_telemetry=False),
    )
