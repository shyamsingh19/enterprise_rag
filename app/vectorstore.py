"""Shared Chroma vector store instance.

Embeddings come from Nomic Embed Text via Ollama for local dev, or from `fastembed`
(a small local ONNX model, no torch, no external API) for RAM-constrained deployments
like Render's free tier. Selected via EMBEDDING_PROVIDER.
"""
from functools import lru_cache

from chromadb.config import Settings as ChromaSettings
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from langchain_ollama import OllamaEmbeddings

from app.config import settings


@lru_cache
def get_embeddings() -> Embeddings:
    if settings.embedding_provider == "fastembed":
        from langchain_community.embeddings import FastEmbedEmbeddings

        return FastEmbedEmbeddings(model_name=settings.fastembed_model)

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
