"""Central configuration, loaded from environment variables / .env."""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # Providers: "ollama" runs everything locally (default, used for local dev).
    # "groq" (LLM) / "fastembed" (embeddings) are for deployments too small to
    # host Ollama + Llama3-8B, e.g. Render's free tier.
    llm_provider: str = "ollama"  # "ollama" | "groq"
    embedding_provider: str = "ollama"  # "ollama" | "fastembed"

    # Ollama
    ollama_base_url: str = "http://localhost:11434"
    llm_model: str = "llama3:8b"
    embedding_model: str = "nomic-embed-text"

    # Groq (hosted Llama 3 inference, free tier)
    groq_api_key: str = ""
    groq_model: str = "llama3-8b-8192"

    # fastembed (local ONNX embeddings, no torch, no external API)
    fastembed_model: str = "BAAI/bge-small-en-v1.5"

    # Chroma
    chroma_persist_dir: str = "./chroma_data"
    chroma_collection_name: str = "knowledge_base"
    # If the collection is empty at startup, auto-index this directory. Covers
    # ephemeral-disk deployments (e.g. Render free tier) where the vector store
    # is wiped on every cold start.
    auto_ingest_dir: str = "./data/documents"

    # Redis: standard TCP (redis-py) by default, used for local dev. If Upstash
    # REST credentials are set, session storage uses Upstash's REST API instead
    # (no persistent TCP connection needed -- a better fit for platforms like
    # Render's free tier).
    redis_url: str = "redis://localhost:6379/0"
    upstash_redis_rest_url: str = ""
    upstash_redis_rest_token: str = ""
    session_history_max_turns: int = 6
    session_ttl_seconds: int = 60 * 60 * 24  # 1 day

    # Chunking (measured in tokens via tiktoken)
    chunk_size_tokens: int = 800
    chunk_overlap_tokens: int = 120

    # Retrieval
    retrieval_top_k: int = 4
    similarity_score_threshold: float = 0.5


settings = Settings()
