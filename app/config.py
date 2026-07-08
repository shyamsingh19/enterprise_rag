"""Central configuration, loaded from environment variables / .env."""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # Ollama
    ollama_base_url: str = "http://localhost:11434"
    llm_model: str = "llama3:8b"
    embedding_model: str = "nomic-embed-text"

    # Chroma
    chroma_persist_dir: str = "./chroma_data"
    chroma_collection_name: str = "knowledge_base"

    # Redis
    redis_url: str = "redis://localhost:6379/0"
    session_history_max_turns: int = 6
    session_ttl_seconds: int = 60 * 60 * 24  # 1 day

    # Chunking (measured in tokens via tiktoken)
    chunk_size_tokens: int = 800
    chunk_overlap_tokens: int = 120

    # Retrieval
    retrieval_top_k: int = 4
    similarity_score_threshold: float = 0.5


settings = Settings()
