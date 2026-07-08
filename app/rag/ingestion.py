"""Indexing phase: load, chunk, embed, and store documents in Chroma."""
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import settings
from app.vectorstore import get_vectorstore

# Markdown is loaded as plain text (skips the heavyweight `unstructured` dependency);
# the recursive splitter still respects paragraph/heading boundaries reasonably well.
_LOADERS_BY_SUFFIX = {
    ".txt": TextLoader,
    ".md": TextLoader,
    ".pdf": PyPDFLoader,
}

_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    encoding_name="cl100k_base",
    chunk_size=settings.chunk_size_tokens,
    chunk_overlap=settings.chunk_overlap_tokens,
)


def load_file(path: Path) -> list[Document]:
    loader_cls = _LOADERS_BY_SUFFIX.get(path.suffix.lower())
    if loader_cls is None:
        raise ValueError(f"Unsupported file type: {path.suffix}")
    return loader_cls(str(path)).load()


def chunk_documents(documents: list[Document]) -> list[Document]:
    return _splitter.split_documents(documents)


def index_documents(documents: list[Document]) -> int:
    """Chunk documents and add them to the vector store. Returns number of chunks indexed."""
    chunks = chunk_documents(documents)
    if not chunks:
        return 0
    get_vectorstore().add_documents(chunks)
    return len(chunks)


def index_text(text: str, source_name: str) -> int:
    """Convenience wrapper for indexing raw text (used by the /ingest endpoint)."""
    return index_documents([Document(page_content=text, metadata={"source": source_name})])


def index_file(path: Path, source_name: str | None = None) -> int:
    """Index a file on disk. `source_name` overrides the metadata source (path.name by
    default), which matters when `path` is a temp file standing in for an upload."""
    documents = load_file(path)
    for doc in documents:
        doc.metadata["source"] = source_name or path.name
    return index_documents(documents)


def index_directory(directory: Path) -> dict[str, int]:
    """Index every supported file in a directory. Used by the standalone ingestion script."""
    results: dict[str, int] = {}
    for path in sorted(directory.iterdir()):
        if path.suffix.lower() in _LOADERS_BY_SUFFIX:
            results[path.name] = index_file(path)
    return results
