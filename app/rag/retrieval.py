"""Query phase: retrieve relevant chunks from Chroma, filtering out weak matches."""
from langchain_core.documents import Document

from app.config import settings
from app.vectorstore import get_vectorstore


async def retrieve(question: str) -> list[tuple[Document, float]]:
    """Return (chunk, similarity_score) pairs above the similarity threshold, best first.

    Filtering weak matches here (rather than trusting the LLM to ignore irrelevant
    context) is the main lever we have against hallucination on out-of-corpus questions.
    """
    results = await get_vectorstore().asimilarity_search_with_relevance_scores(
        question, k=settings.retrieval_top_k
    )
    return [(doc, score) for doc, score in results if score >= settings.similarity_score_threshold]


def format_context(chunks: list[tuple[Document, float]]) -> str:
    if not chunks:
        return "No relevant documents were found."
    return "\n\n".join(
        f"[Source: {doc.metadata.get('source', 'unknown')}]\n{doc.page_content}" for doc, _ in chunks
    )
