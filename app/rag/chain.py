"""Generation phase: grounded prompt -> LLM, wired together as a LangChain LCEL chain.

The LLM is Llama 3 8B via Ollama for local dev, or Groq's hosted Llama 3 for
deployments too small to host Ollama themselves (e.g. Render's free tier).
Selected via LLM_PROVIDER.
"""
from functools import lru_cache
from typing import AsyncIterator

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama

from app.config import settings
from app.rag.prompts import ANSWER_PROMPT


@lru_cache
def get_llm() -> BaseChatModel:
    if settings.llm_provider == "groq":
        from langchain_groq import ChatGroq

        return ChatGroq(model=settings.groq_model, api_key=settings.groq_api_key, temperature=0)

    return ChatOllama(
        model=settings.llm_model,
        base_url=settings.ollama_base_url,
        temperature=0,
    )


@lru_cache
def get_chain():
    return ANSWER_PROMPT | get_llm() | StrOutputParser()


def _to_messages(chat_history: list[tuple[str, str]]) -> list[HumanMessage | AIMessage]:
    return [
        HumanMessage(content=content) if role == "human" else AIMessage(content=content)
        for role, content in chat_history
    ]


async def stream_answer(
    question: str, context: str, chat_history: list[tuple[str, str]]
) -> AsyncIterator[str]:
    """Stream the answer token-by-token, grounded in the given context."""
    chain = get_chain()
    async for token in chain.astream(
        {
            "question": question,
            "context": context,
            "chat_history": _to_messages(chat_history),
        }
    ):
        yield token
