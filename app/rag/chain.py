"""Generation phase: grounded prompt -> LLM, wired together as a LangChain LCEL chain."""
from functools import lru_cache
from typing import AsyncIterator

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama

from app.config import settings
from app.rag.prompts import ANSWER_PROMPT


@lru_cache
def get_llm() -> ChatOllama:
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
