"""Prompt templates that ground the LLM in retrieved context."""
from langchain_core.prompts import ChatPromptTemplate

SYSTEM_PROMPT = """You are an enterprise knowledge assistant. Answer the user's question using \
ONLY the information in the "Context" section below.

Rules:
- If the context does not contain enough information to answer, say "I don't know based on the \
available documents." Do not guess or use outside knowledge.
- Keep answers concise and factual.
- When helpful, mention which source the information came from.

Context:
{context}"""

ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("placeholder", "{chat_history}"),
        ("human", "{question}"),
    ]
)
