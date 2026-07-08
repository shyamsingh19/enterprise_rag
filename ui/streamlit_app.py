"""Minimal web client for the Enterprise Knowledge Assistant.

Talks to the FastAPI /query endpoint (server-sent events) and streams the
answer into the chat as it's generated. Server URL comes from FASTAPI_SERVER
in .env. A plain web app (not a desktop toolkit like PyQt) so it can be
deployed for free on Streamlit Community Cloud.

Run with: streamlit run ui/streamlit_app.py
"""
import json
import os

import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
# get from env file
SERVER_URL = os.getenv("FASTAPI_SERVER", "http://localhost:8000").rstrip("/")

st.set_page_config(page_title="Enterprise Knowledge Assistant", page_icon="📚")
st.title("Enterprise Knowledge Assistant")
st.caption(f"Connected to {SERVER_URL}")

if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if message.get("sources"):
            st.caption("Sources: " + ", ".join(message["sources"]))


def stream_answer(question: str):
    """Yields answer tokens as they arrive; session_id and sources are stashed
    on st.session_state as a side effect once their SSE events are seen."""
    response = requests.post(
        f"{SERVER_URL}/query",
        json={"question": question, "session_id": st.session_state.session_id},
        stream=True,
        timeout=(15, 120),
    )
    response.raise_for_status()

    event, data_lines = None, []
    for line in response.iter_lines(decode_unicode=True):
        if line == "":
            data = "\n".join(data_lines)
            if event == "session":
                st.session_state.session_id = data
            elif event == "sources":
                sources = json.loads(data) if data else []
                st.session_state["_last_sources"] = [
                    f"{s['source']} ({s['score']})" for s in sources
                ]
            elif event == "token":
                yield json.loads(data)
            event, data_lines = None, []
        elif line.startswith("event:"):
            event = line[len("event:") :].strip()
        elif line.startswith("data:"):
            data_lines.append(line[len("data:") :].strip())


question = st.chat_input("Ask a question about your documents...")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        st.session_state["_last_sources"] = []
        try:
            answer = st.write_stream(stream_answer(question))
        except requests.RequestException as exc:
            answer = f"Could not reach the server: {exc}"
            st.error(answer)
        sources = st.session_state.get("_last_sources", [])
        if sources:
            st.caption("Sources: " + ", ".join(sources))

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )
