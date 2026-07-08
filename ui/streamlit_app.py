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

# st.session_state resets on every browser refresh, but the backend keeps
# conversation history in Redis keyed by session_id -- so we stash session_id
# in the URL to survive a refresh, and rehydrate the displayed messages from
# Redis (via /sessions/{id}) the first time we see it.
if "session_id" not in st.session_state:
    st.session_state.session_id = st.query_params.get("session_id")

if "messages" not in st.session_state:
    st.session_state.messages = []
    if st.session_state.session_id:
        try:
            resp = requests.get(f"{SERVER_URL}/sessions/{st.session_state.session_id}", timeout=10)
            resp.raise_for_status()
            for turn in resp.json()["turns"]:
                role = "user" if turn["role"] == "human" else "assistant"
                st.session_state.messages.append({"role": role, "content": turn["content"]})
        except requests.RequestException:
            pass  # session expired or unreachable -- just start fresh


def _set_session_id(session_id: str) -> None:
    st.session_state.session_id = session_id
    st.query_params["session_id"] = session_id


with st.sidebar:
    if st.button("New chat"):
        st.session_state.session_id = None
        st.session_state.messages = []
        st.query_params.clear()
        st.rerun()

    st.header("Add documents")

    uploaded = st.file_uploader("Upload a file", type=["txt", "md", "pdf"])
    if uploaded is not None and st.button("Ingest file"):
        try:
            resp = requests.post(
                f"{SERVER_URL}/ingest/file",
                files={"file": (uploaded.name, uploaded.getvalue())},
                timeout=60,
            )
            resp.raise_for_status()
            result = resp.json()
            st.success(f"Indexed {result['chunks_indexed']} chunk(s) from {result['source_name']}")
        except requests.RequestException as exc:
            st.error(f"Ingest failed: {exc}")

    st.divider()
    st.caption("...or paste raw text")
    text = st.text_area("Text", label_visibility="collapsed", placeholder="Paste text to index...")
    source_name = st.text_input("Source name", placeholder="e.g. policy-update.txt")
    if st.button("Ingest text"):
        if not text.strip() or not source_name.strip():
            st.warning("Enter both text and a source name.")
        else:
            try:
                resp = requests.post(
                    f"{SERVER_URL}/ingest/text",
                    json={"text": text, "source_name": source_name},
                    timeout=60,
                )
                resp.raise_for_status()
                result = resp.json()
                st.success(
                    f"Indexed {result['chunks_indexed']} chunk(s) from {result['source_name']}"
                )
            except requests.RequestException as exc:
                st.error(f"Ingest failed: {exc}")

    st.divider()
    st.header("Manage documents")
    try:
        sources = requests.get(f"{SERVER_URL}/documents", timeout=10).json()
    except requests.RequestException:
        sources = []
        st.caption("Could not load document list.")

    if sources:
        to_delete = st.selectbox("Indexed documents", sources)
        if st.button("Delete selected"):
            try:
                resp = requests.delete(f"{SERVER_URL}/documents/{to_delete}", timeout=30)
                resp.raise_for_status()
                st.success(f"Deleted {resp.json()['chunks_deleted']} chunk(s) from {to_delete}")
                st.rerun()
            except requests.RequestException as exc:
                st.error(f"Delete failed: {exc}")
    else:
        st.caption("No documents indexed yet.")

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
                _set_session_id(data)
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
            with st.spinner("Thinking..."):
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
