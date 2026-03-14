from __future__ import annotations

import shutil
from pathlib import Path

import requests
import streamlit as st

from config import (
    APP_SUBTITLE,
    APP_TITLE,
    ASSETS_DIR,
    CHROMA_DIR,
    HF_EMBED_MODEL,
    MAX_SOURCE_PREVIEW_CHARS,
    OLLAMA_BASE_URL,
    OLLAMA_CHAT_MODEL,
    UPLOAD_DIR,
)
from rag import answer_question, build_vectorstore, load_pdf_documents, load_vectorstore, split_documents


st.set_page_config(
    page_title=APP_TITLE,
    page_icon="📘",
    layout="wide",
    initial_sidebar_state="expanded",
)


def load_css() -> None:
    css_path = ASSETS_DIR / "custom.css"
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)


load_css()

if "messages" not in st.session_state:
    st.session_state.messages = []

if "vector_ready" not in st.session_state:
    st.session_state.vector_ready = False

if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = []


def check_ollama_health() -> None:
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        if response.status_code != 200:
            st.error("Ollama is installed but not responding correctly.")
            st.stop()
    except Exception:
        st.error("Could not connect to Ollama. Start it first.")
        st.stop()


def save_uploaded_files(uploaded_files) -> list[Path]:
    saved_paths: list[Path] = []
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    for uploaded_file in uploaded_files:
        destination = UPLOAD_DIR / uploaded_file.name
        with open(destination, "wb") as f:
            f.write(uploaded_file.getbuffer())
        saved_paths.append(destination)

    return saved_paths


def reset_knowledge_base() -> None:
    if CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR)
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    if UPLOAD_DIR.exists():
        shutil.rmtree(UPLOAD_DIR)
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    st.session_state.vector_ready = False
    st.session_state.indexed_files = []
    st.session_state.messages = []


def render_hero() -> None:
    st.markdown(
        f"""
        <div class="hero-card">
            <div class="app-title">{APP_TITLE}</div>
            <div class="app-subtitle">{APP_SUBTITLE}</div>
            <div class="status-pill">Local RAG • Ollama Chat • HF Embeddings • No API cost</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metrics() -> None:
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="small-label">Knowledge Base</div>
                <div class="big-metric">{'Ready' if st.session_state.vector_ready else 'Not indexed'}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="small-label">Indexed Documents</div>
                <div class="big-metric">{len(st.session_state.indexed_files)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="small-label">Chat Model</div>
                <div class="big-metric">{OLLAMA_CHAT_MODEL}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="small-label">Embed Model</div>
                <div class="big-metric">{HF_EMBED_MODEL.split('/')[-1]}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_sidebar():
    with st.sidebar:
        st.markdown("## Workspace")
        st.caption("Upload documents, build the index, and manage the assistant.")

        st.markdown('<div class="upload-card">', unsafe_allow_html=True)
        uploaded_files = st.file_uploader(
            "Upload policy / SOP PDFs",
            type=["pdf"],
            accept_multiple_files=True,
        )

        process_clicked = st.button("Process Documents", use_container_width=True)
        clear_clicked = st.button("Clear Knowledge Base", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.divider()

        st.divider()
        st.markdown("### Indexed files")
        if st.session_state.indexed_files:
            for file_name in st.session_state.indexed_files:
                st.markdown(f"- {file_name}")
        else:
            st.caption("No files indexed yet.")

    return uploaded_files, process_clicked, clear_clicked


def process_documents(uploaded_files) -> None:
    if not uploaded_files:
        st.warning("Please upload at least one PDF before processing.")
        return

    with st.status("Processing documents...", expanded=True) as status:
        st.write("Saving uploaded files...")
        file_paths = save_uploaded_files(uploaded_files)

        st.write("Loading PDF documents...")
        docs = load_pdf_documents(file_paths)

        st.write("Splitting into chunks...")
        chunks = split_documents(docs)

        st.write("Building vector store...")
        build_vectorstore(chunks)

        st.session_state.vector_ready = True
        st.session_state.indexed_files = [path.name for path in file_paths]

        status.update(label="Documents indexed successfully.", state="complete")


def render_sources(source_documents) -> None:
    st.markdown("### Sources used")
    for i, doc in enumerate(source_documents, start=1):
        source_name = doc.metadata.get("file_name", "Unknown source")
        page = doc.metadata.get("page", "N/A")
        preview = doc.page_content[:MAX_SOURCE_PREVIEW_CHARS].strip()

        st.markdown(
            f"""
            <div class="source-card">
                <div><strong>Source {i}</strong></div>
                <div class="source-meta">{source_name} • Page {page}</div>
                <div>{preview}...</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


check_ollama_health()
render_hero()
render_metrics()

uploaded_files, process_clicked, clear_clicked = render_sidebar()

if clear_clicked:
    reset_knowledge_base()
    st.success("Knowledge base cleared.")

if process_clicked:
    process_documents(uploaded_files)

if not st.session_state.vector_ready:
    chroma_sqlite = CHROMA_DIR / "chroma.sqlite3"
    if chroma_sqlite.exists():
        st.session_state.vector_ready = True

st.markdown("### Chat")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("View sources"):
                render_sources(message["sources"])

prompt = st.chat_input("Ask a question about the indexed policy or SOP documents...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if not st.session_state.vector_ready:
        warning_msg = "Please upload and process documents first."
        st.session_state.messages.append({"role": "assistant", "content": warning_msg})
        with st.chat_message("assistant"):
            st.warning(warning_msg)
    else:
        with st.chat_message("assistant"):
            with st.spinner("Retrieving relevant policy sections..."):
                vectorstore = load_vectorstore()
                result = answer_question(vectorstore, prompt)
                answer = result["answer"]
                source_documents = result["source_documents"]

            st.markdown(answer)
            with st.expander("View sources"):
                render_sources(source_documents)

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
                "sources": source_documents,
            }
        )