"""Streamlit web app for the local RAG personal memory system."""
import json
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
from src.config import (
    CORPUS_PATH,
    LLM_MODEL_PATH,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    DEFAULT_RETRIEVAL_K,
    STREAMLIT_PORT,
)
from src.data_generator import generate_corpus, save_corpus
from src.model_loader import load_llm
from src.rag_engine import RAGEngine

st.set_page_config(page_title="Personal Memory RAG", page_icon="🧠", layout="wide")

# ------------------------------------------------------------------
# Cached resources
# ------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading local LLM & index...")
def get_engine():
    llm = load_llm()
    engine = RAGEngine(llm)
    if CORPUS_PATH.exists():
        corpus = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))
        engine.index_corpus(corpus)
    return engine


# ------------------------------------------------------------------
# Session state
# ------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
    # Showcase mode: pre-populate with a demo conversation
    demo_path = Path(__file__).resolve().parent.parent / "data" / "demo_result.json"
    if os.getenv("SHOWCASE_MODE") and demo_path.exists():
        demo = json.loads(demo_path.read_text(encoding="utf-8"))
        st.session_state.messages = [
            {"role": "user", "content": demo["query"]},
            {"role": "assistant", "content": demo},
        ]

# ------------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------------
with st.sidebar:
    st.title("🧠 Personal Memory RAG")
    st.markdown("Local RAG with **Gemma 2 2B IT**")
    st.divider()

    # Model status
    if LLM_MODEL_PATH.exists():
        st.success("Model found")
    else:
        st.error("Model missing — run `scripts/setup.sh`")

    # Data controls
    st.subheader("Data")
    if st.button("🔄 Regenerate Synthetic Data"):
        with st.spinner("Generating corpus..."):
            corpus = generate_corpus()
            save_corpus(CORPUS_PATH, corpus)
            st.cache_resource.clear()
        st.success(f"Generated {len(corpus)} documents")

    if st.button("📦 Rebuild Index"):
        if not CORPUS_PATH.exists():
            st.error("No corpus found. Regenerate data first.")
        else:
            with st.spinner("Rebuilding index..."):
                st.cache_resource.clear()
                engine = get_engine()
            st.success(f"Index rebuilt: **{engine.store.count()}** chunks")

    st.divider()

    # Hyperparameters
    st.subheader("Settings")
    temperature = st.slider("Temperature", 0.0, 1.5, DEFAULT_TEMPERATURE, 0.1)
    top_p = st.slider("Top-p", 0.1, 1.0, DEFAULT_TOP_P, 0.05)
    retrieval_k = st.slider("Retrieval K", 1, 10, DEFAULT_RETRIEVAL_K, 1)

    st.divider()
    st.caption(f"Port: {STREAMLIT_PORT}")

# ------------------------------------------------------------------
# Main chat
# ------------------------------------------------------------------
st.header("Chat with your Memory")

if not LLM_MODEL_PATH.exists():
    st.error("LLM model not found. Please run `scripts/setup.sh` first.")
    st.stop()

try:
    engine = get_engine()
except Exception as e:
    st.error(f"Failed to initialize engine: {e}")
    st.stop()

# Demo quick-queries
cols = st.columns(3)
demo_queries = [
    "What meetings do I have this week?",
    "Summarize my health notes",
    "Any pending action items?",
]
for col, q in zip(cols, demo_queries):
    if col.button(q, use_container_width=True):
        st.session_state.messages.append({"role": "user", "content": q})
        with st.spinner("Retrieving & generating..."):
            result = engine.chat(q, k=retrieval_k, temperature=temperature, top_p=top_p)
        st.session_state.messages.append({"role": "assistant", "content": result})

# Display history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            st.markdown(msg["content"]["answer"])
            with st.expander("Sources"):
                for i, src in enumerate(msg["content"]["sources"], 1):
                    st.markdown(
                        f"**[{i}]** `{src['metadata']['title']}` ({src['metadata']['doc_type']}) — "
                        f"distance `{src['distance']:.3f}`"
                    )
                    st.caption(src["text"][:300] + "...")
        else:
            st.markdown(msg["content"])

# Input
if prompt := st.chat_input("Ask anything about your personal memory..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = engine.chat(prompt, k=retrieval_k, temperature=temperature, top_p=top_p)
        st.markdown(result["answer"])
        with st.expander("Sources"):
            for i, src in enumerate(result["sources"], 1):
                st.markdown(
                    f"**[{i}]** `{src['metadata']['title']}` ({src['metadata']['doc_type']}) — "
                    f"distance `{src['distance']:.3f}`"
                )
                st.caption(src["text"][:300] + "...")
    st.session_state.messages.append({"role": "assistant", "content": result})
