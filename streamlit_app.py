import streamlit as st
import json
import asyncio
import tempfile
import os

from page_index.providers import get_provider
from page_index.engine import PageIndexEngine

st.set_page_config(page_title="Page Index RAG (Vectorless)")
st.title("Page Index RAG — Markdown to Tree (Vectorless)")

st.sidebar.header("LLM Provider")
provider_name = st.sidebar.selectbox("Provider", options=["mock", "openai", "gemini"], index=0)
api_key = st.sidebar.text_input("API Key (optional)", type="password")

uploaded = st.file_uploader("Upload a Markdown (.md) or PDF file", type=["md", "pdf"])
query = st.text_input("User query")

show_reasoning = st.checkbox("Show reasoning steps", value=True)

if uploaded is not None and query:
    md_bytes = uploaded.getvalue()
    md_text = md_bytes.decode("utf-8")
    suffix = ".md"
    mode = "w"
    if uploaded.name.lower().endswith('.pdf'):
        suffix = ".pdf"
        mode = "wb"
    with tempfile.NamedTemporaryFile(mode, suffix=suffix, delete=False) as tmp:
        if mode == "w":
            tmp.write(md_text)
        else:
            tmp.write(uploaded.getvalue())
        tmp_path = tmp.name

    st.info(f"Saved to {tmp_path}")

    provider = get_provider(provider_name, api_key=api_key)
    engine = PageIndexEngine(provider=provider, model=None)

    if st.button("Run Retrieval"):
        with st.spinner("Running pipeline..."):
            result = asyncio.run(engine.answer(tmp_path, query))

        st.subheader("Query Decomposition")
        st.json(result.get('decomposition'))

        st.subheader("Candidate Pages")
        st.write(result.get('candidates'))

        st.subheader("LLM Guided Selection")
        if show_reasoning:
            st.write(result.get('guided_selection'))
        else:
            st.write([s['page'] for s in result.get('guided_selection', {}).get('selected', [])])

        st.subheader("Context Sources")
        st.write(result.get('context_sources'))

        st.subheader("Final Answer")
        st.json(result.get('answer'))

    if os.path.exists(tmp_path):
        os.remove(tmp_path)
