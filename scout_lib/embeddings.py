import streamlit as st
from chromadb.utils import embedding_functions


@st.cache_resource
def get_embedder():
    """Same model retriever.py uses, so the "retrieval space" plot is the
    actual space Chroma searched — not an approximation."""
    return embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")