import streamlit as st

def inject_head_metadata():
    html_content = """
    <script>
    const doc = window.parent.document;
    doc.title = "Scout — Traced RAG";
    doc.documentElement.lang = "en";
    </script>
    """
    st.iframe(html_content, height=1)