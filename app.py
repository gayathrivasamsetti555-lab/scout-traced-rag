"""
Scout — A research assistant that shows its work.

Streamlit port of the Scout HTML/JS demo. Reuses the same
retrieve() / generate() functions as app.py so this can sit alongside
(or replace) the plain chat UI, but renders the "traced retrieval"
experience: a synthesized answer with citations, a retrieval-space
scatter plot, and ranked passage cards with similarity bars.

File layout:
    scout_app.py            <- this file: session state, layout, callbacks
    scout_lib/
        head_meta.py         inject_head_metadata()
        embeddings.py        get_embedder()
        scoring.py           match_pct, rag_quality_score, retrieval_rationale
        text_format.py       source labels, citation resolution, sentence highlighting
        visuals.py           flow diagram, coverage map, retrieval-space + board figures
        styles.py            CSS theme

Drop this file (and scout_lib/) next to app.py, retriever.py, generator.py,
ingest.py. Run with: streamlit run scout_app.py
"""

import subprocess
from pathlib import Path
import sys
import streamlit as st

from retriever import retrieve
from generator import generate

from scout_lib.head_meta import inject_head_metadata
from scout_lib.embeddings import get_embedder
from scout_lib.scoring import match_pct, rag_quality_score, retrieval_rationale
from scout_lib.text_format import (
    source_display,
    source_title_only,
    resolve_source_citations,
    highlight_best_sentence,
)
from scout_lib.visuals import (
    render_flow_diagram,
    render_coverage_map,
    relevance_color,
    relevance_stars,
    build_retrieval_space_fig,
    build_selection_board_fig,
)
from scout_lib.styles import apply_styles

# ── Auto-ingest on first startup (same as app.py) ─────────────────────
CHROMA_DIR = "chroma_db"
if not Path(CHROMA_DIR).exists():
    with st.spinner("Indexing documents for the first time (30-60 seconds)..."):
        subprocess.run([sys.executable, "ingest.py"], check=True)

# ── Page config ────────────────────────────────────────────────────────
st.set_page_config(page_title="Scout — Traced RAG", page_icon="📜", layout="wide")

inject_head_metadata()
apply_styles()

CANDIDATE_POOL_SIZE = 12   # how many candidates the retriever pulls before cutting down
SELECTED_COUNT = 5         # how many actually get passed to the generator


def section_head(num: int, title: str):
    st.markdown(
        f"""<div class="section-head"><span class="section-num">{num}</span><h3>{title}</h3></div>""",
        unsafe_allow_html=True,
    )


def run_trace(q: str):
    with st.spinner("tracing nearest passages in embedding space…"):
        try:
            candidates = retrieve(q, n_results=CANDIDATE_POOL_SIZE)
        except Exception as e:
            st.session_state.scout_result = {
                "question": q, "candidates": [], "chunks": [],
                "result": {
                    "answer": f"⚠️ Couldn't retrieve passages from the archive: {e}. "
                              "Check that chroma_db exists and ingest.py has run.",
                    "sources": [], "error": "retrieval_failed",
                },
            }
            return
        selected = candidates[:SELECTED_COUNT]
        result = generate(q, selected)
        if not result.get("error"):
            result["answer"] = resolve_source_citations(result["answer"], selected)
    st.session_state.scout_result = {
        "question": q,
        "candidates": candidates,   # full pool considered
        "chunks": selected,         # subset actually used for the answer
        "result": result,
    }


def regenerate_answer():
    """Retry just the generation step against the passages already
    retrieved — no need to re-query Chroma or re-embed anything."""
    payload = st.session_state.get("scout_result")
    if payload is None:
        return
    result = generate(payload["question"], payload["chunks"])
    if not result.get("error"):
        result["answer"] = resolve_source_citations(result["answer"], payload["chunks"])
    payload["result"] = result
    st.session_state.scout_result = payload


def set_question_and_trace(text: str):
    # Runs BEFORE the rerun that follows a button click, so it's safe to
    # write to the widget's session_state key here (unlike doing it further
    # down in the normal script body, which raises a StreamlitAPIException
    # once the text_input widget already exists in that run).
    st.session_state.question_box = text
    st.session_state.pending_trace = True


# ── Masthead ────────────────────────────────────────────────────────────
data_files = list(Path("data").glob("*.*"))
st.markdown(
    f"""
    <div class="scout-mast">
        <h1>Scout</h1>
        <div class="scout-tagline">A research assistant that shows its work — not just the
        answer, but the thread that led there.</div>
        <div class="scout-dateline">
            <span>{len(data_files)} SOURCE FILES INDEXED</span>
            <span>RAG RETRIEVAL, VISUALIZED</span>
            <span>ANSWERS COME ONLY FROM LOADED DOCUMENTS</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.subheader(f"Loaded documents ({len(data_files)})")
    for f in data_files:
        st.write(f"• {f.name}")
    st.divider()
    if st.button("🗑️ Clear"):
        st.session_state.pop("scout_result", None)
        st.rerun()

# ── Query panel ──────────────────────────────────────────────────────────
st.markdown("<span class='panel-label'>Ask the archive</span>", unsafe_allow_html=True)

if "question_box" not in st.session_state:
    st.session_state.question_box = ""
if "pending_trace" not in st.session_state:
    st.session_state.pending_trace = False

examples = [
    "how does attention work in transformers?",
    "why do language models hallucinate?",
    "what is the best way to chunk documents?",
    "how do vector databases scale?",
]

query_col, btn_col = st.columns([5, 1])
with query_col:
    question = st.text_input(
        "Question", key="question_box", label_visibility="collapsed",
        placeholder="e.g. how does chunking affect retrieval precision?",
    )
with btn_col:
    trace_clicked = st.button("Trace it →", width="stretch")

chip_cols = st.columns(len(examples))
for col, ex in zip(chip_cols, examples):
    col.button(ex, key=f"chip_{ex}", on_click=set_question_and_trace, args=(ex,))

# ── Retrieval + generation ───────────────────────────────────────────────
if trace_clicked and question.strip():
    run_trace(question.strip())
elif st.session_state.pending_trace:
    run_trace(st.session_state.question_box)
    st.session_state.pending_trace = False

# ── Render results ────────────────────────────────────────────────────────
if "scout_result" in st.session_state:
    payload = st.session_state.scout_result
    chunks = payload["chunks"]
    result = payload["result"]
    candidates = payload["candidates"]
    total_n = len(candidates)

    if total_n == 0:
        # Retrieval itself failed (see run_trace) — nothing downstream
        # (board, coverage map, passage cards) has data to work with.
        st.markdown(
            f"""
            <div class="answer-card" style="border-left:4px solid var(--rust);">
                <p style="font-size:15px; font-family:'Inter',sans-serif;">{result['answer']}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.stop()

    quality = rag_quality_score(chunks)

    # ── 1. Answer + retrieval space ──────────────────────────────────────
    section_head(1, "Synthesized answer")
    st.markdown(
        f"""
        <div class="quality-badge">
            <span class="num">{quality['score']}</span>
            <span class="lbl">RAG quality: {quality['label']}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns(2)

    with left:
        if result.get("error"):
            st.markdown(
                f"""
                <div class="answer-card" style="border-left:4px solid var(--rust);">
                    <p style="font-size:15px; font-family:'Inter',sans-serif;">⚠️ {result['answer']}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.button("🔁 Retry generation", key="retry_generation",
                      on_click=regenerate_answer, width="stretch")
        else:
            st.markdown(
                f"""
                <div class="answer-card">
                    <p>{result['answer']}</p>
                    <div class="answer-note">
                    Answer generated from the retrieved passages shown at right — no information
                    outside the loaded documents was used.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        st.markdown(
            f"""<div class="rationale-box">{retrieval_rationale(payload['question'], chunks, quality)}</div>""",
            unsafe_allow_html=True,
        )

    with right:
        st.markdown("<span class='panel-label'>Retrieval space</span>", unsafe_allow_html=True)
        ef = get_embedder()
        fig = build_retrieval_space_fig(candidates, chunks, payload["question"], ef)
        st.plotly_chart(fig, width="stretch", theme=None)
        st.caption("🔵 your question · 🟠 retrieved chunk · ⚪ other candidates — hover any point for details")

    # ── 2. Chunk Visual Board ─────────────────────────────────────────────
    section_head(2, "Chunk Visual Board — how chunks were selected")
# 🎯 FIXED: Compute clean 0-100 percentages using the updated match_pct logic
    from scout_lib.scoring import match_pct
    pcts = [match_pct(c["distance"]) for c in candidates]

    # Render board figure safely
    board_fig = build_selection_board_fig(candidates, chunks, pcts)
    st.plotly_chart(board_fig, use_container_width=True, theme=None)
    st.markdown(
        f"""<div class="board-note">
        Retrieved {total_n} candidates by vector distance to your question's embedding,
        then kept the closest {len(chunks)} (darker rust = closer match) to pass to the
        generator. Lighter bars were considered but cut — see the sidebar to explore them anyway.
        </div>""",
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<span class='panel-label'>Document coverage map</span>", unsafe_allow_html=True)
    st.markdown('<div class="board-card">', unsafe_allow_html=True)
    st.plotly_chart(render_coverage_map(candidates, data_files), width="stretch", theme=None)
    st.markdown(
        """<div class="board-note">
        How much of this trace's candidate pool came from each loaded file. Files with no bar
        weren't matched at all for this question — see the sidebar for a nudge to explore them.
        </div>""",
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # ── 3. Retrieved passages ─────────────────────────────────────────────
    section_head(3, "Retrieved passages, ranked")
    ef = get_embedder()
    p_cols = st.columns(2)
    for i, c in enumerate(chunks):
        pct = match_pct(c["distance"])
        edge = relevance_color(pct)
        highlighted = highlight_best_sentence(c["text"], payload["question"], ef)
        with p_cols[i % 2]:
            st.markdown(
                f"""
                <div class="passage-card" style="border-color:{edge};">
                    <div class="passage-section">{source_display(c['source'])}</div>
                    <div class="passage-title">Rank #{i+1}</div>
                    <div class="passage-text">{highlighted}</div>
                    <div class="score-num">Similarity distance: {c['distance']:.3f} (lower = closer)</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with st.expander("📋 Raw sources"):
        for s in result.get("sources", []):
            st.write(f"• {s}")

    # ── Recommendations sidebar ───────────────────────────────────────────
    runner_ups = candidates[len(chunks):]
    explored_sources = {c["source"] for c in candidates}
    unexplored = [f.name for f in data_files if f.name not in explored_sources]

    with st.sidebar:
        st.divider()
        st.markdown("<span class='panel-label'>What to explore next</span>", unsafe_allow_html=True)

        st.markdown("**Runner-up passages**")
        if runner_ups:
            for j, c in enumerate(runner_ups[:4]):
                pct = match_pct(c["distance"])
                st.markdown(
                    f"""
                    <div class="sidebar-rec-card">
                        <div class="sidebar-rec-tag">rank #{len(chunks) + j + 1} · <span class="stars">{relevance_stars(pct)}</span></div>
                        {source_display(c['source'])}
                        <div class="source-byline">distance {c['distance']:.3f}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.button("Ask about this →", key=f"rec_pass_{j}",
                          on_click=set_question_and_trace,
                          args=(f"Tell me more about what's in {c['source']}",),
                          width="stretch")
        else:
            st.caption("No runner-ups — the whole pool was used.")

        st.markdown("**Unexplored source files**")
        if unexplored:
            for f_name in unexplored[:6]:
                st.markdown(
                    f"""
                    <div class="sidebar-rec-card">
                        <div class="sidebar-rec-tag">0 chunks this trace</div>
                        {source_display(f_name)}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.button(f"Ask about {f_name} →", key=f"rec_file_{f_name}",
                          on_click=set_question_and_trace,
                          args=(f"What does {f_name} say?",),
                          width="stretch")
        else:
            st.caption("Every file has shown up in a trace so far.")
else:
    st.info("Ask a question above and click **Trace it →** to see the retrieval, visualized.")
