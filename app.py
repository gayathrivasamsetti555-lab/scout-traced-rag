import subprocess
from pathlib import Path
import sys
import streamlit as st
import time
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
    "How does attention work in transformers?",
    "Why do low-rank adaptation (LoRA) matrices target specific weight matrices in the self-attention module while freezing MLPs?",
    "What is the best way to chunk documents?",
    "How does position bias (the 'lost in the middle' phenomenon) affect a language model's ability to utilize retrieved context?",
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
    # Calculate metrics for the Reason breakdown
    score = quality['score']
    if score >= 80:
        confidence = "High"
        color = "#2ecc71"  # Green
    elif score >= 60:
        confidence = "Medium"
        color = "#f1c40f"  # Yellow
    else:
        confidence = "Low"
        color = "#e74c3c"  # Red

    # Based on your app setup:
    # 12 total candidates retrieved, 5 kept in chunks, unique papers counted from source files
    total_candidates = 12  
    passed_to_gemini = len(chunks)
    papers_contributed = len(set(c.get('source_file', '') for c in chunks))
    ret_start = time.perf_counter()
# ... your retrieval logic ...
    retrieval_time = time.perf_counter() - ret_start

    gen_start = time.perf_counter()
    # ... your generator logic ...
    generation_time = time.perf_counter() - gen_start

    # 2. Calculate similarity stats from distances (converting distance to similarity score)
    similarities = [c.get('distance', 0) for c in chunks] if 'chunks' in locals() and chunks else []
    highest_sim = (1.0 - min(similarities)) if similarities else 0.0
    avg_sim = (1.0 - (sum(similarities) / len(similarities))) if similarities else 0.0

    total_candidates = len(candidates) if 'candidates' in locals() else len(chunks)
    passed_to_gemini = len(chunks) if 'chunks' in locals() else 0
    papers_contributed = len(set(c.get('source', '') for c in chunks)) if 'chunks' in locals() else 0

    st.markdown(
        f"""
        <div style="border: 1px solid #333; padding: 15px; border-radius: 8px; margin-bottom: 20px; background-color: #111;">
            <h4 style="margin: 0 0 12px 0; color: #fff;">RAG Quality Metrics</h4>
            <div style="display: flex; gap: 40px; flex-wrap: wrap;">
                <div>
                    <small style="color: #888; text-transform: uppercase; font-size: 11px; letter-spacing: 0.5px;">Confidence</small>
                    <div style="font-size: 22px; font-weight: bold; color: {color}; margin-top: 4px;">{confidence}</div>
                </div>
                <div>
                    <small style="color: #888; text-transform: uppercase; font-size: 11px; letter-spacing: 0.5px;">Similarity Stats</small>
                    <ul style="margin: 4px 0 0 0; padding-left: 18px; font-size: 14px; color: #ccc; list-style-type: disc; line-height: 1.6;">
                        <li>Highest: <b>{highest_sim:.3f}</b></li>
                        <li>Average: <b>{avg_sim:.3f}</b></li>
                    </ul>
                </div>
                <div>
                    <small style="color: #888; text-transform: uppercase; font-size: 11px; letter-spacing: 0.5px;">Performance</small>
                    <ul style="margin: 4px 0 0 0; padding-left: 18px; font-size: 14px; color: #ccc; list-style-type: disc; line-height: 1.6;">
                        <li>Retrieval: <b>{retrieval_time:.3f}s</b></li>
                        <li>Generation: <b>{generation_time:.3f}s</b></li>
                    </ul>
                </div>
                <div>
                    <small style="color: #888; text-transform: uppercase; font-size: 11px; letter-spacing: 0.5px;">Breakdown</small>
                    <ul style="margin: 4px 0 0 0; padding-left: 18px; font-size: 14px; color: #ccc; list-style-type: disc; line-height: 1.6;">
                        <li><b>{papers_contributed}</b> papers contributed</li>
                        <li><b>{total_candidates}</b> retrieved chunks</li>
                        <li><b>{passed_to_gemini}</b> passed to Gemini</li>
                    </ul>
                </div>
            </div>
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
        # Increase figure height to make the retrieval space prominent
        fig = build_retrieval_space_fig(candidates, chunks, payload["question"], ef)
        fig.update_layout(height=480, margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig, use_container_width=True, theme=None)
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
