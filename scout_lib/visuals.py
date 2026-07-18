"""Chart/diagram builders. Nothing here touches st.session_state — each
function takes plain data in and returns a figure or an HTML/SVG string,
so it can be tested or reused outside the Streamlit callback that renders it.
"""

import numpy as np
import plotly.graph_objects as go
from sklearn.decomposition import PCA

from .text_format import truncate_label, source_title_only


def relevance_color(pct: float) -> str:
    """Continuous rust -> pale-rule gradient by match strength, instead of
    a plain top/not-top binary."""
    rust = (193, 80, 45)
    pale = (217, 208, 184)
    r, g, b = (round(pale[i] + (rust[i] - pale[i]) * pct) for i in range(3))
    return f"rgb({r},{g},{b})"


def relevance_stars(pct: float, max_stars: int = 5) -> str:
    """Star count derived directly from match_pct's relative ranking within
    the pool — a visual shorthand for relevance, not a fabricated rating."""
    filled = max(0, min(max_stars, round(pct * max_stars)))
    return "★" * filled + "☆" * (max_stars - filled)


def render_flow_diagram(n_candidates: int, n_selected: int) -> str:
    """Small SVG showing the retrieval pipeline: question -> embed -> search
    -> rank -> select -> generate, so the selection board has a visual
    map of the process, not just a bar chart of the outcome."""
    steps = [
        ("Question", "your typed query"),
        ("Embed", "MiniLM-L6-v2 vector"),
        (f"Search ({n_candidates})", "Chroma nearest-neighbor"),
        ("Rank", "sort by vector distance"),
        (f"Select ({n_selected})", "keep closest matches"),
        ("Generate", "Gemini answers"),
    ]
    box_w, box_h, gap = 118, 62, 22
    total_w = len(steps) * box_w + (len(steps) - 1) * gap
    parts = [f'<svg viewBox="0 0 {total_w} 100" xmlns="http://www.w3.org/2000/svg" '
              'style="width:100%;height:auto;">',
             '<defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="6" refY="4" '
             'orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="#5b5a4f"/></marker></defs>']
    x = 0
    highlight = {2, 4}  # Search and Select steps stand out
    for i, (title, sub) in enumerate(steps):
        is_hl = i in highlight
        fill, stroke, tcolor = ("#c1502d", "#c1502d", "#faf6ec") if is_hl else ("#faf6ec", "#20211c", "#20211c")
        parts.append(
            f'<rect x="{x}" y="8" width="{box_w}" height="{box_h}" rx="6" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="1.2"/>'
            f'<text x="{x + box_w/2}" y="32" text-anchor="middle" font-family="Fraunces, serif" '
            f'font-size="12.5" font-weight="600" fill="{tcolor}">{title}</text>'
            f'<text x="{x + box_w/2}" y="50" text-anchor="middle" font-family="JetBrains Mono, monospace" '
            f'font-size="8" fill="{tcolor}">{sub}</text>'
        )
        if i < len(steps) - 1:
            ax = x + box_w
            parts.append(
                f'<line x1="{ax}" y1="{8 + box_h/2}" x2="{ax + gap - 5}" y2="{8 + box_h/2}" '
                'stroke="#5b5a4f" stroke-width="1.4" marker-end="url(#arrow)"/>'
            )
        x += box_w + gap
    parts.append("</svg>")
    return "".join(parts)


def render_coverage_map(candidates: list[dict], all_files: list) -> go.Figure:
    """Bar per loaded source file showing how many of the current candidate
    chunks came from it — files at 0 are visible gaps, not just omissions."""
    counts = {f.name: 0 for f in all_files}
    for c in candidates:
        counts[c["source"]] = counts.get(c["source"], 0) + 1
    items = sorted(counts.items(), key=lambda kv: -kv[1])
    fig = go.Figure(go.Bar(
        x=[v for _, v in items], y=[truncate_label(k) for k, _ in items], orientation="h",
        marker=dict(color=["#c1502d" if v > 0 else "#e3dcc7" for _, v in items]),
        hovertext=[k for k, _ in items], hoverinfo="text",
    ))
    fig.update_layout(
        paper_bgcolor="#faf6ec", plot_bgcolor="#faf6ec",
        xaxis=dict(title="chunks in candidate pool", color="#5b5a4f", gridcolor="#e3dcc7"),
        yaxis=dict(autorange="reversed", color="#20211c", tickfont=dict(color="#20211c"), automargin=True),
        height=max(200, 32 * len(items) + 50), margin=dict(l=10, r=10, t=10, b=35),
        font=dict(family="JetBrains Mono, monospace", size=11, color="#5b5a4f"),
    )
    return fig


def build_retrieval_space_fig(candidates: list[dict], chunks: list[dict], question: str, ef) -> go.Figure:
    """PCA-projected scatter of the full candidate pool + the question, in
    the exact embedding space retriever.py searched (not an approximation)."""
    all_texts = [c["text"] for c in candidates]
    if len(all_texts) >= 2:
        vecs = np.array(ef(all_texts + [question]))
        coords = PCA(n_components=2, random_state=0).fit_transform(vecs)
        cand_xy, query_xy = coords[:-1], coords[-1]
    else:
        cand_xy = np.zeros((len(all_texts), 2))
        query_xy = np.zeros(2)

    is_selected = [i < len(chunks) for i in range(len(candidates))]
    titles = [source_title_only(c["source"]) for c in candidates]
    previews = [c["text"][:110].replace("\n", " ").strip() + "…" for c in candidates]

    fig = go.Figure()
    for i, ((px, py), sel) in enumerate(zip(cand_xy, is_selected)):
        fig.add_trace(go.Scatter(
            x=[query_xy[0], px], y=[query_xy[1], py], mode="lines",
            line=dict(color="#d9d0b8", dash="dash" if sel else "dot", width=1.2 if sel else 0.6),
            opacity=1.0 if sel else 0.4, showlegend=False, hoverinfo="skip",
        ))

    other_idx = [i for i, sel in enumerate(is_selected) if not sel]
    sel_idx = [i for i, sel in enumerate(is_selected) if sel]

    if other_idx:
        fig.add_trace(go.Scatter(
            x=[cand_xy[i][0] for i in other_idx], y=[cand_xy[i][1] for i in other_idx],
            mode="markers", marker=dict(size=11, color="#faf6ec", line=dict(color="#8a8776", width=1.3)),
            name="other candidates",
            customdata=[[titles[i], candidates[i]["distance"], previews[i]] for i in other_idx],
            hovertemplate="<b>%{customdata[0]}</b><br>distance: %{customdata[1]:.3f}<br>%{customdata[2]}<extra>not selected</extra>",
        ))
    fig.add_trace(go.Scatter(
        x=[cand_xy[i][0] for i in sel_idx], y=[cand_xy[i][1] for i in sel_idx],
        mode="markers", marker=dict(size=15, color="#d9822b", line=dict(color="#20211c", width=1)),
        name="retrieved chunk",
        customdata=[[titles[i], candidates[i]["distance"], previews[i]] for i in sel_idx],
        hovertemplate="<b>%{customdata[0]}</b><br>distance: %{customdata[1]:.3f}<br>%{customdata[2]}<extra>selected</extra>",
    ))
    fig.add_trace(go.Scatter(
        x=[query_xy[0]], y=[query_xy[1]], mode="markers",
        marker=dict(size=18, color="#3a6ea5", symbol="star", line=dict(color="#20211c", width=1)),
        name="your question", hoverinfo="name",
    ))
    fig.update_layout(
        paper_bgcolor="#faf6ec", plot_bgcolor="#faf6ec",
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        height=320, margin=dict(l=10, r=10, t=10, b=10), showlegend=False,
    )
    return fig


def build_selection_board_fig(candidates: list[dict], chunks: list[dict], pcts: list[float]) -> go.Figure:
    """Horizontal bar per candidate (darker = closer match) with a dashed
    line marking where the selection cutoff fell."""
    total_n = len(candidates)
    labels = [source_title_only(c["source"]) for c in candidates]
    colors = [relevance_color(p) for p in pcts]

    board_fig = go.Figure(go.Bar(
        x=pcts, y=[truncate_label(f"{i+1}. {lbl}") for i, lbl in enumerate(labels)],
        orientation="h", marker=dict(color=colors),
        hovertext=[f"{lbl} — distance: {c['distance']:.3f}"
                   for lbl, c in zip(labels, candidates)],
        hoverinfo="text",
    ))
    board_fig.add_vline(
        x=(pcts[len(chunks) - 1] + pcts[len(chunks)]) / 2 if total_n > len(chunks) else 0,
        line_dash="dash", line_color="#20211c",
        annotation_text="selection cutoff", annotation_font_size=10,
        annotation_font_family="JetBrains Mono",
    )
    board_fig.update_layout(
        paper_bgcolor="#faf6ec", plot_bgcolor="#faf6ec",
        xaxis=dict(visible=False, range=[0, 1]),
        yaxis=dict(autorange="reversed", color="#20211c", tickfont=dict(color="#20211c", size=11), automargin=True),
        height=36 * total_n + 60, margin=dict(l=10, r=10, t=10, b=10),
        showlegend=False, font=dict(family="JetBrains Mono, monospace", color="#20211c"),
    )
    return board_fig