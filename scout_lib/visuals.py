import numpy as np
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from .text_format import truncate_label, source_title_only

def relevance_color(pct: float) -> str:
    """Continuous rust -> pale-rule gradient by match strength."""
    if pct > 1.0:
        pct = pct / 100.0
        
    pct = max(0.0, min(float(pct), 1.0))
    pale = (217, 208, 184)
    rust = (193, 80, 45)
    
    r, g, b = (round(pale[i] + (rust[i] - pale[i]) * pct) for i in range(3))
    return f"rgb({r},{g},{b})"

def relevance_stars(pct: float) -> str:
    """Generates a visual star rating string based on the match score."""
    if pct <= 0:
        return "☆☆☆☆☆"
    if pct > 1.0:
        pct = pct / 100.0
        
    stars = max(1, min(5, round(pct * 5)))
    return "★" * stars + "☆" * (5 - stars)

def build_selection_board_fig(candidates: list[dict], chunks: list[dict], pcts: list[float]) -> go.Figure:
    """Builds a horizontal bar chart where bar lengths reflect actual similarity scores."""
    total_n = len(candidates)
    if total_n == 0:
        return go.Figure()

    valid_pcts = [p / 100.0 if p > 1.0 else p for p in pcts]
    colors = [relevance_color(p) for p in valid_pcts]

    board_fig = go.Figure(go.Bar(
        x=valid_pcts,
        y=[f"Chunk {c.get('id', i)}" for i, c in enumerate(candidates)],
        orientation='h',
        marker=dict(color=colors),
        hoverinfo="text",
        hovertext=[f"Match: {round(p * 100)}% | Source: {source_title_only(c.get('source', 'Unknown'))}" for c, p in zip(candidates, valid_pcts)]
    ))

    cutoff_val = 0.52
    if len(valid_pcts) >= 5:
        sorted_pcts = sorted(valid_pcts, reverse=True)
        cutoff_val = sorted_pcts[4] - 0.005

    board_fig.add_vline(
        x=cutoff_val, 
        line_dash="dash", 
        line_color="#8c8d8a", 
        annotation_text="selection cutoff", 
        annotation_font_size=10,
        annotation_font_family="JetBrains Mono",
        annotation_position="top right"
    )
    
    board_fig.update_layout(
        paper_bgcolor="#faf6ec", 
        plot_bgcolor="#faf6ec",
        xaxis=dict(visible=True, range=[0, 1], gridcolor="#e6dfcc", title="Match Confidence Score"),
        yaxis=dict(autorange="reversed", color="#20211c", tickfont=dict(color="#20211c", size=11), automargin=True),
        height=36 * total_n + 60, 
        margin=dict(l=10, r=10, t=10, b=30),
        showlegend=False, 
        font=dict(family="JetBrains Mono, monospace", color="#20211c"),
    )
    return board_fig

def render_flow_diagram(selected_count: int, total_candidates: int) -> go.Figure:
    """Renders the Sankey or flow pipeline diagram matching your project's interface setup."""
    fig = go.Figure(data=[go.Sankey(
        node = dict(
          pad = 15,
          thickness = 20,
          line = dict(color = "black", width = 0.5),
          label = ["Total Chunks Fetched", "Filtered Candidates", "Target Context Used"],
          color = "#c1502d"
        ),
        link = dict(
          source = [0, 1], 
          target = [1, 2],
          value = [total_candidates, selected_count],
          color = "rgba(217, 208, 184, 0.4)"
        )
    )])
    
    fig.update_layout(
        paper_bgcolor="#faf6ec",
        plot_bgcolor="#faf6ec",
        font=dict(family="JetBrains Mono, monospace", color="#20211c", size=12),
        margin=dict(l=10, r=10, t=20, b=10),
        height=180
    )
    return fig

def render_coverage_map(candidates: list[dict], chunks: list[dict] = None, *args, **kwargs) -> go.Figure:
    """Computes and draws how many candidate chunks originate from each file source."""
    counts = {}
    for c in candidates:
        src = source_title_only(c.get("source", "Unknown"))
        counts[src] = counts.get(src, 0) + 1
        
    sources = list(counts.keys())
    values = [counts[s] for s in sources]
    
    fig = go.Figure(go.Bar(
        x=values,
        y=sources,
        orientation='h',
        marker=dict(color="#c1502d")
    ))
    fig.update_layout(
        paper_bgcolor="#faf6ec",
        plot_bgcolor="#faf6ec",
        xaxis=dict(gridcolor="#e6dfcc", dtick=1),
        yaxis=dict(color="#20211c", automargin=True),
        font=dict(family="JetBrains Mono, monospace", color="#20211c", size=11),
        margin=dict(l=10, r=10, t=10, b=10),
        height=max(140, len(sources) * 35 + 40)
    )
    return fig

def build_retrieval_space_fig(candidates: list[dict], chunks: list[dict], query_text: str, embedding_function) -> go.Figure:
    """Projects high-dimensional vectors cleanly onto an auto-scaled 2D scatter map."""
    fig = go.Figure()
    
    # Base layout decoration variables
    dark_bg = "#1e222b"
    grid_color = "#2c313c"
    
    try:
        query_embedding = embedding_function([query_text])[0]
        candidate_embeddings = [c["embedding"] for c in candidates if "embedding" in c]
        # 🎯 FIX: Elegant styled fallback layout container when no embeddings are parsed
        if not candidate_embeddings:
            fig.update_layout(
                paper_bgcolor="#faf6ec", plot_bgcolor=dark_bg,
                xaxis=dict(visible=False), yaxis=dict(visible=False),
                annotations=[dict(
                    text="No Embedding Coordinates Present<br>for this Query Answer Scope",
                    font=dict(family="JetBrains Mono", size=11, color="#8c8d8a"),
                    showarrow=False
                )],
                margin=dict(l=15, r=15, t=15, b=15), height=260
            )
            return fig
            
        all_vectors = [query_embedding] + candidate_embeddings
        pca = PCA(n_components=2)
        coords = pca.fit_transform(np.array(all_vectors, dtype=np.float32))
    except Exception:
        fig.update_layout(
            paper_bgcolor="#faf6ec", plot_bgcolor=dark_bg,
            xaxis=dict(visible=False), yaxis=dict(visible=False),
            margin=dict(l=15, r=15, t=15, b=15), height=260
        )
        return fig
        
    selected_ids = {c.get("id") for c in chunks}
    unsel_x, unsel_y, unsel_txt = [], [], []
    sel_x, sel_y, sel_txt = [], [], []
    
    chunk_coords = coords[1:]
    for i, c in enumerate(candidates):
        if i >= len(chunk_coords):
            break
        is_sel = c.get("id", i) in selected_ids
        x_val = chunk_coords[i, 0]
        y_val = chunk_coords[i, 1]
        hover_str = f"Chunk {c.get('id', i)} | Dist: {c.get('distance', 0):.3f}"
        
        if is_sel:
            sel_x.append(x_val)
            sel_y.append(y_val)
            sel_txt.append(hover_str)
        else:
            unsel_x.append(x_val)
            unsel_y.append(y_val)
            unsel_txt.append(hover_str)

    if unsel_x:
        fig.add_trace(go.Scatter(
            x=unsel_x, y=unsel_y, mode='markers',
            marker=dict(size=12, color="#8c8d8a", line=dict(color=dark_bg, width=1.5)),
            hoverinfo="text", hovertext=unsel_txt, name="Other Candidates"
        ))

    if sel_x:
        fig.add_trace(go.Scatter(
            x=sel_x, y=sel_y, mode='markers',
            marker=dict(size=14, color="#c1502d", line=dict(color=dark_bg, width=1.5)),
            hoverinfo="text", hovertext=sel_txt, name="Retrieved Passages"
        ))
        
    fig.add_trace(go.Scatter(
        x=[coords[0, 0]], y=[coords[0, 1]], mode='markers',
        marker=dict(size=18, color="#4a7bb0", symbol="star", line=dict(color=dark_bg, width=1.5)),
        hoverinfo="text", hovertext="Query Anchor Point", name="Your Query"
    ))
    
    fig.update_layout(
        paper_bgcolor="#faf6ec",
        plot_bgcolor=dark_bg, 
        xaxis=dict(showgrid=True, gridcolor=grid_color, zeroline=True, zerolinecolor="#8c8d8a", autorange=True),
        yaxis=dict(showgrid=True, gridcolor=grid_color, zeroline=True, zerolinecolor="#8c8d8a", autorange=True),
        showlegend=False,
        margin=dict(l=20, r=20, t=20, b=20),
        height=260
    )
    return fig