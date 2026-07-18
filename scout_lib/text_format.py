"""Text-level formatting helpers: turning raw source strings and model
output into the labels/HTML/highlighting used throughout the UI."""

import re

import numpy as np


def truncate_label(text: str, max_len: int = 34) -> str:
    return text if len(text) <= max_len else text[:max_len - 1].rstrip() + "…"


def parse_source_metadata(source: str) -> dict | None:
    """Best-effort parse of a 'YYYY - Author - Title' style source label
    into structured fields. Returns None (caller falls back to the raw
    string) if it doesn't match — nothing here is invented, only reformatted
    from what's actually encoded in the filename/source string already."""
    m = re.match(r"^\s*(\d{4})\s*-\s*([^-]+?)\s*-\s*(.+?)\s*$", source)
    if not m:
        return None
    year, author, title = m.groups()
    return {"year": year, "author": author.strip(), "title": title.strip()}


def source_display(source: str) -> str:
    """Formatted HTML for a source label: a clean title/author/year card if
    the source string parses as one, otherwise the raw string unchanged."""
    meta = parse_source_metadata(source)
    if not meta:
        return f'<span class="source-raw">{source}</span>'
    return (
        f'<div class="source-meta">'
        f'<div class="source-title">📄 {meta["title"]}</div>'
        f'<div class="source-byline">{meta["author"]} et al. · {meta["year"]}</div>'
        f'</div>'
    )


def source_title_only(source: str) -> str:
    """Short label for chart axes/hovertext — parsed title if available,
    otherwise the raw source string."""
    meta = parse_source_metadata(source)
    return meta["title"] if meta else source


def resolve_source_citations(answer: str, chunks: list[dict]) -> str:
    """generator.py's prompt asks the model to cite '[Source N]', where N is
    the chunk's 1-indexed position in the context block — but that's an
    opaque index the model doesn't always format consistently (brackets
    optional, plural/singular varies) and it means nothing to a reader who
    can't see the original prompt. Replace every 'Source N' reference with
    the actual filename of chunk N, post-hoc and regex-based rather than
    trusting the model to self-report it — LLMs are inconsistent about
    literal formatting instructions, but this substitution is exact."""
    def label_for(n: int) -> str:
        idx = n - 1
        if 0 <= idx < len(chunks):
            return chunks[idx]["source"]
        return f"Source {n}"  # out-of-range fallback; shouldn't normally happen

    return re.sub(
        r"\[?\bSource\s+(\d+)\]?",
        lambda m: label_for(int(m.group(1))),
        answer,
    )


def highlight_best_sentence(text: str, question: str, ef) -> str:
    """Bold+highlight the single sentence in this chunk most similar to the
    question (by embedding cosine similarity), so it's clear at a glance
    which part of the passage actually earned it a spot in the results —
    not just the whole undifferentiated paragraph."""
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]
    if len(sentences) <= 1:
        return text
    vecs = np.array(ef(sentences + [question]))
    sent_vecs, q_vec = vecs[:-1], vecs[-1]
    norms = np.linalg.norm(sent_vecs, axis=1) * np.linalg.norm(q_vec) + 1e-9
    sims = (sent_vecs @ q_vec) / norms
    best_idx = int(np.argmax(sims))
    out = [f'<mark class="hl-sentence">{s}</mark>' if i == best_idx else s
           for i, s in enumerate(sentences)]
    return " ".join(out)