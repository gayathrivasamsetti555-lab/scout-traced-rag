"""Quality scoring for a retrieval+generation trace: how close the matches
were, how many distinct sources they span, and how confident the top hit
was relative to the rest."""

from retriever import MAX_DISTANCE


def match_pct(distance: float, max_distance: float = MAX_DISTANCE) -> float:
    """Absolute relevance percentage: 0% at max_distance (the same cosine
    distance ceiling retriever.py uses to decide "not relevant enough to
    return"), 100% at distance 0.

    This used to normalize *within the current candidate pool*
    (min/max of just this trace's distances) instead of against a fixed
    reference. That meant the closest chunk in ANY pool always scored
    near 100%, even when every candidate in that pool was actually
    irrelevant (e.g. distances of 0.82-0.87 for an off-topic question) —
    which is exactly why "Who is Vasamsetti" showed a "50 · ADEQUATE"
    badge despite retrieving nothing usable. Scaling against an absolute
    ceiling instead means a bad pool now produces a low score, full stop.
    """
    if max_distance <= 0:
        return 0.0
    return max(0.0, min(1.0, 1 - distance / max_distance))


def rag_quality_score(selected: list[dict]) -> dict:
    """A rough, transparent composite score for the retrieval+answer, built
    from three signals: how close the selected chunks are on average, how
    many distinct source documents they span, and how much better the top
    hit is than the rest (a confident single best match vs. a flat tie)."""
    pcts = [match_pct(c["distance"]) for c in selected]
    avg_match = sum(pcts) / len(pcts) if pcts else 0.0
    unique_sources = len({c["source"] for c in selected})
    coverage = min(unique_sources / max(len(selected), 1), 1.0)
    separation = (pcts[0] - pcts[1]) if len(pcts) > 1 else 0.0

    score = 0.7 * avg_match + 0.2 * coverage + 0.1 * min(separation * 2, 1.0)
    score_pct = round(score * 100)

    if score_pct >= 75:
        label = "Strong"
    elif score_pct >= 50:
        label = "Adequate"
    else:
        label = "Weak"

    return {
        "score": score_pct, "label": label, "avg_match": avg_match,
        "unique_sources": unique_sources, "separation": separation,
    }


def retrieval_rationale(question: str, selected: list[dict], quality: dict) -> str:
    """Plain-language explanation of why these chunks were picked, using
    the actual numbers rather than a generic disclaimer."""
    top = selected[0] if selected else None
    lines = [
        f"Your question was embedded with the same model used to index the archive, "
        f"then compared against every indexed chunk using Chroma's configured vector distance "
        f"(lower = closer match)."
    ]
    if top:
        lines.append(
            f"The closest match was **{top['source']}**, which is why it's ranked #1 below."
        )
    if quality["unique_sources"] > 1:
        lines.append(
            f"The selected chunks span {quality['unique_sources']} different source files, "
            f"so the answer isn't resting on a single document."
        )
    else:
        lines.append(
            "All selected chunks came from a single source file — worth a second look if "
            "you expected broader coverage."
        )
    if quality["separation"] < 0.1:
        lines.append(
            "The top few matches were close in score, meaning several passages were nearly "
            "equally relevant rather than one standing out clearly."
        )
    return " ".join(lines)