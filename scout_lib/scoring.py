import math

def match_pct(distance: float) -> float:
    """
    Converts a raw vector distance to a percentage score.
    Assumes standard cosine distance boundaries.
    """
    dist = max(0.0, min(float(distance), 2.0))
    return round((1.0 - (dist / 2.0)) * 100)

def retrieval_rationale(question: str, selected: list[dict], quality: dict = None) -> str:
    """
    Generates a brief structural rationale summary for the retrieved chunks.
    Accepts three arguments to perfectly match the call template inside app.py.
    """
    if not selected:
        return "No context pieces were retrieved."
    return f"Retrieved {len(selected)} unique relevant context chunks from the vector store."

def rag_quality_score(selected: list[dict], generated_text: str = None) -> dict:
    """
    Evaluates the context quality and handles potential generation blockages.
    """
    if generated_text:
        text_lower = generated_text.lower()
        if "rate limit" in text_lower or "429" in text_lower or "error" in text_lower:
            return {
                "score": 0,
                "label": "N/A (API Error)",
                "avg_match": 0.0,
                "unique_sources": 0,
                "separation": 0.0
            }
        
        if "don't have that information" in text_lower:
            return {
                "score": 0,
                "label": "Neutral (No Target Content)",
                "avg_match": 0.0,
                "unique_sources": 0,
                "separation": 0.0
            }

    # Calculate match percentages (values between 0 and 100)
    pcts = [match_pct(c["distance"]) for c in selected]
    
    avg_match = sum(pcts) / len(pcts) if pcts else 0.0
    unique_sources = len({c["source"] for c in selected})
    separation = (pcts[0] - pcts[1]) if len(pcts) > 1 else 0.0
    coverage = min(unique_sources / max(len(selected), 1), 1.0)
    
    # 🎯 FIXED SCORING MATH: Scale metrics up to 100 instead of multiplying by 100 later
    score = 0.7 * avg_match + 20.0 * coverage + 0.1 * min(separation * 2, 100.0)
    score_pct = min(100, max(0, round(score)))
    
    if score_pct >= 75:
        label = "Strong"
    elif score_pct >= 50:
        label = "Adequate"
    else:
        label = "Weak"
        
    return {
        "score": score_pct,
        "label": label,
        "avg_match": avg_match,
        "unique_sources": unique_sources,
        "separation": separation,
    }