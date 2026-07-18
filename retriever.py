# retriever.py
import numpy as np
import chromadb
from chromadb.utils import embedding_functions

CHROMA_DIR = 'chroma_db'
COLLECTION = 'my_docs'
EMBED_MODEL = 'all-MiniLM-L6-v2'

_collection = None
_ef = None


def _get_embedding_function():
    global _ef
    if _ef is None:
        _ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
    return _ef


def get_collection():
    """
    Cached collection handle. Creates the collection with cosine distance
    if it doesn't exist yet. NOTE: if 'my_docs' already exists with the
    default L2 space, this will NOT retroactively fix it — Chroma can't
    change an existing collection's distance metric. Delete and re-index
    (see rebuild_index() below) if you're migrating an old collection.
    """
    global _collection
    if _collection is not None:
        return _collection

    ef = _get_embedding_function()
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    existing = [c.name for c in client.list_collections()]
    if COLLECTION in existing:
        col = client.get_collection(COLLECTION, embedding_function=ef)
        space = (col.metadata or {}).get('hnsw:space', 'l2 (default)')
        if space != 'cosine':
            print(
                f"[retriever] WARNING: collection '{COLLECTION}' is using "
                f"distance space '{space}', not cosine. Distances won't be "
                f"bounded to [0, 1] and quality-score thresholds tuned for "
                f"cosine will be miscalibrated. Re-index with rebuild_index() "
                f"to fix this."
            )
    else:
        col = client.create_collection(
            COLLECTION,
            embedding_function=ef,
            metadata={'hnsw:space': 'cosine'},
        )

    _collection = col
    return _collection


def rebuild_index(documents: list[str], metadatas: list[dict], ids: list[str]):
    """
    Drops and recreates the collection with cosine distance, then re-adds
    the given chunks. Run this once when migrating off the default L2 space.
    """
    global _collection
    ef = _get_embedding_function()
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    if COLLECTION in [c.name for c in client.list_collections()]:
        client.delete_collection(COLLECTION)

    col = client.create_collection(
        COLLECTION,
        embedding_function=ef,
        metadata={'hnsw:space': 'cosine'},
    )
    col.add(documents=documents, metadatas=metadatas, ids=ids)
    _collection = col
    return col


def _mmr_select(query_emb, candidates, n_results, lambda_mult=0.5):
    """
    Maximal Marginal Relevance: greedily picks chunks that are close to the
    query but not redundant with chunks already selected. Prevents
    near-duplicate passages (e.g. two chunks from the same source covering
    the same content) from filling all the result slots.
    """
    if not candidates:
        return []

    query_emb = np.array(query_emb)
    cand_embs = np.array([c['embedding'] for c in candidates])

    def cos_sim(a, b):
        denom = (np.linalg.norm(a) * np.linalg.norm(b)) or 1e-8
        return float(np.dot(a, b) / denom)

    relevance = [cos_sim(query_emb, e) for e in cand_embs]

    selected_idx = []
    remaining_idx = list(range(len(candidates)))

    while remaining_idx and len(selected_idx) < n_results:
        if not selected_idx:
            best = max(remaining_idx, key=lambda i: relevance[i])
        else:
            def mmr_score(i):
                redundancy = max(
                    cos_sim(cand_embs[i], cand_embs[j]) for j in selected_idx
                )
                return lambda_mult * relevance[i] - (1 - lambda_mult) * redundancy

            best = max(remaining_idx, key=mmr_score)

        selected_idx.append(best)
        remaining_idx.remove(best)

    return [candidates[i] for i in selected_idx]


MAX_DISTANCE = 0.55
# Cosine distance ceiling — candidates farther than this from the query
# are treated as "not actually relevant" and dropped before selection,
# rather than being force-included just to fill n_results. Without this,
# a query with no good match in the corpus still returns n_results
# chunks (all weakly related at best), MMR just diversifies *among* the
# irrelevant options instead of recognizing there's nothing good to pick.
# This value is a starting point, not a universal constant — inspect the
# `distance` values your corpus actually produces for queries you know
# ARE answerable vs. ones that AREN'T, and set the cutoff between those
# two clusters rather than trusting this default blindly.


def retrieve(question: str, n_results: int = 5, fetch_k: int = 20,
             max_distance: float = MAX_DISTANCE) -> list[dict]:
    """
    Returns list of dicts: [{text, source, distance}]
    Lower distance = more relevant (requires the collection to use cosine
    space — see get_collection()). May return fewer than n_results (even
    zero) if nothing in the corpus clears max_distance — that's
    intentional: it's a signal to the caller that retrieval found nothing
    usable, rather than silently handing over weak matches.

    Over-fetches `fetch_k` candidates, filters out anything farther than
    max_distance, then applies MMR so the final results are relevant AND
    non-redundant, instead of just returning the raw top-k by distance
    (which tends to cluster near-duplicates from the same source at the
    top).
    """
    col = get_collection()
    ef = _get_embedding_function()
    query_emb = ef([question])[0]

    results = col.query(
        query_texts=[question],
        n_results=min(fetch_k, col.count()),
        include=['documents', 'metadatas', 'distances', 'embeddings'],
    )

    candidates = []
    for i, doc in enumerate(results['documents'][0]):
        distance = round(results['distances'][0][i], 3)
        if distance > max_distance:
            continue
        candidates.append({
            'text': doc,
            'source': results['metadatas'][0][i]['source'],
            'distance': distance,
            'embedding': results['embeddings'][0][i],
        })

    selected = _mmr_select(query_emb, candidates, n_results)

    # strip the embedding before returning — callers don't need it
    return [
        {'text': c['text'], 'source': c['source'], 'distance': c['distance']}
        for c in selected
    ]


# ── TEST ────────────────────────────────────────────────────────────
if __name__ == '__main__':
    test_questions = [
        'What is retrieval augmented generation?',
        'How do LLM agents work?',
        'What are the limitations of fine-tuning?'
    ]
    for q in test_questions:
        print(f'\nQ: {q}')
        for r in retrieve(q):
            print(f"  [{r['distance']}] {r['source']}: {r['text'][:100]}...")