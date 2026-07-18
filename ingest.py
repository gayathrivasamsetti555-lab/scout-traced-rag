# ingest.py
import re
from pathlib import Path
import chromadb
from chromadb.utils import embedding_functions
from pypdf import PdfReader

DATA_DIR   = 'data'
CHROMA_DIR = 'chroma_db'
COLLECTION = 'my_docs'
CHUNK_SIZE = 800   # target characters per chunk — chunks may run a little
                   # over this since whole paragraphs/sentences are kept
                   # intact rather than cut at a fixed offset
CHUNK_OVER = 100   # overlap carried into the next chunk, for context

PARAGRAPH_SEP = re.compile(r'\n\s*\n')
SENTENCE_SEP = re.compile(r'(?<=[.!?])\s+')
BOILERPLATE_HEADING = re.compile(
    r'\n\s*(references|bibliography|acknowledgments|acknowledgements|'
    r'author contributions)\s*\n',
    re.IGNORECASE,
)
MIN_LETTER_RATIO = 0.5  # below this, a chunk is mostly digits/symbols
                        # (tables, hyperparameter dumps) rather than prose


def load_pdf(path: str) -> str:
    reader = PdfReader(path)
    return '\n'.join(p.extract_text() or '' for p in reader.pages)


def load_txt(path: str) -> str:
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        return f.read()


def find_boilerplate_cutoff(text: str):
    """
    Locates the earliest of References / Bibliography / Acknowledgments /
    Author Contributions headings in the back half of the document, and
    returns its offset so everything from there onward can be excluded.
    Uses the *earliest* match rather than just "References" specifically,
    because papers commonly order these as: ...Conclusion, Acknowledgments,
    References, Appendix — so Acknowledgments needs to be the cutoff, not
    References, or the acknowledgments text still leaks into a chunk.
    Only trusts matches past the 40% mark so a paper that happens to
    discuss "acknowledgments" mid-body doesn't get truncated early.
    Returns None if no heading is found.
    """
    matches = [m for m in BOILERPLATE_HEADING.finditer(text) if m.start() > len(text) * 0.3]
    if not matches:
        return None
    return min(m.start() for m in matches)


def strip_references(text: str) -> str:
    """
    Drops everything from the earliest References/Bibliography/
    Acknowledgments/Author-Contributions heading onward. Without this,
    citation-dense reference lists and acknowledgment name-lists end up as
    high-scoring chunks purely on lexical density, even though they carry
    no answerable content — that's what put a pure citation list and an
    acknowledgments paragraph into the top retrieved results for unrelated
    queries.
    """
    cutoff = find_boilerplate_cutoff(text)
    return text[:cutoff].strip() if cutoff else text


def _pack(pieces: list[str], sep: str, chunk_size: int, overlap: int) -> list[str]:
    """Greedily packs whole pieces (paragraphs, sentences, or words) into
    chunks up to chunk_size, carrying trailing pieces into the next chunk
    for overlap. Pieces are never split internally by this function."""
    chunks, current, current_len = [], [], 0
    for piece in pieces:
        piece_len = len(piece) + (len(sep) if current else 0)
        if current and current_len + piece_len > chunk_size:
            chunks.append(sep.join(current).strip())
            carried, carried_len = [], 0
            for p in reversed(current):
                if carried_len >= overlap:
                    break
                carried.insert(0, p)
                carried_len += len(p) + len(sep)
            current, current_len = carried, carried_len
        current.append(piece)
        current_len += piece_len
    if current:
        chunks.append(sep.join(current).strip())
    return chunks


def _split_long_piece(piece: str, chunk_size: int, overlap: int) -> list[str]:
    """Fallback for a single sentence that's still longer than chunk_size
    on its own — splits on word boundaries, never mid-word."""
    return _pack(piece.split(' '), ' ', chunk_size, overlap)


def _is_low_quality_chunk(text: str) -> bool:
    """
    Flags chunks that are mostly digits/symbols rather than prose — e.g. a
    results table or hyperparameter dump extracted as flat text
    ("0.66±0.01 1.31±0.06 1.32±0.07 BBB (Blundell et al., 2015)...").
    These score well on embedding similarity for numeric/technical queries
    but carry no readable content for the generator to answer from.
    Calibrated against real examples: normal prose sits around 0.80-0.83
    letter density regardless of topic; table dumps collapse to ~0.12.
    """
    if not text:
        return True
    letters = sum(c.isalpha() for c in text)
    return (letters / len(text)) < MIN_LETTER_RATIO


def chunk_text(text: str, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVER) -> list[str]:
    """
    Structure-aware chunking: splits on paragraph breaks first, falling
    back to sentence boundaries and then word boundaries only for
    oversized paragraphs/sentences. This replaces a fixed text[start:end]
    slice, which cuts wherever the character count happens to land —
    including mid-word or mid-tag (e.g. severing "[ISSUP=Supported]" into
    "upported]"). Boundaries here always fall on whitespace.
    """
    text = text.strip()
    if not text:
        return []

    paragraphs = [p.strip() for p in PARAGRAPH_SEP.split(text) if p.strip()]

    pieces = []
    for p in paragraphs:
        if len(p) <= chunk_size:
            pieces.append(p)
            continue
        sentences = [s.strip() for s in SENTENCE_SEP.split(p) if s.strip()]
        for s in sentences:
            if len(s) <= chunk_size:
                pieces.append(s)
            else:
                pieces.extend(_split_long_piece(s, chunk_size, overlap))

    chunks = _pack(pieces, ' ', chunk_size, overlap)
    return [c for c in chunks if len(c) > 50 and not _is_low_quality_chunk(c)]


def main():
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name='all-MiniLM-L6-v2'  # free, downloads ~90MB once
    )
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass  # collection didn't exist yet — fine on a first run

    # cosine space: keeps 'distance' bounded to [0, 1] so downstream
    # quality-score thresholds (in retriever.py) are meaningful. The
    # default L2 space produces distances that don't map cleanly to a
    # 0-1 "how relevant is this" scale.
    col = client.create_collection(
        COLLECTION,
        embedding_function=ef,
        metadata={'hnsw:space': 'cosine'},
    )

    docs, ids, metas = [], [], []
    chunk_id = 0

    for file in Path(DATA_DIR).iterdir():
        if file.suffix.lower() == '.pdf':
            text = load_pdf(str(file))
        elif file.suffix.lower() == '.txt':
            text = load_txt(str(file))
        else:
            continue

        if not text.strip():
            print(f'WARNING: {file.name} produced no extractable text '
                  f'(likely a scanned/image PDF) — skipping. Run OCR first '
                  f'if this file needs to be searchable.')
            continue

        text = strip_references(text)

        print(f'Processing {file.name}: {len(text)} chars (after stripping references)')
        chunks = chunk_text(text)

        for chunk in chunks:
            docs.append(chunk)
            ids.append(f'chunk_{chunk_id}')
            metas.append({'source': file.name, 'chunk': chunk_id})
            chunk_id += 1

    if not docs:
        print('No chunks produced — check DATA_DIR and file contents.')
        return

    # Add in batches of 100 (ChromaDB limit)
    batch = 100
    for i in range(0, len(docs), batch):
        col.add(documents=docs[i:i+batch], ids=ids[i:i+batch], metadatas=metas[i:i+batch])
        print(f'  Indexed chunks {i} to {min(i+batch, len(docs))}')

    print(f'\nDone. Total chunks indexed: {len(docs)}')


if __name__ == '__main__':
    main()