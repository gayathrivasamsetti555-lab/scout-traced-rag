# generator.py
import os
from dotenv import load_dotenv
from google import genai
from google.genai import errors

load_dotenv()
client = genai.Client(api_key=os.getenv('GOOGLE_API_KEY'))

MODEL_NAME = os.getenv('GEMINI_MODEL', 'gemini-2.5-flash')


def generate(question: str, chunks: list[dict]) -> dict:
    """
    chunks: list of {text, source, distance} from retriever.py
    Returns: {answer: str, sources: list[str], error: str | None}
    'error' is None on success, otherwise a short machine-readable tag
    ('server_unavailable', 'rate_limited', 'client_error', 'unknown') so
    the UI can decide how to react (e.g. show a retry button).
    """
    context_parts = []
    for i, c in enumerate(chunks):
        context_parts.append(f"[Source {i+1}: {c['source']}]\n{c['text']}")
    context = '\n\n---\n\n'.join(context_parts)

    prompt = f"""You are a precise research assistant. Answer the question using ONLY
the context provided below. Do not use any outside knowledge.

Rules:
- If the answer is in the context, answer clearly and cite which Source you used.
- If the answer is NOT in the context, say exactly: 'I don't have that information in the loaded documents.'
- Keep answers concise: 2–4 sentences unless more detail is needed.
- Always end with: 'Source: [Source N]' or 'Sources: [Source N, Source M]'

Context:
{context}

Question: {question}"""

    sources = list(set(c['source'] for c in chunks))

    try:
        resp = client.models.generate_content(model=MODEL_NAME, contents=prompt)
        return {'answer': resp.text, 'sources': sources, 'error': None}

    except errors.ServerError:
        return {
            'answer': ("Gemini is temporarily overloaded on Google's side (a 503 — high demand, "
                       "not something wrong with your setup or API key). The passages below were "
                       "still retrieved successfully; wait a few seconds and hit retry."),
            'sources': [], 'error': 'server_unavailable',
        }

    except errors.ClientError as e:
        code = getattr(e, 'code', None)
        if code == 429:
            msg = "Hit Gemini's rate limit (429) — you're sending requests faster than your quota allows. Wait a bit and retry."
            tag = 'rate_limited'
        else:
            msg = f"Gemini rejected the request ({code}): {getattr(e, 'message', str(e))}"
            tag = 'client_error'
        return {'answer': msg, 'sources': [], 'error': tag}

    except errors.APIError as e:
        return {
            'answer': f"Gemini API error ({getattr(e, 'code', '?')}): {getattr(e, 'message', str(e))}",
            'sources': [], 'error': 'unknown',
        }


# ── TEST ────────────────────────────────────────────────────────────
if __name__ == '__main__':
    from retriever import retrieve
    q = 'What is the main advantage of RAG over fine-tuning?'
    chunks = retrieve(q)
    result = generate(q, chunks)
    print(result['answer'])