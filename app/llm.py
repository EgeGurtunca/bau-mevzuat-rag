"""Thin Gemini client: embeddings + text generation, with retry on transient errors."""
import time

from google import genai
from google.genai import types

from app import config

try:
    from langfuse import observe
except ImportError:  # tracing is optional (pip install .[tracing])
    def observe(*_a, **_k):
        return lambda f: f

_client: genai.Client | None = None


def client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(api_key=config.GEMINI_API_KEY)
    return _client


def retry(fn, attempts: int = 3):
    """Call fn(); on 429/5xx sleep 2s, 4s and try again. Anything else raises immediately."""
    for i in range(attempts):
        try:
            return fn()
        except Exception as e:
            code = getattr(e, "code", None)
            if code not in (429, 500, 502, 503) or i == attempts - 1:
                raise
            time.sleep(2 ** (i + 1))


@observe()
def embed(texts: list[str], task: str = "RETRIEVAL_DOCUMENT") -> list[list[float]]:
    out: list[list[float]] = []
    for i in range(0, len(texts), 100):  # API batch limit
        batch = texts[i:i + 100]
        res = retry(lambda: client().models.embed_content(
            model=config.EMBED_MODEL,
            contents=batch,
            config=types.EmbedContentConfig(task_type=task, output_dimensionality=config.EMBED_DIM),
        ))
        out.extend(e.values for e in res.embeddings)
    return out


@observe()
def generate(prompt: str, model: str | None = None) -> str:
    res = retry(lambda: client().models.generate_content(
        model=model or config.ANSWER_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(temperature=0),
    ))
    return (res.text or "").strip()
