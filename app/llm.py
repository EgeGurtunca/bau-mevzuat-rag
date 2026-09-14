"""Thin Ollama client (stdlib only): embeddings + text generation, with retry on transient errors."""
import json
import time
import urllib.error
import urllib.request

from app import config

try:
    from langfuse import observe
except ImportError:  # tracing is optional (pip install .[tracing])
    def observe(*_a, **_k):
        return lambda f: f


def _post(path: str, body: dict, timeout: int = 300) -> dict:
    req = urllib.request.Request(
        config.OLLAMA_URL + path,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def retry(fn, attempts: int = 3):
    """Call fn(); on 5xx / connection errors sleep 2s, 4s and try again. Anything else raises immediately."""
    for i in range(attempts):
        try:
            return fn()
        except urllib.error.HTTPError as e:
            if e.code < 500 or i == attempts - 1:
                raise
            time.sleep(2 ** (i + 1))
        except urllib.error.URLError:
            if i == attempts - 1:
                raise
            time.sleep(2 ** (i + 1))


def require_ollama() -> None:
    """Fail fast with a useful message if Ollama is down or a model is missing."""
    try:
        with urllib.request.urlopen(config.OLLAMA_URL + "/api/tags", timeout=5) as r:
            have = {m["name"] for m in json.load(r)["models"]}
    except urllib.error.URLError:
        raise SystemExit(f"Ollama is not reachable at {config.OLLAMA_URL}. Install it from https://ollama.com and start it.")
    have |= {n.removesuffix(":latest") for n in have}
    for m in {config.EMBED_MODEL, config.ANSWER_MODEL, config.JUDGE_MODEL}:
        if m not in have and m.removesuffix(":latest") not in have:
            raise SystemExit(f"Model '{m}' is not pulled. Run: ollama pull {m}")


@observe()
def embed(texts: list[str], task: str = "RETRIEVAL_DOCUMENT") -> list[list[float]]:
    # `task` is kept for API symmetry; bge-m3 uses the same embedding for queries and documents.
    out: list[list[float]] = []
    for i in range(0, len(texts), 32):
        res = retry(lambda: _post("/api/embed", {"model": config.EMBED_MODEL, "input": texts[i:i + 32], "keep_alive": "30m"}))
        out.extend(res["embeddings"])
    return out


@observe()
def generate(prompt: str, model: str | None = None) -> str:
    res = retry(lambda: _post("/api/generate", {
        "model": model or config.ANSWER_MODEL,
        "prompt": prompt,
        "stream": False,
        "keep_alive": "30m",  # avoid the ~2 s reload (50 s cold) between questions
        "options": {"temperature": 0, "num_ctx": 8192, "num_predict": 300},
    }))
    return res["response"].strip()
