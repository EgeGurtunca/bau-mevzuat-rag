"""Cross-encoder reranking: score (question, article) pairs jointly, keep the best k.

Optional: pip install -e ".[rerank]" (sentence-transformers + torch). The model is loaded on first use.
"""
from app import config
from app.chunking import Chunk

_model = None


def _load():
    global _model
    if _model is None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError:
            raise SystemExit('Reranking needs sentence-transformers: pip install -e ".[rerank]"')
        import torch
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        _model = CrossEncoder(config.RERANK_MODEL, max_length=512, model_kwargs={"torch_dtype": dtype})
    return _model


def rerank(question: str, chunks: list[Chunk], k: int) -> list[tuple[Chunk, float]]:
    if not chunks:
        return []
    scores = _load().predict([(question, c.text) for c in chunks])
    ranked = sorted(zip(chunks, (float(s) for s in scores)), key=lambda x: -x[1])
    return ranked[:k]
