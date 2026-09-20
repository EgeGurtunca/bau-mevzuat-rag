"""Dense (Qdrant), sparse (BM25) and hybrid (RRF) retrieval over the chunk index."""
import atexit
import re
import uuid

from qdrant_client import QdrantClient
from rank_bm25 import BM25Okapi

from app import config, llm
from app.chunking import Chunk

DENSE_MIN_SCORE = 0.3  # cosine; below this the hit is noise and we skip the LLM
WORD_RE = re.compile(r"\w+")


def tokenize(text: str) -> list[str]:
    # ponytail: 5-char prefix as a Turkish stemmer; swap for a real stemmer if BM25 recall is the bottleneck
    return [w[:5] for w in WORD_RE.findall(text.lower())]


def rrf(rankings: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion: score(id) = sum over rankings of 1 / (k + rank)."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, cid in enumerate(ranking, start=1):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: -x[1])


def point_id(chunk_id: str) -> str:
    """Qdrant wants int/UUID ids; derive a stable UUID from the chunk id."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


def open_qdrant() -> QdrantClient:
    client = QdrantClient(url=config.QDRANT_URL) if config.QDRANT_URL else QdrantClient(path=config.QDRANT_PATH)
    atexit.register(client.close)  # local mode holds a file lock; release it before interpreter teardown
    return client


def load_chunks() -> list[Chunk]:
    with open(config.CHUNKS_PATH, encoding="utf-8") as f:
        return [Chunk.model_validate_json(line) for line in f if line.strip()]


class Retriever:
    def __init__(self, chunks: list[Chunk], qdrant: QdrantClient | None):
        self.chunks = chunks
        self.by_id = {c.id: c for c in chunks}
        self.qdrant = qdrant
        self.bm25 = BM25Okapi([tokenize(c.text) for c in chunks]) if chunks else None

    @classmethod
    def load(cls) -> "Retriever":
        if not config.CHUNKS_PATH.exists():
            raise SystemExit("Index not found. Run: python -m app.ingest")
        return cls(load_chunks(), open_qdrant())

    def dense(self, question: str, n: int = 10) -> list[tuple[Chunk, float]]:
        vec = llm.embed([question], task="RETRIEVAL_QUERY")[0]
        points = self.qdrant.query_points(config.COLLECTION, query=vec, limit=n).points
        return [(self.by_id[p.payload["id"]], p.score) for p in points
                if p.score >= DENSE_MIN_SCORE and p.payload["id"] in self.by_id]

    def sparse(self, question: str, n: int = 10) -> list[tuple[Chunk, float]]:
        scores = self.bm25.get_scores(tokenize(question))
        ranked = sorted(range(len(scores)), key=lambda i: -scores[i])[:n]
        return [(self.chunks[i], float(scores[i])) for i in ranked if scores[i] > 0]

    def search(self, question: str, mode: str = "hybrid", k: int = 5) -> list[tuple[Chunk, float]]:
        if mode == "dense":
            return self.dense(question)[:k]
        if mode == "bm25":
            return self.sparse(question)[:k]
        if mode == "rerank":
            # wider net first (both retrievers, 10 each), then a cross-encoder reads question + article together
            from app.rerank import rerank
            candidates = {c.id: c for c, _ in self.dense(question, n=10) + self.sparse(question, n=10)}
            return rerank(question, list(candidates.values()), k)
        d, s = self.dense(question), self.sparse(question)
        fused = rrf([[c.id for c, _ in d], [c.id for c, _ in s]])
        top = [cid for cid, _ in fused[:k]]
        # Each retriever's #1 is guaranteed a seat: a rare exact term (BM25 rank 1) must not be
        # outvoted by dense consensus and pushed past k. Observed with "onur öğrencisi" -> Madde 34.
        for must in (d[:1] + s[:1]):
            if must[0].id not in top:
                top[-1] = must[0].id
        scores = dict(fused)
        return [(self.by_id[cid], scores[cid]) for cid in top]
