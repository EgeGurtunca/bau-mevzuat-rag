from fastapi.testclient import TestClient

from app import api
from app.chunking import Chunk


class FakeRetriever:
    chunks: list = []

    def search(self, q, mode="hybrid", k=5):
        return [(Chunk(id="d:5", doc_title="Doc", article_no=5, heading="Notlar", text="Gecme notu 60 puandir."), 0.9)]


class BrokenRetriever(FakeRetriever):
    def search(self, q, mode="hybrid", k=5):
        raise RuntimeError("qdrant down")


def test_ask_returns_answer_and_citations(monkeypatch):
    api.retriever = FakeRetriever()
    monkeypatch.setattr(api.llm, "generate", lambda *_a, **_k: "Gecme notu 60 puandir [1].")
    r = TestClient(api.app).post("/ask", json={"question": "gecme notu kac?"})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"].endswith("[1].")
    assert body["citations"][0]["article_no"] == 5
    assert body["citations"][0]["heading"] == "Notlar"
    assert body["mode"] == "hybrid"
    assert body["latency_ms"] >= 0


def test_ask_rejects_bad_mode():
    api.retriever = FakeRetriever()
    r = TestClient(api.app).post("/ask", json={"question": "soru soru", "mode": "magic"})
    assert r.status_code == 422


def test_ask_returns_503_on_backend_failure():
    api.retriever = BrokenRetriever()
    r = TestClient(api.app).post("/ask", json={"question": "soru soru"})
    assert r.status_code == 503


def test_index_page():
    r = TestClient(api.app).get("/")
    assert r.status_code == 200 and "BAU" in r.text
