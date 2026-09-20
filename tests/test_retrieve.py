from app.chunking import Chunk
from app.retrieve import Retriever, rrf, tokenize

CHUNKS = [
    Chunk(id="d:1", doc_title="D", article_no=1, text="Yıllık izin süresi yirmi gündür."),
    Chunk(id="d:2", doc_title="D", article_no=2, text="Sınavlar dönem sonunda yapılır."),
    Chunk(id="d:3", doc_title="D", article_no=3, text="Kayıt yenileme her dönem başında zorunludur."),
]


def test_rrf_order():
    fused = rrf([["a", "b", "c"], ["b", "c"]])
    assert [i for i, _ in fused] == ["b", "c", "a"]


def test_tokenize_turkish_prefix():
    assert tokenize("Sınavlara girmeyen öğrenciler") == ["sınav", "girme", "öğren"]


def test_bm25_finds_matching_chunk():
    top = Retriever(CHUNKS, qdrant=None).search("sınav ne zaman yapılır", mode="bm25", k=1)
    assert top[0][0].id == "d:2"


def test_bm25_no_match_returns_empty():
    assert Retriever(CHUNKS, qdrant=None).search("kütüphane", mode="bm25", k=1) == []


def test_hybrid_fuses_dense_and_sparse(monkeypatch):
    r = Retriever(CHUNKS, qdrant=None)
    monkeypatch.setattr(r, "dense", lambda q, n=10: [(CHUNKS[2], 0.9), (CHUNKS[1], 0.5)])
    ids = [c.id for c, _ in r.search("sınav ne zaman yapılır", mode="hybrid", k=3)]
    assert ids[0] == "d:2"  # ranked by both -> first
    assert set(ids) == {"d:2", "d:3"}


def test_hybrid_keeps_each_retrievers_top_hit(monkeypatch):
    many = [Chunk(id=f"d:{i}", doc_title="D", text=f"metin {i}") for i in range(10)]
    r = Retriever(many, qdrant=None)
    # dense ranks 0..7; sparse's #1 is 9, which dense never returns -> RRF alone drops it past k
    monkeypatch.setattr(r, "dense", lambda q, n=10: [(many[i], 1.0) for i in range(8)])
    monkeypatch.setattr(r, "sparse", lambda q, n=10: [(many[9], 5.0), (many[0], 1.0), (many[1], 1.0)])
    ids = [c.id for c, _ in r.search("x", mode="hybrid", k=5)]
    assert "d:9" in ids and len(ids) == 5 and ids[0] == "d:0"


def test_rerank_mode_scores_union_of_candidates(monkeypatch):
    from app import rerank as rerank_mod
    many = [Chunk(id=f"d:{i}", doc_title="D", text=f"metin {i}") for i in range(6)]
    r = Retriever(many, qdrant=None)
    monkeypatch.setattr(r, "dense", lambda q, n=10: [(many[0], 1.0), (many[1], 0.9)])
    monkeypatch.setattr(r, "sparse", lambda q, n=10: [(many[1], 5.0), (many[5], 1.0)])

    class FakeModel:
        def predict(self, pairs):
            return [len(text) for _, text in pairs]  # "metin 5" and "metin 1" tie on length -> stable order

    monkeypatch.setattr(rerank_mod, "_load", lambda: FakeModel())
    hits = r.search("x", mode="rerank", k=2)
    assert {c.id for c, _ in hits} <= {"d:0", "d:1", "d:5"} and len(hits) == 2
    assert all(isinstance(s, float) for _, s in hits)
