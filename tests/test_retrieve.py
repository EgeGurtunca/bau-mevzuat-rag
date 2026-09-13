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
