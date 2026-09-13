from app import answer as answer_mod
from app.answer import NOT_FOUND, answer, build_prompt, parse_citations
from app.chunking import Chunk

C = [Chunk(id=f"d:{i}", doc_title="Doc", article_no=i, text=f"madde {i}") for i in (1, 2, 3)]


def test_parse_citations_keeps_valid_drops_invalid():
    text, cites = parse_citations("Cevap [1] ve [3] ve [9].", C)
    assert [c.id for c in cites] == ["d:1", "d:3"]
    assert "[9]" not in text and "[1]" in text


def test_parse_citations_dedupes():
    _, cites = parse_citations("[2] a [2] b", C)
    assert [c.id for c in cites] == ["d:2"]


def test_prompt_numbers_chunks():
    p = build_prompt("soru?", C)
    assert "[2] Doc, Madde 2: madde 2" in p
    assert NOT_FOUND in p
    assert p.rstrip().endswith("CEVAP:")


def test_answer_without_chunks_skips_llm(monkeypatch):
    def boom(*_a, **_k):
        raise AssertionError("llm called")

    monkeypatch.setattr(answer_mod.llm, "generate", boom)
    assert answer("soru?", []) == (NOT_FOUND, [])


def test_answer_maps_llm_output(monkeypatch):
    monkeypatch.setattr(answer_mod.llm, "generate", lambda *_a, **_k: "Madde ikiye göre [2].")
    text, cites = answer("soru?", C)
    assert text == "Madde ikiye göre [2]."
    assert [c.id for c in cites] == ["d:2"]
