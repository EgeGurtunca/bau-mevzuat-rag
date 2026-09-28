from app import answer as answer_mod
from app.answer import NOT_FOUND, answer, build_prompt, parse_citations
from app.chunking import Chunk

C = [Chunk(id=f"d:{i}", doc_title="Doc", article_no=i, text=f"madde {i}") for i in (1, 2, 3)]


def test_parse_citations_renumbers_and_drops_invalid():
    text, cites = parse_citations("Cevap [3] ve [1] ve [9] ve [3].", C)
    assert [c.id for c in cites] == ["d:3", "d:1"]
    assert text == "Cevap [1] ve [2] ve  ve [1]."


def test_parse_citations_dedupes():
    _, cites = parse_citations("[2] a [2] b", C)
    assert [c.id for c in cites] == ["d:2"]


def test_prompt_numbers_chunks():
    p = build_prompt("soru?", C)
    assert "[2] Doc, Madde 2: madde 2" in p
    assert NOT_FOUND in p
    assert p.index("[2] Doc") < p.index("SORU: soru?") < p.index("ALINTI:")  # instructions after the context


def test_answer_without_chunks_skips_llm(monkeypatch):
    def boom(*_a, **_k):
        raise AssertionError("llm called")

    monkeypatch.setattr(answer_mod.llm, "generate", boom)
    assert answer("soru?", []) == (NOT_FOUND, [])


def test_answer_maps_llm_output(monkeypatch):
    monkeypatch.setattr(answer_mod.llm, "generate", lambda *_a, **_k: "Madde ikiye göre [2].")
    text, cites = answer("soru?", C)
    assert text == "Madde ikiye göre [1]."  # renumbered to match the citation list
    assert [c.id for c in cites] == ["d:2"]


def test_parse_citations_handles_comma_lists():
    text, cites = parse_citations("Ücret ödenir [2, 3]. Süre bir yıldır [7, 7].", C)
    assert text == "Ücret ödenir [1][2]. Süre bir yıldır ."
    assert [c.id for c in cites] == ["d:2", "d:3"]


def test_only_the_answer_after_the_quote_is_returned(monkeypatch):
    out = "ALINTI: Süre iki yarıyıldır [3].\n**CEVAP:** En fazla iki yarıyıl [3]."
    monkeypatch.setattr(answer_mod.llm, "generate", lambda *_a, **_k: out)
    text, cites = answer("soru?", C)
    assert text == "En fazla iki yarıyıl [1]." and [c.id for c in cites] == ["d:3"]
    monkeypatch.setattr(answer_mod.llm, "generate", lambda *_a, **_k: f'CEVAP: "{NOT_FOUND}"')
    assert answer("soru?", C) == (NOT_FOUND, [])


def test_sources_come_from_the_quote_when_the_answer_cites_nothing(monkeypatch):
    out = "ALINTI: Ücret alınmaz [2].\nCEVAP: Ücret ödemezsiniz."
    monkeypatch.setattr(answer_mod.llm, "generate", lambda *_a, **_k: out)
    text, cites = answer("soru?", C)
    assert text == "Ücret ödemezsiniz." and [c.id for c in cites] == ["d:2"]


def test_an_answer_that_only_says_the_documents_dont_say_becomes_not_found(monkeypatch):
    def fake(prompt, **_k):
        if "SOMUT ya da YOK" in prompt:  # the second call judges the finished answer
            return "**YOK**" if "belirtilmemiştir" in prompt else "SOMUT"
        return fake.out
    monkeypatch.setattr(answer_mod.llm, "generate", fake)
    fake.out = "ALINTI: x [1].\nCEVAP: Kesin bir tarih belirtilmemiştir [1]."
    assert answer("soru?", C) == (NOT_FOUND, [])
    fake.out = "ALINTI: x [1].\nCEVAP: Evet, iki yarıyıl [1]."
    assert answer("soru?", C)[0] == "Evet, iki yarıyıl [1]."


def test_a_cut_off_answer_is_not_shown(monkeypatch):
    out = 'ALINTI: "Öğrenciler takip etmekle yükümlüdür." [1] "Öğrenciler takip etmekle yükümlüdür." [1] "Öğ'
    monkeypatch.setattr(answer_mod.llm, "generate", lambda *_a, **_k: out)
    assert answer("soru?", C) == (NOT_FOUND, [])
