from eval import run_eval
from eval.run_eval import judge, mrr, recall_at_k


def test_recall_and_mrr():
    assert recall_at_k(["a"], ["x", "a", "y"], k=5) == 1.0
    assert recall_at_k(["a"], ["x", "y"], k=5) == 0.0
    assert recall_at_k(["a"], ["x", "y", "z", "w", "v", "a"], k=5) == 0.0
    assert mrr(["a"], ["x", "a", "y"]) == 0.5
    assert mrr(["a"], []) == 0.0


def test_judge_parses_verdicts(monkeypatch):
    monkeypatch.setattr(run_eval.llm, "generate",
                        lambda *_a, **_k: "faithfulness: EVET\ncorrectness: HAYIR")
    assert judge("s", "r", "src", "a") == (1.0, 0.0)


from app import llm
from app.answer import NOT_FOUND
from app.chunking import Chunk


class FakeRetriever:
    def search(self, q, mode="hybrid", k=5):
        return [(Chunk(id="d:1", doc_title="D", article_no=1, text="Madde bir."), 1.0)]


QS = [{"question": "q1?", "expected_chunk_ids": ["d:1"], "reference_answer": "bir"},
      {"question": "q2?", "expected_chunk_ids": [], "reference_answer": None}]


def test_two_phases_never_mix_models(monkeypatch):
    calls = []

    def fake_generate(prompt, model=None):
        calls.append(model)
        if model == "judge-m":
            return "faithfulness: EVET\ncorrectness: HAYIR"
        return "Madde bir [1]." if "q1?" in prompt else NOT_FOUND

    monkeypatch.setattr(llm, "generate", fake_generate)
    monkeypatch.setattr(run_eval.config, "JUDGE_MODEL", "judge-m")
    records = run_eval.generate_answers(QS, FakeRetriever(), ["hybrid"], model="answer-m")
    assert set(calls) == {"answer-m"}          # phase 1: answer model only
    calls.clear()
    run_eval.judge_answers(records)
    assert calls == ["judge-m"]                # phase 2: judge only, and only for the answerable question
    assert run_eval.summarize(records)["hybrid"] == {
        "recall@5": 1.0, "mrr": 1.0, "faithfulness": 1.0, "correctness": 0.0, "abstention": 1.0}


def test_no_judge_makes_no_model_calls(monkeypatch):
    def boom(*_a, **_k):
        raise AssertionError("model called")
    monkeypatch.setattr(llm, "generate", boom)
    records = run_eval.generate_answers(QS, FakeRetriever(), ["bm25"], model="m", with_answers=False)
    assert all("answer" not in r for r in records) and records[0]["mrr"] == 1.0


def test_complete_at_k_needs_every_expected_article():
    assert run_eval.complete_at_k(["a", "b"], ["a", "x", "b"]) == 1.0
    assert run_eval.complete_at_k(["a", "b"], ["a", "x", "y"]) == 0.0


def test_by_tag_splits_answerable_and_traps():
    recs = [
        {"mode": "hybrid", "answerable": True, "tags": ["multi"], "recall@5": 1.0, "complete@5": 0.0, "correctness": 1.0},
        {"mode": "hybrid", "answerable": False, "tags": ["trap"], "answer": NOT_FOUND},
        {"mode": "hybrid", "answerable": False, "tags": ["trap"], "answer": "uydurma"},
    ]
    t = run_eval.by_tag(recs)
    assert t["hybrid / multi"] == {"n": 1, "recall@5": 1.0, "complete@5": 0.0, "correctness": 1.0}
    assert t["hybrid / trap"] == {"n": 2, "abstention": 0.5}


def test_not_found_on_answerable_question_is_wrong_without_asking_the_judge(monkeypatch):
    def boom(*_a, **_k):
        raise AssertionError("judge called")
    monkeypatch.setattr(llm, "generate", boom)
    recs = [{"mode": "m", "answerable": True, "answer": NOT_FOUND, "sources": [], "question": "q", "reference": "r"}]
    run_eval.judge_answers(recs)
    assert (recs[0]["faithfulness"], recs[0]["correctness"]) == (1.0, 0.0)
