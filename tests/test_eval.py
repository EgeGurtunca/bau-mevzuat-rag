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
