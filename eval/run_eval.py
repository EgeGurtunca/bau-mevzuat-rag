"""Retrieval + answer quality per mode. Run: python -m eval.run_eval [--modes dense,bm25,hybrid] [--no-judge]"""
import argparse
import json
import time
from datetime import date

from app import config, llm
from app.answer import answer
from app.retrieve import Retriever

JUDGE = """Soru: {question}
Referans cevap: {reference}
Kaynak metinler:
{sources}
Verilen cevap: {answer}

İki soruya sadece EVET veya HAYIR ile, tam olarak şu biçimde cevap ver:
faithfulness: <EVET|HAYIR>
correctness: <EVET|HAYIR>

faithfulness: verilen cevaptaki her iddia kaynak metinlerle destekleniyor mu?
correctness: verilen cevap referans cevapla özde uyuşuyor mu?"""


def recall_at_k(expected: list[str], got: list[str], k: int = 5) -> float:
    return 1.0 if any(e in got[:k] for e in expected) else 0.0


def mrr(expected: list[str], got: list[str]) -> float:
    for i, g in enumerate(got, start=1):
        if g in expected:
            return 1.0 / i
    return 0.0


def judge(question: str, reference: str, sources: str, ans: str) -> tuple[float, float]:
    out = llm.generate(JUDGE.format(question=question, reference=reference, sources=sources, answer=ans),
                       model=config.JUDGE_MODEL).lower()
    return (1.0 if "faithfulness: evet" in out else 0.0, 1.0 if "correctness: evet" in out else 0.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", default="dense,bm25,hybrid")
    ap.add_argument("--no-judge", action="store_true", help="retrieval metrics only (no answer generation)")
    args = ap.parse_args()
    config.require_api_key()
    with open(config.ROOT / "eval" / "questions.jsonl", encoding="utf-8") as f:
        qs = [json.loads(line) for line in f if line.strip()]
    r = Retriever.load()
    results: dict[str, dict[str, float]] = {}
    for mode in args.modes.split(","):
        agg = {"recall@5": 0.0, "mrr": 0.0, "faithfulness": 0.0, "correctness": 0.0}
        for q in qs:
            hits = r.search(q["question"], mode=mode, k=10)
            ids = [c.id for c, _ in hits]
            agg["recall@5"] += recall_at_k(q["expected_chunk_ids"], ids)
            agg["mrr"] += mrr(q["expected_chunk_ids"], ids)
            if not args.no_judge:
                ans, cites = answer(q["question"], [c for c, _ in hits[:5]])
                sources = "\n".join(f"- {c.text}" for c in cites) or "(yok)"
                faith, corr = judge(q["question"], q["reference_answer"], sources, ans)
                agg["faithfulness"] += faith
                agg["correctness"] += corr
                time.sleep(6)  # free-tier RPM
        results[mode] = {k: round(v / len(qs), 3) for k, v in agg.items()}
        print(f"{mode}: {results[mode]}", flush=True)
    print("\n| mode | Recall@5 | MRR | Faithfulness | Correctness |\n|---|---|---|---|---|")
    for mode, m in results.items():
        print(f"| {mode} | {m['recall@5']} | {m['mrr']} | {m['faithfulness']} | {m['correctness']} |")
    out = config.ROOT / "eval" / "results" / f"{date.today()}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"n": len(qs), "judge": not args.no_judge, "results": results}, indent=2),
                   encoding="utf-8")
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
