"""Retrieval + answer quality per mode, plus abstention on unanswerable questions.

Two phases, so only one LLM is in VRAM at a time:
  1. answer: retrieve for every question, generate every answer with the answer model, write them to a file,
     then unload the answer model.
  2. judge:  read that file and grade every answer with the judge model.
Answers stay on disk, so a different judge (or a fix to the judge prompt) needs only phase 2.

Run: python -m eval.run_eval [--modes dense,bm25,hybrid] [--model NAME] [--no-judge]
     python -m eval.run_eval --phase judge --answers eval/results/<date>-answers.jsonl
"""
import argparse
import json
import time
from datetime import date
from pathlib import Path

from app import config, llm
from app.answer import NOT_FOUND, answer
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


def generate_answers(qs: list[dict], retriever, modes: list[str], model: str | None, with_answers: bool = True) -> list[dict]:
    """Phase 1. One record per (mode, question): retrieval metrics, plus the answer when with_answers."""
    records = []
    for mode in modes:
        for q in qs:
            answerable = bool(q["expected_chunk_ids"])
            hits = retriever.search(q["question"], mode=mode, k=10 if answerable else 5)
            rec = {"mode": mode, "question": q["question"], "answerable": answerable, "answer_model": model,
                   "reference": q.get("reference_answer"), "retrieved": [c.id for c, _ in hits]}
            if answerable:
                rec["recall@5"] = recall_at_k(q["expected_chunk_ids"], rec["retrieved"])
                rec["mrr"] = mrr(q["expected_chunk_ids"], rec["retrieved"])
            if with_answers:
                ans, cites = answer(q["question"], [c for c, _ in hits[:5]], model=model)
                rec["answer"] = ans
                rec["sources"] = [c.text for c in cites]
            records.append(rec)
    return records


def judge_answers(records: list[dict]) -> list[dict]:
    """Phase 2. Grades answerable records in place; unanswerable ones need no judge (exact-string abstention)."""
    for rec in records:
        if rec["answerable"] and "answer" in rec:
            sources = "\n".join(f"- {t}" for t in rec["sources"]) or "(yok)"
            rec["faithfulness"], rec["correctness"] = judge(rec["question"], rec["reference"], sources, rec["answer"])
    return records


def summarize(records: list[dict]) -> dict[str, dict[str, float]]:
    results = {}
    for mode in dict.fromkeys(r["mode"] for r in records):
        ans = [r for r in records if r["mode"] == mode and r["answerable"]]
        una = [r for r in records if r["mode"] == mode and not r["answerable"]]
        mean = lambda key, rs: round(sum(r.get(key, 0.0) for r in rs) / len(rs), 3) if rs else 0.0
        results[mode] = {
            "recall@5": mean("recall@5", ans), "mrr": mean("mrr", ans),
            "faithfulness": mean("faithfulness", ans), "correctness": mean("correctness", ans),
            "abstention": round(sum(r.get("answer") == NOT_FOUND for r in una) / len(una), 3) if una else 0.0,
        }
        for r in una:
            if "answer" in r and r["answer"] != NOT_FOUND:
                print(f"  [{mode}] did not abstain: {r['question']} -> {r['answer']}", flush=True)
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", default="dense,bm25,hybrid", help="add rerank if sentence-transformers is installed")
    ap.add_argument("--model", default=config.ANSWER_MODEL, help="model that writes the answers (phase 1)")
    ap.add_argument("--phase", choices=["all", "answer", "judge"], default="all")
    ap.add_argument("--answers", help="answers file to judge (phase judge); default: today's")
    ap.add_argument("--no-judge", action="store_true", help="retrieval metrics only (no answer generation)")
    args = ap.parse_args()
    results_dir = config.ROOT / "eval" / "results"
    results_dir.mkdir(exist_ok=True)
    answers_path = Path(args.answers) if args.answers else results_dir / f"{date.today()}-answers.jsonl"

    if args.phase in ("all", "answer"):
        llm.require_ollama(args.model)
        with open(config.ROOT / "eval" / "questions.jsonl", encoding="utf-8") as f:
            qs = [json.loads(line) for line in f if line.strip()]
        t0 = time.perf_counter()
        records = generate_answers(qs, Retriever.load(), args.modes.split(","), args.model, with_answers=not args.no_judge)
        llm.unload(args.model)
        llm.unload(config.EMBED_MODEL)
        with open(answers_path, "w", encoding="utf-8") as f:
            f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in records)
        print(f"phase 1 (answers, {args.model}): {len(records)} records in {time.perf_counter() - t0:.0f} s -> {answers_path}", flush=True)
    else:
        with open(answers_path, encoding="utf-8") as f:
            records = [json.loads(line) for line in f if line.strip()]

    if args.phase in ("all", "judge") and not args.no_judge:
        llm.require_ollama(config.JUDGE_MODEL)
        t0 = time.perf_counter()
        records = judge_answers(records)
        llm.unload(config.JUDGE_MODEL)
        with open(answers_path, "w", encoding="utf-8") as f:
            f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in records)
        print(f"phase 2 (judge, {config.JUDGE_MODEL}): {time.perf_counter() - t0:.0f} s", flush=True)

    results = summarize(records)
    print("\n| mode | Recall@5 | MRR | Faithfulness | Correctness | Abstention |\n|---|---|---|---|---|---|")
    for mode, m in results.items():
        print(f"| {mode} | {m['recall@5']} | {m['mrr']} | {m['faithfulness']} | {m['correctness']} | {m['abstention']} |")
    n_ans = sum(r["answerable"] for r in records) // max(len(results), 1)
    out = results_dir / f"{date.today()}.json"
    out.write_text(json.dumps({"n_answerable": n_ans, "n_unanswerable": len(records) // max(len(results), 1) - n_ans,
                               "answer_model": records[0].get("answer_model") if records else None, "judge": None if args.no_judge else config.JUDGE_MODEL,
                               "results": results}, indent=2), encoding="utf-8")
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
