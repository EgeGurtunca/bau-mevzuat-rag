"""How much do we trust the judge? Run it on answers I graded by hand (eval/judge_labels.jsonl).

Run: python -m eval.check_judge [--model NAME]
"""
import argparse
import json

from app import config, llm
from eval.run_eval import judge


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=config.JUDGE_MODEL)
    args = ap.parse_args()
    llm.require_ollama(args.model)
    with open(config.ROOT / "eval" / "judge_labels.jsonl", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    agree = fp = fn = 0
    for r in rows:
        if r["answer"].startswith("Bu konuda mevzuatta bilgi bulamadım"):
            got = 0.0  # decided by rule in run_eval, not by the judge
        else:
            sources = "\n".join(f"- {t}" for t in r["sources"]) or "(yok)"
            got = judge(r["question"], r["reference"], sources, r["answer"], model=args.model)[1]
        ok = bool(got) == bool(r["correct"])
        agree += ok
        fp += got and not r["correct"]
        fn += (not got) and r["correct"]
        if not ok:
            print(f"  x judge={int(got)} hand={r['correct']} | {r['question'][:55]} | {r['answer'][:80]}")
    llm.unload(args.model)
    print(f"{args.model}: agreement {agree}/{len(rows)} = {agree / len(rows):.2f} | "
          f"said correct but wrong: {fp} | said wrong but correct: {fn}")


if __name__ == "__main__":
    main()
