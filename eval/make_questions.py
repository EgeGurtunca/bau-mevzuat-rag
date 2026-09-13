"""Draft eval questions from random chunks. A human must review the output before it becomes questions.jsonl.

Run: python -m eval.make_questions
"""
import json
import random
import time

from app import config, llm
from app.retrieve import load_chunks

PROMPT = """Aşağıdaki yönetmelik maddesini oku. Bir öğrencinin bu maddeyle ilgili sorabileceği,
maddeyi okumadan cevaplanamayacak TEK bir doğal Türkçe soru ve 1-2 cümlelik doğru cevabını yaz.
Soru maddeyi veya yönetmeliği adıyla anmasın; öğrenci gibi sor.
Sadece şu JSON'u döndür: {{"question": "...", "answer": "..."}}

MADDE:
{text}"""


def main(n: int = 40, seed: int = 7) -> None:
    config.require_api_key()
    chunks = [c for c in load_chunks() if len(c.text.split()) > 30]
    random.Random(seed).shuffle(chunks)
    out = config.ROOT / "eval" / "questions.draft.jsonl"
    written = 0
    with open(out, "w", encoding="utf-8") as f:
        for c in chunks[:n]:
            raw = llm.generate(PROMPT.format(text=c.text), model=config.JUDGE_MODEL)
            raw = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            try:
                qa = json.loads(raw)
            except json.JSONDecodeError:
                continue
            f.write(json.dumps({"question": qa["question"], "expected_chunk_ids": [c.id],
                                "reference_answer": qa["answer"]}, ensure_ascii=False) + "\n")
            written += 1
            time.sleep(4)  # free-tier RPM
    print(f"wrote {written} drafts to {out}; review them and save the good ones as eval/questions.jsonl")


if __name__ == "__main__":
    main()
