# bau-mevzuat-rag

Ask a question about Bahçeşehir University's regulations and get an answer that points at the exact article
it came from. Runs entirely on my laptop — no API keys, no cloud.

<!-- docs/demo.gif -->

I built this because I kept getting the same questions from friends ("can I freeze my registration?",
"what happens if I miss the final?") and the answers are all in two long documents nobody reads. It's also
the second project in a series where I'm working through the LLM stack one layer at a time — this one is
the retrieval layer. I wanted to actually understand chunking, hybrid search and evaluation rather than call
a framework, so there's no LangChain here; every step is a short file I wrote and can explain.

**Stack:** Python 3.12 · FastAPI · Ollama (`qwen2.5:7b` for answers, `bge-m3` for embeddings) · Qdrant ·
rank-bm25 · pytest · Docker

## Running it

```bash
ollama pull bge-m3 && ollama pull qwen2.5:7b                      # https://ollama.com — about 6 GB
python -m venv .venv && .venv/Scripts/pip install -e ".[dev]"   # Linux/mac: .venv/bin/pip
python -m app.ingest                                              # data/raw/* -> chunks + Qdrant
uvicorn app.api:app --reload                                      # http://localhost:8000
```

Anything with ~8 GB of GPU or unified memory is fine. It works on CPU too, just slower. Models can be swapped
through `.env` (see `.env.example`) — any Ollama chat/embedding model works.

The two regulations in `data/raw/` are the raw HTML from [mevzuat.gov.tr](https://www.mevzuat.gov.tr)
(nos. 33950 and 42316). Drop any other `.pdf`, `.docx`, `.html` or `.txt` in there and re-run ingest; the
file name becomes the document title in citations.

### API

```
POST /ask   {"question": "...", "mode": "hybrid" | "dense" | "bm25", "k": 5}
        ->  {"answer": "... [1] ...", "citations": [{doc_title, article_no, heading, text}], "mode", "latency_ms"}
GET  /health
GET  /       the chat page
```

### Docker

```bash
docker compose up -d                                   # app + Qdrant; Ollama stays on the host
docker compose run --rm app python -m app.ingest
```

Qdrant's embedded mode (the default) only allows one process at a time, so you can't run the API and the
eval together. Compose runs Qdrant as a proper server and removes that limit.

## How it works

```
data/raw/*  ──ingest──►  article chunks ("MADDE n")  ──embed──►  Qdrant
                                                    └──────────►  BM25 (in-process)

question ──► dense top-10 ─┐
         └─► BM25  top-10 ─┴─► RRF fusion ─► top-5 ─► prompt ─► Ollama ─► answer + [n] citations
```

**Chunking.** Turkish regulations are written as `MADDE 5 – (1) ... (2) ...`, so one article is one chunk.
That's the natural unit of meaning and the natural unit to cite. Articles over 600 words get split at
paragraph `(n)` boundaries; anything without article structure falls back to 400-word windows with overlap.

**Retrieval.** Three modes so I could compare them: dense (bge-m3, 1024-d, in Qdrant), BM25 over 5-character
prefixes (a cheap Turkish stemmer — `sınavlara`, `sınavın`, `sınav` all become `sınav`), and hybrid, which
fuses the two with Reciprocal Rank Fusion. Hybrid is the default, with one rule on top of plain RRF: each
retriever's #1 result is guaranteed a slot in the top-k. I added that after "onur öğrencisi" (honour student)
— BM25 put the one article containing "onur" at rank 1, dense didn't have it in its top 10 at all, and RRF
averaged it out to rank 7, so the model never saw it and confidently invented a GPA threshold. With the
rule it's in the top 5 and the model correctly says the regulation doesn't specify one.

**Grounding.** The model only sees the retrieved articles, has to tag each claim with `[n]`, and has to answer
exactly `Bu konuda yönetmeliklerde bilgi bulamadım.` when the context doesn't cover the question. Citation
markers that don't map to a retrieved chunk get stripped, and the rest are renumbered so the answer and the
citation list always agree. If retrieval comes back empty, the model isn't called at all.

**Guardrails.** Question length 3–500 chars, `k ≤ 10`, retry with backoff on 5xx/connection errors, 503
instead of a stack trace when the backend is down.

**Latency.** About 1 s per question on an RTX 4090 laptop (0.1 s retrieval, the rest is the 7B model). My
first version took 9.7 s and I assumed that was just local inference. Measuring showed otherwise: every Ollama
call had a fixed ~2 s cost, and it turned out to be `localhost` resolving to `::1` first on Windows while
Ollama only listens on IPv4 — two calls per question, four seconds of nothing. `127.0.0.1` fixed it. The
other half was the model writing 400-token essays; capping the prompt at "3–4 sentences" brought generation
to ~50 tokens. Both were invisible until I timed each stage separately. A side effect I didn't expect: the
shorter answers also scored higher on faithfulness (0.933 → 0.967) — less room to drift from the source.

## Evaluation

I didn't want to ship a chatbot that "seems fine", so there's an eval harness:

```bash
python -m eval.make_questions      # drafts 40 Q/A pairs from random articles -> eval/questions.draft.jsonl
# I review these by hand and keep the good ones -> eval/questions.jsonl
python -m eval.run_eval --no-judge # retrieval metrics only
python -m eval.run_eval            # + LLM-judged faithfulness / correctness
```

| mode | Recall@5 | MRR | Faithfulness | Correctness |
|---|---|---|---|---|
| dense | 1.000 | **0.864** | 0.967 | 0.967 |
| bm25 | 0.933 | 0.756 | 1.000 | 1.000 |
| hybrid | 1.000 | 0.861 | 0.967 | 0.967 |

_30 human-reviewed questions over 100 article chunks. `qwen2.5:7b` answers and judges, `bge-m3` embeds.
Raw numbers per run are in `eval/results/`._

- **Recall@5 / MRR** — is the article the question was written from in the top 5, and how high up.
- **Faithfulness** — every claim in the answer is backed by a cited article (LLM judge, yes/no).
- **Correctness** — the answer matches the reference I checked by hand (LLM judge, yes/no).

What I take from the table: BM25 alone misses 2 of 30 questions (paraphrases with no shared stem), so
dense and hybrid clearly win on retrieval. The judge columns differ by a single question (1/30 = 0.033),
which is noise at this sample size — once the right article is in the top 5, answer quality is the same
across modes. I keep hybrid as the default because it has dense's recall and stays robust on exact-term
queries (article numbers, grade letters) where BM25 is strongest.

**Two honest caveats.** The questions are generated *from* their target article, so they share its
vocabulary, which flatters retrieval — Recall@5 saturating on a 100-chunk corpus says more about the test
set than the system. MRR is the number I actually watch. And 30 questions is small; the judge columns move
by one question between runs.

The most useful thing the eval did was catch chunking bugs. The regulations come as Word-exported HTML that
wraps lines *inside* `<span>`s, so a naive HTML-to-text split headings like "Dersten çekilme" over two lines
and leaked the next article's heading into the previous chunk. Section headers ("BEŞİNCİ BÖLÜM …") leaked the
same way. Each cleanup moved dense MRR: 0.825 → 0.847 → 0.864. Chunk quality *is* retrieval quality.

## Tests

```bash
pytest
```

31 tests, no network: chunking edge cases, RRF and the top-hit guarantee, BM25, citation parsing and
renumbering, the API with the LLM and retriever mocked, eval metrics.

## What's next

- A harder eval set: paraphrased questions, and questions with no answer in the corpus (to measure abstention)
- A reranker as a fourth mode in the table
- `nomic-embed-text` as a second embedding row, to show the Turkish-vs-English gap with numbers
- Multi-turn chat with question rewriting
- Telegram front end so actual students use it

## Layout

```
app/        config, llm (Ollama + retry), chunking, ingest, retrieve, answer, api
static/     the chat page, plain HTML/JS
eval/       question drafting, eval runner, results
tests/      pytest
data/raw/   source regulations
data/index/ chunks.jsonl + Qdrant storage (generated, gitignored)
```
