# bau-mevzuat-rag

Turkish RAG question-answering over Bahçeşehir University regulations. Every answer cites the exact
article (`Madde n`) it came from, and three retrieval strategies — dense, BM25, hybrid (RRF) — are
compared on a human-verified eval set.

<!-- demo: docs/demo.gif -->

**Stack:** Python 3.12 · FastAPI · Ollama (`qwen2.5:7b`, `bge-m3`) · Qdrant · rank-bm25 · pytest · Docker.
Fully local — no API keys, no per-token cost. No LangChain — every step is ~50 lines you can read.

## Quickstart

```bash
ollama pull bge-m3 && ollama pull qwen2.5:7b                      # https://ollama.com — ~6 GB total
python -m venv .venv && .venv/Scripts/pip install -e ".[dev]"   # Linux/mac: .venv/bin/pip
python -m app.ingest                                              # data/raw/* -> chunks + Qdrant (local mode)
uvicorn app.api:app --reload                                      # http://localhost:8000
```

Runs on any machine with ~8 GB of GPU/unified memory; on CPU it works but answers take longer.
Models are overridable via `.env` (see `.env.example`) — any Ollama chat/embedding model works.

Drop regulation files (`.pdf`, `.docx`, `.html`, `.txt`) into `data/raw/`. The file name becomes the
document title shown in citations, so name them like `BAU Önlisans ve Lisans Yönetmeliği.pdf`.

### API

```
POST /ask   {"question": "...", "mode": "hybrid" | "dense" | "bm25", "k": 5}
        ->  {"answer": "... [1] ...", "citations": [{doc_title, article_no, heading, text}], "mode", "latency_ms"}
GET  /health
GET  /       chat page
```

### Docker

```bash
docker compose up -d                                   # app + Qdrant; Ollama stays on the host
docker compose run --rm app python -m app.ingest
```

Qdrant's embedded local mode (the default) allows one process at a time — you can't run the API and the eval
together. The Compose setup runs Qdrant as a server, which removes that limit.

## How it works

```
data/raw/*  ──ingest──►  article chunks ("MADDE n")  ──embed──►  Qdrant
                                                    └──────────►  BM25 (in-process)

question ──► dense top-10 ─┐
         └─► BM25  top-10 ─┴─► RRF fusion ─► top-5 ─► prompt ─► Ollama ─► answer + [n] citations
```

- **Chunking:** one chunk per article. Regulations are written as `MADDE 5 – (1) ... (2) ...`, so an article
  is the natural unit of meaning and the natural unit to cite. Articles over 600 words are split at paragraph
  `(n)` boundaries; documents without article structure fall back to 400-word windows with 50-word overlap.
- **Retrieval:** dense = bge-m3 embeddings (1024-d, multilingual, strong on Turkish) in Qdrant; BM25 over 5-character prefixes (a cheap Turkish
  stemmer — `sınavlara`, `sınavın`, `sınav` all become `sınav`); hybrid = Reciprocal Rank Fusion of both lists.
- **Grounding:** the model only sees the retrieved articles, must mark every claim with `[n]`, and must answer
  exactly `Bu konuda yönetmeliklerde bilgi bulamadım.` when the context is insufficient. Citation markers that
  don't map to a retrieved chunk are stripped. If retrieval returns nothing, the LLM is not called at all.
- **Guardrails:** question length 3–500 chars, `k ≤ 10`, retry with backoff on 5xx/connection errors, 503 on backend failure.

## Evaluation

```bash
python -m eval.make_questions      # drafts 40 Q/A pairs from random articles -> eval/questions.draft.jsonl
# review by hand, keep ~30 -> eval/questions.jsonl
python -m eval.run_eval --no-judge # retrieval metrics only
python -m eval.run_eval            # + LLM-judged faithfulness / correctness
```

| mode | Recall@5 | MRR | Faithfulness | Correctness |
|---|---|---|---|---|
| dense | 1.000 | 0.825 | 0.967 | 0.967 |
| bm25 | 0.933 | 0.739 | 0.900 | 0.900 |
| hybrid | 1.000 | **0.828** | 0.967 | **1.000** |

_n = 30 human-verified questions over 100 article chunks; `qwen2.5:7b` answers and judges, `bge-m3` embeds.
Run on 2026-09-14, raw numbers in `eval/results/`._

BM25 alone misses 2/30 (paraphrased questions with no shared stem); dense finds everything but ranks it
lower on average; fusing the two gives the best ranking and the only mode with no incorrect answers.

- **Recall@5 / MRR** — is the article the question was written from in the top 5, and how high.
- **Faithfulness** — every claim in the answer is supported by the cited articles (LLM judge, binary).
- **Correctness** — the answer agrees with the human-verified reference (LLM judge, binary).

Questions are drafted by the model from a random article, then reviewed and edited by a human (9 of 39 drafts
were dropped as meta, garbled or duplicate; 11 were rewritten). A question the model wrote *and* graded would
be circular; the human pass breaks that loop.

**Known limitation:** because each question is generated *from* its target article, it shares that article's
vocabulary, which flatters retrieval — Recall@5 saturates on a 100-chunk corpus. MRR is the more discriminating
number here. A harder set (paraphrased student questions, questions with no answer in the corpus) is on the roadmap.

## Tests

```bash
pytest
```

27 tests, no network: chunking, RRF, BM25, citation parsing, API (LLM and retriever mocked), eval metrics.

## Tracing (optional)

`pip install -e ".[tracing]"` and set `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY`; every `embed` and
`generate` call is traced. Without the keys nothing changes.

## Roadmap

- Harder eval set: paraphrased questions + unanswerable questions (measures abstention)
- Reranker (LLM listwise or cross-encoder) as a fourth mode in the eval table
- Second embedding model (`nomic-embed-text`) as a Turkish-vs-English baseline row
- Multi-turn chat with question rewriting
- Telegram bot front end

## Project layout

```
app/        config, llm (Ollama + retry), chunking, ingest, retrieve, answer, api
static/     single-page chat UI
eval/       question drafting, eval runner, results
tests/      pytest
data/raw/   source regulations (you supply these)
data/index/ chunks.jsonl + Qdrant local storage (generated)
```
