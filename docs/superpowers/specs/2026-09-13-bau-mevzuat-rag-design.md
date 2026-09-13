# bau-mevzuat-rag — Design

Date: 2026-09-13
Status: approved (brainstorm)

## Goal

A Turkish RAG question-answering service over Bahçeşehir University regulations
(yönetmelik/yönerge). Answers cite the exact article (madde). The project exists
to demonstrate retrieval engineering: three retrieval modes (dense, BM25, hybrid)
compared on a human-verified eval set, with faithfulness/correctness judged.

Second project in a five-project AI progression (API → retrieval → agent →
system → own model). Code and README in English; UI and corpus in Turkish.

## Decisions

| Decision | Choice | Why |
|---|---|---|
| Corpus | BAU regulations, files dropped into `data/raw/` | Public, student-relevant, `MADDE n` structure gives natural chunks and citations |
| LLM | Gemini `gemini-2.5-flash` (free tier) | Zero cost, good Turkish |
| Embeddings | Gemini `gemini-embedding-001` | Same client, no local model download |
| Vector store | Qdrant — local mode in dev (`path=`), service in Compose (`url=`) | Same API both ways; Qdrant is the keyword employers list |
| Sparse retrieval | `rank_bm25` in-process over `chunks.jsonl` | Corpus is ~1k chunks; no extra service |
| Fusion | Reciprocal Rank Fusion (k=60) | Simple, no tuning, standard |
| Framework | None (no LangChain) | Every step is ~50 lines; the point is to own the mechanics. LangGraph arrives in project 3 |
| UI | FastAPI + one static HTML page | API layer for CV; page for demo GIF |
| Tracing | Langfuse Cloud free tier, optional via env | Self-hosted Langfuse is heavy (ClickHouse etc.) |
| Out of scope (phase 2) | Reranker, Telegram bot, multi-turn chat | Reranker costs 5-10 extra calls/question on free tier |

## Layout

```
bau-mevzuat-rag/
  app/
    config.py     env: GEMINI_API_KEY, QDRANT_PATH | QDRANT_URL, model names, LANGFUSE_*
    llm.py        Gemini client: embed(texts) -> list[vec], generate(prompt) -> str; retry on 429/5xx
    chunking.py   text -> list[Chunk] (article-based, window fallback)
    ingest.py     data/raw/* -> data/index/chunks.jsonl + Qdrant collection
    retrieve.py   dense / bm25 / hybrid -> list[Chunk] with scores
    answer.py     chunks + question -> prompt -> answer text + citations
    api.py        FastAPI: POST /ask, GET / (serves static/index.html)
  static/index.html
  data/raw/       source documents (pdf/docx/html/txt), committed
  data/index/     chunks.jsonl, qdrant/ — gitignored
  eval/
    make_questions.py   draft Q/A from random chunks (LLM) -> questions.draft.jsonl
    questions.jsonl     30 human-verified items
    run_eval.py         metrics table per mode -> stdout + eval/results/<date>.json
  tests/          pytest, no network
  Dockerfile, docker-compose.yml, pyproject.toml, .env.example, README.md
```

## Data model

```python
class Chunk(BaseModel):
    id: str            # "<doc_slug>:<article_no>" or "<doc_slug>:w<n>"
    doc_title: str
    article_no: int | None
    heading: str | None   # e.g. "Sınavlar"
    text: str
    source_url: str | None
```

## Ingest (one-time)

1. For each file in `data/raw/`: extract text by extension — pypdf, python-docx,
   BeautifulSoup for html, plain read for txt. Unparseable file → log and skip.
2. `chunking.split(text, doc_title)`:
   - If the text contains `MADDE \d+` headings: one chunk per article. The heading
     line immediately above the article (e.g. "Sınavlar") is captured as `heading`.
     Articles longer than ~600 words are split at fıkra boundaries `(\d+)`.
   - Otherwise: 400-word windows with 50-word overlap.
3. Write `data/index/chunks.jsonl`.
4. Embed in batches of 100 via `llm.embed`; upsert into Qdrant collection
   `bau_mevzuat` (cosine, payload = chunk fields).

## Query

`POST /ask` body `{question: str, mode: "dense"|"bm25"|"hybrid" = "hybrid", k: int = 5}`

- dense: embed question → Qdrant top-10.
- bm25: `rank_bm25.BM25Okapi` over lowercased whitespace tokens of all chunks → top-10.
- hybrid: both lists → RRF → top-k.
- If the best score is below a threshold (dense cosine < 0.3 or bm25 == 0), return
  the not-found answer without calling the LLM.
- `answer.build_prompt(question, chunks)`: numbered context blocks
  `[1] <doc_title>, Madde <n>: <text>`; instruction (Turkish): answer only from
  context, cite with `[n]`, if unsupported say exactly
  "Bu konuda yönetmeliklerde bilgi bulamadım."
- `answer.parse_citations(text, chunks)`: `[n]` present in text and in range →
  citation; out-of-range markers are stripped from the answer.
- Response: `{answer, citations: [{doc_title, article_no, text}], mode, latency_ms}`.

## Eval

- `make_questions.py`: sample 40 chunks, ask Gemini for one student-style
  question + short answer per chunk → `questions.draft.jsonl`. The human edits
  and keeps 30 → `questions.jsonl` with `{question, expected_chunk_ids, reference_answer}`.
- `run_eval.py --modes dense,bm25,hybrid`:
  - Retrieval: Recall@5 (any expected id in top 5), MRR.
  - Answer: Gemini judge, two binary scores — faithfulness (every claim supported
    by cited chunks) and correctness (agrees with reference_answer).
  - Output: markdown table (mode × metric) + `eval/results/<date>.json`.
  - Sleep between calls; backoff on 429. 30 × 3 modes fits the free tier.

## Error handling

- Missing `GEMINI_API_KEY` → fail at startup with a clear message.
- Gemini 429/5xx → 3 attempts with exponential backoff; then HTTP 503
  "Şu an yanıt üretilemiyor".
- Empty/low-score retrieval → not-found answer, no LLM call.
- Ingest never aborts on one bad file.

## Tests (pytest, no network)

- `test_chunking.py`: 3-article sample → 3 chunks with correct `article_no` and
  `heading`; text without MADDE → window chunks with overlap.
- `test_retrieve.py`: RRF on a hand-computed example.
- `test_answer.py`: `[1][3]` → two citations; `[9]` stripped.
- `test_api.py`: monkeypatch `llm.embed`/`llm.generate` and the retriever →
  `POST /ask` returns 200 and the response schema.

## Docker

- `Dockerfile`: `python:3.12-slim`, install from `pyproject.toml`, `uvicorn app.api:app`.
- `docker-compose.yml`: `app` (port 8000, env from `.env`) + `qdrant` (volume).
- Index is not baked into the image: `docker compose run app python -m app.ingest`.

## Dependencies

google-genai, qdrant-client, rank-bm25, fastapi, uvicorn, pydantic, pypdf,
python-docx, beautifulsoup4, python-dotenv, langfuse (optional), pytest, httpx.

## Git workflow

Feature branch → PR → merge, small commits. README: demo GIF, 3-command run,
eval table, "Design decisions" section (article chunking, RRF, why no framework).
