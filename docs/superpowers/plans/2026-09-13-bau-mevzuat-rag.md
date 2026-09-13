# bau-mevzuat-rag Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turkish RAG Q&A over BAU regulations with article citations, three retrieval modes (dense/BM25/hybrid) and a human-verified eval harness.

**Architecture:** Thin custom pipeline, no framework. `ingest` turns `data/raw/*` into article chunks → Gemini embeddings → Qdrant. `retrieve` offers dense (Qdrant), BM25 (`rank_bm25`) and hybrid (RRF). `answer` prompts Gemini with numbered chunks and maps `[n]` markers to citations. FastAPI exposes `POST /ask` and one static page. `eval/` measures Recall@5, MRR, faithfulness, correctness per mode.

**Tech Stack:** Python 3.12, google-genai, qdrant-client, rank-bm25, FastAPI, pypdf, python-docx, beautifulsoup4, pytest.

## Global Constraints

- Python 3.12 at `C:/Users/egee8/AppData/Local/Programs/Python/Python312/python.exe`; project venv at `.venv`.
- No LangChain/LlamaIndex. No network in tests.
- Models: `gemini-embedding-001` (768 dims), `gemini-2.5-flash` (answers), `gemini-2.5-flash-lite` (judge, question drafting).
- Qdrant collection `bau_mevzuat`, cosine. Dev: `QDRANT_PATH=data/index/qdrant`; Compose: `QDRANT_URL=http://qdrant:6333`.
- Not-found sentence, exact: `Bu konuda yönetmeliklerde bilgi bulamadım.`
- Code/README English, UI Turkish. Commits: feature branches, small commits.

---

### Task 1: Scaffold, config, Gemini client

**Files:**
- Create: `pyproject.toml`, `.gitignore`, `.env.example`, `app/__init__.py`, `app/config.py`, `app/llm.py`
- Test: `tests/test_llm.py`

**Interfaces:**
- Produces: `config.GEMINI_API_KEY, EMBED_MODEL, EMBED_DIM, ANSWER_MODEL, JUDGE_MODEL, COLLECTION, CHUNKS_PATH, QDRANT_PATH, QDRANT_URL, RAW_DIR, ROOT`, `config.require_api_key()`, `llm.embed(texts: list[str], task="RETRIEVAL_DOCUMENT") -> list[list[float]]`, `llm.generate(prompt: str, model: str | None = None) -> str`, `llm.retry(fn)`.

- [ ] **Step 1: pyproject, gitignore, env example**

`pyproject.toml`:
```toml
[project]
name = "bau-mevzuat-rag"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
  "google-genai>=1.0",
  "qdrant-client>=1.12",
  "rank-bm25>=0.2",
  "fastapi>=0.115",
  "uvicorn>=0.30",
  "pydantic>=2",
  "pypdf>=5",
  "python-docx>=1.1",
  "beautifulsoup4>=4.12",
  "python-dotenv>=1.0",
]
[project.optional-dependencies]
dev = ["pytest>=8", "httpx>=0.27"]
tracing = ["langfuse>=3"]

[tool.setuptools]
packages = ["app"]

[tool.pytest.ini_options]
pythonpath = ["."]
```

`.gitignore`:
```
.venv/
__pycache__/
.pytest_cache/
*.egg-info/
.env
data/index/
eval/results/
eval/questions.draft.jsonl
```

`.env.example`:
```
GEMINI_API_KEY=your-key
# QDRANT_URL=http://localhost:6333   # unset -> local mode at data/index/qdrant
# LANGFUSE_PUBLIC_KEY=
# LANGFUSE_SECRET_KEY=
# LANGFUSE_HOST=https://cloud.langfuse.com
```

- [ ] **Step 2: venv + install**

```bash
"C:/Users/egee8/AppData/Local/Programs/Python/Python312/python.exe" -m venv .venv
.venv/Scripts/python -m pip install -e ".[dev]"
```

- [ ] **Step 3: config.py**

```python
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
EMBED_MODEL = os.getenv("EMBED_MODEL", "gemini-embedding-001")
EMBED_DIM = 768
ANSWER_MODEL = os.getenv("ANSWER_MODEL", "gemini-2.5-flash")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "gemini-2.5-flash-lite")
COLLECTION = "bau_mevzuat"
RAW_DIR = ROOT / "data" / "raw"
CHUNKS_PATH = ROOT / "data" / "index" / "chunks.jsonl"
QDRANT_PATH = str(ROOT / "data" / "index" / "qdrant")
QDRANT_URL = os.getenv("QDRANT_URL")  # None -> local mode


def require_api_key() -> None:
    if not GEMINI_API_KEY:
        raise SystemExit("GEMINI_API_KEY is not set. Copy .env.example to .env and fill it in.")
```

- [ ] **Step 4: failing test for retry**

`tests/test_llm.py`:
```python
import pytest
from app import llm


class Boom(Exception):
    code = 429


def test_retry_succeeds_after_transient_errors(monkeypatch):
    monkeypatch.setattr(llm.time, "sleep", lambda s: None)
    calls = []

    def fn():
        calls.append(1)
        if len(calls) < 3:
            raise Boom()
        return "ok"

    assert llm.retry(fn) == "ok"
    assert len(calls) == 3


def test_retry_gives_up(monkeypatch):
    monkeypatch.setattr(llm.time, "sleep", lambda s: None)

    def fn():
        raise Boom()

    with pytest.raises(Boom):
        llm.retry(fn)
```

- [ ] **Step 5: run, expect ImportError**

`.venv/Scripts/python -m pytest tests/test_llm.py -v`

- [ ] **Step 6: llm.py**

```python
import time
from google import genai
from google.genai import types
from app import config

try:
    from langfuse import observe
except ImportError:  # tracing is optional
    def observe(*_a, **_k):
        return lambda f: f

_client = None


def client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(api_key=config.GEMINI_API_KEY)
    return _client


def retry(fn, attempts: int = 3):
    for i in range(attempts):
        try:
            return fn()
        except Exception as e:
            code = getattr(e, "code", None)
            if code not in (429, 500, 502, 503) or i == attempts - 1:
                raise
            time.sleep(2 ** (i + 1))


@observe()
def embed(texts: list[str], task: str = "RETRIEVAL_DOCUMENT") -> list[list[float]]:
    out: list[list[float]] = []
    for i in range(0, len(texts), 100):
        batch = texts[i:i + 100]
        res = retry(lambda: client().models.embed_content(
            model=config.EMBED_MODEL,
            contents=batch,
            config=types.EmbedContentConfig(task_type=task, output_dimensionality=config.EMBED_DIM),
        ))
        out.extend(e.values for e in res.embeddings)
    return out


@observe()
def generate(prompt: str, model: str | None = None) -> str:
    res = retry(lambda: client().models.generate_content(
        model=model or config.ANSWER_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(temperature=0),
    ))
    return (res.text or "").strip()
```

- [ ] **Step 7: run tests, expect PASS; commit**

```bash
git add pyproject.toml .gitignore .env.example app tests
git commit -m "feat: scaffold project, config and Gemini client with retry"
```

---

### Task 2: Chunking

**Files:**
- Create: `app/chunking.py`
- Test: `tests/test_chunking.py`

**Interfaces:**
- Produces: `class Chunk(BaseModel): id, doc_title, article_no: int|None, heading: str|None, text, source_url: str|None`; `split(text: str, doc_title: str, slug: str) -> list[Chunk]`; `slugify(name) -> str`.

- [ ] **Step 1: failing tests**

```python
from app.chunking import split, slugify

SAMPLE = """BİRİNCİ BÖLÜM
Amaç
MADDE 1 – (1) Bu Yönetmeliğin amacı eğitim esaslarını düzenlemektir.
Kapsam
MADDE 2 – (1) Bu Yönetmelik lisans öğrencilerini kapsar.
(2) Yüksek lisans öğrencilerini kapsamaz.
Dayanak
MADDE 3 - (1) 2547 sayılı Kanuna dayanır.
"""


def test_article_split():
    chunks = split(SAMPLE, "Test Yönetmeliği", "test")
    assert [c.article_no for c in chunks] == [1, 2, 3]
    assert chunks[1].heading == "Kapsam"
    assert chunks[1].id == "test:2"
    assert "(2) Yüksek lisans" in chunks[1].text
    assert chunks[0].text.startswith("MADDE 1")


def test_window_fallback():
    words = [f"w{i}" for i in range(1000)]
    chunks = split(" ".join(words), "Doc", "doc")
    assert chunks[0].article_no is None
    assert chunks[0].id == "doc:w0"
    assert len(chunks[0].text.split()) == 400
    assert chunks[1].text.split()[0] == "w350"
    assert len(chunks) == 3


def test_slugify():
    assert slugify("BAU Önlisans ve Lisans Yönetmeliği") == "bau-onlisans-ve-lisans-yonetmeligi"
```

- [ ] **Step 2: run, expect FAIL**

- [ ] **Step 3: chunking.py**

```python
import re
import unicodedata
from pydantic import BaseModel

ARTICLE_RE = re.compile(r"^\s*MADDE\s+(\d+)\s*[-–—]?", re.MULTILINE | re.IGNORECASE)
FIKRA_RE = re.compile(r"(?=\(\d+\)\s)")
WINDOW, OVERLAP, MAX_ARTICLE_WORDS = 400, 50, 600
TR_MAP = str.maketrans("çğıöşüÇĞİÖŞÜ", "cgiosuCGIOSU")


class Chunk(BaseModel):
    id: str
    doc_title: str
    article_no: int | None = None
    heading: str | None = None
    text: str
    source_url: str | None = None


def slugify(name: str) -> str:
    s = name.translate(TR_MAP)
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


def split(text: str, doc_title: str, slug: str) -> list[Chunk]:
    matches = list(ARTICLE_RE.finditer(text))
    if not matches:
        return _windows(text, doc_title, slug)
    chunks: list[Chunk] = []
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[m.start():end].strip()
        no = int(m.group(1))
        heading = _heading_before(text, m.start())
        parts = _split_long(body)
        for j, part in enumerate(parts):
            cid = f"{slug}:{no}" if len(parts) == 1 else f"{slug}:{no}.{j + 1}"
            chunks.append(Chunk(id=cid, doc_title=doc_title, article_no=no, heading=heading, text=part))
    return chunks


def _heading_before(text: str, pos: int) -> str | None:
    lines = [l.strip() for l in text[:pos].splitlines() if l.strip()]
    if not lines:
        return None
    line = lines[-1]
    if len(line) > 80 or ARTICLE_RE.match(line) or line.startswith("("):
        return None
    return line


def _split_long(body: str) -> list[str]:
    if len(body.split()) <= MAX_ARTICLE_WORDS:
        return [body]
    parts, cur = [], ""
    for piece in FIKRA_RE.split(body):
        if cur and len((cur + piece).split()) > MAX_ARTICLE_WORDS:
            parts.append(cur.strip())
            cur = ""
        cur += piece
    if cur.strip():
        parts.append(cur.strip())
    return parts


def _windows(text: str, doc_title: str, slug: str) -> list[Chunk]:
    words = text.split()
    chunks, start, n = [], 0, 0
    while start < len(words):
        chunks.append(Chunk(id=f"{slug}:w{n}", doc_title=doc_title, text=" ".join(words[start:start + WINDOW])))
        if start + WINDOW >= len(words):
            break
        start += WINDOW - OVERLAP
        n += 1
    return chunks
```

- [ ] **Step 4: run, expect PASS; commit** `feat: article-based chunking with window fallback`

---

### Task 3: Retrieval (dense, BM25, hybrid/RRF)

**Files:**
- Create: `app/retrieve.py`
- Test: `tests/test_retrieve.py`

**Interfaces:**
- Consumes: `Chunk`, `llm.embed`, `config`.
- Produces: `tokenize(text) -> list[str]`, `rrf(rankings: list[list[str]], k=60) -> list[tuple[str, float]]`, `point_id(chunk_id) -> str`, `open_qdrant()`, `load_chunks()`, `class Retriever(chunks, qdrant)` with `.search(question, mode, k) -> list[tuple[Chunk, float]]`, `Retriever.load()`.

- [ ] **Step 1: failing tests**

```python
from app.retrieve import rrf, tokenize, Retriever
from app.chunking import Chunk


def test_rrf_order():
    fused = rrf([["a", "b", "c"], ["b", "c"]])
    assert [i for i, _ in fused] == ["b", "c", "a"]


def test_tokenize_turkish_prefix():
    assert tokenize("Sınavlara girmeyen öğrenciler") == ["sınav", "girme", "öğren"]


def test_bm25_finds_matching_chunk():
    chunks = [
        Chunk(id="d:1", doc_title="D", article_no=1, text="Yıllık izin süresi yirmi gündür."),
        Chunk(id="d:2", doc_title="D", article_no=2, text="Sınavlar dönem sonunda yapılır."),
    ]
    r = Retriever(chunks, qdrant=None)
    top = r.search("sınav ne zaman yapılır", mode="bm25", k=1)
    assert top[0][0].id == "d:2"


def test_bm25_no_match_returns_empty():
    chunks = [Chunk(id="d:1", doc_title="D", text="Yıllık izin süresi yirmi gündür.")]
    assert Retriever(chunks, qdrant=None).search("kütüphane", mode="bm25", k=1) == []
```

- [ ] **Step 2: run, expect FAIL**

- [ ] **Step 3: retrieve.py**

```python
import re
import uuid
from rank_bm25 import BM25Okapi
from qdrant_client import QdrantClient
from app import config, llm
from app.chunking import Chunk

DENSE_MIN_SCORE = 0.3
WORD_RE = re.compile(r"\w+")


def tokenize(text: str) -> list[str]:
    # ponytail: 5-char prefix as a Turkish stemmer; swap for a real stemmer if BM25 recall is the bottleneck
    return [w[:5] for w in WORD_RE.findall(text.lower())]


def rrf(rankings: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, cid in enumerate(ranking, start=1):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: -x[1])


def point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


def open_qdrant() -> QdrantClient:
    return QdrantClient(url=config.QDRANT_URL) if config.QDRANT_URL else QdrantClient(path=config.QDRANT_PATH)


def load_chunks() -> list[Chunk]:
    with open(config.CHUNKS_PATH, encoding="utf-8") as f:
        return [Chunk.model_validate_json(line) for line in f if line.strip()]


class Retriever:
    def __init__(self, chunks: list[Chunk], qdrant: QdrantClient | None):
        self.chunks = chunks
        self.by_id = {c.id: c for c in chunks}
        self.qdrant = qdrant
        self.bm25 = BM25Okapi([tokenize(c.text) for c in chunks]) if chunks else None

    @classmethod
    def load(cls) -> "Retriever":
        if not config.CHUNKS_PATH.exists():
            raise SystemExit("Index not found. Run: python -m app.ingest")
        return cls(load_chunks(), open_qdrant())

    def dense(self, question: str, n: int = 10) -> list[tuple[Chunk, float]]:
        vec = llm.embed([question], task="RETRIEVAL_QUERY")[0]
        res = self.qdrant.query_points(config.COLLECTION, query=vec, limit=n).points
        return [(self.by_id[p.payload["id"]], p.score) for p in res
                if p.score >= DENSE_MIN_SCORE and p.payload["id"] in self.by_id]

    def sparse(self, question: str, n: int = 10) -> list[tuple[Chunk, float]]:
        scores = self.bm25.get_scores(tokenize(question))
        ranked = sorted(range(len(scores)), key=lambda i: -scores[i])[:n]
        return [(self.chunks[i], float(scores[i])) for i in ranked if scores[i] > 0]

    def search(self, question: str, mode: str = "hybrid", k: int = 5) -> list[tuple[Chunk, float]]:
        if mode == "dense":
            return self.dense(question)[:k]
        if mode == "bm25":
            return self.sparse(question)[:k]
        d, s = self.dense(question), self.sparse(question)
        fused = rrf([[c.id for c, _ in d], [c.id for c, _ in s]])
        return [(self.by_id[cid], score) for cid, score in fused[:k]]
```

- [ ] **Step 4: run, expect PASS; commit** `feat: dense, BM25 and RRF hybrid retrieval`

---

### Task 4: Answer generation with citations

**Files:**
- Create: `app/answer.py`
- Test: `tests/test_answer.py`

**Interfaces:**
- Consumes: `Chunk`, `llm.generate`.
- Produces: `NOT_FOUND`, `build_prompt(question, chunks) -> str`, `parse_citations(text, chunks) -> tuple[str, list[Chunk]]`, `answer(question, chunks) -> tuple[str, list[Chunk]]`.

- [ ] **Step 1: failing tests**

```python
import pytest
from app import answer as answer_mod
from app.answer import parse_citations, build_prompt, answer, NOT_FOUND
from app.chunking import Chunk

C = [Chunk(id=f"d:{i}", doc_title="Doc", article_no=i, text=f"madde {i}") for i in (1, 2, 3)]


def test_parse_citations_keeps_valid_drops_invalid():
    text, cites = parse_citations("Cevap [1] ve [3] ve [9].", C)
    assert [c.id for c in cites] == ["d:1", "d:3"]
    assert "[9]" not in text and "[1]" in text


def test_prompt_numbers_chunks():
    p = build_prompt("soru?", C)
    assert "[2] Doc, Madde 2: madde 2" in p
    assert NOT_FOUND in p


def test_answer_without_chunks_skips_llm(monkeypatch):
    def boom(*_a, **_k):
        raise AssertionError("llm called")
    monkeypatch.setattr(answer_mod.llm, "generate", boom)
    assert answer("soru?", []) == (NOT_FOUND, [])
```

- [ ] **Step 2: run, expect FAIL**

- [ ] **Step 3: answer.py**

```python
import re
from app import llm
from app.chunking import Chunk

NOT_FOUND = "Bu konuda yönetmeliklerde bilgi bulamadım."
CITE_RE = re.compile(r"\[(\d+)\]")

PROMPT = """Sen Bahçeşehir Üniversitesi yönetmelikleri konusunda yardımcı bir asistansın.
Sadece aşağıdaki BAĞLAM'daki bilgilere dayanarak Türkçe cevap ver.
Her iddianın sonuna dayandığı kaynağın numarasını [n] biçiminde ekle.
Bağlam soruyu cevaplamaya yetmiyorsa sadece şunu yaz: "{not_found}"

BAĞLAM:
{context}

SORU: {question}
CEVAP:"""


def build_prompt(question: str, chunks: list[Chunk]) -> str:
    ctx = "\n\n".join(
        f"[{i}] {c.doc_title}, Madde {c.article_no}: {c.text}" if c.article_no else f"[{i}] {c.doc_title}: {c.text}"
        for i, c in enumerate(chunks, start=1)
    )
    return PROMPT.format(not_found=NOT_FOUND, context=ctx, question=question)


def parse_citations(text: str, chunks: list[Chunk]) -> tuple[str, list[Chunk]]:
    used: list[Chunk] = []
    for n in CITE_RE.findall(text):
        i = int(n)
        if 1 <= i <= len(chunks) and chunks[i - 1] not in used:
            used.append(chunks[i - 1])
    cleaned = CITE_RE.sub(lambda m: m.group(0) if 1 <= int(m.group(1)) <= len(chunks) else "", text)
    return cleaned.strip(), used


def answer(question: str, chunks: list[Chunk]) -> tuple[str, list[Chunk]]:
    if not chunks:
        return NOT_FOUND, []
    text = llm.generate(build_prompt(question, chunks))
    return parse_citations(text, chunks)
```

- [ ] **Step 4: run, expect PASS; commit** `feat: cited answer generation`

---

### Task 5: Ingest

**Files:**
- Create: `app/ingest.py`
- Test: `tests/test_ingest.py`

**Interfaces:**
- Consumes: `chunking.split/slugify`, `llm.embed`, `retrieve.open_qdrant/point_id`.
- Produces: `read_text(path) -> str`, `build_chunks(raw_dir) -> list[Chunk]`, `main()`.

- [ ] **Step 1: failing test**

```python
from pathlib import Path
from app.ingest import build_chunks


def test_build_chunks_from_txt(tmp_path: Path):
    (tmp_path / "Test Yonetmeligi.txt").write_text(
        "Amaç\nMADDE 1 – (1) Amaç budur.\nMADDE 2 – (1) Kapsam budur.", encoding="utf-8")
    (tmp_path / "broken.pdf").write_bytes(b"not a pdf")
    chunks = build_chunks(tmp_path)
    assert [c.id for c in chunks] == ["test-yonetmeligi:1", "test-yonetmeligi:2"]
    assert chunks[0].doc_title == "Test Yonetmeligi"
```

- [ ] **Step 2: run, expect FAIL**

- [ ] **Step 3: ingest.py**

```python
import logging
import sys
from pathlib import Path
from qdrant_client.models import Distance, PointStruct, VectorParams
from app import config, llm
from app.chunking import Chunk, slugify, split
from app.retrieve import open_qdrant, point_id

log = logging.getLogger(__name__)


def read_text(path: Path) -> str:
    ext = path.suffix.lower()
    if ext == ".pdf":
        from pypdf import PdfReader
        return "\n".join(p.extract_text() or "" for p in PdfReader(str(path)).pages)
    if ext == ".docx":
        from docx import Document
        return "\n".join(p.text for p in Document(str(path)).paragraphs)
    if ext in (".html", ".htm"):
        from bs4 import BeautifulSoup
        return BeautifulSoup(path.read_text(encoding="utf-8", errors="ignore"), "html.parser").get_text("\n")
    return path.read_text(encoding="utf-8", errors="ignore")


def build_chunks(raw_dir: Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(raw_dir.iterdir()):
        if not path.is_file():
            continue
        try:
            text = read_text(path)
        except Exception as e:  # one bad file must not abort ingest
            log.warning("skipping %s: %s", path.name, e)
            continue
        got = split(text, path.stem, slugify(path.stem))
        log.info("%s -> %d chunks", path.name, len(got))
        chunks.extend(got)
    return chunks


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    config.require_api_key()
    chunks = build_chunks(config.RAW_DIR)
    if not chunks:
        sys.exit(f"No documents found in {config.RAW_DIR}")
    config.CHUNKS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(config.CHUNKS_PATH, "w", encoding="utf-8") as f:
        f.writelines(c.model_dump_json() + "\n" for c in chunks)
    vectors = llm.embed([c.text for c in chunks])
    q = open_qdrant()
    if q.collection_exists(config.COLLECTION):
        q.delete_collection(config.COLLECTION)
    q.create_collection(config.COLLECTION, vectors_config=VectorParams(size=config.EMBED_DIM, distance=Distance.COSINE))
    q.upsert(config.COLLECTION, points=[
        PointStruct(id=point_id(c.id), vector=v, payload=c.model_dump()) for c, v in zip(chunks, vectors)
    ])
    log.info("indexed %d chunks into %s", len(chunks), config.COLLECTION)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: run, expect PASS; commit** `feat: ingest documents into chunks.jsonl and Qdrant`

---

### Task 6: API + static page

**Files:**
- Create: `app/api.py`, `static/index.html`
- Test: `tests/test_api.py`

**Interfaces:**
- Consumes: `Retriever`, `answer.answer`, `config.require_api_key`.
- Produces: `POST /ask` `{question, mode, k}` → `{answer, citations[{doc_title, article_no, heading, text}], mode, latency_ms}`; `GET /` → html; `GET /health`.

- [ ] **Step 1: failing test**

```python
from fastapi.testclient import TestClient
from app import api
from app.chunking import Chunk


class FakeRetriever:
    chunks = []

    def search(self, q, mode="hybrid", k=5):
        return [(Chunk(id="d:5", doc_title="Doc", article_no=5, text="Gecme notu 60 puandir."), 0.9)]


def test_ask_returns_answer_and_citations(monkeypatch):
    api.retriever = FakeRetriever()
    monkeypatch.setattr(api.llm, "generate", lambda *_a, **_k: "Gecme notu 60 puandir [1].")
    r = TestClient(api.app).post("/ask", json={"question": "gecme notu kac?"})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"].endswith("[1].")
    assert body["citations"][0]["article_no"] == 5
    assert body["mode"] == "hybrid"


def test_index_page():
    r = TestClient(api.app).get("/")
    assert r.status_code == 200 and "BAU" in r.text
```

- [ ] **Step 2: run, expect FAIL**

- [ ] **Step 3: api.py**

```python
import time
from contextlib import asynccontextmanager
from typing import Literal
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from app import config, llm
from app.answer import answer
from app.retrieve import Retriever

retriever: Retriever | None = None


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global retriever
    config.require_api_key()
    retriever = Retriever.load()
    yield


app = FastAPI(title="bau-mevzuat-rag", lifespan=lifespan)


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    mode: Literal["dense", "bm25", "hybrid"] = "hybrid"
    k: int = Field(default=5, ge=1, le=10)


class Citation(BaseModel):
    doc_title: str
    article_no: int | None
    heading: str | None
    text: str


class AskResponse(BaseModel):
    answer: str
    citations: list[Citation]
    mode: str
    latency_ms: int


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    t0 = time.perf_counter()
    try:
        hits = retriever.search(req.question, mode=req.mode, k=req.k)
        text, cites = answer(req.question, [c for c, _ in hits])
    except Exception as e:
        raise HTTPException(503, f"Şu an yanıt üretilemiyor: {type(e).__name__}")
    return AskResponse(
        answer=text,
        citations=[Citation(**c.model_dump()) for c in cites],
        mode=req.mode,
        latency_ms=int((time.perf_counter() - t0) * 1000),
    )


@app.get("/health")
def health() -> dict:
    return {"ok": True, "chunks": len(retriever.chunks) if retriever else 0}


@app.get("/")
def index() -> FileResponse:
    return FileResponse(config.ROOT / "static" / "index.html")
```

- [ ] **Step 4: static/index.html** — Turkish UI, no framework: textarea, mode select (hybrid/dense/bm25), "Sor" button, answer box, collapsible citation list, `fetch("/ask")`, error line on non-200.

- [ ] **Step 5: run, expect PASS; commit** `feat: FastAPI /ask endpoint and chat page`

---

### Task 7: Eval harness

**Files:**
- Create: `eval/__init__.py`, `eval/make_questions.py`, `eval/run_eval.py`
- Test: `tests/test_eval.py` (metric functions only)

**Interfaces:**
- Produces: `run_eval.recall_at_k(expected: list[str], got: list[str], k=5) -> float`, `run_eval.mrr(expected, got) -> float`, CLI `python -m eval.run_eval --modes dense,bm25,hybrid [--no-judge]`, CLI `python -m eval.make_questions`.

- [ ] **Step 1: failing tests**

```python
from eval.run_eval import recall_at_k, mrr


def test_recall_and_mrr():
    assert recall_at_k(["a"], ["x", "a", "y"], k=5) == 1.0
    assert recall_at_k(["a"], ["x", "y"], k=5) == 0.0
    assert mrr(["a"], ["x", "a", "y"]) == 0.5
    assert mrr(["a"], []) == 0.0
```

- [ ] **Step 2: run, expect FAIL**

- [ ] **Step 3: make_questions.py**

```python
"""Draft eval questions from random chunks. A human must review the output."""
import json
import random
import time
from app import config, llm
from app.retrieve import load_chunks

PROMPT = """Aşağıdaki yönetmelik maddesini oku. Bir öğrencinin bu maddeyle ilgili sorabileceği,
maddeyi okumadan cevaplanamayacak TEK bir doğal Türkçe soru ve 1-2 cümlelik doğru cevabını yaz.
Sadece şu JSON'u döndür: {{"question": "...", "answer": "..."}}

MADDE:
{text}"""


def main(n: int = 40, seed: int = 7) -> None:
    config.require_api_key()
    chunks = [c for c in load_chunks() if len(c.text.split()) > 30]
    random.Random(seed).shuffle(chunks)
    out = config.ROOT / "eval" / "questions.draft.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for c in chunks[:n]:
            raw = llm.generate(PROMPT.format(text=c.text), model=config.JUDGE_MODEL)
            raw = raw.strip().removeprefix("```json").removesuffix("```").strip()
            try:
                qa = json.loads(raw)
            except json.JSONDecodeError:
                continue
            f.write(json.dumps({"question": qa["question"], "expected_chunk_ids": [c.id],
                                "reference_answer": qa["answer"]}, ensure_ascii=False) + "\n")
            time.sleep(4)  # free-tier RPM
    print(f"wrote {out}; review it and save the good ones as eval/questions.jsonl")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: run_eval.py**

```python
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

İki soruya sadece EVET veya HAYIR ile, şu biçimde cevap ver:
faithfulness: <EVET|HAYIR>  (verilen cevaptaki her iddia kaynak metinlerle destekleniyor mu?)
correctness: <EVET|HAYIR>   (verilen cevap referans cevapla özde uyuşuyor mu?)"""


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
        qs = [json.loads(l) for l in f if l.strip()]
    r = Retriever.load()
    results = {}
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
                f_, c_ = judge(q["question"], q["reference_answer"], sources, ans)
                agg["faithfulness"] += f_
                agg["correctness"] += c_
                time.sleep(6)  # free-tier RPM
        results[mode] = {k: round(v / len(qs), 3) for k, v in agg.items()}
    print("| mode | Recall@5 | MRR | Faithfulness | Correctness |\n|---|---|---|---|---|")
    for mode, m in results.items():
        print(f"| {mode} | {m['recall@5']} | {m['mrr']} | {m['faithfulness']} | {m['correctness']} |")
    out = config.ROOT / "eval" / "results" / f"{date.today()}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"n": len(qs), "results": results}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: run tests, expect PASS; commit** `feat: eval harness (recall@5, MRR, LLM-judged faithfulness/correctness)`

---

### Task 8: Docker + README

**Files:**
- Create: `Dockerfile`, `docker-compose.yml`, `README.md`

- [ ] **Step 1: Dockerfile**

```dockerfile
FROM python:3.12-slim
WORKDIR /srv
COPY pyproject.toml .
COPY app app
COPY static static
COPY eval eval
RUN pip install --no-cache-dir .
EXPOSE 8000
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
```

- [ ] **Step 2: docker-compose.yml**

```yaml
services:
  app:
    build: .
    ports: ["8000:8000"]
    env_file: .env
    environment:
      QDRANT_URL: http://qdrant:6333
    volumes:
      - ./data:/srv/data
    depends_on: [qdrant]
  qdrant:
    image: qdrant/qdrant:latest
    volumes:
      - qdrant_data:/qdrant/storage
volumes:
  qdrant_data:
```

- [ ] **Step 3: README.md** — what/why, demo GIF placeholder `docs/demo.gif`, quickstart (3 commands), Docker, eval table (filled after Task 9), Design decisions (article chunking, RRF, prefix stemmer, no framework), Roadmap (reranker, Telegram).

- [ ] **Step 4: commit** `docs: README, Dockerfile, compose`

---

### Task 9: Real data + eval run (needs API key and documents)

- [ ] Put regulation files into `data/raw/` (user supplies; file names become titles).
- [ ] `python -m app.ingest` → check chunk count and a few `article_no` values in `data/index/chunks.jsonl`.
- [ ] `uvicorn app.api:app --reload` → ask 5 questions in the page, sanity check citations.
- [ ] `python -m eval.make_questions` → review draft → save 30 as `eval/questions.jsonl`; commit.
- [ ] `python -m eval.run_eval --no-judge` first (cheap), then full run → paste table into README; commit.
