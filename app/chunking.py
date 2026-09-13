"""Split regulation text into article ("MADDE n") chunks; fall back to word windows."""
import re
import unicodedata

from pydantic import BaseModel

ARTICLE_RE = re.compile(r"^\s*MADDE\s+(\d+)\s*[-–—]?", re.MULTILINE | re.IGNORECASE)
FIKRA_RE = re.compile(r"(?=\(\d+\)\s)")  # paragraph markers "(1) ", "(2) " inside an article
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
    """The short non-empty line right above an article is its heading (e.g. 'Sınavlar')."""
    lines = [line.strip() for line in text[:pos].splitlines() if line.strip()]
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
