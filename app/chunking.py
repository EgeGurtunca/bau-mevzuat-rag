"""Split regulation text into article ("MADDE n") chunks; fall back to word windows."""
import re
import unicodedata

from pydantic import BaseModel

ARTICLE_RE = re.compile(r"^\s*(?:(GEÇİCİ|GECICI|EK)\s+)?MADDE\s+(\d+)\s*[-–—]?", re.MULTILINE | re.IGNORECASE)
FIKRA_RE = re.compile(r"(?=\(\d+\)\s)")  # paragraph markers "(1) ", "(2) " inside an article
APPENDIX_RE = re.compile(r"^.*(?:İŞLENEMEYEN HÜKÜMLER|YÜRÜRLÜĞE GİRİŞ TARİHLERİNİ GÖSTERİR TABLO|Yayımlandığı Resmî Gazete.?nin)",
                         re.MULTILINE)
WINDOW, OVERLAP, MAX_ARTICLE_WORDS = 400, 50, 600
TR_MAP = str.maketrans("çğıöşüÇĞİÖŞÜ", "cgiosuCGIOSU")


class Chunk(BaseModel):
    id: str
    doc_title: str
    article_no: int | None = None
    article_kind: str | None = None  # None (ordinary), "Geçici" or "Ek"
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
    text = _strip_appendix(text, matches[0].start())
    matches = list(ARTICLE_RE.finditer(text))
    chunks: list[Chunk] = []
    seen: set[str] = set()
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[m.start():end].strip()
        no = int(m.group(2))
        kind = _kind(m.group(1))
        heading = _heading_before(text, m.start())
        body = _strip_trailing_headings(body)  # the next article's (and section's) headings sit inside our slice
        parts = _split_long(body)
        base = f"{slug}:{kind[0].lower() if kind else ''}{no}"
        while base in seen:  # same article number twice (e.g. an amending law quoted in full) -> keep both, distinct ids
            base += "-dup"
        seen.add(base)
        for j, part in enumerate(parts):
            cid = base if len(parts) == 1 else f"{base}.{j + 1}"
            chunks.append(Chunk(id=cid, doc_title=doc_title, article_no=no, article_kind=kind, heading=heading, text=part))
    return chunks


def _kind(prefix: str | None) -> str | None:
    if not prefix:
        return None
    return "Ek" if prefix.upper() == "EK" else "Geçici"


def article_label(kind: str | None, no: int | None) -> str:
    """'Madde 5', 'Geçici Madde 2', 'Ek Madde 1'."""
    return f"{kind} Madde {no}" if kind else f"Madde {no}"


def _heading_before(text: str, pos: int) -> str | None:
    """The short non-empty line right above an article is its heading (e.g. 'Sınavlar')."""
    lines = [line.strip() for line in text[:pos].splitlines() if line.strip()]
    if not lines:
        return None
    line = lines[-1]
    if len(line) > 80 or ARTICLE_RE.match(line) or line.startswith("("):
        return None
    return line


def _looks_like_heading(line: str) -> bool:
    """Short line, no sentence punctuation at the end, not a numbered/lettered paragraph."""
    return (len(line) <= 80 and not line.endswith((".", ":", ";", ","))
            and not line.startswith("(") and not re.match(r"^[a-zçğıöşü]\)", line))


def _strip_trailing_headings(body: str, max_lines: int = 3) -> str:
    """Drop heading-looking lines from the end of an article (they belong to the next article/section)."""
    lines = body.rstrip().splitlines()
    while len(lines) > 1 and max_lines and _looks_like_heading(lines[-1].strip()):
        lines.pop()
        max_lines -= 1
    return "\n".join(lines).rstrip()


def _strip_appendix(text: str, first_article: int) -> str:
    """mevzuat.gov.tr appends amending laws ("... KANUNA İŞLENEMEYEN HÜKÜMLER") and an effective-date table
    after the last article; their own 'Madde n' lines would collide with the real ones. Cut there."""
    m = APPENDIX_RE.search(text, first_article)
    return text[:m.start()] if m else text


def _split_long(body: str) -> list[str]:
    if len(body.split()) <= MAX_ARTICLE_WORDS:
        return [body]
    pieces = FIKRA_RE.split(body)
    if len(pieces) < 2:  # no "(n)" paragraph markers (the Constitution) -> split on lines instead
        pieces = [line + "\n" for line in body.splitlines()]
    parts, cur = [], ""
    for piece in pieces:
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
