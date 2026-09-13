"""data/raw/* -> data/index/chunks.jsonl + Qdrant collection. Run: python -m app.ingest"""
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
        soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="ignore"), "html.parser")
        for tag in soup(["script", "style"]):
            tag.decompose()
        return soup.get_text("\n")
    return path.read_text(encoding="utf-8", errors="ignore")


def build_chunks(raw_dir: Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(raw_dir.iterdir()):
        if not path.is_file():
            continue
        try:
            text = read_text(path)
        except Exception as e:  # one bad file must not abort the whole ingest
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
