"""Grounded answer generation: numbered context -> Gemini -> answer text + cited chunks."""
import re

from app import llm
from app.chunking import Chunk

NOT_FOUND = "Bu konuda yönetmeliklerde bilgi bulamadım."
CITE_RE = re.compile(r"\[(\d+)\]")

PROMPT = """Sen Bahçeşehir Üniversitesi yönetmelikleri konusunda yardımcı bir asistansın.
Sadece aşağıdaki BAĞLAM'daki bilgilere dayanarak cevap ver. Soru hangi dildeyse o dilde cevap ver.
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
    """Map [n] markers to chunks and renumber them 1..k in order of first appearance,
    so the answer's markers match the returned citation list. Markers the model made up are stripped."""
    order: dict[int, int] = {}  # prompt index -> citation number
    for n in CITE_RE.findall(text):
        i = int(n)
        if 1 <= i <= len(chunks) and i not in order:
            order[i] = len(order) + 1
    cleaned = CITE_RE.sub(lambda m: f"[{order[int(m.group(1))]}]" if int(m.group(1)) in order else "", text)
    return cleaned.strip(), [chunks[i - 1] for i in order]


def answer(question: str, chunks: list[Chunk]) -> tuple[str, list[Chunk]]:
    if not chunks:  # nothing retrieved -> no LLM call, saves quota
        return NOT_FOUND, []
    text = llm.generate(build_prompt(question, chunks))
    return parse_citations(text, chunks)
