"""Grounded answer generation: numbered context -> LLM -> answer text + cited chunks."""
import re

from app import llm
from app.chunking import Chunk, article_label

NOT_FOUND = "Bu konuda mevzuatta bilgi bulamadım."
CITE_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")  # [2] or [1, 3]

PROMPT = """Sen Türk mevzuatı (Anayasa ve Bahçeşehir Üniversitesi yönetmelikleri) konusunda yardımcı bir asistansın.
Sadece aşağıdaki BAĞLAM'daki bilgilere dayanarak cevap ver. Soru hangi dildeyse o dilde cevap ver.
Kısa ve net ol: en fazla 3-4 cümle, madde işareti veya başlık kullanma.
Her iddianın sonuna dayandığı kaynağın numarasını köşeli parantez içinde ekle, örneğin: "... ücretin %25'ini öder [2]."
Kaynak numaraları 1 ile {n} arasındadır; başka numara kullanma.
Bağlam soruyu cevaplamaya yetmiyorsa sadece şunu yaz: "{not_found}"

BAĞLAM:
{context}

SORU: {question}
CEVAP:"""


def build_prompt(question: str, chunks: list[Chunk]) -> str:
    ctx = "\n\n".join(
        f"[{i}] {c.doc_title}, {article_label(c.article_kind, c.article_no)}: {c.text}" if c.article_no else f"[{i}] {c.doc_title}: {c.text}"
        for i, c in enumerate(chunks, start=1)
    )
    return PROMPT.format(not_found=NOT_FOUND, context=ctx, question=question, n=len(chunks))


def parse_citations(text: str, chunks: list[Chunk]) -> tuple[str, list[Chunk]]:
    """Map [n] markers to chunks and renumber them 1..k in order of first appearance,
    so the answer's markers match the returned citation list. Markers the model made up are stripped."""
    order: dict[int, int] = {}  # prompt index -> citation number

    def nums(group: str) -> list[int]:
        return [int(x) for x in group.split(",")]

    for group in CITE_RE.findall(text):
        for i in nums(group):
            if 1 <= i <= len(chunks) and i not in order:
                order[i] = len(order) + 1

    def renumber(m: re.Match) -> str:
        return "".join(f"[{order[i]}]" for i in dict.fromkeys(nums(m.group(1))) if i in order)

    return CITE_RE.sub(renumber, text).strip(), [chunks[i - 1] for i in order]


def answer(question: str, chunks: list[Chunk], model: str | None = None) -> tuple[str, list[Chunk]]:
    if not chunks:  # nothing retrieved -> no LLM call, saves quota
        return NOT_FOUND, []
    text = llm.generate(build_prompt(question, chunks), model=model)
    return parse_citations(text, chunks)
