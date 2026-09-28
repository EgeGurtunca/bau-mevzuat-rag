"""Grounded answer generation: numbered context -> LLM -> answer text + cited chunks."""
import re

from app import llm
from app.chunking import Chunk, article_label

NOT_FOUND = "Bu konuda mevzuatta bilgi bulamadım."
CITE_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")  # [2] or [1, 3]

# Instructions come after the context: with them on top, the 7B model had forgotten them ~5000 tokens later.
PROMPT = """Sen Türk mevzuatı (Anayasa ve Bahçeşehir Üniversitesi yönetmelikleri) konusunda yardımcı bir asistansın.
Aşağıda numaralı mevzuat maddeleri ve bir öğrencinin sorusu var.

BAĞLAM:
{context}

SORU: {question}

Sadece BAĞLAM'daki maddelere dayanarak cevap ver. Öğrenciler günlük dille sorar, maddeler aynı şeyi resmi
terimlerle anlatır: soruyu anlamca karşılayan bir madde varsa onu kullan, kelimelerin aynı olmasını bekleme.

Cevabını iki bölüm halinde yaz:
ALINTI: Soruyu cevaplayan cümleleri bağlamdan kelimesi kelimesine kopyala, her birinin sonuna kaynak numarasını
ekle, örneğin "... [2]". Sadece gereken cümleleri al, bütün maddeyi kopyalama. Soru birden fazla şey soruyorsa
her biri için alıntı yap.
CEVAP: Alıntılara dayanan kısa ve net cevap: en fazla 3-4 cümle, madde işareti veya başlık yok. Sayıları, süreleri
ve yüzdeleri alıntıdan aynen al. Her iddianın sonuna kaynak numarasını ekle, örneğin "[2]". Soru hangi dildeyse
cevabı o dilde yaz.
Kaynak numaraları 1 ile {n} arasındadır; başka numara kullanma.

Maddeler sorunun istediği bilgiyi vermiyorsa bölümleri yazma, sadece şunu yaz: "{not_found}\""""
ANSWER_RE = re.compile(r"\**CEVAP\**\s*:\**", re.IGNORECASE)
ALINTI_RE = re.compile(r"\**ALINTI\**\s*:", re.IGNORECASE)

# Told to write NOT_FOUND when the articles don't answer, gemma4 wrote "no date is specified, the Rektörlük
# announces it" instead. Asking it to judge the context up front (a KARAR: EVET/HAYIR field) fixed those but
# also refused paraphrased questions whose article was right there. Judging the finished answer is easier:
# does it say something concrete, or only that the documents don't say? The same sentence ("the university
# announces the exam dates") answers "when are finals held?" and dodges "what date do this term's finals
# start?", so the check has to look at what the question asks for, not only at the answer.
CHECK = """Bir öğrenci şunu sordu: {question}
Verilen cevap: {answer}

Cevap, sorulan şeyi veriyor mu?
- Soru genel bir kuralı soruyorsa (ne zaman, kim belirler, nasıl yapılır, zorunlu mu), bir kural söylemek yeterlidir;
  "X tarafından belirlenir" ya da "duyurulan zamanlarda yapılır" da bir cevaptır: SOMUT.
- Soru belirli bir bilgi istiyorsa (belirli bir tarih, bir tutar, bir kişinin adı) ve cevap bunu vermeyip sadece
  kimin belirlediğini ya da duyurduğunu söylüyorsa: YOK.
- Cevap sadece mevzuatta bir hüküm bulunmadığını ya da bilginin belirtilmediğini söylüyorsa: YOK.
Tek kelime yaz: SOMUT ya da YOK."""


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
    # The model copies the relevant sentences first and answers from them; only the answer is shown (the cited
    # articles are shown anyway). Answering directly, the 7B model misread numbers in articles it had in front of it.
    text = llm.generate(build_prompt(question, chunks), model=model, num_predict=1000)
    *quote, ans = ANSWER_RE.split(text)
    ans = ans.strip()
    if ans.strip('"').startswith(NOT_FOUND):
        return NOT_FOUND, []
    if not quote and ALINTI_RE.search(ans):  # cut off before CEVAP (a quote that looped or copied a whole article)
        return NOT_FOUND, []
    check = llm.generate(CHECK.format(question=question, answer=ans), model=model, num_predict=5)
    if check.strip().strip("*").upper().startswith("YOK"):
        return NOT_FOUND, []
    ans, cites = parse_citations(ans, chunks)
    if not cites and quote:  # some models cite in the quote only; the quoted articles are still the sources
        cites = parse_citations(quote[0], chunks)[1]
    return ans, cites
