from app.chunking import slugify, split

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
    assert chunks[0].doc_title == "Test Yönetmeliği"


def test_long_article_is_split_at_fikra():
    fikra = " ".join(["kelime"] * 250)
    text = "MADDE 7 – (1) " + fikra + " (2) " + fikra + " (3) " + fikra
    chunks = split(text, "Doc", "doc")
    assert [c.id for c in chunks] == ["doc:7.1", "doc:7.2"]
    assert all(c.article_no == 7 for c in chunks)
    assert chunks[1].text.startswith("(3)")


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


def test_next_heading_not_leaked_into_previous_chunk():
    chunks = split(SAMPLE, "T", "t")
    assert not chunks[0].text.rstrip().endswith("Kapsam")
    assert not chunks[1].text.rstrip().endswith("Dayanak")


def test_section_heading_not_leaked_into_previous_chunk():
    text = "MADDE 35 – (1) Son fıkra.\nBEŞİNCİ BÖLÜM\nDiploma Hakkı ve Diplomalar\nDiploma hakkı\nMADDE 36 – (1) Metin."
    chunks = split(text, "T", "t")
    assert chunks[0].text == "MADDE 35 – (1) Son fıkra."
    assert chunks[1].heading == "Diploma hakkı"


def test_gecici_and_ek_madde_are_separate_articles():
    text = "Madde 177 – Son madde.\nGeçici Madde 1 – Geçici hüküm.\nEk Madde 2 – Ek hüküm."
    chunks = split(text, "Anayasa", "anayasa")
    assert [(c.id, c.article_no, c.article_kind) for c in chunks] == [
        ("anayasa:177", 177, None), ("anayasa:g1", 1, "Geçici"), ("anayasa:e2", 2, "Ek")]
    assert chunks[0].text == "Madde 177 – Son madde."


def test_appendix_after_last_article_is_dropped():
    text = ("Madde 1 – Birinci.\nMadde 2 – İkinci.\n"
            "18/10/1982 TARİHLİ VE 2709 SAYILI KANUNA İŞLENEMEYEN HÜKÜMLER\n"
            "1- 4121 sayılı Kanunun hükmüdür.\nMadde 16 – Bu Kanunun halkoylamasına sunulması halinde.")
    chunks = split(text, "A", "a")
    assert [c.id for c in chunks] == ["a:1", "a:2"]
    assert chunks[1].text == "Madde 2 – İkinci."


def test_duplicate_article_numbers_get_distinct_ids():
    chunks = split("Madde 4 – Bir.\nMadde 4 – İki.", "A", "a")
    assert [c.id for c in chunks] == ["a:4", "a:4-dup"]


def test_long_article_without_fikra_markers_splits_on_lines():
    line = " ".join(["kelime"] * 250)
    text = "Geçici Madde 20 – " + line + "\n" + line + "\n" + line
    chunks = split(text, "A", "a")
    assert [c.id for c in chunks] == ["a:g20.1", "a:g20.2"]
    assert all(len(c.text.split()) <= 600 for c in chunks)
