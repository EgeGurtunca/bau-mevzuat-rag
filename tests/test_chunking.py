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
