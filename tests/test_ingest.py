from pathlib import Path

from app.ingest import build_chunks, read_text


def test_build_chunks_from_txt_and_skips_broken_files(tmp_path: Path):
    (tmp_path / "Test Yonetmeligi.txt").write_text(
        "Amaç\nMADDE 1 – (1) Amaç budur.\nMADDE 2 – (1) Kapsam budur.", encoding="utf-8")
    (tmp_path / "broken.pdf").write_bytes(b"not a pdf")
    chunks = build_chunks(tmp_path)
    assert [c.id for c in chunks] == ["test-yonetmeligi:1", "test-yonetmeligi:2"]
    assert chunks[0].doc_title == "Test Yonetmeligi"


def test_read_html_strips_tags(tmp_path: Path):
    p = tmp_path / "x.html"
    p.write_text("<html><body><p>MADDE 1 – (1) Metin.</p><script>x()</script></body></html>", encoding="utf-8")
    assert "MADDE 1" in read_text(p) and "<p>" not in read_text(p)
