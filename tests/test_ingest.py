from pathlib import Path

from app.ingest import build_chunks, read_text


def test_build_chunks_from_txt_and_skips_broken_files(tmp_path: Path):
    (tmp_path / "Test Yonetmeligi.txt").write_text(
        "Amaç\nMADDE 1 – (1) Amaç budur.\nMADDE 2 – (1) Kapsam budur.", encoding="utf-8")
    (tmp_path / "broken.pdf").write_bytes(b"not a pdf")
    chunks = build_chunks(tmp_path)
    assert [c.id for c in chunks] == ["test-yonetmeligi:1", "test-yonetmeligi:2"]
    assert chunks[0].doc_title == "Test Yonetmeligi"


def test_read_html_one_line_per_block_with_collapsed_whitespace(tmp_path: Path):
    p = tmp_path / "x.html"
    p.write_text(
        "<html><body><p><b><span>Dersten\nçekilme</span></b></p>"
        "<p><span>MADDE 25\n–</span><span> (1) Metin.</span></p><script>x()</script></body></html>",
        encoding="utf-8")
    assert read_text(p) == "Dersten çekilme\nMADDE 25 – (1) Metin."
