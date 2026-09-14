import urllib.error

import pytest

from app import llm


def http_error(code: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError("http://x", code, "err", {}, None)


def test_retry_succeeds_after_transient_errors(monkeypatch):
    monkeypatch.setattr(llm.time, "sleep", lambda s: None)
    calls = []

    def fn():
        calls.append(1)
        if len(calls) < 3:
            raise http_error(503)
        return "ok"

    assert llm.retry(fn) == "ok"
    assert len(calls) == 3


def test_retry_gives_up(monkeypatch):
    monkeypatch.setattr(llm.time, "sleep", lambda s: None)

    def fn():
        raise urllib.error.URLError("connection refused")

    with pytest.raises(urllib.error.URLError):
        llm.retry(fn)


def test_retry_does_not_retry_client_errors(monkeypatch):
    calls = []

    def fn():
        calls.append(1)
        raise http_error(404)

    with pytest.raises(urllib.error.HTTPError):
        llm.retry(fn)
    assert len(calls) == 1


def test_embed_batches_and_flattens(monkeypatch):
    seen = []

    def fake_post(path, body, timeout=300):
        seen.append(len(body["input"]))
        return {"embeddings": [[0.0] * 3 for _ in body["input"]]}

    monkeypatch.setattr(llm, "_post", fake_post)
    out = llm.embed([f"t{i}" for i in range(70)])
    assert len(out) == 70 and seen == [32, 32, 6]


def test_generate_returns_stripped_response(monkeypatch):
    monkeypatch.setattr(llm, "_post", lambda path, body, timeout=300: {"response": "  cevap \n"})
    assert llm.generate("soru") == "cevap"
