import pytest

from app import llm


class Boom(Exception):
    code = 429


def test_retry_succeeds_after_transient_errors(monkeypatch):
    monkeypatch.setattr(llm.time, "sleep", lambda s: None)
    calls = []

    def fn():
        calls.append(1)
        if len(calls) < 3:
            raise Boom()
        return "ok"

    assert llm.retry(fn) == "ok"
    assert len(calls) == 3


def test_retry_gives_up(monkeypatch):
    monkeypatch.setattr(llm.time, "sleep", lambda s: None)

    def fn():
        raise Boom()

    with pytest.raises(Boom):
        llm.retry(fn)


def test_retry_does_not_retry_non_transient(monkeypatch):
    calls = []

    def fn():
        calls.append(1)
        raise ValueError("bad request")

    with pytest.raises(ValueError):
        llm.retry(fn)
    assert len(calls) == 1
