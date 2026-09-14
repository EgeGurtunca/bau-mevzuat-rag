"""FastAPI app: POST /ask, GET /health, GET / (chat page). Run: uvicorn app.api:app --reload"""
import time
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from app import config, llm
from app.answer import answer
from app.retrieve import Retriever

retriever: Retriever | None = None


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global retriever
    llm.require_ollama()
    retriever = Retriever.load()
    yield


app = FastAPI(title="bau-mevzuat-rag", lifespan=lifespan)


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    mode: Literal["dense", "bm25", "hybrid"] = "hybrid"
    k: int = Field(default=5, ge=1, le=10)


class Citation(BaseModel):
    doc_title: str
    article_no: int | None
    heading: str | None
    text: str


class AskResponse(BaseModel):
    answer: str
    citations: list[Citation]
    mode: str
    latency_ms: int


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    t0 = time.perf_counter()
    try:
        hits = retriever.search(req.question, mode=req.mode, k=req.k)
        text, cites = answer(req.question, [c for c, _ in hits])
    except Exception as e:
        raise HTTPException(503, f"Şu an yanıt üretilemiyor: {type(e).__name__}")
    return AskResponse(
        answer=text,
        citations=[Citation(**c.model_dump()) for c in cites],
        mode=req.mode,
        latency_ms=int((time.perf_counter() - t0) * 1000),
    )


@app.get("/health")
def health() -> dict:
    return {"ok": True, "chunks": len(retriever.chunks) if retriever else 0}


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(config.ROOT / "static" / "index.html")
