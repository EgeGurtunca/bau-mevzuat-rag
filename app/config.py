import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
EMBED_MODEL = os.getenv("EMBED_MODEL", "gemini-embedding-001")
EMBED_DIM = 768
ANSWER_MODEL = os.getenv("ANSWER_MODEL", "gemini-2.5-flash")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "gemini-2.5-flash-lite")
COLLECTION = "bau_mevzuat"
RAW_DIR = ROOT / "data" / "raw"
CHUNKS_PATH = ROOT / "data" / "index" / "chunks.jsonl"
QDRANT_PATH = str(ROOT / "data" / "index" / "qdrant")
QDRANT_URL = os.getenv("QDRANT_URL")  # None -> Qdrant local mode at QDRANT_PATH


def require_api_key() -> None:
    if not GEMINI_API_KEY:
        raise SystemExit("GEMINI_API_KEY is not set. Copy .env.example to .env and fill it in.")
