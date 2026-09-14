import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
# 127.0.0.1, not localhost: on Windows "localhost" tries ::1 first and Ollama only listens on IPv4 -> ~2 s per request
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
EMBED_MODEL = os.getenv("EMBED_MODEL", "bge-m3")
EMBED_DIM = 1024
ANSWER_MODEL = os.getenv("ANSWER_MODEL", "qwen2.5:7b")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", ANSWER_MODEL)
COLLECTION = "bau_mevzuat"
RAW_DIR = ROOT / "data" / "raw"
CHUNKS_PATH = ROOT / "data" / "index" / "chunks.jsonl"
QDRANT_PATH = str(ROOT / "data" / "index" / "qdrant")
QDRANT_URL = os.getenv("QDRANT_URL")  # None -> Qdrant local mode at QDRANT_PATH
