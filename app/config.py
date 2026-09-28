import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
# 127.0.0.1, not localhost: on Windows "localhost" tries ::1 first and Ollama only listens on IPv4 -> ~2 s per request
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
EMBED_MODEL = os.getenv("EMBED_MODEL", "bge-m3")
EMBED_DIM = 1024
ANSWER_MODEL = os.getenv("ANSWER_MODEL", "gemma4:12b")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "gemma4:12b")  # 48/51 agreement with my hand grades, no false "correct" (eval/check_judge.py)
OLLAMA_KEEP_ALIVE = os.getenv("OLLAMA_KEEP_ALIVE", "3s")  # VRAM is freed 3s after the last call; a reload costs ~2s
RERANK_MODEL = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")  # optional, see app/rerank.py
CONTEXT_K = int(os.getenv("CONTEXT_K", "6"))  # articles given to the answer model; 3 hard-set misses sat at rank 6
COLLECTION = "bau_mevzuat"
RAW_DIR = ROOT / "data" / "raw"
CHUNKS_PATH = ROOT / "data" / "index" / "chunks.jsonl"
QDRANT_PATH = str(ROOT / "data" / "index" / "qdrant")
QDRANT_URL = os.getenv("QDRANT_URL")  # None -> Qdrant local mode at QDRANT_PATH
