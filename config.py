import os
from pathlib import Path
from dotenv import load_dotenv
PROJECT_ROOT = Path(__file__).resolve().parent

load_dotenv(dotenv_path=PROJECT_ROOT / ".env")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5")
GALILEO_API_KEY = os.getenv("GALILEO_API_KEY")
GALILEO_PROJECT = os.getenv("GALILEO_PROJECT")
GALILEO_LOG_STREAM = os.getenv("GALILEO_LOG_STREAM")
COLLECTION_NAME = "Rama"
# Qdrant Cloud Inference: embeddings are generated server-side (no local model, no external API).
DENSE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SPARSE_MODEL = "Qdrant/bm25"
DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"
EMBEDDING_SIZE = 384  # all-MiniLM-L6-v2 output dimension
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200
