import os
from pathlib import Path
from dotenv import load_dotenv
PROJECT_ROOT = Path(__file__).resolve().parent

load_dotenv(dotenv_path=PROJECT_ROOT / ".env")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5")
GALILEO_API_KEY = os.getenv("GALILEO_API_KEY")
GALILEO_PROJECT = os.getenv("GALILEO_PROJECT")
GALILEO_LOG_STREAM = os.getenv("GALILEO_LOG_STREAM")
GALILEO_CONSOLE_URL = os.getenv("GALILEO_CONSOLE_URL")
EMBEDDING_MODEL = "models/gemini-embedding-001"
COLLECTION_NAME = "my_documents"
EMBEDDING_SIZE = 3072
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200
