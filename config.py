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
COMPONENT_IMAGES_DIR = PROJECT_ROOT / "data" / "components"
COMPONENT_COLLECTION_NAME = os.getenv("COMPONENT_COLLECTION_NAME", "Rama_components")
CLIP_MODEL_NAME = os.getenv("CLIP_MODEL_NAME", "clip-ViT-B-32")
CLIP_VECTOR_DIM = 512
VISION_MODEL = os.getenv("VISION_MODEL", ANTHROPIC_MODEL)
VISION_CAPTION_MODEL = os.getenv("VISION_CAPTION_MODEL", VISION_MODEL)
VISION_RERANK_MODEL = os.getenv("VISION_RERANK_MODEL", VISION_MODEL)
COMPONENT_RERANK_POOL = int(os.getenv("COMPONENT_RERANK_POOL", "3"))
COMPONENT_CLIP_CANDIDATES = int(os.getenv("COMPONENT_CLIP_CANDIDATES", "8"))
COMPONENT_IMAGE_PUBLIC_BASE_URL = os.getenv("COMPONENT_IMAGE_PUBLIC_BASE_URL", "")
CATALOG_IMAGE_TEMP_UPLOAD = os.getenv("CATALOG_IMAGE_TEMP_UPLOAD", "true").strip().lower() in {
    "1", "true", "yes", "on",
}

# Email (Brevo)
BREVO_API_KEY = os.getenv("BREVO_API_KEY", "")
BREVO_FROM_EMAIL = os.getenv("BREVO_FROM_EMAIL", "")
BREVO_FROM_NAME = os.getenv("BREVO_FROM_NAME", "Product Distribution Team")

# Apollo
APOLLO_API_KEY = os.getenv("APOLLO_API_KEY", "")

# Galileo UI
GALILEO_DEBUG_URLS = os.getenv("GALILEO_DEBUG_URLS", "false").strip().lower() in {"1", "true", "yes", "on"}
GALILEO_EVAL_MODE = os.getenv("GALILEO_EVAL_MODE", "false").strip().lower() in {"1", "true", "yes", "on"}

# Salesforce (re-export for non-client callers)
SALESFORCE_BACKEND = os.getenv("SALESFORCE_BACKEND", "")
