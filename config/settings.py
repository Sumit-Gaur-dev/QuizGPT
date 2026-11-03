"""
Central configuration settings for Quiz RAG.
Stores model names, limits, and chunking parameters in one place.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load local environment variables from .env if present
load_dotenv()

# Base paths
BASE_DIR = Path(__file__).parent.parent.resolve()
STORAGE_DIR = BASE_DIR / "storage"
CHROMA_PERSIST_DIR = BASE_DIR / os.getenv("CHROMA_PERSIST_DIR", "storage/chroma_db")

# Upload and Document Constraints
MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "10"))
MAX_FILES_PER_SESSION = int(os.getenv("MAX_FILES_PER_SESSION", "5"))
ALLOWED_EXTENSIONS = [".pdf"]

# Modular Embedding Provider Configuration
# Supported providers: "huggingface", "gemini", "openai"
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "huggingface").lower().strip()

# Per-provider default models
HUGGINGFACE_EMBEDDING_MODEL = os.getenv(
    "HUGGINGFACE_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)
GEMINI_EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "text-embedding-004")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

# LLM Generation Settings (Phase 6)
GOOGLE_API_KEY = (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY") or "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip()

# Chunking Defaults (measured in characters)
CHUNK_SIZE_CHARS = int(os.getenv("CHUNK_SIZE_CHARS", "1000"))
CHUNK_OVERLAP_CHARS = int(os.getenv("CHUNK_OVERLAP_CHARS", "150"))
RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "5"))


def has_valid_gemini_key() -> bool:
    """Return True if a valid (non-placeholder) Gemini API key is configured."""
    key = GOOGLE_API_KEY
    if not key or key in ("your_gemini_api_key_here", "your_api_key_here"):
        return False
    return True
