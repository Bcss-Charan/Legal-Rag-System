from pathlib import Path
import os
from urllib.parse import quote_plus

from dotenv import load_dotenv

# Base Directory
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

def _get_mongo_uri():
    uri = (
        os.getenv("MONGO_URI")
        or os.getenv("MONGODB_URI")
        or "mongodb://localhost:27017/"
    ).strip()
    password = os.getenv("MONGO_PWD") or os.getenv("MONGO_PASSWORD")

    if password:
        encoded_password = quote_plus(password.strip())
        for placeholder in ("<db_password>", "<password>", "${MONGO_PWD}", "$MONGO_PWD"):
            uri = uri.replace(placeholder, encoded_password)

    return uri


# MongoDB
MONGO_URI = _get_mongo_uri()
DATABASE_NAME = os.getenv(
    "DATABASE_NAME",
    os.getenv("MONGODB_DB_NAME", "legal_database")
)
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "laws_sections")

# Embedding Model
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2"
)

# FAISS Index Path
FAISS_INDEX_PATH = BASE_DIR / os.getenv("FAISS_INDEX_PATH", "faiss_index")

# Chunking
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "900"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))

# Retrieval
DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", "5"))
MAX_TOP_K = int(os.getenv("MAX_TOP_K", "20"))
SIMILARITY_SCORE_THRESHOLD = float(
    os.getenv("SIMILARITY_SCORE_THRESHOLD", "1.35")
)

# Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_FALLBACK_MODEL = os.getenv("GEMINI_FALLBACK_MODEL", "gemini-2.5-flash-lite")
GEMINI_TEMPERATURE = float(os.getenv("GEMINI_TEMPERATURE", "0.1"))
GEMINI_MAX_OUTPUT_TOKENS = int(os.getenv("GEMINI_MAX_OUTPUT_TOKENS", "800"))
GEMINI_MAX_RETRIES = int(os.getenv("GEMINI_MAX_RETRIES", "2"))
