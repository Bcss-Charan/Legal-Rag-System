from pathlib import Path
import logging

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from app.config import (
    EMBEDDING_MODEL,
    FAISS_INDEX_PATH
)

from app.rag.embedding import load_chunked_documents
from app.rag.exceptions import RAGSetupError


logger = logging.getLogger(__name__)

_embedding_model = None
_vector_store = None


def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL
        )
    return _embedding_model


def create_faiss_index():
    global _vector_store

    documents = load_chunked_documents()

    vector_store = FAISS.from_documents(
        documents,
        get_embedding_model()
    )

    # Create directory if not exists
    Path(FAISS_INDEX_PATH).mkdir(
        parents=True,
        exist_ok=True
    )

    # Save FAISS index
    vector_store.save_local(
        str(FAISS_INDEX_PATH)
    )
    _vector_store = vector_store

    logger.info("FAISS index created with %s chunks at %s", len(documents), FAISS_INDEX_PATH)
    print(f"FAISS Index Created Successfully with {len(documents)} chunks")


def load_faiss_index():
    global _vector_store
    if _vector_store is not None:
        return _vector_store

    index_file = Path(FAISS_INDEX_PATH) / "index.faiss"
    metadata_file = Path(FAISS_INDEX_PATH) / "index.pkl"

    if not index_file.exists() or not metadata_file.exists():
        raise RAGSetupError(
            "FAISS index is missing. Run `python scripts/build_faiss.py` before using /ask."
        )

    _vector_store = FAISS.load_local(
        str(FAISS_INDEX_PATH),
        get_embedding_model(),
        allow_dangerous_deserialization=True
    )

    return _vector_store


def similarity_search(query, k=5, metadata_filter=None):
    vector_store = load_faiss_index()
    search_kwargs = {
        "query": query,
        "k": k,
    }

    if metadata_filter:
        search_kwargs["filter"] = metadata_filter

    return vector_store.similarity_search_with_score(**search_kwargs)
