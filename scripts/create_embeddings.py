from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.rag.embedding import load_chunked_documents
from app.rag.vector_store import get_embedding_model


def main():
    documents = load_chunked_documents()
    texts = [document.page_content for document in documents]
    embeddings = get_embedding_model().embed_documents(texts)
    print(f"Created {len(embeddings)} embeddings from {len(documents)} chunks")


if __name__ == "__main__":
    main()
