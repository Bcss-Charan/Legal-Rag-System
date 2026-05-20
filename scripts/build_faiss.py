from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.rag.vector_store import create_faiss_index


def main():
    try:
        create_faiss_index()
    except Exception as exc:
        print(f"Could not build FAISS index: {exc}")
        raise


if __name__ == "__main__":
    main()
