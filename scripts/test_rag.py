from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.rag.rag_pipeline import rag_pipeline
from app.rag.retrieval import retrieve_documents
from app.rag.exceptions import LLMConnectionError, RAGSetupError


TEST_QUERIES = [
    "What is section 30 in BSA?",
]


def print_retrieval(query, top_k):
    print(f"\nQuery: {query}")
    print("Retrieval:")
    retrieved = retrieve_documents(query, k=top_k)

    if not retrieved:
        print("  No documents retrieved.")
        return

    for index, item in enumerate(retrieved, start=1):
        metadata = item["document"].metadata
        print(
            "  "
            f"{index}. {metadata.get('law_name')} "
            f"Section {metadata.get('section_number')} - "
            f"{metadata.get('section_title')} "
            f"(score={item['score']}, source={item.get('source')})"
        )


def print_answer(query, top_k):
    try:
        result = rag_pipeline(query, top_k=top_k)
    except (LLMConnectionError, RAGSetupError) as exc:
        print("\nGemini Answer:")
        print(f"  Skipped: {exc}")
        return

    print("\nGemini Answer:")
    print(result["answer"])
    print("\nReturned Documents:")
    for index, doc in enumerate(result["retrieved_documents"], start=1):
        print(
            "  "
            f"{index}. {doc['law_name']} Section {doc['section_number']} "
            f"(score={doc['score']})"
        )


def main():
    top_k = 5
    for query in TEST_QUERIES:
        print_retrieval(query, top_k=top_k)
        print_answer(query, top_k=top_k)


if __name__ == "__main__":
    main()
