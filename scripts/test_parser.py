from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.rag.retrieval import extract_law_name, extract_section_number


TEST_QUERIES = [
    "what is section 30 in bns?",
    "what is sec 30 in bns?",
    "what is sec. 30 in bns?",
    "what is s. 30 in bns?",
    "What is section 30 in BSA?",
]


def main():
    for query in TEST_QUERIES:
        print(
            f"{query} -> "
            f"law={extract_law_name(query)}, "
            f"section={extract_section_number(query)}"
        )


if __name__ == "__main__":
    main()
