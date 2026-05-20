import logging
import re

from app.config import SIMILARITY_SCORE_THRESHOLD
from app.database import find_section
from app.rag.embedding import mongo_doc_to_document
from app.rag.vector_store import similarity_search


logger = logging.getLogger(__name__)

LAW_ALIASES = {
    "bns": "BNS",
    "bharatiya nyaya sanhita": "BNS",
    "bnss": "BNSS",
    "bharatiya nagarik suraksha sanhita": "BNSS",
    "bsa": "BSA",
    "bharatiya sakshya adhiniyam": "BSA",
}


def extract_law_name(query):
    normalized_query = query.lower()

    for alias, law_name in LAW_ALIASES.items():
        if re.search(rf"\b{re.escape(alias)}\b", normalized_query):
            return law_name

    return None


def extract_section_number(query):
    match = re.search(
        r"\b(?:section|sec\.?|s\.)\s+(\d+[a-z]?)\b",
        query,
        flags=re.IGNORECASE
    )
    if match:
        return match.group(1).upper()

    return None


def asks_for_section(query):
    return bool(
        re.search(r"\b(?:section|sec\.?|s\.)\b", query, flags=re.IGNORECASE)
    )


def format_result(document, score, source):
    return {
        "document": document,
        "score": float(score),
        "source": source,
    }


def retrieve_exact_section(query):
    section_number = extract_section_number(query)
    if not section_number:
        return []

    law_name = extract_law_name(query)
    doc = find_section(law_name, section_number)
    if not doc:
        logger.info(
            "No exact section found for law=%s section=%s",
            law_name,
            section_number
        )
        return []

    document = mongo_doc_to_document(doc)
    document.metadata["chunk_id"] = None
    return [format_result(document, 0.0, "mongodb_exact")]


def retrieve_documents(query, k=5):
    law_name = extract_law_name(query)
    section_number = extract_section_number(query)
    exact_results = retrieve_exact_section(query)
    if exact_results:
        return exact_results

    # If the user asked for an exact law section, do not answer from a different
    # law via semantic fallback. This prevents BNS/BNSS/BSA cross-contamination.
    if law_name and section_number:
        return []

    if law_name and asks_for_section(query):
        return []

    metadata_filter = {"law_name": law_name} if law_name else None
    results = similarity_search(query=query, k=k, metadata_filter=metadata_filter)

    filtered_results = [
        format_result(document, score, "faiss")
        for document, score in results
        if float(score) <= SIMILARITY_SCORE_THRESHOLD
    ]

    logger.info(
        "Retrieved %s/%s documents for query=%r",
        len(filtered_results),
        len(results),
        query
    )
    return filtered_results
