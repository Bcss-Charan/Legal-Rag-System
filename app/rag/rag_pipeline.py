from app.rag.gemini_client import generate_legal_answer
from app.rag.retrieval import retrieve_documents


NOT_FOUND_MESSAGE = "Information not found in retrieved legal context."


def format_context(documents):
    context_parts = []

    for index, item in enumerate(documents, start=1):
        doc = item["document"]
        metadata = doc.metadata

        context_parts.append(
            f"""
[Retrieved Document D{index}]
Source: {item.get("source", "")}
Similarity Score: {item["score"]}
Law Name: {metadata.get("law_name", "")}
Section Number: {metadata.get("section_number", "")}
Section Title: {metadata.get("section_title", "")}
Chapter: {metadata.get("chapter", "")}
Chapter Title: {metadata.get("chapter_title", "")}

{doc.page_content}
""".strip()
        )

    return "\n\n---\n\n".join(context_parts)


def retrieved_document_payload(item):
    doc = item["document"]
    metadata = doc.metadata

    return {
        "section_number": metadata.get("section_number"),
        "section_title": metadata.get("section_title"),
        "law_name": metadata.get("law_name"),
        "chunk_id": metadata.get("chunk_id"),
        "score": item["score"],
        "source": item.get("source"),
        "preview": doc.page_content[:300],
    }


def rag_pipeline(question, top_k=5):
    retrieved_docs = retrieve_documents(question, k=top_k)
    if not retrieved_docs:
        return {
            "question": question,
            "answer": NOT_FOUND_MESSAGE,
            "retrieved_documents": []
        }

    context = format_context(retrieved_docs)
    answer = generate_legal_answer(
        question=question,
        context=context
    )

    return {
        "question": question,
        "answer": answer,
        "retrieved_documents": [
            retrieved_document_payload(item)
            for item in retrieved_docs
        ]
    }
