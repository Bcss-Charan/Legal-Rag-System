import logging

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import CHUNK_OVERLAP, CHUNK_SIZE
from app.database import get_all_documents
from app.rag.exceptions import RAGSetupError


logger = logging.getLogger(__name__)


def create_combined_text(doc):
    subsections_text = []

    for subsection in doc.get("subsections", []):
        subsection_number = subsection.get("subsection_number", "")
        subsection_text = subsection.get("text", "")
        subsections_text.append(f"{subsection_number} {subsection_text}".strip())

        for clause in subsection.get("clauses", []):
            clause_number = clause.get("clause_number", "")
            clause_text = clause.get("text", "")
            subsections_text.append(f"  {clause_number} {clause_text}".strip())

    for clause in doc.get("clauses", []):
        clause_number = clause.get("clause_number", "")
        clause_text = clause.get("text", "")
        subsections_text.append(f"{clause_number} {clause_text}".strip())

    return f"""
Law Name: {doc.get('law_name', '')}
Chapter: {doc.get('chapter', '')}
Chapter Title: {doc.get('chapter_title', '')}
Section Number: {doc.get('section_number', '')}
Section Title: {doc.get('section_title', '')}

Section Text:
{doc.get('section_text', '')}

Subsections and Clauses:
{chr(10).join(subsections_text)}

Punishment:
{doc.get('punishment', '')}
""".strip()


def document_metadata(doc, source_index=None):
    metadata = {
        "law_name": doc.get("law_name"),
        "section_number": str(doc.get("section_number", "")),
        "section_title": doc.get("section_title"),
        "chapter": doc.get("chapter"),
        "chapter_title": doc.get("chapter_title"),
        "mongo_id": str(doc.get("_id", "")),
    }

    if source_index is not None:
        metadata["source_index"] = source_index

    return metadata


def mongo_doc_to_document(doc, source_index=None):
    return Document(
        page_content=create_combined_text(doc),
        metadata=document_metadata(doc, source_index=source_index)
    )


def load_documents():
    mongo_docs = get_all_documents()
    if not mongo_docs:
        raise RAGSetupError(
            "No legal documents found in MongoDB. Check DATABASE_NAME and COLLECTION_NAME in .env."
        )

    documents = []
    for doc_index, doc in enumerate(mongo_docs):
        text = create_combined_text(doc)
        if not text:
            continue

        documents.append(
            Document(
                page_content=text,
                metadata=document_metadata(doc, source_index=doc_index)
            )
        )

    if not documents:
        raise RAGSetupError("MongoDB documents did not contain indexable legal text.")

    logger.info("Loaded %s legal documents from MongoDB", len(documents))
    return documents


def load_chunked_documents():
    documents = load_documents()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=[
            "\nSection Text:",
            "\nSubsections and Clauses:",
            "\nPunishment:",
            "\n\n",
            "\n",
            ". ",
            " ",
            ""
        ]
    )
    chunks = splitter.split_documents(documents)

    for chunk_index, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = chunk_index

    if not chunks:
        raise RAGSetupError("No text chunks were created from MongoDB documents.")

    logger.info("Created %s chunks from %s legal documents", len(chunks), len(documents))
    return chunks
