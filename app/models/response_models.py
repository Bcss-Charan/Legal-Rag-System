from pydantic import BaseModel


class RetrievedDocument(BaseModel):
    section_number: str | None = None
    section_title: str | None = None
    law_name: str | None = None
    chunk_id: int | None = None
    score: float
    source: str | None = None
    preview: str


class QueryResponse(BaseModel):
    question: str
    answer: str
    retrieved_documents: list[RetrievedDocument]
