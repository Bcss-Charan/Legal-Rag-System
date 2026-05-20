from pydantic import BaseModel, Field

from app.config import DEFAULT_TOP_K, MAX_TOP_K


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int = Field(default=DEFAULT_TOP_K, ge=1, le=MAX_TOP_K)
