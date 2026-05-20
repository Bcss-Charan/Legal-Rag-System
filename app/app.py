import logging

from fastapi import FastAPI

from app.api.routes import router


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)

app = FastAPI(
    title="Legal RAG API",
    version="2.0.0",
    description="MongoDB + FAISS retrieval pipeline with Gemini grounded generation.",
)
app.include_router(router)
