from html import escape

from fastapi import APIRouter, Body, HTTPException, Query
from fastapi.responses import HTMLResponse

from app.config import DEFAULT_TOP_K, MAX_TOP_K
from app.models.request_models import QueryRequest
from app.models.response_models import QueryResponse
from app.rag.exceptions import LLMConnectionError, RAGSetupError
from app.rag.fir_validator import validate_fir
from app.rag.rag_pipeline import rag_pipeline


router = APIRouter()


@router.get("/")
def home():
    return {
        "message": "Legal RAG API Running"
    }


def run_question(question, top_k):
    try:
        result = rag_pipeline(question, top_k=top_k)
    except RAGSetupError as exc:
        raise HTTPException(
            status_code=503,
            detail=str(exc)
        ) from exc
    except LLMConnectionError as exc:
        raise HTTPException(
            status_code=503,
            detail=str(exc)
        ) from exc

    return result


@router.post("/ask", response_model=QueryResponse)
def ask_question(request: QueryRequest):
    return run_question(request.question, request.top_k)


@router.post("/validate-fir")
def validate_fir_endpoint(
    fir_text: str = Body(
        ...,
        media_type="text/plain",
        description="Paste FIR, complaint, or petition text here.",
    ),
    top_k: int = Query(default=DEFAULT_TOP_K, ge=1, le=MAX_TOP_K)
):
    try:
        return validate_fir(fir_text, top_k=top_k)
    except RAGSetupError as exc:
        raise HTTPException(
            status_code=503,
            detail=str(exc)
        ) from exc
    except LLMConnectionError as exc:
        raise HTTPException(
            status_code=503,
            detail=str(exc)
        ) from exc


@router.get("/ask/{question:path}", response_model=QueryResponse)
def ask_question_from_url(
    question: str,
    top_k: int = Query(default=DEFAULT_TOP_K, ge=1, le=MAX_TOP_K)
):
    return run_question(question, top_k)


@router.get("/view/{question:path}", response_class=HTMLResponse)
def view_answer_in_browser(
    question: str,
    top_k: int = Query(default=DEFAULT_TOP_K, ge=1, le=MAX_TOP_K)
):
    result = run_question(question, top_k)
    answer = escape(result["answer"]).replace("\n", "<br>")

    retrieved_items = []
    for item in result["retrieved_documents"]:
        retrieved_items.append(
            f"""
            <li>
                <strong>{escape(str(item.get("law_name") or ""))}
                Section {escape(str(item.get("section_number") or ""))}</strong><br>
                {escape(str(item.get("section_title") or ""))}<br>
                <small>score={escape(str(item.get("score")))} | source={escape(str(item.get("source") or ""))}</small>
            </li>
            """
        )

    retrieved_html = "\n".join(retrieved_items) or "<li>No documents retrieved.</li>"

    return f"""
    <!doctype html>
    <html>
    <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>Legal RAG Answer</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                line-height: 1.6;
                max-width: 900px;
                margin: 32px auto;
                padding: 0 18px;
                color: #1f2937;
                background: #f8fafc;
            }}
            main {{
                background: #ffffff;
                border: 1px solid #e5e7eb;
                border-radius: 8px;
                padding: 24px;
            }}
            h1 {{
                font-size: 24px;
                margin-top: 0;
            }}
            .question {{
                color: #475569;
                margin-bottom: 24px;
            }}
            .answer {{
                white-space: normal;
                font-size: 16px;
            }}
            ul {{
                padding-left: 22px;
            }}
            li {{
                margin-bottom: 12px;
            }}
        </style>
    </head>
    <body>
        <main>
            <h1>Legal RAG Answer</h1>
            <div class="question"><strong>Question:</strong> {escape(question)}</div>
            <div class="answer">{answer}</div>
            <h2>Retrieved Sources</h2>
            <ul>{retrieved_html}</ul>
        </main>
    </body>
    </html>
    """
