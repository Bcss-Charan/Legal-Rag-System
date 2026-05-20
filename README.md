# Legal RAG Project

Production-ready lightweight Legal RAG API using MongoDB for legal source data, sentence-transformer embeddings, FAISS vector search, and Gemini API for grounded answer generation.

## Architecture

```text
User Query
  -> exact law/section parser
  -> MongoDB exact section lookup when possible
  -> query embedding
  -> FAISS vector similarity search
  -> retrieved legal context with scores
  -> Gemini API grounded prompt
  -> legal answer or "Information not found in retrieved legal context."
```

## Folder Structure

```text
app/
  app.py                 FastAPI app factory
  main.py                Compatibility entry point
  config.py              Environment-based settings
  database.py            MongoDB connection and exact section lookup
  api/routes.py          API endpoints
  models/                Request and response schemas
  rag/
    embedding.py         MongoDB document formatting and chunking
    vector_store.py      FAISS index creation and loading
    retrieval.py         Exact lookup + vector retrieval
    gemini_client.py     Gemini API client
    rag_pipeline.py      End-to-end RAG orchestration
scripts/
  build_faiss.py         Rebuild vector index
  create_embeddings.py   Embedding smoke test
  test_rag.py            Retrieval + grounded answer validation
```

## Setup

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
```

Edit `.env`:

```env
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-2.5-flash
GEMINI_FALLBACK_MODEL=gemini-2.5-flash-lite
```

Google's current Gemini SDK pattern uses `from google import genai` and `client.models.generate_content(...)`. The default model is `gemini-2.5-flash`; switch `GEMINI_MODEL` to a Pro model if you prefer higher reasoning quality over speed/cost.
If Gemini returns a temporary high-demand `503`, the app retries and then falls back to `GEMINI_FALLBACK_MODEL`.

## Build FAISS Index

Start MongoDB first and make sure `.env` points to the database and collection containing BNS, BNSS, and BSA sections.

```powershell
python scripts/build_faiss.py
```

This reads MongoDB, chunks legal sections, creates embeddings, and stores the FAISS index in `faiss_index/`.

## Run API

```powershell
python run.py
```

API docs:

```text
http://127.0.0.1:8000/docs
```

Example request:

```json
{
  "question": "What is section 30 in BSA?",
  "top_k": 5
}
```

Shareable browser URL:

```text
http://127.0.0.1:8000/ask/what%20is%20section%2030%20in%20BSA?top_k=5
```

Formatted browser page:

```text
http://127.0.0.1:8000/view/what%20is%20section%2030%20in%20BSA?top_k=5
```

## Test RAG Quality

```powershell
python scripts/test_rag.py
```

The test script prints:

- retrieved law name, section number, title, source, and similarity score
- Gemini's final grounded answer
- returned documents included in the API response

## Validate FIR Sections

Use this endpoint to check whether applied FIR sections match the incident facts:

```text
POST http://127.0.0.1:8000/validate-fir?top_k=5
```

Send the FIR JSON as the request body. The validator retrieves:

- sections already applied in the FIR
- possible sections listed in the FIR payload
- additional semantically similar legal sections from FAISS

Then Gemini compares the FIR facts only against the retrieved legal context.

Smoke test:

```powershell
python scripts/test_fir_validation.py
```

## Best Practices

- Rebuild FAISS whenever MongoDB legal data changes.
- Keep exact section lookup before vector search for legal section queries.
- Use law filters so BNS, BNSS, and BSA do not contaminate each other.
- Keep Gemini temperature low, such as `0.1`, for legal answers.
- Return the required fallback sentence when retrieved context is missing.
- Log retrieval counts and generation failures for debugging.
- Add curated evaluation queries for each act and compare expected sections.
