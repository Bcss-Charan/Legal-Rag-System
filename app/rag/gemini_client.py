import logging
import time

from google import genai
from google.genai import types

from app.config import (
    GEMINI_API_KEY,
    GEMINI_FALLBACK_MODEL,
    GEMINI_MAX_RETRIES,
    GEMINI_MAX_OUTPUT_TOKENS,
    GEMINI_MODEL,
    GEMINI_TEMPERATURE,
)
from app.rag.exceptions import LLMConnectionError, RAGSetupError


logger = logging.getLogger(__name__)

_client = None


def is_retryable_error(exc):
    message = str(exc).lower()
    return (
        "503" in message
        or "unavailable" in message
        or "high demand" in message
        or "429" in message
        or "resource_exhausted" in message
    )


def get_gemini_client():
    global _client
    if not GEMINI_API_KEY:
        raise RAGSetupError("GEMINI_API_KEY is missing. Add it to .env.")

    if _client is None:
        _client = genai.Client(api_key=GEMINI_API_KEY)

    return _client


def generate_legal_answer(question, context):
    prompt = f"""
You are a legal AI assistant for Indian legal materials.

Generate a clean, structured legal response.

Rules:
- Use ONLY the retrieved context.
- Do NOT use outside knowledge, assumptions, commentary, or legal advice.
- Do NOT dump the entire section text.
- Summarize clearly in plain English.
- Avoid repetition.
- Do not invent conditions, exceptions, punishments, illustrations, or consequences.
- If the retrieved context does not contain the answer, respond exactly:
Information not found in retrieved legal context.
- If punishment is not mentioned, say exactly:
No punishment specified in this section.
- If the question asks for a specific subsection or clause such as "(a)",
  "(b)", "subsection (1)", or "clause (a)", focus the answer on that exact
  part and include its condition and punishment when present in context.
- When subsections or clauses are present in retrieved context, explain the
  relevant ones under Key Conditions or Punishment instead of merging them into
  one generic summary.
- Keep response concise and readable.

Answer format:
Use these headings exactly:

{{LAW NAME}} Section {{SECTION NUMBER}}

Section Title

{{section title}}

Summary

{{2-4 plain English sentences based only on the retrieved context.}}

Key Conditions

{{bullet points for conditions explicitly present in the context. If none are present, write "No specific conditions mentioned in retrieved context."}}

Exceptions

{{bullet points for exceptions explicitly present in the context. If none are present, write "No specific exceptions mentioned in retrieved context."}}

Punishment

{{punishment explicitly stated in the context. If none is stated, write "No specific punishment mentioned in this section."}}

Source

{{law name}}, Section {{section number}}. Include retrieved document id in parentheses.

Retrieved Context:
{context}

Question:
{question}

Answer:
""".strip()

    models_to_try = [GEMINI_MODEL]
    if GEMINI_FALLBACK_MODEL and GEMINI_FALLBACK_MODEL not in models_to_try:
        models_to_try.append(GEMINI_FALLBACK_MODEL)

    last_error = None
    for model in models_to_try:
        for attempt in range(GEMINI_MAX_RETRIES + 1):
            try:
                response = get_gemini_client().models.generate_content(
                    model=model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=GEMINI_TEMPERATURE,
                        max_output_tokens=GEMINI_MAX_OUTPUT_TOKENS,
                    ),
                )
                answer = getattr(response, "text", None)
                if not answer:
                    raise LLMConnectionError("Gemini returned an empty response.")

                return answer.strip()
            except Exception as exc:
                last_error = exc
                if not is_retryable_error(exc):
                    logger.exception("Gemini generation failed with non-retryable error")
                    raise LLMConnectionError(f"Gemini generation failed: {exc}") from exc

                logger.warning(
                    "Gemini model %s failed on attempt %s/%s: %s",
                    model,
                    attempt + 1,
                    GEMINI_MAX_RETRIES + 1,
                    exc
                )
                if attempt < GEMINI_MAX_RETRIES:
                    time.sleep(1.5 * (attempt + 1))
                    continue

                break

    raise LLMConnectionError(f"Gemini generation failed after retries: {last_error}")


def generate_with_gemini(prompt, max_output_tokens=None):
    models_to_try = [GEMINI_MODEL]
    if GEMINI_FALLBACK_MODEL and GEMINI_FALLBACK_MODEL not in models_to_try:
        models_to_try.append(GEMINI_FALLBACK_MODEL)

    last_error = None
    for model in models_to_try:
        for attempt in range(GEMINI_MAX_RETRIES + 1):
            try:
                response = get_gemini_client().models.generate_content(
                    model=model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=GEMINI_TEMPERATURE,
                        max_output_tokens=max_output_tokens or GEMINI_MAX_OUTPUT_TOKENS,
                    ),
                )
                answer = getattr(response, "text", None)
                if not answer:
                    raise LLMConnectionError("Gemini returned an empty response.")

                return answer.strip()
            except Exception as exc:
                last_error = exc
                if not is_retryable_error(exc):
                    logger.exception("Gemini generation failed with non-retryable error")
                    raise LLMConnectionError(f"Gemini generation failed: {exc}") from exc

                logger.warning(
                    "Gemini model %s failed on attempt %s/%s: %s",
                    model,
                    attempt + 1,
                    GEMINI_MAX_RETRIES + 1,
                    exc
                )
                if attempt < GEMINI_MAX_RETRIES:
                    time.sleep(1.5 * (attempt + 1))
                    continue

                break

    raise LLMConnectionError(f"Gemini generation failed after retries: {last_error}")
