import re

from deep_translator import GoogleTranslator


def chunk_text(text, max_chars=4500):
    text = str(text or "")
    if len(text) <= max_chars:
        return [text]

    chunks = []
    current = []
    current_length = 0

    for part in re.split(r"(\s+)", text):
        if current_length + len(part) > max_chars and current:
            chunks.append("".join(current).strip())
            current = []
            current_length = 0

        current.append(part)
        current_length += len(part)

    if current:
        chunks.append("".join(current).strip())

    return [chunk for chunk in chunks if chunk]


def translate_to_english(raw_text):
    raw_text = str(raw_text or "").strip()
    if not raw_text:
        return ""

    try:
        translator = GoogleTranslator(source="auto", target="en")
        translated_chunks = [
            translator.translate(chunk)
            for chunk in chunk_text(raw_text)
            if chunk.strip()
        ]
        return " ".join(
            chunk
            for chunk in translated_chunks
            if chunk
        ).strip()
    except Exception:
        return raw_text
