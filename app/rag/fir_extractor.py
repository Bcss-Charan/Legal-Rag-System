import json
import re

from app.rag.gemini_client import generate_with_gemini
from app.rag.translation import translate_to_english


LAW_NAMES = {"BNS", "BNSS", "BSA"}


def collect_text(value):
    if value is None:
        return ""

    if isinstance(value, str):
        return value

    if isinstance(value, dict):
        return " ".join(collect_text(item) for item in value.values())

    if isinstance(value, list):
        return " ".join(collect_text(item) for item in value)

    return str(value)


def normalize_law(law):
    law = str(law or "BNS").upper().strip()
    return law if law in LAW_NAMES else "BNS"


def normalize_section(section):
    return {
        "law": normalize_law(section.get("law") or section.get("act")),
        "section": str(section.get("section") or section.get("section_number") or "").strip(),
        "clause": section.get("clause"),
        "title": section.get("title") or section.get("section_name") or section.get("section_title"),
    }


def parse_json_response(text):
    cleaned = text.strip()
    fence_match = re.search(
        r"```(?:json)?\s*(.*?)\s*```",
        cleaned,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if fence_match:
        cleaned = fence_match.group(1).strip()

    return json.loads(cleaned)


def extract_with_regex(raw_text):
    references = []

    def add_series(series, law):
        number_pattern = re.compile(r"\b(\d+[A-Za-z]?)(?:\s*\(([a-zA-Z0-9]+)\))?")
        for number_match in number_pattern.finditer(series):
            section_number, clause = number_match.groups()
            references.append({
                "law": normalize_law(law),
                "section": section_number.upper(),
                "clause": f"({clause})" if clause else None,
                "title": None,
            })

    us_pattern = re.compile(
        r"\b(?:u/s|under\s+section)\s*"
        r"((?:\d+[A-Za-z]?(?:\s*\([a-zA-Z0-9]+\))?\s*(?:,|and|&)?\s*)+)"
        r"(?:\s*(?:of|under|&)?\s*(BNS|BNSS|BSA))?",
        flags=re.IGNORECASE,
    )
    for match in us_pattern.finditer(raw_text):
        series, law = match.groups()
        add_series(series, law or "BNS")

    law_series_pattern = re.compile(
        r"\b(BNS|BNSS|BSA)\s*(?:sections?|secs?\.?|s\.)?\s*"
        r"((?:\d+[A-Za-z]?(?:\s*\([a-zA-Z0-9]+\))?\s*(?:,|and|&)?\s*)+)",
        flags=re.IGNORECASE,
    )
    for match in law_series_pattern.finditer(raw_text):
        law, series = match.groups()
        add_series(series, law)

    default_bns_pattern = re.compile(
        r"\b(?:sections?|secs?\.?|s\.)\s*"
        r"((?:\d+[A-Za-z]?(?:\s*\([a-zA-Z0-9]+\))?\s*(?:,|and|&)?\s*)+)",
        flags=re.IGNORECASE,
    )
    for match in default_bns_pattern.finditer(raw_text):
        add_series(match.group(1), "BNS")

    return dedupe_sections(references)


def dedupe_sections(sections):
    seen = set()
    deduped = []

    for section in sections:
        normalized = normalize_section(section)
        key = (
            normalized.get("law"),
            normalized.get("section"),
            normalized.get("clause"),
        )
        if not normalized.get("section") or key in seen:
            continue
        seen.add(key)
        deduped.append(normalized)

    return deduped


def fallback_extraction(fir_data, translated_text=None):
    raw_text = collect_text(fir_data)
    searchable_text = " ".join(
        item
        for item in [raw_text, translated_text]
        if item
    )
    return {
        "sections": extract_with_regex(searchable_text),
        "crime_facts": [],
        "normalized_english_text": translated_text or raw_text,
    }


def extract_fir_details(fir_data):
    raw_text = collect_text(fir_data)
    translated_text = translate_to_english(raw_text)
    prompt = f"""
You are an FIR/petition extraction assistant.

Task:
Read the English FIR/petition text below.
Extract ONLY these two things:
1. Legal sections mentioned.
2. Crime facts normalized into concise English.

Rules:
- Do not validate the case.
- Do not suggest sections.
- Do not add legal conclusions beyond the facts.
- Preserve BNS, BNSS, and BSA law names when mentioned.
- If only a section number is mentioned without a law, use BNS.
- Handle police endorsement formats like "U/s 125(a) & BNS".
- The text has already been translated to English before this prompt.
- Return ONLY valid JSON. Do not use markdown fences.

Output shape:
{{
  "sections": [
    {{
      "law": "BNS",
      "section": "125",
      "clause": "(a)",
      "title": null
    }}
  ],
  "crime_facts": [
    "rash and negligent driving",
    "lorry hit two-wheeler",
    "victim suffered serious leg injury"
  ],
  "normalized_english_text": ""
}}

English FIR/Petition Text:
{translated_text}
""".strip()

    try:
        extracted = parse_json_response(
            generate_with_gemini(prompt, max_output_tokens=1600)
        )
    except Exception:
        return fallback_extraction(fir_data, translated_text=translated_text)

    sections = dedupe_sections(extracted.get("sections", []))
    crime_facts = [
        str(item).strip()
        for item in extracted.get("crime_facts", [])
        if str(item).strip()
    ]
    normalized_text = str(extracted.get("normalized_english_text") or "").strip()

    fallback = fallback_extraction(fir_data, translated_text=translated_text)
    if not sections:
        sections = fallback["sections"]
    if not normalized_text:
        normalized_text = " ".join(crime_facts) or translated_text or fallback["normalized_english_text"]

    return {
        "sections": sections,
        "crime_facts": crime_facts,
        "normalized_english_text": normalized_text,
    }
