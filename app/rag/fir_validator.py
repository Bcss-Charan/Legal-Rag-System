import json
import logging
import re

from app.config import SIMILARITY_SCORE_THRESHOLD
from app.database import find_section
from app.rag.embedding import mongo_doc_to_document
from app.rag.fir_extractor import extract_fir_details
from app.rag.gemini_client import generate_with_gemini
from app.rag.retrieval import format_result
from app.rag.vector_store import similarity_search


logger = logging.getLogger(__name__)

LAW_NAMES = {"BNS", "BNSS", "BSA"}


def normalize_fir_input(fir_data):
    if isinstance(fir_data, str):
        return {"text": fir_data}

    if isinstance(fir_data, dict):
        return fir_data

    return {"text": collect_text(fir_data)}


def incident_description(fir_data):
    extracted = fir_data.get("_extracted_fir_details", {})
    if extracted:
        return (
            extracted.get("normalized_english_text")
            or " ".join(extracted.get("crime_facts", []))
            or ""
        )

    details = fir_data.get("incident_details", {})
    return (
        fir_data.get("incident_description")
        or fir_data.get("complaint_description")
        or fir_data.get("description")
        or details.get("description")
        or collect_crime_text(fir_data)
        or ""
    )


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


def collect_crime_text(value):
    ignored_keys = {
        "applied_sections",
        "sections_applied",
        "possible_sections",
        "offence_summary",
        "section",
        "section_number",
        "section_name",
        "section_title",
        "title",
        "act",
        "law",
        "law_name",
    }

    if value is None:
        return ""

    if isinstance(value, str):
        return value

    if isinstance(value, dict):
        return " ".join(
            collect_crime_text(item)
            for key, item in value.items()
            if key not in ignored_keys
        )

    if isinstance(value, list):
        return " ".join(collect_crime_text(item) for item in value)

    return str(value)


def incident_location(fir_data):
    details = fir_data.get("incident_details", {})
    return fir_data.get("location") or details.get("location") or ""


def normalize_section(section):
    law = (
        section.get("law")
        or section.get("act")
        or section.get("law_name")
        or "BNS"
    )
    law = str(law).upper().strip()
    if law not in LAW_NAMES:
        law = "BNS"

    return {
        "law": law,
        "section": str(section.get("section") or section.get("section_number") or "").strip(),
        "title": section.get("title") or section.get("section_name") or section.get("section_title"),
    }


def extract_section_references(text):
    references = []

    def add_series(series, law):
        number_pattern = re.compile(r"\b(\d+[A-Za-z]?)(?:\s*\(([a-zA-Z0-9]+)\))?")
        for number_match in number_pattern.finditer(series):
            section_number, clause = number_match.groups()
            references.append({
                "law": law or "BNS",
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
    for match in us_pattern.finditer(text):
        series, law = match.groups()
        add_series(series, law.upper() if law else "BNS")

    law_series_pattern = re.compile(
        r"\b(BNS|BNSS|BSA)\s*(?:sections?|secs?\.?|s\.)?\s*((?:\d+[A-Za-z]?(?:\s*\([a-zA-Z0-9]+\))?\s*(?:,|and|&)?\s*)+)",
        flags=re.IGNORECASE,
    )
    for match in law_series_pattern.finditer(text):
        law, series = match.groups()
        add_series(series, law.upper())

    default_bns_pattern = re.compile(
        r"\b(?:sections?|secs?\.?|s\.)\s*((?:\d+[A-Za-z]?(?:\s*\([a-zA-Z0-9]+\))?\s*(?:,|and|&)?\s*)+)",
        flags=re.IGNORECASE,
    )
    for match in default_bns_pattern.finditer(text):
        add_series(match.group(1), "BNS")

    return references


def dedupe_sections(sections):
    seen = set()
    deduped = []

    for section in sections:
        normalized = normalize_section(section)
        key = (
            normalized.get("law"),
            normalized.get("section"),
            section.get("clause"),
        )
        if not normalized.get("section") or key in seen:
            continue
        seen.add(key)
        if section.get("clause"):
            normalized["clause"] = section.get("clause")
        deduped.append(normalized)

    return deduped


def get_applied_sections(fir_data):
    extracted = fir_data.get("_extracted_fir_details", {})
    if extracted.get("sections"):
        return dedupe_sections(extracted["sections"])

    structured_sections = [
        normalize_section(section)
        for section in (
            fir_data.get("sections_applied")
            or fir_data.get("applied_sections")
            or []
        )
    ]

    if structured_sections:
        return dedupe_sections(structured_sections)

    return dedupe_sections(extract_section_references(collect_text(fir_data)))


def get_possible_sections(fir_data):
    offence_summary = fir_data.get("offence_summary", {})
    return [
        normalize_section(section)
        for section in (
            offence_summary.get("actual_possible_offence")
            or offence_summary.get("possible_sections")
            or fir_data.get("possible_sections")
            or []
        )
    ]


def retrieve_section_reference(section, source):
    normalized = normalize_section(section)
    law = normalized.get("law")
    number = normalized.get("section")
    if not law or not number:
        return None

    doc = find_section(law, number)
    if not doc:
        return None

    document = mongo_doc_to_document(doc)
    document.metadata["chunk_id"] = None
    return format_result(document, 0.0, source)


def dedupe_results(results):
    seen = set()
    deduped = []

    for item in results:
        metadata = item["document"].metadata
        key = (
            metadata.get("law_name"),
            str(metadata.get("section_number")),
        )
        if key in seen:
            continue

        seen.add(key)
        deduped.append(item)

    return deduped


def retrieved_payload(item):
    doc = item["document"]
    metadata = doc.metadata
    return {
        "law_name": metadata.get("law_name"),
        "section_number": metadata.get("section_number"),
        "section_title": metadata.get("section_title"),
        "score": item["score"],
        "source": item.get("source"),
        "preview": doc.page_content[:500],
    }


def limit_validation_results(results, max_items=8):
    source_priority = {
        "applied_section_exact": 0,
        "semantic_crime_candidate": 1,
        "inferred_candidate_exact": 2,
        "faiss_candidate": 2,
    }
    return sorted(
        results,
        key=lambda item: (
            source_priority.get(item.get("source"), 9),
            float(item.get("score", 0.0)),
        )
    )[:max_items]


def format_validation_context(results):
    context_parts = []

    for index, item in enumerate(results, start=1):
        doc = item["document"]
        metadata = doc.metadata
        context_parts.append(
            f"""
[Retrieved Legal Section D{index}]
Retrieval Source: {item.get("source")}
Similarity Score: {item["score"]}
Law Name: {metadata.get("law_name", "")}
Section Number: {metadata.get("section_number", "")}
Section Title: {metadata.get("section_title", "")}

{doc.page_content[:1200]}
""".strip()
        )

    return "\n\n---\n\n".join(context_parts)


def build_fir_search_query(fir_data):
    extracted = fir_data.get("_extracted_fir_details", {})
    crime_facts = " ".join(extracted.get("crime_facts", []))
    return " ".join(
        [
            "BNS offence legal section for these FIR crime facts",
            crime_facts,
            incident_description(fir_data),
            incident_location(fir_data),
        ]
    ).strip()


def extracted_crime_facts(fir_data):
    extracted = fir_data.get("_extracted_fir_details", {})
    if extracted.get("crime_facts"):
        return extracted["crime_facts"]

    description = incident_description(fir_data)
    return [description] if description else ["crime could not be inferred from provided FIR facts"]


def retrieve_semantic_candidates(fir_data, top_k, source="semantic_crime_candidate"):
    search_query = build_fir_search_query(fir_data)
    results = similarity_search(
        query=search_query,
        k=top_k,
        metadata_filter={"law_name": "BNS"}
    )

    return [
        format_result(document, score, source)
        for document, score in results
        if float(score) <= SIMILARITY_SCORE_THRESHOLD
    ]


def normalized_fir_payload(fir_data):
    return {
        "sections": get_applied_sections(fir_data),
        "crime": {
            "description": incident_description(fir_data),
            "detected_crimes": extracted_crime_facts(fir_data),
        },
    }


def build_validation_prompt(fir_data, context, retrieved_documents):
    normalized_fir = normalized_fir_payload(fir_data)
    return f"""
You are a legal FIR section validation assistant for Indian criminal law.

Task:
Validate whether the extracted section numbers match the extracted crime.

Hard rules:
- Use ONLY the extracted sections, extracted crime, and retrieved legal context below.
- Extracted crime facts may be translated from Telugu or another language into English.
- Do not use outside legal knowledge.
- Do not invent sections.
- If a section is not present in retrieved context, say it could not be verified.
- This is an analytical validation, not legal advice.
- A section is not correct merely because it exists in the database.
- Mark each applied section by whether its legal ingredients match the FIR crime facts.
- Use these statuses only: "Correct", "Partially Correct", "Incorrect".
- "Correct" means the section clearly matches a crime fact.
- "Partially Correct" means it touches the facts but is not the main or complete offence.
- "Incorrect" means a required legal ingredient is missing from the FIR facts.
- Identify irrelevant, incomplete, and missing sections.
- If no law is provided with a section number, it has already been treated as BNS.
- Suggested sections must come from retrieved legal context only.
- crime_detected must be copied from Extracted FIR Details crime.detected_crimes
  and must not include applied section names unless the incident facts support them.
- If an extracted section includes a clause, such as BNS 125(a), validate that
  clause against the facts when the retrieved context contains clause text.
- For every applied section, compare the section ingredients with the extracted facts.
- If a required ingredient is missing from the extracted facts, mark that section Incorrect or Partially Correct.
- Suggested sections must be selected from the retrieved legal context that better matches the extracted facts.
- Keep every explanation under 35 words.
- Keep reason under 45 words.
- Keep crime_detected to the main crime categories, not every factual detail.

Output format:
Return ONLY valid JSON. Do not use markdown fences. Use this exact shape:
{{
  "crime_detected": [],
  "applied_sections": [
    {{
      "section": "BNS 303 - Theft",
      "status": "Correct / Partially Correct / Incorrect",
      "explanation": ""
    }}
  ],
  "suggested_sections": [
    {{
      "section": "BNS 309 - Robbery",
      "explanation": ""
    }}
  ],
  "final_result": "Correct Match / Partial Match / Incorrect Match",
  "reason": ""
}}

Extracted FIR Details:
{json.dumps(normalized_fir, ensure_ascii=False, indent=2)}

Retrieved Document Metadata:
{json.dumps(retrieved_documents, ensure_ascii=False, indent=2)}

Retrieved Legal Context:
{context}

Answer:
""".strip()


def parse_validation_json(validation_text):
    cleaned = validation_text.strip()
    fence_match = re.search(
        r"```(?:json)?\s*(.*?)\s*```",
        cleaned,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if fence_match:
        cleaned = fence_match.group(1).strip()

    return json.loads(cleaned)


def repair_validation_json(validation_text):
    prompt = f"""
Convert the following malformed or truncated validation response into valid JSON.

Rules:
- Return ONLY valid JSON.
- Use exactly this shape:
{{
  "crime_detected": [],
  "applied_sections": [
    {{
      "section": "",
      "status": "Correct / Partially Correct / Incorrect",
      "explanation": ""
    }}
  ],
  "suggested_sections": [
    {{
      "section": "",
      "explanation": ""
    }}
  ],
  "final_result": "Correct Match / Partial Match / Incorrect Match",
  "reason": ""
}}
- If data is missing because the input is truncated, complete conservatively from what is present.
- Keep explanations short.

Malformed response:
{validation_text}
""".strip()
    return generate_with_gemini(prompt, max_output_tokens=2000)


def validate_fir(fir_data, top_k=5):
    fir_data = normalize_fir_input(fir_data)
    extracted_details = extract_fir_details(fir_data)
    fir_data = {
        **fir_data,
        "_extracted_fir_details": extracted_details,
    }
    applied_sections = get_applied_sections(fir_data)

    retrieved = []
    issues = []

    if not incident_description(fir_data):
        issues.append("Missing incident description.")

    if not applied_sections:
        issues.append("Missing applied sections.")

    for section in applied_sections:
        item = retrieve_section_reference(section, "applied_section_exact")
        if item:
            retrieved.append(item)
        else:
            issues.append(
                f"Applied section could not be found in database: "
                f"{section.get('law')} {section.get('section')}"
            )

    retrieved.extend(
        retrieve_semantic_candidates(
            fir_data,
            top_k=max(top_k, 8),
            source="semantic_crime_candidate",
        )
    )
    retrieved = dedupe_results(retrieved)
    if len(retrieved) < top_k:
        retrieved.extend(
            retrieve_semantic_candidates(
                fir_data,
                top_k=top_k,
                source="faiss_candidate",
            )
        )
        retrieved = dedupe_results(retrieved)
    retrieved = limit_validation_results(retrieved)

    if not retrieved:
        return {
            "project_status": "Issues Found",
            "issues": issues or ["Information not found in retrieved legal context."],
            "fir_validation": {
                "crime_detected": [],
                "applied_sections": applied_sections,
                "suggested_sections": [],
                "final_result": "Incorrect Match",
                "reason": "Information not found in retrieved legal context.",
            },
        }

    context = format_validation_context(retrieved)
    retrieved_documents = [
        retrieved_payload(item)
        for item in retrieved
    ]
    prompt = build_validation_prompt(fir_data, context, retrieved_documents)
    validation_text = generate_with_gemini(prompt, max_output_tokens=4000)

    try:
        validation = parse_validation_json(validation_text)
        return {
            "project_status": "Issues Found" if issues else "Working Correctly",
            "issues": issues,
            "fir_validation": validation,
        }
    except (json.JSONDecodeError, TypeError, AttributeError):
        try:
            repaired_text = repair_validation_json(validation_text)
            validation = parse_validation_json(repaired_text)
            return {
                "project_status": "Issues Found" if issues else "Working Correctly",
                "issues": issues,
                "fir_validation": validation,
            }
        except (json.JSONDecodeError, TypeError, AttributeError):
            issues.append("LLM did not return valid JSON.")

    return {
        "project_status": "Issues Found" if issues else "Working Correctly",
        "issues": issues,
        "fir_validation": validation_text,
    }
