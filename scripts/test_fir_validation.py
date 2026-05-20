from pathlib import Path
import json
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.rag.fir_validator import validate_fir


SAMPLE_FIR = {
    "fir_number": "FIR-2026-0147",
    "police_station": "Madhapur Police Station",
    "date_of_registration": "2026-05-19",
    "complainant": {
        "name": "Ravi Kumar",
        "age": 34,
        "address": "Hyderabad, Telangana"
    },
    "accused": {
        "name": "Unknown Person"
    },
    "incident_details": {
        "date": "2026-05-18",
        "time": "09:30 PM",
        "location": "Residential Apartment, Hyderabad",
        "description": (
            "The complainant stated that an unknown person entered his apartment "
            "during the night and stole a gold chain, a laptop, and cash worth "
            "Rs. 45,000 from the bedroom cupboard. The accused allegedly escaped "
            "from the location before local residents noticed the incident."
        )
    },
    "sections_applied": [
        {
            "law": "BNS",
            "section": "323",
            "title": "Voluntarily causing hurt"
        },
        {
            "law": "BNS",
            "section": "351",
            "title": "Criminal intimidation"
        }
    ],
    "offence_summary": {
        "actual_possible_offence": [
            {
                "law": "BNS",
                "section": "303",
                "title": "Theft"
            },
            {
                "law": "BNS",
                "section": "305",
                "title": "Theft in dwelling house"
            }
        ],
        "problem": (
            "Applied sections do not match the crime described in the FIR. "
            "The FIR describes theft inside a dwelling house, but hurt and "
            "criminal intimidation sections were applied instead."
        )
    },
    "investigating_officer": {
        "name": "Inspector Suresh",
        "badge_id": "TG-4582"
    }
}


def main():
    result = validate_fir(SAMPLE_FIR, top_k=5)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
