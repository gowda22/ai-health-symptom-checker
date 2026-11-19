import re
from typing import Tuple, Optional

# Critical red-flag symptom patterns that bypass the LLM immediately
EMERGENCY_PATTERNS = [
    (
        r"\b(chest pain|chest tightness|crushing chest)\b",
        "Potential cardiac emergency (chest pain/tightness detected).",
    ),
    (
        r"\b(shortness of breath|can't breathe|cannot breathe|difficulty breathing|gasping)\b",
        "Acute respiratory distress detected.",
    ),
    (
        r"\b(slurred speech|face drooping|one side weak|sudden numbness|paralysis)\b",
        "Potential stroke warning signs (FAST criteria).",
    ),
    (
        r"\b(coughing blood|vomiting blood|severe bleeding|uncontrolled bleeding)\b",
        "Severe hemorrhage/bleeding symptoms detected.",
    ),
    (
        r"\b(suicidal|want to die|kill myself|self harm)\b",
        "Acute mental health crisis detected.",
    ),
    (
        r"\b(stiff neck).*(high fever)|(high fever).*(stiff neck)\b",
        "Potential meningitis warning signs (fever + stiff neck).",
    ),
]


def check_red_flags(text: str) -> Tuple[bool, Optional[str]]:
    normalized = text.lower().strip()
    for pattern, reason in EMERGENCY_PATTERNS:
        if re.search(pattern, normalized):
            return True, reason
    return False, None
