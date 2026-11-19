from pydantic import BaseModel, Field
from typing import List, Literal, Optional


class SymptomRequest(BaseModel):
    symptoms_text: str = Field(..., min_length=3, description="Patient's description of symptoms")
    age: int = Field(..., ge=0, le=120)
    duration_days: int = Field(..., ge=0)


class ConditionMatch(BaseModel):
    condition: str
    reasoning: str
    source_citation: str


class TriageResponse(BaseModel):
    is_emergency: bool
    urgency_level: Literal[
        "Emergency (Call Ambulance / ER)",
        "Doctor Visit (24-48 hrs)",
        "Self-Care",
    ]
    emergency_reason: Optional[str] = None
    possible_conditions: List[ConditionMatch] = []
    recommended_next_steps: str
    follow_up_questions: List[str] = []
    medical_disclaimer: str = (
        "DISCLAIMER: This tool is for educational triage support only and is NOT a medical diagnosis. "
        "Always consult a licensed healthcare professional."
    )
