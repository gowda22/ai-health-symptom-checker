from fastapi import FastAPI, HTTPException
from app.schemas import SymptomRequest, TriageResponse
from app.guardrails import check_red_flags
from app.rag_engine import MedicalRAGEngine

app = FastAPI(title="AI Health Symptom Checker API", version="1.0.0")
rag_engine = MedicalRAGEngine()


@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.post("/triage", response_model=TriageResponse)
def evaluate_symptoms(request: SymptomRequest):
    # Step 1: Deterministic Red-Flag Safety Check (Bypasses LLM)
    is_emergency, reason = check_red_flags(request.symptoms_text)
    if is_emergency:
        return TriageResponse(
            is_emergency=True,
            urgency_level="Emergency (Call Ambulance / ER)",
            emergency_reason=reason,
            possible_conditions=[],
            recommended_next_steps=(
                "Seek immediate emergency medical care or call local emergency services right away. "
                "Do not drive yourself."
            ),
            follow_up_questions=[],
        )

    # Step 2: Grounded RAG Inference
    try:
        return rag_engine.generate_triage(
            symptoms_text=request.symptoms_text,
            age=request.age,
            duration_days=request.duration_days,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
