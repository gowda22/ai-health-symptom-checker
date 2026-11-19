from app.guardrails import check_red_flags
from app.rag_engine import MedicalRAGEngine


def test_emergency_guardrail():
    is_emerg, reason = check_red_flags(
        "I have severe chest pain and shortness of breath"
    )
    assert is_emerg is True
    assert "cardiac" in reason.lower()


def test_non_emergency_query():
    is_emerg, _ = check_red_flags("I have a mild runny nose and sneezing")
    assert is_emerg is False


def test_rag_retrieval_grounding():
    engine = MedicalRAGEngine()
    context = engine.retrieve_context(
        "throbbing headache one side sensitivity to light"
    )
    assert "Migraine" in context
    assert "MedlinePlus" in context
