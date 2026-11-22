import os
import streamlit as st
from app.guardrails import check_red_flags
from app.rag_engine import MedicalRAGEngine

st.set_page_config(
    page_title="AI Health Triage Assistant", page_icon="🩺", layout="centered"
)

if "GROQ_API_KEY" in st.secrets:
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]


@st.cache_resource
def load_engine_v2():
    return MedicalRAGEngine()


st.title("🩺 AI Health Symptom Checker")
st.caption(
    "Grounded Clinical Triage System • Powered by RAG & Deterministic Safety Guardrails"
)

with st.form("symptom_form"):
    symptoms = st.text_area(
        "Describe your symptoms in detail:",
        placeholder="e.g., I have had a throbbing headache on one side, nausea, and sensitivity to light since morning.",
    )
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", min_value=1, max_value=120, value=25)
    with col2:
        duration = st.number_input(
            "Duration (days)", min_value=0, max_value=365, value=1
        )

    submitted = st.form_submit_button("Analyze Symptoms", type="primary")

if submitted:
    if len(symptoms.strip()) < 3:
        st.warning("Please enter a valid symptom description.")
    else:
        with st.spinner("Checking clinical guidelines and safety guardrails..."):
            try:
                is_emergency, reason = check_red_flags(symptoms)

                if is_emergency:
                    st.error("🚨 **Emergency (Call Ambulance / ER)**")
                    st.markdown(f"**Trigger Reason:** {reason}")
                    st.markdown(
                        "**Immediate Action:** Seek immediate emergency medical care or call local emergency services right away. Do not drive yourself."
                    )
                else:
                    rag_engine = load_engine_v2()
                    result = rag_engine.generate_triage(
                        symptoms_text=symptoms,
                        age=age,
                        duration_days=duration,
                    )

                    if "Doctor" in result.urgency_level:
                        st.warning(f"⚠️ **Recommended Triage:** {result.urgency_level}")
                    else:
                        st.success(f"✅ **Recommended Triage:** {result.urgency_level}")

                    st.subheader("Possible Matching Conditions")
                    for cond in result.possible_conditions:
                        with st.expander(f"📌 {cond.condition}", expanded=True):
                            st.write(f"**Clinical Reasoning:** {cond.reasoning}")
                            st.caption(f"**Verified Source:** {cond.source_citation}")

                    st.subheader("Recommended Next Steps")
                    st.write(result.recommended_next_steps)

                    if result.follow_up_questions:
                        st.subheader("Clarifying Questions for Your Doctor")
                        for q in result.follow_up_questions:
                            st.markdown(f"- {q}")

                st.divider()
                st.caption(
                    "DISCLAIMER: This tool is for educational triage support only and is NOT a medical diagnosis. Always consult a licensed healthcare professional."
                )
            except Exception as e:
                st.error(f"Error analyzing symptoms: {e}")
