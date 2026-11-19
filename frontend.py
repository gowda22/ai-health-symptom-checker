import streamlit as st
import requests
import os

API_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")

st.set_page_config(
    page_title="AI Health Triage Assistant", page_icon="🩺", layout="centered"
)

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
                res = requests.post(
                    f"{API_URL}/triage",
                    json={
                        "symptoms_text": symptoms,
                        "age": age,
                        "duration_days": duration,
                    },
                    timeout=30,
                )
                if res.status_code != 200:
                    st.error(f"Backend error ({res.status_code}): {res.text}")
                else:
                    data = res.json()

                    if data["is_emergency"]:
                        st.error(f"🚨 **{data['urgency_level']}**")
                        st.markdown(f"**Trigger Reason:** {data['emergency_reason']}")
                        st.markdown(
                            f"**Immediate Action:** {data['recommended_next_steps']}"
                        )
                    else:
                        if "Doctor" in data["urgency_level"]:
                            st.warning(
                                f"⚠️ **Recommended Triage:** {data['urgency_level']}"
                            )
                        else:
                            st.success(
                                f"✅ **Recommended Triage:** {data['urgency_level']}"
                            )

                        st.subheader("Possible Matching Conditions")
                        for cond in data["possible_conditions"]:
                            with st.expander(f"📌 {cond['condition']}", expanded=True):
                                st.write(
                                    f"**Clinical Reasoning:** {cond['reasoning']}"
                                )
                                st.caption(
                                    f"**Verified Source:** {cond['source_citation']}"
                                )

                        st.subheader("Recommended Next Steps")
                        st.write(data["recommended_next_steps"])

                        if data["follow_up_questions"]:
                            st.subheader("Clarifying Questions for Your Doctor")
                            for q in data["follow_up_questions"]:
                                st.markdown(f"- {q}")

                    st.divider()
                    st.caption(data["medical_disclaimer"])
            except Exception as e:
                st.error(f"Could not connect to backend API: {e}")
