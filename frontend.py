import os
import streamlit as st
from app.guardrails import check_red_flags
from app.rag_engine import MedicalRAGEngine

st.set_page_config(
    page_title="AI Health Triage Assistant", page_icon="🩺", layout="centered"
)

# Load GROQ_API_KEY from Streamlit Cloud Secrets into environment variables
if "GROQ_API_KEY" in st.secrets:
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

@st.cache_resource
def load_engine():
    return MedicalRAGEngine()