# 🩺 AI Health Symptom Checker (Grounded Clinical Triage System)

A full-stack clinical triage assistant built with **FastAPI**, **ChromaDB (RAG)**, **Groq (Llama-3.3-70B)**, and **Streamlit**. Designed to demonstrate responsible healthcare AI through deterministic red-flag safety guardrails, source-cited retrieval-augmented generation (RAG), and strict Pydantic schema validation.

---

## 🏗️ System Architecture

```text
User Input (Streamlit UI)
        │
        ▼
FastAPI Backend (/triage)
        │
        ├──► [1. Deterministic Red-Flag Guardrail] ──► Emergency Detected? ──► Immediate ER Alert (Bypasses LLM)
        │
        └──► [2. Hybrid Semantic Retrieval (ChromaDB)]
                        │
                        ▼
             [3. Grounded LLM Inference (Groq Llama-3.3)]
                        │
                        ▼
             [4. Pydantic Structured JSON + Citations] ──► Rendered Triage Card
```

### Key Engineering Highlights
1. **Deterministic Safety Layer (`app/guardrails.py`):** Evaluates incoming text against high-risk clinical red flags (cardiac chest pain, stroke FAST criteria, respiratory distress, meningitis signs) before invoking any generative model.
2. **Zero-Hallucination Grounding (`app/rag_engine.py`):** Embeds curated clinical guidelines (MedlinePlus, WHO, CDC, NIDDK) using `all-MiniLM-L6-v2` into ChromaDB and restricts LLM generation strictly to retrieved documents.
3. **Structured Output Contracts (`app/schemas.py`):** Enforces typed JSON responses with urgency routing (`Emergency`, `Doctor Visit (24-48 hrs)`, `Self-Care`), clinical reasoning, and verifiable source citations.

---

## 🚀 Quickstart Guide

### 1. Clone & Set Up Environment
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure API Key
1. Generate a free API key at [console.groq.com](https://console.groq.com).
2. Copy `.env.example` to `.env` and add your key:
```bash
cp .env.example .env
```
Edit `.env`:
```env
GROQ_API_KEY=gsk_your_actual_key_here
```

### 3. Run the Application
Open **two terminals** inside the project directory:

* **Terminal 1 (FastAPI Backend):**
  ```bash
  uvicorn app.main:app --reload
  ```
* **Terminal 2 (Streamlit Frontend):**
  ```bash
  streamlit run frontend.py
  ```

Open `http://localhost:8501` in your browser.

---

## 🧪 Running Verification Tests
Run the automated test suite to verify red-flag emergency detection and vector database grounding:
```bash
pytest
```
