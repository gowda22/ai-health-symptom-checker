import json
import os
from pathlib import Path
import chromadb
from groq import Groq
from dotenv import load_dotenv
from app.schemas import TriageResponse

load_dotenv()


class MedicalRAGEngine:
    def __init__(self, kb_path: str = "data/medical_kb.json"):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY is missing. Add it to .env locally or Streamlit Secrets in the cloud."
            )

        self.groq_client = Groq(api_key=api_key)
        self.model_name = "llama-3.1-8b-instant"

        self.chroma_client = chromadb.Client()
        self.collection = self.chroma_client.get_or_create_collection(
            name="clinical_guidelines"
        )

        self._ingest_knowledge_base(kb_path)

    def _ingest_knowledge_base(self, kb_path: str):
        if self.collection.count() > 0:
            return

        file_path = Path(kb_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Knowledge base not found at {kb_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            records = json.load(f)

        documents = []
        metadatas = []
        ids = []

        for idx, item in enumerate(records):
            symptoms_list = item.get("symptoms", [])
            symptoms_str = (
                ", ".join(symptoms_list)
                if isinstance(symptoms_list, list)
                else str(symptoms_list)
            )
            text_chunk = (
                f"Condition: {item.get('condition', 'Unknown')}\n"
                f"Symptoms: {symptoms_str}\n"
                f"Risk Factors: {item.get('risk_factors', 'General population')}\n"
                f"Clinical Guidance: {item.get('guidance', item.get('description', ''))}\n"
                f"Default Urgency: {item.get('urgency', 'Doctor Visit (24-48 hrs)')}"
            )
            documents.append(text_chunk)
            metadatas.append(
                {
                    "condition": item.get("condition", "Unknown"),
                    "source": item.get("source", "Verified Medical Guidelines"),
                    "urgency": item.get("urgency", "Doctor Visit (24-48 hrs)"),
                }
            )
            ids.append(str(item.get("id", f"doc_{idx}")))

        self.collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids,
        )

    def retrieve_context(self, query: str, top_k: int = 3) -> str:
        results = self.collection.query(query_texts=[query], n_results=top_k)

        context_blocks = []
        for idx, doc in enumerate(results["documents"][0]):
            source = results["metadatas"][0][idx]["source"]
            context_blocks.append(f"\n{doc}")

        return "\n\n---\n\n".join(context_blocks)

    def generate_triage(
        self, symptoms_text: str, age: int = None, duration_days: int = None
    ) -> TriageResponse:
        retrieved_context = self.retrieve_context(symptoms_text, top_k=3)

        patient_meta = []
        if age:
            patient_meta.append(f"Age: {age}")
        if duration_days:
            patient_meta.append(f"Duration: {duration_days} days")
        meta_str = ", ".join(patient_meta) if patient_meta else "Not provided"

        system_prompt = (
            "You are a cautious, evidence-based clinical triage assistant. "
            "Analyze the patient's symptoms STRICTLY using the provided Verified Medical Context. "
            "Do NOT invent diseases or treatments outside the provided context. "
            "Always cite the exact source provided in the context for each condition. "
            "Return ONLY valid JSON matching the exact schema requested."
        )

        user_prompt = f"""
Patient Info: {meta_str}
Reported Symptoms: {symptoms_text}

Verified Medical Context:
{retrieved_context}

Respond in valid JSON matching this structure:
{{
  "urgency_level": "Doctor Visit (24-48 hrs)" or "Self-Care",
  "possible_conditions": [
    {{
      "condition": "Name from context",
      "reasoning": "Why symptoms match the context",
      "source_citation": "Exact source name from context"
    }}
  ],
  "recommended_next_steps": "Clear, safe next steps based on context",
  "follow_up_questions": ["Question 1", "Question 2"]
}}
"""

        completion = self.groq_client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
            response_format={"type": "json_object"},
        )

        raw_json = completion.choices[0].message.content
        return TriageResponse.model_validate_json(raw_json)