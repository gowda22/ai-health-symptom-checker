import json
import os
import chromadb
from chromadb.utils import embedding_functions
from groq import Groq
from dotenv import load_dotenv
from app.schemas import TriageResponse

load_dotenv()


class MedicalRAGEngine:
    def __init__(self, kb_path: str = "data/medical_kb.json"):
        self.client = chromadb.PersistentClient(path="./chroma_db")
        self.embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        self.collection = self.client.get_or_create_collection(
            name="medical_guidelines",
            embedding_function=self.embed_fn,
        )
        self._load_knowledge_base(kb_path)
        api_key = os.getenv("GROQ_API_KEY")
        self.groq_client = Groq(api_key=api_key) if api_key else None

    def _load_knowledge_base(self, kb_path: str):
        if self.collection.count() > 0:
            return
        with open(kb_path, "r", encoding="utf-8") as f:
            records = json.load(f)

        documents, metadatas, ids = [], [], []
        for item in records:
            doc_text = (
                f"Condition: {item['condition']}. "
                f"Symptoms: {item['symptoms']}. "
                f"Clinical Guidance: {item['guidance']}. "
                f"Standard Urgency: {item['urgency']}."
            )
            documents.append(doc_text)
            metadatas.append({"condition": item["condition"], "source": item["source"]})
            ids.append(item["id"])

        self.collection.add(documents=documents, metadatas=metadatas, ids=ids)

    def retrieve_context(self, query: str, top_k: int = 3) -> str:
        results = self.collection.query(query_texts=[query], n_results=top_k)
        context_blocks = []
        for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
            context_blocks.append(f"[Source: {meta['source']}]\n{doc}")
        return "\n\n".join(context_blocks)

    def generate_triage(
        self, symptoms_text: str, age: int, duration_days: int
    ) -> TriageResponse:
        if not self.groq_client:
            raise ValueError(
                "GROQ_API_KEY is missing. Copy .env.example to .env and add your Groq API key."
            )

        retrieved_docs = self.retrieve_context(symptoms_text)

        system_prompt = """
        You are a clinical triage assistant. Answer STRICTLY using ONLY the provided medical context below.
        Do NOT invent conditions or treatments outside the provided context.
        Return valid JSON matching this exact schema:
        {
          "is_emergency": false,
          "urgency_level": "Doctor Visit (24-48 hrs)" or "Self-Care",
          "emergency_reason": null,
          "possible_conditions": [
            {
              "condition": "Name from context",
              "reasoning": "Why user symptoms match",
              "source_citation": "Exact Source from context"
            }
          ],
          "recommended_next_steps": "Actionable advice strictly from context",
          "follow_up_questions": ["Question 1 to clarify symptoms", "Question 2"]
        }
        """

        user_prompt = f"""
        Patient Info: Age {age}, Duration: {duration_days} days.
        Reported Symptoms: {symptoms_text}

        Retrieved Medical Context:
        {retrieved_docs}
        """

        completion = self.groq_client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
            response_format={"type": "json_object"},
        )

        raw_json = completion.choices[0].message.content
        parsed = json.loads(raw_json)
        return TriageResponse(**parsed)
