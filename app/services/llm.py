import os
from typing import Optional

from .vector_store import VectorStoreService

try:
    from langchain.chat_models import ChatOpenAI
except ImportError:
    ChatOpenAI = None

try:
    from langchain.chat_models import GoogleGemini
except ImportError:
    GoogleGemini = None

class LLMService:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")

        self.model_name = "fallback"
        self.llm = None

        if self.api_key and ChatOpenAI is not None:
            self.llm = ChatOpenAI(openai_api_key=self.api_key, temperature=0.0)
            self.model_name = "openai-chat"
        elif os.getenv("GOOGLE_API_KEY") and GoogleGemini is not None:
            self.llm = GoogleGemini()
            self.model_name = "google-gemini"

    def generate_response(self, prompt: str) -> str:
        if self.llm is None:
            # fallback behavior for development without cloud credentials
            return f"[fallback] {prompt}"

        try:
            return self.llm.predict(prompt)
        except Exception as e:
            return f"Error from LLM: {e}"

    def combined_response(self, query: str, vector_store: VectorStoreService, top_k: int = 3) -> dict:
        retrieved = vector_store.similarity_search(query, k=top_k)
        context = "\n\n".join([item.get("text", "") for item in retrieved])
        final_prompt = f"You are a helpful assistant. Use the following context to answer the question.\n\nContext:\n{context}\n\nQuestion:\n{query}\n\nAnswer:"

        answer = self.generate_response(final_prompt)
        return {
            "answer": answer,
            "model": self.model_name,
            "retrieved_docs": retrieved,
        }
