from __future__ import annotations

from typing import List, Optional
import os

from dotenv import load_dotenv

load_dotenv()

try:
    from google import genai
    GEMINI_AVAILABLE = True
except Exception:
    genai = None
    GEMINI_AVAILABLE = False


class GeminiEmbedder:
    def __init__(self, api_key: Optional[str] = None, model: str = "models/text-embedding-004") -> None:
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        self.model = model

        if not GEMINI_AVAILABLE:
            raise RuntimeError("google-genai is not available")
        if not self.api_key:
            raise RuntimeError("No Google Gemini API key found. Set GOOGLE_API_KEY in .env.")

        self.client = genai.Client(api_key=self.api_key)

    def embed_text(self, text: str) -> List[float]:
        response = self.client.models.embed_content(
            model=self.model,
            contents=text,
        )

        if hasattr(response, "embedding") and response.embedding:
            return list(response.embedding.values)

        if hasattr(response, "embeddings") and response.embeddings:
            emb = response.embeddings[0]
            return list(emb.values)

        raise RuntimeError("Could not extract embedding from Gemini response")

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        return [self.embed_text(t) for t in texts]