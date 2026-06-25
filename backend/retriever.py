from __future__ import annotations

from typing import List

from backend.models import DocumentChunk, RetrievalResult
from backend.embeddings import GeminiEmbedder
from backend.vectorstore import ChromaVectorStore


class ChromaRetriever:
    def __init__(self, embedder: GeminiEmbedder, vectorstore: ChromaVectorStore, top_k: int = 4) -> None:
        self.embedder = embedder
        self.vectorstore = vectorstore
        self.top_k = top_k

    def retrieve(self, query: str, top_k: int | None = None) -> RetrievalResult:
        k = top_k or self.top_k
        query_embedding = self.embedder.embed_text(query)
        chunks: List[DocumentChunk] = self.vectorstore.query(query_embedding=query_embedding, top_k=k)
        return RetrievalResult(query=query, chunks=chunks)
