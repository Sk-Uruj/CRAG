from __future__ import annotations

from typing import List, Optional
from pathlib import Path

import chromadb

from backend.models import DocumentChunk
from backend.config import settings


class ChromaVectorStore:
    def __init__(self, persist_dir: Optional[Path] = None, collection_name: Optional[str] = None) -> None:
        self.persist_dir = persist_dir or settings.CHROMA_DIR
        self.collection_name = collection_name or settings.CHROMA_COLLECTION_NAME

        self.client = chromadb.PersistentClient(path=str(self.persist_dir))
        self.collection = self.client.get_or_create_collection(name=self.collection_name)

    def add_chunks(self, chunks: List[DocumentChunk], embeddings: List[List[float]]) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("Chunks and embeddings must have the same length")

        ids = [c.chunk_id for c in chunks]
        documents = [c.text for c in chunks]
        metadatas = [dict(c.metadata) for c in chunks]

        self.collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    def query(self, query_embedding: List[float], top_k: int = 4) -> List[DocumentChunk]:
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "ids", "distances"],
        )

        chunks: List[DocumentChunk] = []
        ids = results.get("ids", [[]])[0]
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]

        for i, chunk_id in enumerate(ids):
            meta = metas[i] or {}
            chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    source=meta.get("source", "unknown"),
                    text=docs[i],
                    metadata=meta,
                )
            )

        return chunks

    def count(self) -> int:
        return self.collection.count()

    def reset(self) -> None:
        self.client.delete_collection(name=self.collection_name)
        self.collection = self.client.get_or_create_collection(name=self.collection_name)
