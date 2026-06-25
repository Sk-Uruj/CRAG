from __future__ import annotations

from dataclasses import asdict
from typing import Dict, List

from backend.models import DocumentChunk, RetrievalLabel, RetrievalResult
from backend.retriever import ChromaRetriever
from backend.evaluator import RetrievalEvaluator
from backend.generator import ResponseGenerator


class CRAGOrchestrator:
    def __init__(self, retriever: ChromaRetriever, evaluator: RetrievalEvaluator, generator: ResponseGenerator) -> None:
        self.retriever = retriever
        self.evaluator = evaluator
        self.generator = generator

    def run(self, query: str) -> Dict[str, any]:
        retrieval = self.retriever.retrieve(query=query)
        evaluation = self.evaluator.evaluate(query=query, chunks=retrieval.chunks)

        if evaluation.label == RetrievalLabel.CORRECT:
            final_chunks = [
                DocumentChunk(
                    chunk_id=f"clean_{i+1}",
                    source="internal_cleaned",
                    text=text,
                    metadata={"type": "cleaned"},
                )
                for i, text in enumerate(evaluation.cleaned_context)
            ]
            generation = self.generator.generate(query, final_chunks, evaluation.label, evaluation.confidence)
            return self._package(query, retrieval, evaluation, generation, [])

        if evaluation.label == RetrievalLabel.INCORRECT:
            rewritten = evaluation.rewritten_query or query
            web_chunks = self._simulate_web_search(rewritten)
            generation = self.generator.generate(rewritten, web_chunks, evaluation.label, evaluation.confidence)
            return self._package(query, retrieval, evaluation, generation, web_chunks)

        web_chunks = self._simulate_web_search(evaluation.rewritten_query or query)
        local_chunks = [
            DocumentChunk(
                chunk_id=f"partial_{i+1}",
                source="internal_partial",
                text=text,
                metadata={"type": "partial"},
            )
            for i, text in enumerate(evaluation.cleaned_context)
        ]
        blended = local_chunks + web_chunks
        generation = self.generator.generate(query, blended, evaluation.label, evaluation.confidence)
        return self._package(query, retrieval, evaluation, generation, web_chunks)

    def _simulate_web_search(self, query: str) -> List[DocumentChunk]:
        return [
            DocumentChunk(
                chunk_id="web_1",
                source="web_search",
                text=f"Web result: updated information relevant to '{query}'.",
                metadata={"rank": 1, "source_type": "web"},
            ),
            DocumentChunk(
                chunk_id="web_2",
                source="web_search",
                text=f"Additional web result: authoritative guidance for '{query}'.",
                metadata={"rank": 2, "source_type": "web"},
            ),
        ]

    def _package(self, query, retrieval, evaluation, generation, web_chunks):
        return {
            "query": query,
            "retrieved_chunks": [asdict(c) for c in retrieval.chunks],
            "evaluation": evaluation.model_dump(),
            "web_chunks": [asdict(c) for c in web_chunks],
            "generation": generation,
        }
