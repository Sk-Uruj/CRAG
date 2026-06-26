from __future__ import annotations

from typing import Any, Dict, List

from backend.models import DocumentChunk, RetrievalLabel


class ResponseGenerator:
    def generate(
        self,
        query: str,
        context_chunks: List[DocumentChunk],
        mode: RetrievalLabel,
        confidence: float,
    ) -> Dict[str, Any]:
        trace = [f"[{c.source}::{c.chunk_id}] {c.text}" for c in context_chunks]
        return {
            "answer": " | ".join(trace),
            "mode": mode.value,
            "confidence": confidence,
            "sources": sorted(list({c.source for c in context_chunks})),
            "trace": trace,
            "question": query,
        }
