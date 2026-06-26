from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

from backend.models import DocumentChunk, EvaluationResult, RetrievalLabel

load_dotenv()

try:
    from google import genai
    GEMINI_AVAILABLE = True
except Exception:
    genai = None
    GEMINI_AVAILABLE = False


class RetrievalEvaluator:
    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        api_key: Optional[str] = None,
        max_output_tokens: int = 400,
        temperature: float = 0.0,
    ) -> None:
        self.model = model
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        self.max_output_tokens = max_output_tokens
        self.temperature = temperature

        if not GEMINI_AVAILABLE:
            self.client = None
        elif not self.api_key:
            self.client = None
        else:
            self.client = genai.Client(api_key=self.api_key)

    def evaluate(self, query: str, chunks: List[DocumentChunk]) -> EvaluationResult:
        if self.client is None:
            return self._fallback_result(query=query, chunks=chunks, reason="Gemini client unavailable or API key missing.")

        try:
            prompt = self._build_prompt(query, chunks)
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config={
                    "temperature": self.temperature,
                    "max_output_tokens": self.max_output_tokens,
                },
            )

            raw_text = self._extract_text(response)
            print("=== RAW GEMINI RESPONSE ===")
            print(raw_text)

            parsed = self._extract_json(raw_text)
            print("=== PARSED JSON ===")
            print(parsed)

            parsed = self._normalize(parsed)
            result = EvaluationResult.model_validate(parsed)

            if result.label == RetrievalLabel.CORRECT:
                result.rewritten_query = None
            elif not result.rewritten_query:
                result.rewritten_query = self._rewrite_query_local(query)

            return result

        except Exception as exc:
            return self._fallback_result(
                query=query,
                chunks=chunks,
                reason=f"Gemini evaluation failed: {type(exc).__name__}: {exc}",
            )

    def _build_prompt(self, query: str, chunks: List[DocumentChunk]) -> str:
        payload = {
            "task": "Evaluate the semantic relevance of retrieved document chunks against the user query.",
            "rules": [
                "Return only one raw JSON object and nothing else.",
                "Do not add markdown fences or code blocks.",
                "Keep reasons short and limited to at most 3 strings.",
                "Keep cleaned_context short and limited to at most 3 strings.",
                "Be conservative; if unsure choose AMBIGUOUS.",
                "rewritten_query must be null for CORRECT.",
                "rewritten_query is required for INCORRECT or AMBIGUOUS.",
            ],
            "query": query,
            "retrieved_chunks": [
                {
                    "chunk_id": c.chunk_id,
                    "source": c.source,
                    "text": c.text,
                    "metadata": c.metadata,
                }
                for c in chunks
            ],
            "required_json_schema": {
                "label": "CORRECT | INCORRECT | AMBIGUOUS",
                "confidence": "float between 0 and 1",
                "reasons": ["short string"],
                "cleaned_context": ["short string"],
                "rewritten_query": "string or null",
            },
        }
        return json.dumps(payload, indent=2, ensure_ascii=False)

    def _extract_text(self, response: Any) -> str:
        if response is None:
            raise ValueError("Empty Gemini response")

        if hasattr(response, "text") and isinstance(response.text, str) and response.text.strip():
            return response.text.strip()

        parts = []
        if hasattr(response, "candidates") and response.candidates:
            for c in response.candidates:
                content = getattr(c, "content", None)
                if content and getattr(content, "parts", None):
                    for p in content.parts:
                        txt = getattr(p, "text", None)
                        if txt:
                            parts.append(txt)

        out = "\n".join(parts).strip()
        if not out:
            raise ValueError("Could not extract text from Gemini response")
        return out

    def _extract_json(self, raw: str) -> Dict[str, Any]:
        raw = raw.strip()
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
        raw = re.sub(r"\s*```$", "", raw, flags=re.IGNORECASE).strip()

        try:
            return json.loads(raw)
        except Exception:
            pass

        start = raw.find("{")
        if start == -1:
            raise ValueError(f"No JSON object found in model response: {raw[:300]}")

        depth = 0
        in_string = False
        escape = False

        for i in range(start, len(raw)):
            ch = raw[i]
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
            else:
                if ch == '"':
                    in_string = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        candidate = raw[start : i + 1]
                        return json.loads(candidate)

        raise ValueError(f"Could not find a complete JSON object in model response: {raw[:300]}")

    def _normalize(self, parsed: Dict[str, Any]) -> Dict[str, Any]:
        label = str(parsed.get("label", "AMBIGUOUS")).replace("RetrievalLabel.", "").upper()
        if label not in {e.value for e in RetrievalLabel}:
            label = RetrievalLabel.AMBIGUOUS.value
        parsed["label"] = label

        try:
            parsed["confidence"] = float(parsed.get("confidence", 0.5))
        except Exception:
            parsed["confidence"] = 0.5

        reasons = parsed.get("reasons") or ["Model returned invalid or missing reasons."]
        parsed["reasons"] = [str(x).strip() for x in reasons if str(x).strip()][:3]

        cleaned = parsed.get("cleaned_context") or []
        parsed["cleaned_context"] = [str(x).strip() for x in cleaned if str(x).strip()][:3]

        rq = parsed.get("rewritten_query")
        parsed["rewritten_query"] = str(rq).strip() if rq is not None and str(rq).strip() else None
        return parsed

    def _fallback_result(self, query: str, chunks: List[DocumentChunk], reason: str) -> EvaluationResult:
        return EvaluationResult(
            label=RetrievalLabel.AMBIGUOUS,
            confidence=0.35,
            reasons=["Safe fallback activated.", reason],
            cleaned_context=self._local_cleanup(chunks),
            rewritten_query=self._rewrite_query_local(query),
        )

    def _local_cleanup(self, chunks: List[DocumentChunk]) -> List[str]:
        cleaned = []
        for c in chunks:
            text = re.sub(r"\s+", " ", c.text).strip()
            if text:
                cleaned.append(text)
        return cleaned[:3]

    def _rewrite_query_local(self, query: str) -> str:
        return f"{query.strip().rstrip('?')} authoritative policy or official documentation"