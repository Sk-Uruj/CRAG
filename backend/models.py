from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class RetrievalLabel(str, Enum):
    CORRECT = "CORRECT"
    INCORRECT = "INCORRECT"
    AMBIGUOUS = "AMBIGUOUS"


@dataclass
class DocumentChunk:
    chunk_id: str
    source: str
    text: str
    metadata: Dict[str, Any]


@dataclass
class RetrievalResult:
    query: str
    chunks: List[DocumentChunk]


class EvaluationResult(BaseModel):
    label: RetrievalLabel
    confidence: float = Field(ge=0.0, le=1.0)
    reasons: List[str]
    cleaned_context: List[str]
    rewritten_query: Optional[str] = None


class GenerationResult(BaseModel):
    answer: str
    sources: List[str]
    trace: List[str]
    question: str
    mode: RetrievalLabel
    confidence: float
