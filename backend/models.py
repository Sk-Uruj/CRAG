from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


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

    @field_validator("reasons")
    @classmethod
    def validate_reasons(cls, v: List[str]) -> List[str]:
        if not isinstance(v, list) or not v:
            raise ValueError("reasons must be a non-empty list")
        return [str(x).strip() for x in v if str(x).strip()][:3]

    @field_validator("cleaned_context")
    @classmethod
    def validate_cleaned_context(cls, v: List[str]) -> List[str]:
        if not isinstance(v, list):
            raise ValueError("cleaned_context must be a list")
        return [str(x).strip() for x in v if str(x).strip()][:3]

    @field_validator("rewritten_query")
    @classmethod
    def validate_rewritten_query(cls, v: Optional[str]) -> Optional[str]:
        if v is not None:
            v = v.strip()
            if not v:
                return None
        return v


class GenerationResult(BaseModel):
    answer: str
    sources: List[str]
    trace: List[str]
    question: str
    mode: RetrievalLabel
    confidence: float
