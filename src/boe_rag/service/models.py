"""Request and response models for the RAG service API.

Pydantic models double as the FastAPI schema and the engine's return types, so
the contract is defined once and validated at the boundary.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

#: Hard cap on how many passages a single request may ask for.
MAX_K = 20


class Source(BaseModel):
    """A retrieved passage cited as a source.

    Attributes:
        chunk_id: The chunk's stable id.
        citation: Human-readable citation (e.g. law and article).
        text: The passage text.
        url: Link back to the source document on boe.es.
        score: Retrieval/rerank score (higher is more relevant).
    """

    chunk_id: str
    citation: str
    text: str
    url: str
    score: float


class AskRequest(BaseModel):
    """A question to answer with grounded, cited sources.

    Attributes:
        question: The user's natural-language question.
        k: Number of passages to ground the answer in.
    """

    question: str = Field(min_length=1, max_length=1000)
    k: int = Field(default=5, ge=1, le=MAX_K)


class SearchRequest(BaseModel):
    """A query for raw passage retrieval (no generation).

    Attributes:
        query: The search query.
        k: Number of passages to return.
    """

    query: str = Field(min_length=1, max_length=1000)
    k: int = Field(default=10, ge=1, le=MAX_K)


class AnswerResponse(BaseModel):
    """A grounded answer with its supporting sources.

    Attributes:
        answer: The generated answer (or a refusal when unsupported).
        refused: Whether the model declined for lack of grounding.
        sources: The passages the answer is grounded in (empty if refused).
    """

    answer: str
    refused: bool
    sources: list[Source]


class SearchResponse(BaseModel):
    """The passages retrieved for a query.

    Attributes:
        query: The query that was searched.
        results: The retrieved passages, most relevant first.
    """

    query: str
    results: list[Source]


class HealthResponse(BaseModel):
    """Service health and readiness.

    Attributes:
        status: ``ok`` when the service is ready to serve.
        num_chunks: Number of indexed corpus chunks.
    """

    status: str
    num_chunks: int
