from typing import Literal

from pydantic import BaseModel


class HistoryMessage(BaseModel):
    role: Literal['user', 'assistant']
    content: str


class IngestResponse(BaseModel):
    document_id: str
    filename: str
    chunks: int
    message: str


class ChunkSource(BaseModel):
    document_id: str
    filename: str
    page: int | None
    chunk_index: int
    score: float


class QueryRequest(BaseModel):
    question: str
    document_ids: list[str] | None = None
    history: list[HistoryMessage] = []


class QueryResponse(BaseModel):
    answer: str
    sources: list[ChunkSource]
    tokens_used: int
    latency_ms: float
    intent: str = 'doc_question'


class DocumentInfo(BaseModel):
    document_id: str
    filename: str
    chunks: int


class DocumentListResponse(BaseModel):
    documents: list[DocumentInfo]
