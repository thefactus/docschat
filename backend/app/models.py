from pydantic import BaseModel


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


class QueryResponse(BaseModel):
    answer: str
    sources: list[ChunkSource]
    tokens_used: int
    latency_ms: float


class DocumentInfo(BaseModel):
    document_id: str
    filename: str
    chunks: int


class DocumentListResponse(BaseModel):
    documents: list[DocumentInfo]
