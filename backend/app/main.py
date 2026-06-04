import time
import uuid
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.models import (
    DocumentListResponse,
    IngestResponse,
    QueryRequest,
    QueryResponse,
)

log = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    from app.retrieval.store import init_db
    log.info("startup", database_url=settings.database_url.split("@")[-1])
    init_db()
    yield
    log.info("shutdown")


app = FastAPI(title="DocChat", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ingest", response_model=IngestResponse)
async def ingest(file: UploadFile = File(...)):
    from app.ingestion.pipeline import ingest_document

    filename = file.filename or "unnamed"
    allowed = {".pdf", ".txt", ".md"}
    suffix = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if suffix not in allowed:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {suffix or 'none'}")

    content = await file.read()

    max_bytes = settings.max_upload_mb * 1024 * 1024
    if len(content) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds {settings.max_upload_mb}MB limit",
        )

    document_id = str(uuid.uuid4())
    log.info("ingest.start", document_id=document_id, filename=filename, size=len(content))

    chunk_count = await ingest_document(
        content=content,
        filename=filename,
        document_id=document_id,
        suffix=suffix,
    )

    if chunk_count == 0:
        raise HTTPException(
            status_code=422,
            detail="No extractable text found. Scanned/image-only PDFs are not supported.",
        )

    log.info("ingest.done", document_id=document_id, chunks=chunk_count)

    return IngestResponse(
        document_id=document_id,
        filename=filename,
        chunks=chunk_count,
        message="Document indexed successfully",
    )


@app.get("/documents", response_model=DocumentListResponse)
def documents():
    from app.retrieval.store import list_documents
    return DocumentListResponse(documents=list_documents())


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    from app.generation.openai import generate
    from app.retrieval.planner import retrieve_with_planning

    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    t0 = time.perf_counter()
    log.info("query.start", question=request.question[:120])

    chunks = retrieve_with_planning(request.question, document_ids=request.document_ids)

    if not chunks:
        return QueryResponse(
            answer=(
                "I don't have enough information in the uploaded documents"
                " to answer this question."
            ),
            sources=[],
            tokens_used=0,
            latency_ms=round((time.perf_counter() - t0) * 1000, 1),
        )

    response = await generate(question=request.question, chunks=chunks)

    latency = round((time.perf_counter() - t0) * 1000, 1)
    log.info(
        "query.done",
        question=request.question[:120],
        chunks_retrieved=len(chunks),
        tokens_used=response.tokens_used,
        latency_ms=latency,
    )

    # tokens_used == 0 means the low-confidence guardrail fired — no grounded answer.
    # Returning sources alongside a refusal is inconsistent: we're saying "I don't know"
    # while pointing at documents. Return empty sources on refusal.
    sources = [] if response.tokens_used == 0 else [c.to_source() for c in chunks]

    return QueryResponse(
        answer=response.answer,
        sources=sources,
        tokens_used=response.tokens_used,
        latency_ms=latency,
    )
