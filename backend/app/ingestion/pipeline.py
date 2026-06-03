import io
import re
import uuid

import structlog
from openai import OpenAI

from app.config import settings

log = structlog.get_logger()

# chunk_size and chunk_overlap are in characters. Character-based splitting is
# simpler and deterministic — no coupling to a specific tokenizer. At ~4 chars/
# token for English text, chunk_size=1500 yields ~375 tokens, well within
# text-embedding-3-small's 8191-token window. The real quality lever is
# boundary-aware splitting (paragraph → sentence), not the unit of measurement.


def _parse(content: bytes, suffix: str) -> list[tuple[int | None, str]]:
    """Return (page, text) pairs. PDFs yield one pair per page (1-indexed);
    plain text and markdown yield a single pair with page=None."""
    if suffix == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(content))
        result = []
        for i, pdf_page in enumerate(reader.pages, 1):
            text = pdf_page.extract_text() or ""
            if text.strip():
                result.append((i, text))
        return result
    else:
        return [(None, content.decode("utf-8", errors="replace"))]


def _split(text: str) -> list[str]:
    """Split text on paragraph/sentence boundaries, packing segments greedily
    up to chunk_size characters. The tail of each chunk is carried forward as
    overlap so context is not lost at boundaries."""
    size = settings.chunk_size
    overlap = settings.chunk_overlap

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    current = ""

    for para in paragraphs:
        candidate = (current + "\n\n" + para) if current else para
        if current and len(candidate) > size:
            chunks.append(current)
            tail = current[-overlap:] if len(current) > overlap else current
            current = tail.lstrip() + " " + para
        else:
            current = candidate

        # A single paragraph longer than size: split on sentence boundaries.
        while len(current) > size:
            split_at = size
            for sep in (". ", "! ", "? ", "; ", " "):
                idx = current.rfind(sep, size // 2, size)
                if idx != -1:
                    split_at = idx + len(sep)
                    break
            chunks.append(current[:split_at].rstrip())
            tail = current[max(0, split_at - overlap) : split_at]
            current = tail + current[split_at:]

    if current.strip():
        chunks.append(current.strip())

    return [c for c in chunks if len(c) > 30]


def _embed(texts: list[str]) -> list[list[float]]:
    client = OpenAI(api_key=settings.openai_api_key)
    result: list[list[float]] = []
    for i in range(0, len(texts), 100):
        resp = client.embeddings.create(
            model=settings.embedding_model,
            input=texts[i : i + 100],
        )
        result.extend(item.embedding for item in sorted(resp.data, key=lambda x: x.index))
    return result


async def ingest_document(
    content: bytes, filename: str, document_id: str, suffix: str
) -> int:
    from app.retrieval.store import store_chunks

    pages = _parse(content, suffix)

    # Chunk within each page so every chunk carries exact page provenance.
    all_chunks: list[tuple[int | None, str]] = []
    for page, text in pages:
        for chunk_text in _split(text):
            all_chunks.append((page, chunk_text))

    if not all_chunks:
        log.warning("ingest.no_chunks", document_id=document_id, filename=filename)
        return 0

    embeddings = _embed([text for _, text in all_chunks])

    rows = [
        {
            "id": str(uuid.uuid4()),
            "document_id": document_id,
            "filename": filename,
            "page": page,
            "chunk_index": idx,
            "content": text,
            "embedding": embedding,
        }
        for idx, ((page, text), embedding) in enumerate(zip(all_chunks, embeddings))
    ]

    store_chunks(rows)
    log.info("ingest.done", document_id=document_id, chunks=len(rows))
    return len(rows)
