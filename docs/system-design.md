# DocChat System Design

DocChat lets a user upload documents and ask questions about them. The system retrieves
relevant document chunks first, then asks an OpenAI chat model to answer only from that
retrieved context with citations.

OpenAI creates the embeddings; pgvector stores and searches them.

## High-Level Flow

```text
User uploads document
      │
      ▼
Parse PDF / TXT / MD
      │
      ▼
Split into chunks
      │
      ▼
Create embeddings with OpenAI
      │
      ▼
Store chunks in PostgreSQL
      │
      ├─ embedding stored with pgvector
      └─ text indexed with PostgreSQL full-text search
```

```text
User asks a question
      │
      ▼
Query planner refines/decomposes the question
      │
      ▼
Run retrieval for each query
      │
      ├─ vector search with pgvector
      └─ keyword search with full-text search
      │
      ▼
Merge + dedupe chunks
      │
      ▼
Normalize scores + weighted fusion
      │
      ▼
Send top chunks to OpenAI chat model
      │
      ▼
Return answer with citations
```

## Architecture

```text
                  ┌──────────────────┐
                  │      User        │
                  └────────┬─────────┘
                           │
                           ▼
                  ┌──────────────────┐
                  │  Next.js UI      │
                  └───────┬──────────┘
                          │
             ┌────────────┴────────────┐
             ▼                         ▼
      POST /ingest                POST /query
             │                         │
             ▼                         ▼
   ┌─────────────────┐       ┌─────────────────────┐
   │ Parse + chunk   │       │ Query planner       │
   │ documents       │       │ OpenAI chat model   │
   └────────┬────────┘       └──────────┬──────────┘
            │                           │
            ▼                           ▼
   ┌─────────────────┐       ┌─────────────────────┐
   │ OpenAI          │       │ Hybrid retrieval    │
   │ embeddings      │       │ vector + keyword    │
   └────────┬────────┘       └──────────┬──────────┘
            │                           │
            ▼                           ▼
   ┌───────────────────────────────────────────────┐
   │ PostgreSQL                                    │
   │ documents + chunks + pgvector + full-text     │
   └───────────────────────┬───────────────────────┘
                           │
                           ▼
                  ┌──────────────────┐
                  │ OpenAI generator │
                  │ grounded answer  │
                  └────────┬─────────┘
                           │
                           ▼
                  Answer + citations
```

## Key Design Choices

**Query planning**

The raw user question is not always the best retrieval query. If the user asks:

```text
What are the pricing terms and termination clauses in this contract?
```

The planner can split it into:

```text
pricing terms
termination clauses
```

That improves recall because each subquery can retrieve the right section independently.
If planning fails, the system falls back to the original question.

This is not an agentic workflow; it is a small retrieval-planning step with structured
JSON output and fallback.

**Hybrid retrieval**

Vector search finds semantically similar chunks. Full-text search catches exact terms such
as product names, clause names, acronyms, dates, and error codes. The system combines both
signals instead of trusting only one retrieval method.

Because pgvector similarity scores and PostgreSQL full-text scores are on different scales,
each result set is normalized before weighted fusion.

**Grounded generation**

The chat model is instructed not to answer from memory. It receives only the top retrieved
chunks and is required by the prompt to cite the sources it used. If the documents do not
contain enough information, it should say so.

## Why This Design

This keeps the first version focused while showing the important RAG engineering decisions:

- improve the query before retrieval;
- retrieve with both semantic and exact-match search;
- normalize and fuse retrieval scores;
- generate answers only from cited document context;
- keep the system easy to run locally with Docker Compose.
