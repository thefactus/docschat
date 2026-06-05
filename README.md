# DocChat: Chat With Your Docs

A conversational assistant that answers questions grounded in your own documents,
with citations. Upload PDFs / text / markdown, ask questions in a chat, and watch
the retrieval pipeline run live.

I chose the **Chat With Your Docs** option because it is the smallest useful surface
that still exposes the hard parts of AI engineering: chunking, hybrid retrieval,
query rewriting, guardrails, streaming UX, evals, and production trade-offs.

This is deliberately a **solid, well-tested core** rather than a feature-maximal
prototype. The scope cuts are explicit in the limitations section.

---

## Author note

I built DocChat to be small enough to understand quickly, but complete enough to
show the important parts of a real RAG product. You can upload a couple of
documents, ask natural questions, and get answers that show where they came from.
The UI is intentionally not just a chat box. It exposes the pipeline live because,
for a RAG system, the answer alone does not tell the full story. I wanted reviewers
to see when the system planned, searched, fused results, checked confidence,
generated, or refused.

The main engineering choice was to keep the system explicit. There is no big
orchestration framework hiding the work: the backend has a planner, hybrid
retrieval, fusion, a confidence guardrail, and generation. Conversation memory is
handled by rewriting follow-up questions into standalone queries before retrieval.
That makes the app feel like chat, but still keeps the retrieval step grounded in
clear document search.

---

## What it does

- **Upload** PDFs, `.txt`, or `.md` and ask questions about them.
- **Grounded answers with citations**: every answer cites the file (and page) it came from. If the documents don't contain the answer, it says so instead of guessing.
- **Real chat**: follow-up questions work. "And how many can they carry over?" is understood in the context of the previous turn.
- **Live pipeline view**: a toggleable diagram + dev log that shows every stage (query planning, hybrid retrieval, fusion, the confidence guardrail, generation) running in real time, with the answer streaming in token by token.
- **Document scoping**: restrict a question to specific documents.

---

## Screenshots

![A grounded answer with the live pipeline trace and a page citation.](screenshots/01-answer-with-pipeline.png)
*A grounded answer with the live pipeline trace and a page citation.*

![The pipeline trace up close.](screenshots/02-pipeline-trace.png)
*The pipeline trace up close — every stage with its real decision, scores, and timing.*

![An off-topic question is refused.](screenshots/03-guardrail-refusal.png)
*An off-topic question: the guardrail refuses and the generation step is skipped.*

---

## Quick start

**Requirements:** Docker + an OpenAI API key.

```bash
cp .env.example .env
# edit .env and set OPENAI_API_KEY=sk-...

docker compose up --build
```

- Frontend: <http://localhost:3000>
- Backend API: <http://localhost:8000> (`/health`, `/ingest`, `/documents`, `/query`, `/query/stream`)

The app boots without a key (`/health` responds), but `/ingest` and `/query` require `OPENAI_API_KEY` to be set.

Then open the frontend, drag a document into the sidebar, and ask a question.

Useful smoke-test prompts:

- `hi`: should short-circuit as a greeting, with no retrieval or generation.
- `What is this app?`: should short-circuit as a meta question.
- Ask a document question, then a follow-up like `and what about the carryover?`
  to exercise history-aware rewriting.
- Ask an off-topic question like `Give me a recipe for carbonara.`: the guardrail
  should refuse and skip the LLM node.

---

## Validation commands

```bash
# frontend compile/type gate
docker compose build frontend

# backend lint, from a local/dev Python environment
pip install -r backend/requirements.txt -r backend/requirements-dev.txt
(cd backend && python -m ruff check .)

# planner unit tests; pytest is also available in the backend container
docker compose exec backend python -m pytest tests/test_planner_intent.py

# full backend test suite, from the local/dev environment with testcontainers
(cd backend && python -m pytest)
```

---

## Architecture

Three containers via `docker compose`:

```
┌───────────────┐      ┌──────────────────────────────┐      ┌──────────────────┐
│  Frontend     │      │  Backend (FastAPI)           │      │  Postgres 16     │
│  Next.js 14   │ ───▶ │                              │ ───▶ │  + pgvector      │
│  (App Router) │      │  ingestion · retrieval ·     │      │                  │
│  Pipeline UI  │ ◀─── │  generation · guardrail      │ ◀─── │  vector + FTS    │
└───────────────┘ SSE  └──────────────┬───────────────┘      └──────────────────┘
                                       │
                                       ▼
                                  OpenAI API
                          (embeddings + chat, streaming)
```

**Request flow for a question (`POST /query` / `/query/stream`):**

1. **Query planning** (`retrieval/planner.py`): one LLM call classifies intent
   (`greeting` / `meta` / `doc_question`) and, using the recent conversation history,
   rewrites the message into a self-contained query (and optionally decomposes it into
   sub-queries). Greetings/meta short-circuit here: no retrieval, no cost.
2. **Hybrid retrieval** (`retrieval/{vector,fts}.py`): per (sub-)query, runs **vector
   search** (pgvector cosine) and **full-text search** (Postgres FTS / BM25-style) in
   parallel.
3. **Fusion** (`retrieval/fusion.py`): min-max normalizes each arm's scores and combines
   them with configurable weights (default 0.7 vector / 0.3 FTS), then truncates to `top_k`.
4. **Confidence guardrail** (`generation/openai.py`): if the best raw cosine score is below
   `low_confidence_threshold` (0.20), it refuses instead of generating.
5. **Generation**: a grounded prompt (answer only from the retrieved chunks, cite sources,
   refuse if insufficient) with recent history for conversational coherence. `/query/stream`
   streams the answer token by token over SSE.

The streaming endpoint (`/query/stream`) mirrors the plain endpoint and is used by
the live pipeline view. `/query` remains the simple request/response path and the
fallback for environments where streaming is unavailable.

---

## RAG / LLM approach & decisions

| Concern | Choice | Why |
|---|---|---|
| **Embeddings** | OpenAI `text-embedding-3-small` (1536-dim) | Cheap, current, good enough; 1536 dims keep the schema light. |
| **Generation / planner** | OpenAI `gpt-4.1-mini` (configurable via env) | Small, fast model sized to the task; same provider as embeddings. |
| **Vector store** | Postgres + pgvector | One datastore for relational + vector + full-text: no separate vector DB to operate. |
| **Orchestration** | Hand-written (no LangChain) | The pipeline is small and explicit; a framework would add indirection without payoff. |

**Chunking.** Character-based, ~1500 chars with 300-char overlap, split on paragraph ->
sentence boundaries (not fixed cuts). Character-based was a deliberate choice for
simplicity and determinism: no coupling to a specific tokenizer; the quality lever is
boundary-aware splitting, not the unit of measurement. PDFs are chunked **per page** so
every chunk carries its page number, which is what makes the citations accurate.

**Retrieval: hybrid.** Vector search handles semantic similarity; full-text (BM25-style)
handles exact terms, names, and acronyms. They're fused with weighted, min-max-normalized
scores. Min-max preserves the *relative spacing* of candidates within a result set (better
than rank-only fusion like RRF) while keeping the configurable weights meaningful.

**Context management.** Conversation history is bounded (last ~6 turns) and used in two
places: the planner rewrites elliptical follow-ups into standalone queries (so *retrieval*
has a topic anchor, not just generation), and generation gets the recent turns for coherence.
History is held by the frontend and sent with each request: the backend is stateless.

**Guardrails.** Two layers: (1) a confidence gate on the raw cosine score before generation,
and (2) the grounded prompt itself, which refuses when the context is insufficient. A refused
answer returns empty sources and zero tokens.

**Quality / evaluation.** A token-aware eval harness (`backend/evals/`) with a golden set
measures retrieval hit-rate, citation accuracy, a lexical answer-match proxy, refusal accuracy,
and scoping correctness. It's how the guardrail threshold was tuned (see below) rather than
guessed. See **Testing & evaluation**.

**Observability.** Structured logging (`structlog`) on every stage; the live pipeline view is
end-user-facing observability: it surfaces each stage's real decision, scores, and timing.

**Why no agent framework?** The project uses one planner call plus an explicit retrieval and
generation pipeline. That shape is small enough that a framework like LangGraph or LangChain
would add abstraction before it adds leverage. If I later added tool use, long-running workflows,
or resumable multi-step tasks, I would revisit that decision.

---

## Testing & evaluation

**Unit / integration tests**:

- `test_chunking.py`, `test_fusion.py`, `test_evals.py`, `test_planner_intent.py`,
  `test_stream_utils.py`: pure functions, no DB, no API.
- `test_db.py`: real pgvector via testcontainers with **fake embeddings**, exercising the
  SQL layer (the `::vector` cast, the FTS `numnode` guard, scoping). Token-free on purpose:
  it tests the wiring, not the model.

**Eval harness** (`python -m evals.run_evals`, needs the DB + an API key):

```
DocChat eval — 15 items (12 answerable, 3 refuse)  top_k=6
  retrieval_hit_rate       12/12    1.00
  citation_accuracy         3/3     1.00
  answer_match (partial)    mean 1.00   n=12
  refusal_accuracy          3/3 refuse + 12/12 answerable   1.00
  scoping_correctness       1/1     1.00
```

**Tuning the guardrail threshold with data.** The harness can sweep the threshold. The score
distribution showed off-topic questions top out well below legitimate document questions, so a
threshold sweep confirmed the safe band:

```
 threshold   ret_hit   refusal   ans_match
      0.05      1.00      1.00        0.95
      0.20      1.00      1.00        1.00
      0.30      1.00      1.00        0.95
      0.40      1.00      0.71        0.73   <- legit questions wrongly refused
```

`0.30` sat only ~0.01 above the weakest legitimate question, so I lowered the threshold to
**0.20** to center it in the gap. That decision came from the eval, not intuition.

---

## Key technical decisions

I chose tools I know well and that keep the system simple, so my time went into the quality
of the RAG rather than fighting the stack.

- **FastAPI**: the minimum needed for a clean, typed API, no boilerplate.
- **Postgres + pgvector**: the choice that saved the most complexity. Instead of running a
  separate vector database, Postgres does relational + vector + full-text in one place, which
  gave me hybrid retrieval without operating a second datastore. I also had prior experience
  with it.
- **OpenAI**: what I've been using for AI systems; embeddings and generation from one provider,
  with native streaming.

If the corpus grew large, I'd revisit a dedicated vector DB and a reranking stage. But for this
scope, one well-understood datastore was the right call.

The main architectural choice was to keep the backend stateless for chat memory. The frontend
sends the last few non-error turns with each request. That keeps the demo simple and avoids
session infrastructure, while still exercising the important behavior: retrieval sees rewritten,
self-contained follow-up questions instead of ambiguous fragments.

---

## Engineering standards

**Followed:** containerized one-command setup; typed Python (Pydantic) and TypeScript; linting
(ruff) and a compile/type gate in CI; token-free unit + integration tests; an eval harness as a
first-class quality discipline; structured logging; conventional commits; secrets kept out of the
repo (`.env` gitignored, `.env.example` only).

**Skipped (and acknowledged):** no auth / multi-tenancy; no persistent sessions (chat history is
client-side and ephemeral); no rate limiting; connection-per-request instead of a pool; no queued
ingestion for large batches; the eval's LLM-judge tier exists but is opt-in. These are deliberate
scope cuts for a take-home, not oversights.

---

## How I used AI tools

I treated development as a **team of agents that I orchestrated**, not as autocomplete.

- **I** owned the system design, the features, the improvements, and the overall plan: every
  product and architecture decision was mine.
- An **orchestrator agent** (Claude Code) wrote the plan with me and ran reviews.
- A **coder agent** (Claude Code) implemented.
- A third agent (**Codex**) added an independent review pass.

The rule that mattered most: **nothing was "done" until it actually ran.** The orchestrator didn't
trust "the tests pass". It brought up `docker compose`, ran the eval, and exercised the app. That
caught bugs static review missed: a committed syntax error that broke boot, a pgvector type cast, a
non-existent SQL function. All were invisible to the unit tests, and all were found only by running it.

**Do's:** review against criteria, not vibes; verify by running; isolate new features from verified
ones. **Don'ts:** accept "done" without executing; let an agent make product decisions.

That workflow was useful, but only because I kept the bar concrete: code review, unit tests,
Docker builds, eval runs, and browser smoke tests. The AI agents accelerated implementation;
they did not replace engineering judgment.

---

## What I'd do differently with more time

- **Reranking stage**: a cross-encoder to re-score retrieved candidates for precision. I
  *deliberately didn't* add one: the eval showed strong retrieval recall (1.0) on this corpus, so it
  wasn't justified. With a larger corpus where ordering matters more, it would be the next lever.
- **A more discriminating eval set**: the current golden set passes at ~1.0, so it doesn't separate
  good from better. Harder cases (ambiguous questions, exact-term / FTS-only lookups, near-miss
  off-topic) would let the eval catch quality regressions and justify tuning by contrast.
- **Resumable chats**: conversation history is currently held in the frontend and lost on refresh.
  I'd persist sessions server-side (`conversation_id`) so chats survive reloads and can be resumed.
- **Productionizing**: see below.

---

## Demo notes

The live pipeline is intentionally visible because it makes the system easier to evaluate:

- greeting/meta turns show the planner short-circuit and dim downstream nodes;
- normal document questions show retrieval, fusion, guardrail, generation, and citations;
- follow-ups show the planner rewrite state;
- low-confidence questions show the guardrail refuse path and skip generation.

This is also how I would debug the product with a stakeholder: not by explaining RAG abstractly,
but by showing which stage made which decision.

---

## Productionizing on a hyperscaler

To run this on AWS / GCP / Azure / Cloudflare:

- **Compute:** containers on a managed runtime (ECS/Fargate, Cloud Run, or k8s); the frontend as a
  Next.js standalone build on the same or on Vercel/Cloudflare.
- **Database:** managed Postgres with the pgvector extension (e.g. RDS / Cloud SQL / Aurora), with a
  connection pool (PgBouncer) instead of the current connection-per-request.
- **Secrets:** a secrets manager + rotation instead of `.env`.
- **Scale & cost:** batch/queue ingestion for large uploads; cache embeddings; an HNSW index is
  already created, but I'd tune it and consider a dedicated vector store if the corpus grows; add a
  reranking stage where the data justifies it.
- **Reliability & security:** auth + multi-tenancy, per-tenant rate limiting, input/file validation
  hardening, and an output-validation guardrail.
- **Observability:** ship the structured logs to a backend (e.g. OpenTelemetry -> a tracing/metrics
  stack), with dashboards for latency, token cost, retrieval scores, and refusal rate.
- **CI/CD:** the existing lint + test gates run on GitHub Actions; add the integration suite and a
  gated deploy.

---

## Known limitations

- **Enumeration / completeness**: for "list every passage that mentions X", retrieval surfaces the
  relevant chunks but generation tends to return one and phrase it as definitive. The fix is intent
  routing (a lexical filter for exhaustive term lookup); not implemented.
- **No persistent storage of chats** (see above).
- **Eval corpus** references the sample documents; the larger PDFs used in testing aren't committed.
