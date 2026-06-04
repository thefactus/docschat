#!/usr/bin/env python3
"""
DocChat evaluation harness.

Tier A (default): retrieval hit, citation accuracy, lexical answer match,
                  refusal accuracy, scoping correctness.
Tier B (opt-in):  LLM judge for groundedness and correctness.

Usage (from backend/):
  python -m evals.run_evals
  python -m evals.run_evals --sweep-threshold 0.25,0.30,0.35
  python -m evals.run_evals --sweep-weights 0.6/0.4,0.7/0.3,0.8/0.2
  EVAL_JUDGE=1 python -m evals.run_evals

Requires the live DB (docker compose up -d db) and OPENAI_API_KEY.
The three corpus documents must already be indexed.
"""

import argparse
import asyncio
import itertools
import json
import os
import re
import sys
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Generator

import yaml

# ---------------------------------------------------------------------------
# Path bootstrap — makes app.* importable when run as python -m evals.run_evals
# ---------------------------------------------------------------------------
EVALS_DIR = Path(__file__).parent
BACKEND_DIR = EVALS_DIR.parent
sys.path.insert(0, str(BACKEND_DIR))

import app.generation.openai as _gen_mod  # noqa: E402
from app.config import settings  # noqa: E402
from app.retrieval.planner import retrieve_with_planning  # noqa: E402
from app.retrieval.store import list_documents  # noqa: E402

GOLDEN_PATH = EVALS_DIR / "golden_set.yaml"
OUTPUT_PATH = EVALS_DIR / "last_run.json"
REFUSAL_MARKER = "I don't have enough information"
_NUMBER_ALIASES = {
    "0": "zero",
    "1": "one",
    "2": "two",
    "3": "three",
    "4": "four",
    "5": "five",
    "6": "six",
    "7": "seven",
    "8": "eight",
    "9": "nine",
    "10": "ten",
    "11": "eleven",
    "12": "twelve",
    "13": "thirteen",
    "14": "fourteen",
    "15": "fifteen",
    "20": "twenty",
    "25": "twentyfive",
}


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class Source:
    filename: str
    page: int | None
    score: float
    content: str  # truncated; used for Tier B judge context


@dataclass
class ItemResult:
    id: str
    type: str
    question: str
    answer: str
    tokens_used: int
    refused: bool
    sources: list[Source] = field(default_factory=list)
    retrieval_hit: bool | None = None    # None for refuse items
    citation_hit: bool | None = None     # None if no expect_pages
    answer_hit: float | None = None      # None for refuse; float 0–1 = partial credit
    scoping_ok: bool | None = None       # None if no scope
    judge_groundedness: float | None = None
    judge_correctness: float | None = None
    failures: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Config override context manager (used by sweep)
# ---------------------------------------------------------------------------

@contextmanager
def config_override(
    threshold: float | None = None,
    vec_weight: float | None = None,
    fts_weight: float | None = None,
) -> Generator[None, None, None]:
    old_t = _gen_mod._LOW_CONFIDENCE_THRESHOLD
    old_v = settings.fusion_vector_weight
    old_f = settings.fusion_fts_weight
    try:
        if threshold is not None:
            _gen_mod._LOW_CONFIDENCE_THRESHOLD = threshold
        if vec_weight is not None:
            settings.fusion_vector_weight = vec_weight
        if fts_weight is not None:
            settings.fusion_fts_weight = fts_weight
        yield
    finally:
        _gen_mod._LOW_CONFIDENCE_THRESHOLD = old_t
        settings.fusion_vector_weight = old_v
        settings.fusion_fts_weight = old_f


# ---------------------------------------------------------------------------
# Document resolver (filename → document_id)
# ---------------------------------------------------------------------------

_doc_cache: dict[str, str] | None = None


def resolve_scope(filenames: list[str]) -> list[str]:
    """Map filenames to document_ids using the live DB."""
    global _doc_cache
    if _doc_cache is None:
        _doc_cache = {d.filename: d.document_id for d in list_documents()}
    missing = [f for f in filenames if f not in _doc_cache]
    if missing:
        print(f"  WARNING: scope filenames not found in DB: {missing}")
    return [_doc_cache[f] for f in filenames if f in _doc_cache]


def indexed_filenames() -> set[str]:
    """Return filenames currently indexed in the live DB."""
    global _doc_cache
    if _doc_cache is None:
        _doc_cache = {d.filename: d.document_id for d in list_documents()}
    return set(_doc_cache)


def missing_expected_documents(items: list[dict]) -> list[str]:
    """Find expected/scope documents that are absent from the live corpus."""
    expected: set[str] = set()
    for item in items:
        expected.update(item.get("expect_documents", []))
        expected.update(item.get("scope", []))
    return sorted(expected - indexed_filenames())


# ---------------------------------------------------------------------------
# Single-item runner
# ---------------------------------------------------------------------------

async def run_item(
    item: dict,
    threshold: float | None = None,
    vec_weight: float | None = None,
    fts_weight: float | None = None,
) -> ItemResult:
    scope_ids = resolve_scope(item["scope"]) if "scope" in item else None

    with config_override(threshold, vec_weight, fts_weight):
        chunks = retrieve_with_planning(item["question"], document_ids=scope_ids)
        gen = await _gen_mod.generate(question=item["question"], chunks=chunks)

    sources = [
        Source(
            filename=c.filename,
            page=c.page,
            score=round(c.raw_vec_score, 4),
            content=c.content[:400],
        )
        for c in chunks
    ]
    refused = gen.tokens_used == 0 or REFUSAL_MARKER in gen.answer

    result = ItemResult(
        id=item["id"],
        type=item["type"],
        question=item["question"],
        answer=gen.answer,
        tokens_used=gen.tokens_used,
        refused=refused,
        sources=sources,
    )
    _score(result, item)
    return result


def _score(result: ItemResult, item: dict) -> None:
    """Compute per-item metric verdicts and append to result.failures."""
    source_filenames = {s.filename for s in result.sources}

    if item["type"] == "answerable":
        expected_docs: list[str] = item.get("expect_documents", [])

        # Retrieval hit: all expected documents appear in sources
        result.retrieval_hit = all(f in source_filenames for f in expected_docs)
        if not result.retrieval_hit:
            missing = [f for f in expected_docs if f not in source_filenames]
            result.failures.append(
                f"retrieval_miss: {missing} not in sources "
                f"(got: {sorted(source_filenames)})"
            )

        # Citation accuracy: for each expected doc, at least one source has an expected page
        expected_pages: list[int] | None = item.get("expect_pages")
        if expected_pages:
            ok = True
            for doc in expected_docs:
                pages_found = {
                    s.page for s in result.sources
                    if s.filename == doc and s.page is not None
                }
                if not any(p in expected_pages for p in pages_found):
                    ok = False
                    result.failures.append(
                        f"citation_miss: expected page(s) {expected_pages} for "
                        f"'{doc}', got pages {sorted(pages_found)}"
                    )
            result.citation_hit = ok

        # Partial-credit lexical match: (matched / expected) averaged across items.
        # All-or-nothing was brittle — a single prose variant flips a correct retrieval
        # to failure. Partial credit keeps the metric stable run-to-run while still
        # surfacing per-item misses in the failures list.
        expected_strings: list[str] = item.get("expect_answer_contains", [])
        if expected_strings:
            matched = [s for s in expected_strings if _contains_lexical_fact(result.answer, s)]
            result.answer_hit = round(len(matched) / len(expected_strings), 4)
            missing_strings = [s for s in expected_strings if s not in matched]
            if missing_strings:
                result.failures.append(
                    f"answer_miss ({len(matched)}/{len(expected_strings)} keywords): "
                    f"{missing_strings} not found. "
                    f"answer[:200]: {result.answer[:200]!r}"
                )

        if result.refused:
            result.failures.append("unexpected_refusal: answerable item was refused")

    else:  # "refuse"
        if not result.refused:
            result.failures.append(
                f"unexpected_answer: {result.answer[:100]!r}"
            )

    # Scoping correctness: no sources from outside declared scope
    if "scope" in item:
        scope_set = set(item["scope"])
        leaks = [s.filename for s in result.sources if s.filename not in scope_set]
        result.scoping_ok = len(leaks) == 0
        if not result.scoping_ok:
            result.failures.append(f"scope_leak: out-of-scope sources: {leaks}")


def _compact(text: str) -> str:
    """Lowercase and remove punctuation/space for resilient lexical matching.

    This keeps Tier A cheap while avoiding false negatives like "LangGraph" vs
    "Lang Graph". Raw case-insensitive substring is still checked first.
    """
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def _contains_lexical_fact(answer: str, expected: str) -> bool:
    answer_lower = answer.lower()
    expected_lower = expected.lower()
    compact_answer = _compact(answer)
    compact_expected = _compact(expected)
    if expected_lower in answer_lower or compact_expected in compact_answer:
        return True
    alias = _NUMBER_ALIASES.get(compact_expected)
    return bool(alias and alias in compact_answer)


# ---------------------------------------------------------------------------
# Tier B: LLM judge
# ---------------------------------------------------------------------------

async def run_judge(result: ItemResult) -> None:
    from openai import OpenAI

    client = OpenAI(api_key=settings.openai_api_key)
    context = "\n\n---\n\n".join(
        f"[{s.filename}, p.{s.page}]\n{s.content}" for s in result.sources
    )
    prompt = (
        "Rate this Q&A on two dimensions (1-5 integers).\n\n"
        f"Context:\n{context[:2500]}\n\n"
        f"Question: {result.question}\nAnswer: {result.answer}\n\n"
        "Groundedness (1=unsupported claims, 5=every claim supported by context)\n"
        "Correctness (1=wrong or missing key facts, 5=accurate and complete)\n\n"
        'JSON only: {"groundedness": <1-5>, "correctness": <1-5>}'
    )
    resp = client.chat.completions.create(
        model=settings.generation_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    data = json.loads(resp.choices[0].message.content or "{}")
    result.judge_groundedness = float(data.get("groundedness", 3))
    result.judge_correctness = float(data.get("correctness", 3))


# ---------------------------------------------------------------------------
# Metrics aggregation
# ---------------------------------------------------------------------------

def compute_metrics(results: list[ItemResult]) -> dict:
    answerable = [r for r in results if r.type == "answerable"]
    refuse_items = [r for r in results if r.type == "refuse"]
    cit_items = [r for r in results if r.citation_hit is not None]
    scp_items = [r for r in results if r.scoping_ok is not None]

    def rate(hits: int, n: int) -> float:
        return round(hits / n, 4) if n else 0.0

    ret_hits = sum(1 for r in answerable if r.retrieval_hit)
    cit_hits = sum(1 for r in cit_items if r.citation_hit)
    ans_scores = [r.answer_hit for r in answerable if r.answer_hit is not None]
    ans_mean = round(sum(ans_scores) / len(ans_scores), 4) if ans_scores else 0.0
    ref_correct = sum(1 for r in refuse_items if r.refused)
    not_refused = sum(1 for r in answerable if not r.refused)
    scp_hits = sum(1 for r in scp_items if r.scoping_ok)
    gnd = [r.judge_groundedness for r in results if r.judge_groundedness is not None]
    cor = [r.judge_correctness for r in results if r.judge_correctness is not None]

    return {
        "retrieval_hit_rate": {
            "hits": ret_hits, "n": len(answerable),
            "rate": rate(ret_hits, len(answerable)),
        },
        "citation_accuracy": {
            "hits": cit_hits, "n": len(cit_items),
            "rate": rate(cit_hits, len(cit_items)),
        },
        "answer_match": {
            "mean": ans_mean, "n": len(ans_scores),
        },
        "refusal_accuracy": {
            "refuse_correct": ref_correct,
            "refuse_total": len(refuse_items),
            "answerable_not_refused": not_refused,
            "answerable_total": len(answerable),
            "combined_rate": rate(ref_correct + not_refused, len(results)),
        },
        "scoping_correctness": {
            "hits": scp_hits, "n": len(scp_items),
            "rate": rate(scp_hits, len(scp_items)),
        },
        "judge_groundedness": {
            "mean": round(sum(gnd) / len(gnd), 2) if gnd else None,
            "n": len(gnd),
        },
        "judge_correctness": {
            "mean": round(sum(cor) / len(cor), 2) if cor else None,
            "n": len(cor),
        },
    }


# ---------------------------------------------------------------------------
# Score distribution helper
# ---------------------------------------------------------------------------

def _median(vals: list[float]) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    n = len(s)
    mid = n // 2
    return (s[mid - 1] + s[mid]) / 2.0 if n % 2 == 0 else s[mid]


def _print_score_distribution(results: list[ItemResult]) -> None:
    """Print max off-topic vs min/median answerable raw cosine scores.

    This single comparison is the clearest justification for the 0.30
    guardrail threshold: if the gap between off-topic max and answerable min
    is large and the threshold sits in the middle, the threshold is safe.
    """
    refuse_top = [
        max((s.score for s in r.sources), default=0.0)
        for r in results if r.type == "refuse"
    ]
    answerable_top = [
        max((s.score for s in r.sources), default=0.0)
        for r in results if r.type == "answerable"
    ]
    if not refuse_top or not answerable_top:
        return

    off_max = max(refuse_top)
    ans_min = min(answerable_top)
    ans_med = _median(answerable_top)
    threshold = _gen_mod._LOW_CONFIDENCE_THRESHOLD
    gap = ans_min - off_max

    print(f"\nScore distribution  [guardrail threshold: {threshold:.2f}]")
    print(f"  off-topic top score  (n={len(refuse_top)}):     max={off_max:.2f}")
    print(
        f"  answerable top score (n={len(answerable_top)}):  "
        f"min={ans_min:.2f}  median={ans_med:.2f}"
    )
    print(
        f"  gap={gap:.2f}  "
        f"(threshold {threshold:.2f} is {threshold - off_max:.2f} above off-topic max, "
        f"{ans_min - threshold:.2f} below answerable min)"
    )


# ---------------------------------------------------------------------------
# Scorecard printer
# ---------------------------------------------------------------------------

def print_scorecard(results: list[ItemResult], metrics: dict) -> None:
    answerable = [r for r in results if r.type == "answerable"]
    refuse_items = [r for r in results if r.type == "refuse"]
    m = metrics
    rh = m["retrieval_hit_rate"]
    ca = m["citation_accuracy"]
    am = m["answer_match"]
    ra = m["refusal_accuracy"]
    sc = m["scoping_correctness"]
    gnd = m["judge_groundedness"]
    cor = m["judge_correctness"]

    print(
        f"\nDocChat eval — {len(results)} items "
        f"({len(answerable)} answerable, {len(refuse_items)} refuse)  "
        f"top_k={settings.retrieval_top_k}"
    )
    print(f"  retrieval_hit_rate       {rh['hits']}/{rh['n']}    {rh['rate']:.2f}")
    if ca["n"] > 0:
        print(f"  citation_accuracy         {ca['hits']}/{ca['n']}     {ca['rate']:.2f}")
    print(f"  answer_match (partial)    mean {am['mean']:.2f}   n={am['n']}")
    print(
        f"  refusal_accuracy          "
        f"{ra['refuse_correct']}/{ra['refuse_total']} refuse + "
        f"{ra['answerable_not_refused']}/{ra['answerable_total']} answerable   "
        f"{ra['combined_rate']:.2f}"
    )
    if sc["n"] > 0:
        print(f"  scoping_correctness       {sc['hits']}/{sc['n']}     {sc['rate']:.2f}")
    if gnd["n"] > 0:
        print(f"  [judge] groundedness      mean {gnd['mean']:.1f}/5   (EVAL_JUDGE=1)")
    if cor["n"] > 0:
        print(f"  [judge] correctness       mean {cor['mean']:.1f}/5")

    # Score distribution — the key evidence for the 0.30 guardrail threshold.
    # Off-topic items score low raw cosine (0.05–0.10); answerable items score high
    # (0.6+). The gap between them is why 0.30 is a safe threshold.
    _print_score_distribution(results)

    failures = [r for r in results if r.failures]
    if failures:
        print("\nFailures:")
        for r in failures:
            src_str = ", ".join(f"{s.filename} p.{s.page}" for s in r.sources[:4])
            print(f"  [{r.id}]")
            for note in r.failures:
                print(f"    {note}")
            print(f"    sources: [{src_str}]")
    else:
        print("\nAll items passed ✓")
    print()


# ---------------------------------------------------------------------------
# Standard run
# ---------------------------------------------------------------------------

async def run_standard(items: list[dict], judge: bool) -> None:
    print(f"Running {len(items)} eval items  Tier {'A+B' if judge else 'A'}…\n")
    results: list[ItemResult] = []

    for item in items:
        print(f"  [{item['id']}]", end="", flush=True)
        try:
            result = await run_item(item)
            if judge and item["type"] == "answerable" and not result.refused:
                await run_judge(result)
            print(f"  {'✓' if not result.failures else '✗'}")
        except Exception as exc:
            print(f"  ERROR: {exc}")
            result = ItemResult(
                id=item["id"], type=item["type"],
                question=item["question"], answer="",
                tokens_used=0, refused=False,
                failures=[f"runner_error: {exc}"],
            )
        results.append(result)

    metrics = compute_metrics(results)
    print_scorecard(results, metrics)

    output = {
        "config": {
            "top_k": settings.retrieval_top_k,
            "fusion_vector_weight": settings.fusion_vector_weight,
            "fusion_fts_weight": settings.fusion_fts_weight,
            "low_confidence_threshold": _gen_mod._LOW_CONFIDENCE_THRESHOLD,
        },
        "metrics": metrics,
        "results": [asdict(r) for r in results],
    }
    OUTPUT_PATH.write_text(json.dumps(output, indent=2, default=str))
    print(f"Full results → {OUTPUT_PATH}")


# ---------------------------------------------------------------------------
# Sweep mode
# ---------------------------------------------------------------------------

async def run_sweep(items: list[dict], args: argparse.Namespace) -> None:
    thresholds = (
        [float(t) for t in args.sweep_threshold.split(",")]
        if args.sweep_threshold
        else [_gen_mod._LOW_CONFIDENCE_THRESHOLD]
    )
    weight_pairs = (
        [
            (float(p.split("/")[0]), float(p.split("/")[1]))
            for p in args.sweep_weights.split(",")
        ]
        if args.sweep_weights
        else [(settings.fusion_vector_weight, settings.fusion_fts_weight)]
    )

    n_configs = len(thresholds) * len(weight_pairs)
    print(
        f"\nSweep: {len(thresholds)} threshold(s) × {len(weight_pairs)} weight pair(s) "
        f"= {n_configs} config(s), {n_configs * len(items)} API calls\n"
    )
    header = (
        f"{'threshold':>10}  {'vec/fts':>8}  "
        f"{'ret_hit':>8}  {'refusal':>8}  {'ans_match':>10}"
    )
    print(header)
    print("-" * len(header))

    for threshold, (vec_w, fts_w) in itertools.product(thresholds, weight_pairs):
        results = []
        for item in items:
            try:
                r = await run_item(
                    item, threshold=threshold, vec_weight=vec_w, fts_weight=fts_w
                )
            except Exception as exc:
                r = ItemResult(
                    id=item["id"], type=item["type"],
                    question=item["question"], answer="",
                    tokens_used=0, refused=False,
                    failures=[f"runner_error: {exc}"],
                )
            results.append(r)

        m = compute_metrics(results)
        print(
            f"{threshold:>10.2f}  {vec_w:.1f}/{fts_w:.1f}   "
            f"{m['retrieval_hit_rate']['rate']:>8.2f}  "
            f"{m['refusal_accuracy']['combined_rate']:>8.2f}  "
            f"{m['answer_match']['mean']:>10.2f}"
        )
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "DocChat eval harness. "
            "Requires: docker compose up -d db, OPENAI_API_KEY set, corpus indexed."
        )
    )
    parser.add_argument(
        "--sweep-threshold",
        metavar="T1,T2,...",
        help="Guardrail thresholds to sweep (e.g. 0.25,0.30,0.35)",
    )
    parser.add_argument(
        "--sweep-weights",
        metavar="V1/F1,V2/F2,...",
        help="vec/fts weight pairs to sweep (e.g. 0.6/0.4,0.7/0.3,0.8/0.2)",
    )
    args = parser.parse_args()

    items: list[dict] = yaml.safe_load(GOLDEN_PATH.read_text())
    judge = os.environ.get("EVAL_JUDGE", "").strip() == "1"
    missing_docs = missing_expected_documents(items)
    if missing_docs:
        print(
            "Missing expected document(s) in the live index: "
            f"{', '.join(missing_docs)}\n"
            "Index the golden corpus before running evals. Expected corpus:\n"
            "  - backend/evals/sample.txt\n"
            "  - AI_Native Builder.pdf\n"
            "  - Assignment v3.pdf\n",
            file=sys.stderr,
        )
        raise SystemExit(2)

    if args.sweep_threshold or args.sweep_weights:
        await run_sweep(items, args)
    else:
        await run_standard(items, judge)


if __name__ == "__main__":
    asyncio.run(main())
