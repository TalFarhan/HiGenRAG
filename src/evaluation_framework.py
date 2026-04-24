"""
Evaluation Framework – Hit@3 Comparison

Loads SciFact claims with known ground-truth document IDs, runs each
claim as a search query against both the baseline (naive-chunking) and
enriched (generative + boosted) Qdrant collections, and reports the
Hit@3 metric for each approach alongside a causal-diagnosis example.
"""

from __future__ import annotations

import logging
import random
import sys
import textwrap
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import ir_datasets
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from semantic_chunker import EMBEDDING_MODEL_NAME
from retrieval_engine import RetrievalEngine

logger = logging.getLogger(__name__)

QDRANT_DB_PATH = "output/qdrant_db"
ENRICHED_COLLECTION = "enriched_chunks"
BASELINE_COLLECTION = "baseline_chunks"
DATASET_NAME = "scifact"
DOCUMENT_LIMIT = 150
SAMPLE_SIZE = 50
TOP_K = 3
RANDOM_SEED = 42

_SEPARATOR = "\u2550" * 72
_THIN_SEP = "\u2500" * 72


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class EvalResult:
    """Holds per-query evaluation outcome for both approaches."""

    claim_id: str
    claim_text: str
    ground_truth_doc_ids: set[str]
    baseline_hit: bool
    baseline_top_doc_ids: list[str]
    baseline_scores: list[float]
    enriched_hit: bool
    enriched_top_doc_ids: list[str]
    enriched_scores: list[float]


# ---------------------------------------------------------------------------
# 1. Identify indexed document IDs (the 150-doc evidence-filtered subset)
# ---------------------------------------------------------------------------

def _get_evidence_doc_ids(dataset_name: str) -> set[str]:
    """Extract all doc_ids that have positive relevance in the test qrels.

    Mirrors the identical helper used by the ingestion pipelines
    (semantic_chunker / baseline_pipeline) so the evaluation sees the
    exact same document universe.
    """
    canonical = dataset_name.split("/")[-1]
    dataset = ir_datasets.load(f"beir/{canonical}/test")
    evidence_ids: set[str] = set()
    for qrel in dataset.qrels_iter():
        if qrel.relevance > 0:
            evidence_ids.add(qrel.doc_id)
    return evidence_ids


def load_indexed_doc_ids(
    dataset_name: str = DATASET_NAME,
    limit: int = DOCUMENT_LIMIT,
) -> set[str]:
    """Return the doc_ids of the evidence-filtered corpus subset.

    Reproduces the same filtering logic used by the ingestion pipelines:
      1. Collect doc_ids with positive relevance from the test qrels.
      2. Iterate over the ir_datasets corpus and keep only documents
         whose doc_id appears in the evidence set.
      3. Stop once *limit* documents have been collected.
    """
    logger.info(
        "Loading evidence-filtered doc_ids from BeIR/%s corpus (limit=%d)",
        dataset_name, limit,
    )

    evidence_doc_ids = _get_evidence_doc_ids(dataset_name)
    logger.info("Found %d unique doc_ids with evidence in qrels", len(evidence_doc_ids))

    canonical_name = dataset_name.split("/")[-1]
    corpus_dataset = ir_datasets.load(f"beir/{canonical_name}/test")

    doc_ids: set[str] = set()
    for doc in corpus_dataset.docs_iter():
        if doc.doc_id not in evidence_doc_ids:
            continue
        doc_ids.add(doc.doc_id)
        if len(doc_ids) >= limit:
            break

    logger.info("Indexed doc_id set contains %d evidence-filtered documents", len(doc_ids))
    return doc_ids


# ---------------------------------------------------------------------------
# 2. Load claims + ground-truth from SciFact test qrels
# ---------------------------------------------------------------------------

def load_claims_with_ground_truth(
    indexed_doc_ids: set[str],
    sample_size: int = SAMPLE_SIZE,
) -> list[dict]:
    """Load SciFact test claims and keep those whose relevant doc is indexed.

    Uses ``ir_datasets`` to access the official BeIR/SciFact test split
    which provides query texts and relevance judgments (qrels).

    Returns up to *sample_size* claims, each as a dict with keys:
    ``claim_id``, ``text``, ``ground_truth_doc_ids``.
    """
    canonical = DATASET_NAME.split("/")[-1]
    logger.info("Loading %s test split via ir_datasets", canonical)
    dataset = ir_datasets.load(f"beir/{canonical}/test")

    qrels: dict[str, set[str]] = defaultdict(set)
    for qrel in dataset.qrels_iter():
        if qrel.relevance > 0:
            qrels[qrel.query_id].add(qrel.doc_id)

    queries = {q.query_id: q.text for q in dataset.queries_iter()}

    eligible: list[dict] = []
    for qid, text in queries.items():
        relevant = qrels.get(qid, set())
        overlap = relevant & indexed_doc_ids
        if overlap:
            eligible.append(
                {
                    "claim_id": qid,
                    "text": text,
                    "ground_truth_doc_ids": overlap,
                }
            )

    logger.info(
        "Found %d claims with ground-truth docs in the indexed subset", len(eligible),
    )

    random.seed(RANDOM_SEED)
    if len(eligible) > sample_size:
        eligible = random.sample(eligible, sample_size)
        logger.info("Sampled %d claims for evaluation", sample_size)

    return eligible


# ---------------------------------------------------------------------------
# 3. Search helpers (single shared QdrantClient, no locking conflicts)
# ---------------------------------------------------------------------------

def _extract_doc_ids_from_payload(payload: dict) -> list[str]:
    """Safely extract document IDs from a Qdrant payload.

    Tries several possible key names and normalises every value to str
    so that int-typed IDs returned by Qdrant compare correctly against
    the str-typed ground-truth set.
    """
    _CANDIDATE_KEYS = ("doc_ids", "Docs", "docs", "doc_id")
    raw = None
    for key in _CANDIDATE_KEYS:
        raw = payload.get(key)
        if raw is not None:
            break

    if raw is None:
        return []

    if isinstance(raw, list):
        return [str(v) for v in raw]

    return [str(raw)]


def _search_baseline(
    client: QdrantClient,
    model: SentenceTransformer,
    query: str,
    top_k: int = TOP_K,
) -> tuple[list[str], list[float]]:
    """Plain cosine-similarity search against the baseline collection."""
    vec = model.encode(query, convert_to_numpy=True).tolist()
    hits = client.query_points(
        collection_name=BASELINE_COLLECTION,
        query=vec,
        limit=top_k,
    ).points

    doc_ids: list[str] = []
    scores: list[float] = []
    for hit in hits:
        payload = hit.payload or {}
        extracted = _extract_doc_ids_from_payload(payload)
        did = extracted[0] if extracted else ""
        doc_ids.append(did)
        scores.append(round(float(hit.score), 4))
    return doc_ids, scores


def _search_enriched(
    client: QdrantClient,
    model: SentenceTransformer,
    query: str,
    top_k: int = TOP_K,
) -> tuple[list[str], list[float]]:
    """Vector search with entity-aware boosting (mirrors RetrievalEngine)."""
    vec = model.encode(query, convert_to_numpy=True).tolist()

    entity_terms = RetrievalEngine._extract_entity_terms(query)
    fetch_limit = top_k * 3 if entity_terms else top_k

    hits = client.query_points(
        collection_name=ENRICHED_COLLECTION,
        query=vec,
        limit=fetch_limit,
    ).points

    scored: list[tuple[float, list[str]]] = []
    for hit in hits:
        payload = hit.payload or {}
        base_score = float(hit.score)
        keywords = payload.get("keywords", [])
        subtopics = payload.get("subtopics", [])
        boost = RetrievalEngine._compute_boost(entity_terms, keywords, subtopics)
        scored.append((base_score + boost, _extract_doc_ids_from_payload(payload)))

    scored.sort(key=lambda x: x[0], reverse=True)

    result_doc_ids: list[str] = []
    result_scores: list[float] = []
    for score, chunk_doc_ids in scored[:top_k]:
        result_doc_ids.append("|".join(chunk_doc_ids))
        result_scores.append(round(score, 4))

    return result_doc_ids, result_scores


def _check_hit(ground_truth: set[str], returned_doc_ids: list[str]) -> bool:
    """True if any ground-truth doc_id appears in the returned results.

    Handles both single-value IDs (baseline) and pipe-delimited
    multi-doc IDs (enriched chunks that span multiple documents).
    All values are cast to str before comparison to avoid int/str
    mismatches when Qdrant returns numeric IDs.
    """
    gt_normalised = {str(g) for g in ground_truth}
    for entry in returned_doc_ids:
        for did in str(entry).split("|"):
            if str(did) in gt_normalised:
                return True
    return False


# ---------------------------------------------------------------------------
# 4. Run full evaluation
# ---------------------------------------------------------------------------

def run_evaluation(
    db_path: str = QDRANT_DB_PATH,
    sample_size: int = SAMPLE_SIZE,
    top_k: int = TOP_K,
) -> list[EvalResult]:
    """Execute the Hit@3 evaluation and return per-claim results."""
    indexed_doc_ids = load_indexed_doc_ids()
    claims = load_claims_with_ground_truth(indexed_doc_ids, sample_size)

    if not claims:
        logger.error("No eligible claims found — cannot evaluate.")
        return []

    logger.info("Initialising embedding model and Qdrant client")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    client = QdrantClient(path=db_path)

    results: list[EvalResult] = []

    for idx, claim in enumerate(claims, 1):
        gt = claim["ground_truth_doc_ids"]

        bl_doc_ids, bl_scores = _search_baseline(client, model, claim["text"], top_k)
        en_doc_ids, en_scores = _search_enriched(client, model, claim["text"], top_k)

        bl_hit = _check_hit(gt, bl_doc_ids)
        en_hit = _check_hit(gt, en_doc_ids)

        results.append(
            EvalResult(
                claim_id=claim["claim_id"],
                claim_text=claim["text"],
                ground_truth_doc_ids=gt,
                baseline_hit=bl_hit,
                baseline_top_doc_ids=bl_doc_ids,
                baseline_scores=bl_scores,
                enriched_hit=en_hit,
                enriched_top_doc_ids=en_doc_ids,
                enriched_scores=en_scores,
            )
        )

        if idx % 10 == 0 or idx == len(claims):
            logger.info("Evaluated %d / %d claims", idx, len(claims))

    client.close()
    return results


# ---------------------------------------------------------------------------
# 5. Pretty-print comparison report
# ---------------------------------------------------------------------------

def print_report(results: list[EvalResult]) -> None:
    """Print a formatted comparison table and a diagnostic example."""
    n = len(results)
    if n == 0:
        print("No evaluation results to report.")
        return

    bl_hits = sum(1 for r in results if r.baseline_hit)
    en_hits = sum(1 for r in results if r.enriched_hit)
    bl_pct = bl_hits / n * 100
    en_pct = en_hits / n * 100

    both_hit = sum(1 for r in results if r.baseline_hit and r.enriched_hit)
    only_bl = sum(1 for r in results if r.baseline_hit and not r.enriched_hit)
    only_en = sum(1 for r in results if not r.baseline_hit and r.enriched_hit)
    neither = sum(1 for r in results if not r.baseline_hit and not r.enriched_hit)

    delta = en_pct - bl_pct
    delta_sign = "+" if delta >= 0 else ""
    winner = "Enriched" if en_pct > bl_pct else ("Baseline" if bl_pct > en_pct else "Tie")

    print()
    print(_SEPARATOR)
    print("  EVALUATION REPORT  —  Hit@3 Comparison")
    print(_SEPARATOR)
    print()

    print(f"  Claims evaluated : {n}")
    print(f"  Top-K            : {TOP_K}")
    print()

    hdr  = f"  {'Method':<28} {'Hits':>6} {'Miss':>6} {'Hit@3':>8}"
    print(hdr)
    print(f"  {'─' * 52}")
    print(f"  {'Baseline (naive chunking)':<28} {bl_hits:>6} {n - bl_hits:>6} {bl_pct:>7.1f}%")
    print(f"  {'Enriched (generative+boost)':<28} {en_hits:>6} {n - en_hits:>6} {en_pct:>7.1f}%")
    print(f"  {'─' * 52}")
    print(f"  {'Delta':<28} {'':>6} {'':>6} {delta_sign}{delta:.1f}pp")
    print(f"  {'Winner':<28} {'':>6} {'':>6} {winner:>8}")
    print()

    print(f"  Overlap breakdown:")
    print(f"    Both hit      : {both_hit:>4}")
    print(f"    Only Baseline  : {only_bl:>4}")
    print(f"    Only Enriched  : {only_en:>4}")
    print(f"    Neither        : {neither:>4}")
    print()

    # --- Causal diagnosis example: baseline miss, enriched hit -----------
    diagnostic = next(
        (r for r in results if not r.baseline_hit and r.enriched_hit), None,
    )

    print(_THIN_SEP)
    print("  CAUSAL DIAGNOSIS EXAMPLE")
    print(f"  (Baseline MISS  /  Enriched HIT)")
    print(_THIN_SEP)

    if diagnostic is None:
        print("  No such example found in this run.")
    else:
        wrapped = textwrap.fill(diagnostic.claim_text, width=66, initial_indent="    ", subsequent_indent="    ")
        gt_str = ", ".join(sorted(diagnostic.ground_truth_doc_ids))

        print()
        print(f"  Claim ID        : {diagnostic.claim_id}")
        print(f"  Ground-truth doc: {gt_str}")
        print(f"  Claim text:")
        print(wrapped)
        print()

        print(f"  Baseline top-{TOP_K} doc_ids (MISS):")
        for i, (did, sc) in enumerate(
            zip(diagnostic.baseline_top_doc_ids, diagnostic.baseline_scores), 1,
        ):
            marker = " <-- GT" if did in diagnostic.ground_truth_doc_ids else ""
            print(f"    #{i}  doc_id={did:<8}  score={sc:.4f}{marker}")

        print()
        print(f"  Enriched top-{TOP_K} doc_ids (HIT):")
        for i, (did_str, sc) in enumerate(
            zip(diagnostic.enriched_top_doc_ids, diagnostic.enriched_scores), 1,
        ):
            parts = did_str.split("|")
            is_gt = any(d in diagnostic.ground_truth_doc_ids for d in parts)
            marker = " <-- GT" if is_gt else ""
            print(f"    #{i}  doc_ids={did_str:<16}  score={sc:.4f}{marker}")

        print()
        print("  Why the enriched approach won here:")
        print(
            "    The generative pipeline produces semantically cohesive chunks\n"
            "    enriched with topic labels, keywords, and subtopics. Combined\n"
            "    with entity-aware boosting, the correct document surfaces in the\n"
            "    top results even when the naive fixed-size chunks scatter the\n"
            "    relevant content across multiple low-scoring fragments."
        )

    print()
    print(_SEPARATOR)
    print()


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    results = run_evaluation()
    print_report(results)


if __name__ == "__main__":
    main()
