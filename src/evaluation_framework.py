"""
Evaluation Framework – retrieval metrics (Hit@K, MRR, Precision@K)

Loads SciFact claims with known ground-truth document IDs, runs each
claim as a search query against both the baseline (naive-chunking) and
enriched Qdrant collections. Baseline uses dense vectors only; enriched
uses hybrid dense retrieval plus payload full-text match on generative
anchors, fused with weighted reciprocal rank fusion (dense vs lexical).
Reports Hit@K, MRR, and
Precision@K alongside a causal-diagnosis example.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
import sys
import textwrap
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import ir_datasets
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchText
from sentence_transformers import CrossEncoder, SentenceTransformer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from semantic_chunker import EMBEDDING_MODEL_NAME

logger = logging.getLogger(__name__)

QDRANT_DB_PATH = "output/qdrant_db"
ENRICHED_COLLECTION = "enriched_chunks"
BASELINE_COLLECTION = "baseline_chunks"
DATASET_NAME = "scifact"
# None = treat every evidence-filtered corpus doc as indexed (SciFact: 283 docs,
# 300 claims with gold in that set). Use an int to mirror a capped ingest.
DOCUMENT_LIMIT: int | None = None
# None = evaluate every eligible claim (gold doc in indexed subset). Use
# ``--sample-size N`` for a fixed-size subset; sampling is deterministic
# after a stable sort and ``random.seed``.
DEFAULT_CLAIM_SAMPLE_SIZE: int | None = None
TOP_K = 3
RANDOM_SEED = 42
RRF_K = 60
# Weighted RRF: dense (vector) ranks vs lexical / overlap ranks.
RRF_DENSE_WEIGHT = 0.7
RRF_LEXICAL_WEIGHT = 0.3
ENRICHED_LEXICAL_PREFETCH_MIN = 32
ENRICHED_PREFETCH_FACTOR = 20

_LEX_TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
_SEPARATOR = "=" * 72
_THIN_SEP = "-" * 72


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
# 1. Identify indexed document IDs (evidence-filtered subset; optionally capped)
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
    limit: int | None = DOCUMENT_LIMIT,
) -> set[str]:
    """Return the doc_ids of the evidence-filtered corpus subset.

    Reproduces the same filtering logic used by the ingestion pipelines:
      1. Collect doc_ids with positive relevance from the test qrels.
      2. Iterate over the ir_datasets corpus and keep only documents
         whose doc_id appears in the evidence set.
      3. If *limit* is an int, stop once that many documents have been collected;
         if *limit* is ``None``, include every matching document.
    """
    lim_desc = "all" if limit is None else str(limit)
    logger.info(
        "Loading evidence-filtered doc_ids from BeIR/%s corpus (limit=%s)",
        dataset_name,
        lim_desc,
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
        if limit is not None and len(doc_ids) >= limit:
            break

    logger.info("Indexed doc_id set contains %d evidence-filtered documents", len(doc_ids))
    return doc_ids


# ---------------------------------------------------------------------------
# 2. Load claims + ground-truth from SciFact test qrels
# ---------------------------------------------------------------------------

def load_claims_with_ground_truth(
    indexed_doc_ids: set[str],
    sample_size: int | None = DEFAULT_CLAIM_SAMPLE_SIZE,
    random_seed: int = RANDOM_SEED,
) -> list[dict]:
    """Load SciFact test claims and keep those whose relevant doc is indexed.

    Uses ``ir_datasets`` to access the official BeIR/SciFact test split
    which provides query texts and relevance judgments (qrels).

    Eligible claims are sorted by ``claim_id`` so subsampling order does not
    depend on corpus iteration order. If *sample_size* is ``None`` or not
    smaller than the eligible count, every eligible claim is returned.
    Otherwise ``random.seed(random_seed)`` is applied and *sample_size*
    claims are drawn with ``random.sample``.

    Each returned dict has keys: ``claim_id``, ``text``,
    ``ground_truth_doc_ids``.
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

    eligible.sort(key=lambda row: str(row["claim_id"]))
    n_eligible = len(eligible)
    if sample_size is None or sample_size >= n_eligible:
        logger.info("Evaluating all %d eligible claims", n_eligible)
        return eligible

    if sample_size < 1:
        logger.error("sample_size must be >= 1 when limiting claims; got %s", sample_size)
        return []

    random.seed(random_seed)
    eligible = random.sample(eligible, sample_size)
    logger.info(
        "Sampled %d / %d eligible claims for evaluation (seed=%d)",
        sample_size,
        n_eligible,
        random_seed,
    )
    return eligible


# ---------------------------------------------------------------------------
# 3. Search helpers (single shared QdrantClient, no locking conflicts)
# ---------------------------------------------------------------------------

def _chunk_text_from_enriched_payload(payload: dict) -> str:
    """Return chunk body text stored on the enriched Qdrant payload."""
    t = payload.get("text")
    return str(t).strip() if t is not None else ""


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


def _normalized_lex_tokens(text: str) -> set[str]:
    """Lowercase alphanumeric tokens with length >= 2 (aligns with index min_token_len)."""
    return {
        m.group(0).lower()
        for m in _LEX_TOKEN_RE.finditer(text)
        if len(m.group(0)) >= 2
    }


def _lexical_overlap_score(claim_text: str, payload: dict) -> int:
    """Count overlapping lexical tokens between the claim and anchor/keyword text."""
    claim_toks = _normalized_lex_tokens(claim_text)
    if not claim_toks:
        return 0
    anchor = payload.get("anchor_queries") or []
    pieces: list[str] = []
    for a in anchor:
        if isinstance(a, str) and a.strip():
            pieces.append(a)
    for kw in payload.get("keywords") or []:
        if isinstance(kw, str) and kw.strip():
            pieces.append(kw)
    field_toks = _normalized_lex_tokens(" ".join(pieces))
    return len(claim_toks & field_toks)


def _reciprocal_rank_fusion(
    ranked_lists: list[list[str]],
    k: int = RRF_K,
    list_weights: list[float] | None = None,
) -> list[tuple[str, float]]:
    """Fuse ranked lists with weighted RRF: sum_i w_i / (k + rank_i).

    When *list_weights* is omitted, each list contributes equally (legacy
    behaviour). When provided, len(list_weights) must equal len(ranked_lists).
    """
    n_lists = len(ranked_lists)
    if n_lists == 0:
        return []
    if list_weights is None:
        weights = [1.0] * n_lists
    else:
        if len(list_weights) != n_lists:
            raise ValueError(
                f"list_weights length ({len(list_weights)}) must match "
                f"ranked_lists ({n_lists})",
            )
        weights = list(list_weights)
    scores: dict[str, float] = defaultdict(float)
    for weight, ranked in zip(weights, ranked_lists):
        for rank, raw_id in enumerate(ranked, start=1):
            pid = str(raw_id)
            scores[pid] += weight * (1.0 / (k + rank))
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))


def _search_baseline(
    client: QdrantClient,
    model: SentenceTransformer,
    query: str,
    top_k: int = TOP_K,
    collection_name: str = BASELINE_COLLECTION,
) -> tuple[list[str], list[float]]:
    """Plain cosine-similarity (dense_only) search against any collection.

    Each returned doc id is pipe-joined when the chunk spans multiple
    documents so multi-doc payloads (sentence window / semantic chunker)
    are compared correctly by ``_check_hit``.
    """
    vec = model.encode(query, convert_to_numpy=True).tolist()
    hits = client.query_points(
        collection_name=collection_name,
        query=vec,
        limit=top_k,
    ).points

    doc_ids: list[str] = []
    scores: list[float] = []
    for hit in hits:
        payload = hit.payload or {}
        extracted = _extract_doc_ids_from_payload(payload)
        joined = "|".join(extracted) if extracted else ""
        doc_ids.append(joined)
        scores.append(round(float(hit.score), 4))
    return doc_ids, scores


def _search_enriched(
    client: QdrantClient,
    model: SentenceTransformer,
    query: str,
    collect_limit: int = TOP_K,
    collection_name: str = ENRICHED_COLLECTION,
) -> tuple[list[str], list[float], list[str]]:
    """Hybrid dense + payload ``MatchText`` on anchors/keywords, fused with weighted RRF.

    Returns fused (doc_id rows, RRF scores, chunk texts) up to *collect_limit* rows.
    Operates on any collection name; the caller is responsible for
    ensuring the collection actually carries lexical payload fields.
    """
    claim_text = query or ""
    vec = model.encode(claim_text, convert_to_numpy=True).tolist()
    fetch_limit = max(collect_limit * ENRICHED_PREFETCH_FACTOR, ENRICHED_LEXICAL_PREFETCH_MIN)

    dense_hits = client.query_points(
        collection_name=collection_name,
        query=vec,
        limit=fetch_limit,
    ).points

    dense_ranked = [str(h.id) for h in dense_hits]
    payload_by_id: dict[str, dict] = {}
    for h in dense_hits:
        payload_by_id[str(h.id)] = dict(h.payload or {})

    ranked_lists: list[list[str]] = [dense_ranked]
    match_text = claim_text.strip()

    if match_text:
        text_filter = Filter(
            should=[
                FieldCondition(key="anchor_queries", match=MatchText(text=match_text)),
                FieldCondition(key="keywords", match=MatchText(text=match_text)),
            ],
        )
        try:
            scroll_points, _ = client.scroll(
                collection_name=collection_name,
                scroll_filter=text_filter,
                limit=fetch_limit,
                with_payload=True,
            )
        except Exception as exc:
            logger.warning("Lexical scroll failed; using dense ranks only: %s", exc)
            scroll_points = []
    else:
        scroll_points = []

    if scroll_points:
        dedup: dict[str, object] = {}
        for rec in scroll_points:
            dedup[str(rec.id)] = rec
        scored = sorted(
            dedup.values(),
            key=lambda r: (
                _lexical_overlap_score(claim_text, getattr(r, "payload", None) or {}),
                str(r.id),
            ),
            reverse=True,
        )
        lexical_ranked = [str(r.id) for r in scored[:fetch_limit]]
        if lexical_ranked:
            ranked_lists.append(lexical_ranked)
            for r in scored:
                pid = str(r.id)
                if pid not in payload_by_id:
                    payload_by_id[pid] = dict(getattr(r, "payload", None) or {})

    rrf_weights = (
        [RRF_DENSE_WEIGHT, RRF_LEXICAL_WEIGHT]
        if len(ranked_lists) > 1
        else [RRF_DENSE_WEIGHT]
    )
    fused = _reciprocal_rank_fusion(ranked_lists, k=RRF_K, list_weights=rrf_weights)

    result_doc_ids: list[str] = []
    result_scores: list[float] = []
    result_texts: list[str] = []
    for pid, rrf_score in fused:
        payload = payload_by_id.get(pid)
        if payload is None:
            continue
        chunk_doc_ids = _extract_doc_ids_from_payload(payload)
        result_doc_ids.append("|".join(chunk_doc_ids))
        result_scores.append(round(rrf_score, 6))
        result_texts.append(_chunk_text_from_enriched_payload(payload))
        if len(result_doc_ids) >= collect_limit:
            break

    return result_doc_ids, result_scores, result_texts


def _cross_encoder_rerank_top_k(
    cross_encoder: CrossEncoder,
    claim_text: str,
    doc_ids: list[str],
    candidate_texts: list[str],
    top_k: int,
) -> tuple[list[str], list[float]]:
    """Score (claim, chunk) pairs with a cross-encoder and keep the top-*top_k* rows."""
    if not doc_ids:
        return [], []
    n = min(len(doc_ids), len(candidate_texts))
    doc_ids = doc_ids[:n]
    candidate_texts = candidate_texts[:n]
    pairs = [[claim_text, ct] for ct in candidate_texts]
    # Batch scoring; order matches *doc_ids*.
    ce_scores = cross_encoder.predict(pairs)
    order = sorted(range(n), key=lambda i: float(ce_scores[i]), reverse=True)
    ranked_docs: list[str] = []
    ranked_scores: list[float] = []
    for i in order[:top_k]:
        ranked_docs.append(doc_ids[i])
        ranked_scores.append(round(float(ce_scores[i]), 6))
    return ranked_docs, ranked_scores


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


def _first_relevant_rank(
    ground_truth: set[str],
    returned_doc_ids: list[str],
    top_k: int,
) -> int | None:
    """1-based rank of the first retrieved item that matches any GT doc (within top_k)."""
    gt_normalised = {str(g) for g in ground_truth}
    for rank, entry in enumerate(returned_doc_ids[:top_k], start=1):
        for did in str(entry).split("|"):
            if str(did) in gt_normalised:
                return rank
    return None


def reciprocal_rank_score(
    ground_truth: set[str],
    returned_doc_ids: list[str],
    top_k: int,
) -> float:
    """MRR contribution for one query: 1/rank of first relevant hit in top-K, else 0."""
    r = _first_relevant_rank(ground_truth, returned_doc_ids, top_k)
    if r is None:
        return 0.0
    return 1.0 / r


def precision_at_k_score(
    ground_truth: set[str],
    returned_doc_ids: list[str],
    top_k: int,
) -> float:
    """Precision@K for one query: (relevant slots in top-K) / K."""
    if top_k <= 0:
        return 0.0
    gt_normalised = {str(g) for g in ground_truth}
    relevant_slots = 0
    for entry in returned_doc_ids[:top_k]:
        for did in str(entry).split("|"):
            if str(did) in gt_normalised:
                relevant_slots += 1
                break
    return relevant_slots / top_k


# ---------------------------------------------------------------------------
# 4. Run full evaluation
# ---------------------------------------------------------------------------

def _qdrant_collection_names(client: QdrantClient) -> set[str]:
    return {c.name for c in client.get_collections().collections}


def run_evaluation(
    db_path: str = QDRANT_DB_PATH,
    document_limit: int | None = DOCUMENT_LIMIT,
    sample_size: int | None = DEFAULT_CLAIM_SAMPLE_SIZE,
    top_k: int = TOP_K,
    random_seed: int = RANDOM_SEED,
    compare_baseline: bool = True,
    rerank_model: str | None = None,
    rerank_candidates: int = 40,
) -> tuple[list[EvalResult], bool]:
    """Execute retrieval evaluation (Hit@K, MRR, Precision@K) vs SciFact test qrels.

    Returns ``(results, compare_baseline_used)``. If the baseline collection is
    missing, baseline comparison is skipped and the second value is ``False``.
    """
    indexed_doc_ids = load_indexed_doc_ids(limit=document_limit)
    claims = load_claims_with_ground_truth(
        indexed_doc_ids, sample_size=sample_size, random_seed=random_seed,
    )

    if not claims:
        logger.error("No eligible claims found — cannot evaluate.")
        return [], compare_baseline

    logger.info("Initialising embedding model and Qdrant client")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    cross_encoder: CrossEncoder | None = None
    if rerank_model:
        logger.info("Initialising cross-encoder re-ranker: %s", rerank_model)
        cross_encoder = CrossEncoder(rerank_model)
    client = QdrantClient(path=db_path)

    existing = _qdrant_collection_names(client)
    if ENRICHED_COLLECTION not in existing:
        client.close()
        raise ValueError(
            f"Qdrant collection {ENRICHED_COLLECTION!r} not found under {db_path!s}. "
            "Run enriched indexing (semantic chunker ingestion) first.",
        )
    if compare_baseline and BASELINE_COLLECTION not in existing:
        logger.warning(
            "Collection %r missing - skipping baseline metrics. "
            "Index it with `python -m src.baseline_pipeline` or pass --enriched-only.",
            BASELINE_COLLECTION,
        )
        compare_baseline = False

    results: list[EvalResult] = []

    for idx, claim in enumerate(claims, 1):
        gt = claim["ground_truth_doc_ids"]

        if compare_baseline:
            bl_doc_ids, bl_scores = _search_baseline(client, model, claim["text"], top_k)
            bl_hit = _check_hit(gt, bl_doc_ids)
        else:
            bl_doc_ids, bl_scores = [], []
            bl_hit = False

        collect_limit = (
            max(rerank_candidates, top_k) if cross_encoder is not None else top_k
        )
        en_doc_ids, en_scores, en_texts = _search_enriched(
            client, model, claim["text"], collect_limit=collect_limit,
        )
        if cross_encoder is not None:
            en_doc_ids, en_scores = _cross_encoder_rerank_top_k(
                cross_encoder,
                claim["text"],
                en_doc_ids,
                en_texts,
                top_k,
            )
        else:
            en_doc_ids = en_doc_ids[:top_k]
            en_scores = en_scores[:top_k]
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
    return results, compare_baseline


# ---------------------------------------------------------------------------
# 5. Pretty-print comparison report
# ---------------------------------------------------------------------------

def print_report(
    results: list[EvalResult],
    top_k: int = TOP_K,
    compare_baseline: bool = True,
    enriched_method_label: str = "Enriched (hybrid w-RRF)",
    rerank_model: str | None = None,
    rerank_candidates: int = 40,
) -> None:
    """Print a formatted comparison table and a diagnostic example."""
    n = len(results)
    if n == 0:
        print("No evaluation results to report.")
        return

    en_hits = sum(1 for r in results if r.enriched_hit)
    en_pct = en_hits / n * 100
    hit_label = f"Hit@{top_k}"
    prec_label = f"P@{top_k}"

    en_mrr = sum(
        reciprocal_rank_score(r.ground_truth_doc_ids, r.enriched_top_doc_ids, top_k)
        for r in results
    ) / n
    en_prec = sum(
        precision_at_k_score(r.ground_truth_doc_ids, r.enriched_top_doc_ids, top_k)
        for r in results
    ) / n

    print()
    print(_SEPARATOR)
    if compare_baseline:
        print(
            f"  EVALUATION REPORT  -  {hit_label}, MRR, {prec_label} "
            "(Baseline vs Enriched, SciFact qrels)",
        )
    else:
        print(
            f"  EVALUATION REPORT  -  {hit_label}, MRR, {prec_label} "
            "(Enriched only, SciFact qrels)",
        )
    print(_SEPARATOR)
    print()

    print(f"  Claims evaluated : {n}")
    print(f"  Top-K            : {top_k}")
    if rerank_model:
        print(f"  Re-ranker model  : {rerank_model}")
        print(f"  Re-rank pool     : {rerank_candidates} candidates → top-{top_k}")
    print()

    table_rule = 72

    if compare_baseline:
        bl_hits = sum(1 for r in results if r.baseline_hit)
        bl_pct = bl_hits / n * 100
        bl_mrr = sum(
            reciprocal_rank_score(r.ground_truth_doc_ids, r.baseline_top_doc_ids, top_k)
            for r in results
        ) / n
        bl_prec = sum(
            precision_at_k_score(r.ground_truth_doc_ids, r.baseline_top_doc_ids, top_k)
            for r in results
        ) / n

        both_hit = sum(1 for r in results if r.baseline_hit and r.enriched_hit)
        only_bl = sum(1 for r in results if r.baseline_hit and not r.enriched_hit)
        only_en = sum(1 for r in results if not r.baseline_hit and r.enriched_hit)
        neither = sum(1 for r in results if not r.baseline_hit and not r.enriched_hit)

        delta = en_pct - bl_pct
        delta_sign = "+" if delta >= 0 else ""
        delta_mrr = en_mrr - bl_mrr
        delta_mrr_sign = "+" if delta_mrr >= 0 else ""
        delta_prec_pp = (en_prec - bl_prec) * 100
        delta_prec_sign = "+" if delta_prec_pp >= 0 else ""
        winner = "Enriched" if en_pct > bl_pct else ("Baseline" if bl_pct > en_pct else "Tie")

        hdr = (
            f"  {'Method':<28} {'Hits':>6} {'Miss':>6} {hit_label:>8} "
            f"{'MRR':>7} {prec_label:>8}"
        )
        print(hdr)
        print(f"  {'-' * table_rule}")
        print(
            f"  {'Baseline (naive chunking)':<28} {bl_hits:>6} {n - bl_hits:>6} "
            f"{bl_pct:>7.1f}% {bl_mrr:>7.3f} {bl_prec * 100:>7.1f}%",
        )
        print(
            f"  {enriched_method_label:<28} {en_hits:>6} {n - en_hits:>6} "
            f"{en_pct:>7.1f}% {en_mrr:>7.3f} {en_prec * 100:>7.1f}%",
        )
        print(f"  {'-' * table_rule}")
        print(
            f"  {'Delta':<28} {'':>6} {'':>6} {delta_sign}{delta:.1f}pp "
            f"{delta_mrr_sign}{delta_mrr:.3f} {delta_prec_sign}{delta_prec_pp:.1f}pp",
        )
        print(f"  {'Winner (by Hit@K)':<28} {'':>6} {'':>6} {winner:>8} {'':>7} {'':>8}")
        print()

        print("  Overlap breakdown:")
        print(f"    Both hit       : {both_hit:>4}")
        print(f"    Only Baseline  : {only_bl:>4}")
        print(f"    Only Enriched  : {only_en:>4}")
        print(f"    Neither        : {neither:>4}")
        print()
    else:
        hdr = (
            f"  {'Method':<28} {'Hits':>6} {'Miss':>6} {hit_label:>8} "
            f"{'MRR':>7} {prec_label:>8}"
        )
        print(hdr)
        print(f"  {'-' * table_rule}")
        print(
            f"  {enriched_method_label:<28} {en_hits:>6} {n - en_hits:>6} "
            f"{en_pct:>7.1f}% {en_mrr:>7.3f} {en_prec * 100:>7.1f}%",
        )
        print(f"  {'-' * table_rule}")
        print()

    # --- Diagnostic example ------------------------------------------------
    if compare_baseline:
        diagnostic = next(
            (r for r in results if not r.baseline_hit and r.enriched_hit), None,
        )
        print(_THIN_SEP)
        print("  CAUSAL DIAGNOSIS EXAMPLE")
        print("  (Baseline MISS  /  Enriched HIT)")
        print(_THIN_SEP)

        if diagnostic is None:
            print("  No such example found in this run.")
        else:
            wrapped = textwrap.fill(
                diagnostic.claim_text, width=66, initial_indent="    ", subsequent_indent="    ",
            )
            gt_str = ", ".join(sorted(diagnostic.ground_truth_doc_ids))

            print()
            print(f"  Claim ID        : {diagnostic.claim_id}")
            print(f"  Ground-truth doc: {gt_str}")
            print("  Claim text:")
            print(wrapped)
            print()

            print(f"  Baseline top-{top_k} doc_ids (MISS):")
            for i, (did, sc) in enumerate(
                zip(diagnostic.baseline_top_doc_ids, diagnostic.baseline_scores), 1,
            ):
                marker = " <-- GT" if did in diagnostic.ground_truth_doc_ids else ""
                print(f"    #{i}  doc_id={did:<8}  score={sc:.4f}{marker}")

            print()
            print(f"  Enriched top-{top_k} doc_ids (HIT):")
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
                "    enriched with anchor queries and keywords. Hybrid retrieval\n"
                "    fuses dense similarity with lexical overlap on those fields\n"
                "    (weighted RRF, dense-weighted), so the correct document surfaces\n"
                "    in the top results even when naive fixed-size chunks scatter the evidence.\n",
            )
    else:
        diagnostic_hit = next((r for r in results if r.enriched_hit), None)
        diagnostic_miss = next((r for r in results if not r.enriched_hit), None)
        print(_THIN_SEP)
        print("  EXAMPLES (Enriched vs ground-truth doc)")
        print(_THIN_SEP)
        for label, diagnostic in ("HIT", diagnostic_hit), ("MISS", diagnostic_miss):
            if diagnostic is None:
                print(f"  No {label} example in this sample.")
                continue
            wrapped = textwrap.fill(
                diagnostic.claim_text, width=66, initial_indent="    ", subsequent_indent="    ",
            )
            gt_str = ", ".join(sorted(diagnostic.ground_truth_doc_ids))
            print()
            print(f"  [{label}] Claim ID : {diagnostic.claim_id}")
            print(f"  Ground-truth doc : {gt_str}")
            print("  Claim text:")
            print(wrapped)
            print(f"  Enriched top-{top_k}:")
            for i, (did_str, sc) in enumerate(
                zip(diagnostic.enriched_top_doc_ids, diagnostic.enriched_scores), 1,
            ):
                parts = did_str.split("|")
                is_gt = any(d in diagnostic.ground_truth_doc_ids for d in parts)
                marker = " <-- GT" if is_gt else ""
                print(f"    #{i}  doc_ids={did_str:<16}  score={sc:.4f}{marker}")
            print()

    print()
    print(_SEPARATOR)
    print()


# ---------------------------------------------------------------------------
# 6. JSON export (summary + per-claim enriched retrieval)
# ---------------------------------------------------------------------------

DEFAULT_JSON_REPORT_PATH = "output/evaluation_report.json"


def _ground_truth_doc_id_string(ground_truth_doc_ids: set[str]) -> str:
    """Stable single string for qrels doc id(s) (comma-separated if multiple)."""
    return ",".join(sorted(str(d) for d in ground_truth_doc_ids))


def _build_summary_metrics_enriched(
    results: list[EvalResult],
    top_k: int,
) -> dict:
    """Aggregate Hit@K, MRR, and Precision@K for the enriched retrieval path."""
    n = len(results)
    if n == 0:
        return {
            "total_claims": 0,
            "hits": 0,
            "misses": 0,
            "hit_at_k": 0.0,
            "mrr": 0.0,
            "precision_at_k": 0.0,
        }
    hits = sum(1 for r in results if r.enriched_hit)
    mrr = (
        sum(
            reciprocal_rank_score(r.ground_truth_doc_ids, r.enriched_top_doc_ids, top_k)
            for r in results
        )
        / n
    )
    prec = (
        sum(
            precision_at_k_score(r.ground_truth_doc_ids, r.enriched_top_doc_ids, top_k)
            for r in results
        )
        / n
    )
    return {
        "total_claims": n,
        "hits": hits,
        "misses": n - hits,
        "hit_at_k": hits / n,
        "mrr": mrr,
        "precision_at_k": prec,
    }


def _build_detailed_results_enriched(
    results: list[EvalResult],
    top_k: int,
) -> list[dict]:
    """One row per claim: enriched top-k, hit flag, and first relevant rank."""
    rows: list[dict] = []
    for r in results:
        top_k_results = [
            {"doc_id": doc_id, "score": score}
            for doc_id, score in zip(r.enriched_top_doc_ids, r.enriched_scores)
        ]
        hit_rank = _first_relevant_rank(
            r.ground_truth_doc_ids, r.enriched_top_doc_ids, top_k,
        )
        rows.append(
            {
                "claim_id": r.claim_id,
                "claim_text": r.claim_text,
                "ground_truth_doc_id": _ground_truth_doc_id_string(r.ground_truth_doc_ids),
                "is_hit": r.enriched_hit,
                "hit_rank": hit_rank,
                "top_k_results": top_k_results,
            },
        )
    return rows


class EvaluationJsonExport:
    """Serialises evaluation results to a JSON file (enriched retrieval metrics)."""

    def __init__(self, results: list[EvalResult], top_k: int) -> None:
        self._results = results
        self._top_k = top_k

    def export_to_json(self, output_path: str | Path) -> None:
        """Write summary_metrics and detailed_results to output_path as UTF-8 JSON."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "summary_metrics": _build_summary_metrics_enriched(self._results, self._top_k),
            "detailed_results": _build_detailed_results_enriched(self._results, self._top_k),
        }
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        logger.info("Wrote evaluation JSON report to %s", path.resolve())


# ---------------------------------------------------------------------------
# 7. Multi-collection ablation evaluation (4x2 matrix)
# ---------------------------------------------------------------------------

# Modes for the ablation matrix: dense_only ignores any payload anchors;
# hybrid combines dense ranks with MatchText ranks on ``anchor_queries`` /
# ``keywords``; ``both`` runs both modes back-to-back on the same collection.
MODE_DENSE_ONLY = "dense_only"
MODE_HYBRID = "hybrid"
MODE_BOTH = "both"
SUPPORTED_MODES: tuple[str, ...] = (MODE_DENSE_ONLY, MODE_HYBRID, MODE_BOTH)

DEFAULT_COMPARISON_JSON = "output/comparison_report.json"
DEFAULT_COMPARISON_MD = "output/comparison_report.md"


def _collection_has_anchor_queries(
    client: QdrantClient,
    collection_name: str,
    probe_limit: int = 32,
) -> bool:
    """Heuristically test whether ``anchor_queries`` exist on a collection's payload.

    Scrolls a tiny window of points and inspects their payloads. Returns
    True as soon as one point has a non-empty ``anchor_queries`` field.
    The hybrid path uses this probe to decide whether to silently fall
    back to ``dense_only`` (see :func:`run_single_collection_evaluation`).
    """
    try:
        records, _ = client.scroll(
            collection_name=collection_name,
            limit=probe_limit,
            with_payload=True,
        )
    except Exception as exc:
        logger.warning(
            "Could not probe %r for anchor_queries (%s); assuming none.",
            collection_name,
            exc,
        )
        return False
    for rec in records:
        payload = getattr(rec, "payload", None) or {}
        value = payload.get("anchor_queries")
        if isinstance(value, list) and any(
            isinstance(v, str) and v.strip() for v in value
        ):
            return True
        if isinstance(value, str) and value.strip():
            return True
    return False


@dataclass
class CollectionEvalSummary:
    """Aggregated metrics for one (collection, mode) cell of the ablation matrix."""

    collection: str
    mode: str
    requested_mode: str
    fell_back_to_dense_only: bool
    total_claims: int
    hits: int
    hit_at_k: float
    mrr: float
    precision_at_k: float
    top_k: int
    detailed_results: list[dict]


def _run_collection_search(
    client: QdrantClient,
    model: SentenceTransformer,
    collection_name: str,
    effective_mode: str,
    claim_text: str,
    top_k: int,
    rerank_candidates: int,
    cross_encoder: CrossEncoder | None,
) -> tuple[list[str], list[float]]:
    """Dispatch one claim to the right search function based on the mode."""
    if effective_mode == MODE_DENSE_ONLY:
        doc_ids, scores = _search_baseline(
            client, model, claim_text, top_k=top_k, collection_name=collection_name,
        )
        return doc_ids, scores

    # Hybrid path (with optional cross-encoder re-rank).
    collect_limit = max(rerank_candidates, top_k) if cross_encoder is not None else top_k
    doc_ids, scores, texts = _search_enriched(
        client,
        model,
        claim_text,
        collect_limit=collect_limit,
        collection_name=collection_name,
    )
    if cross_encoder is not None:
        doc_ids, scores = _cross_encoder_rerank_top_k(
            cross_encoder, claim_text, doc_ids, texts, top_k,
        )
    else:
        doc_ids = doc_ids[:top_k]
        scores = scores[:top_k]
    return doc_ids, scores


def run_single_collection_evaluation(
    client: QdrantClient,
    model: SentenceTransformer,
    collection_name: str,
    mode: str,
    claims: list[dict],
    top_k: int = TOP_K,
    cross_encoder: CrossEncoder | None = None,
    rerank_candidates: int = 40,
) -> CollectionEvalSummary:
    """Evaluate one (collection, mode) cell of the ablation matrix.

    Applies the safe-fallback rule: if ``mode == 'hybrid'`` but the
    collection does not expose any ``anchor_queries`` payload, the
    evaluation transparently switches to ``dense_only`` instead of
    crashing. The ``fell_back_to_dense_only`` flag in the summary
    records this decision for the final report.
    """
    if mode not in (MODE_DENSE_ONLY, MODE_HYBRID):
        raise ValueError(
            f"Unsupported single-cell mode: {mode!r}; use 'dense_only' or 'hybrid'."
        )

    effective_mode = mode
    fell_back = False
    if mode == MODE_HYBRID and not _collection_has_anchor_queries(
        client, collection_name,
    ):
        logger.warning(
            "Collection %r has no anchor_queries payload; hybrid mode falling "
            "back to dense_only.",
            collection_name,
        )
        effective_mode = MODE_DENSE_ONLY
        fell_back = True

    detailed: list[dict] = []
    hits = 0
    mrr_sum = 0.0
    prec_sum = 0.0

    for idx, claim in enumerate(claims, 1):
        gt: set[str] = claim["ground_truth_doc_ids"]
        doc_ids, scores = _run_collection_search(
            client=client,
            model=model,
            collection_name=collection_name,
            effective_mode=effective_mode,
            claim_text=claim["text"],
            top_k=top_k,
            rerank_candidates=rerank_candidates,
            cross_encoder=cross_encoder,
        )
        is_hit = _check_hit(gt, doc_ids)
        if is_hit:
            hits += 1
        mrr_sum += reciprocal_rank_score(gt, doc_ids, top_k)
        prec_sum += precision_at_k_score(gt, doc_ids, top_k)

        detailed.append(
            {
                "claim_id": claim["claim_id"],
                "claim_text": claim["text"],
                "ground_truth_doc_id": _ground_truth_doc_id_string(gt),
                "is_hit": is_hit,
                "hit_rank": _first_relevant_rank(gt, doc_ids, top_k),
                "top_k_results": [
                    {"doc_id": d, "score": s} for d, s in zip(doc_ids, scores)
                ],
            }
        )

        if idx % 50 == 0 or idx == len(claims):
            logger.info(
                "[%s | %s] evaluated %d / %d claims (hits=%d)",
                collection_name,
                effective_mode,
                idx,
                len(claims),
                hits,
            )

    n = len(claims)
    return CollectionEvalSummary(
        collection=collection_name,
        mode=effective_mode,
        requested_mode=mode,
        fell_back_to_dense_only=fell_back,
        total_claims=n,
        hits=hits,
        hit_at_k=(hits / n) if n else 0.0,
        mrr=(mrr_sum / n) if n else 0.0,
        precision_at_k=(prec_sum / n) if n else 0.0,
        top_k=top_k,
        detailed_results=detailed,
    )


def run_ablation_evaluation(
    collections: list[str],
    mode: str = MODE_BOTH,
    db_path: str = QDRANT_DB_PATH,
    document_limit: int | None = DOCUMENT_LIMIT,
    sample_size: int | None = DEFAULT_CLAIM_SAMPLE_SIZE,
    top_k: int = TOP_K,
    random_seed: int = RANDOM_SEED,
    rerank_model: str | None = None,
    rerank_candidates: int = 40,
) -> list[CollectionEvalSummary]:
    """Run the 4x2 ablation matrix and return one summary per cell.

    For ``mode == 'both'`` every collection is evaluated twice (once
    dense_only, once hybrid). For ``mode in {'dense_only', 'hybrid'}``
    each collection is evaluated once with the requested mode (hybrid may
    fall back to dense_only as documented in
    :func:`run_single_collection_evaluation`).
    """
    if mode not in SUPPORTED_MODES:
        raise ValueError(
            f"Unsupported --mode value {mode!r}; expected one of {SUPPORTED_MODES}."
        )
    if not collections:
        raise ValueError("At least one collection name must be supplied.")

    indexed_doc_ids = load_indexed_doc_ids(limit=document_limit)
    claims = load_claims_with_ground_truth(
        indexed_doc_ids, sample_size=sample_size, random_seed=random_seed,
    )
    if not claims:
        logger.error("No eligible claims found - aborting ablation.")
        return []

    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    cross_encoder: CrossEncoder | None = None
    if rerank_model:
        logger.info("Initialising cross-encoder re-ranker: %s", rerank_model)
        cross_encoder = CrossEncoder(rerank_model)

    client = QdrantClient(path=db_path)
    try:
        existing = _qdrant_collection_names(client)
        missing = [c for c in collections if c not in existing]
        if missing:
            client.close()
            raise ValueError(
                f"Missing Qdrant collections: {missing}. Available: {sorted(existing)}",
            )

        modes_per_collection = (
            (MODE_DENSE_ONLY, MODE_HYBRID) if mode == MODE_BOTH else (mode,)
        )

        summaries: list[CollectionEvalSummary] = []
        for coll in collections:
            for run_mode in modes_per_collection:
                logger.info(
                    "=== Ablation cell: collection=%s, mode=%s ===", coll, run_mode,
                )
                summary = run_single_collection_evaluation(
                    client=client,
                    model=model,
                    collection_name=coll,
                    mode=run_mode,
                    claims=claims,
                    top_k=top_k,
                    cross_encoder=cross_encoder,
                    rerank_candidates=rerank_candidates,
                )
                summaries.append(summary)
        return summaries
    finally:
        client.close()


# ---------------------------------------------------------------------------
# 8. Comparison report (JSON + Markdown)
# ---------------------------------------------------------------------------


def _summary_to_dict(summary: CollectionEvalSummary) -> dict:
    """Serialise one cell summary including the per-claim breakdown."""
    return {
        "collection": summary.collection,
        "mode": summary.mode,
        "requested_mode": summary.requested_mode,
        "fell_back_to_dense_only": summary.fell_back_to_dense_only,
        "top_k": summary.top_k,
        "summary_metrics": {
            "total_claims": summary.total_claims,
            "hits": summary.hits,
            "misses": summary.total_claims - summary.hits,
            "hit_at_k": summary.hit_at_k,
            "mrr": summary.mrr,
            "precision_at_k": summary.precision_at_k,
        },
        "detailed_results": summary.detailed_results,
    }


def write_comparison_json(
    summaries: list[CollectionEvalSummary],
    path: str | Path = DEFAULT_COMPARISON_JSON,
    extra_metadata: dict | None = None,
) -> Path:
    """Persist the full ablation report as JSON.

    Always includes per-claim ``detailed_results`` so post-hoc analyses
    (significance tests, error analysis) can be run offline.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload: dict = {
        "metadata": extra_metadata or {},
        "cells": [_summary_to_dict(s) for s in summaries],
    }
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    logger.info("Wrote comparison JSON report to %s", out.resolve())
    return out


def _collections_for_matrix(summaries: list[CollectionEvalSummary]) -> list[str]:
    """Stable ordering: preserve first-occurrence order from *summaries*."""
    seen: list[str] = []
    for s in summaries:
        if s.collection not in seen:
            seen.append(s.collection)
    return seen


def _format_metric_cell(value: float, kind: str = "pct") -> str:
    """Format a metric for the Markdown matrix table."""
    if kind == "pct":
        return f"{value * 100:.1f}%"
    if kind == "mrr":
        return f"{value:.3f}"
    return f"{value:.4f}"


def write_comparison_markdown(
    summaries: list[CollectionEvalSummary],
    path: str | Path = DEFAULT_COMPARISON_MD,
    extra_metadata: dict | None = None,
    auxiliary_stats: dict[str, dict] | None = None,
) -> Path:
    """Render the 4x2 ablation matrix as a Markdown report."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    cells: dict[tuple[str, str], CollectionEvalSummary] = {
        (s.collection, s.mode): s for s in summaries
    }
    collections = _collections_for_matrix(summaries)
    top_k = summaries[0].top_k if summaries else TOP_K

    lines: list[str] = []
    lines.append("# RAG Ablation Study - Comparison Report")
    lines.append("")
    if extra_metadata:
        lines.append("## Run metadata")
        lines.append("")
        for key, value in extra_metadata.items():
            lines.append(f"- **{key}**: {value}")
        lines.append("")

    lines.append(f"## 4 x 2 metric matrix (Hit@{top_k} / MRR / P@{top_k})")
    lines.append("")
    lines.append(
        f"| Collection | dense_only Hit@{top_k} | dense_only MRR | dense_only P@{top_k} | "
        f"hybrid Hit@{top_k} | hybrid MRR | hybrid P@{top_k} | hybrid fallback? |"
    )
    lines.append(
        "|---|---|---|---|---|---|---|---|"
    )
    for coll in collections:
        row = [f"`{coll}`"]
        for mode in (MODE_DENSE_ONLY, MODE_HYBRID):
            s = cells.get((coll, mode))
            if s is None:
                row += ["-", "-", "-"]
                continue
            row += [
                _format_metric_cell(s.hit_at_k, "pct"),
                _format_metric_cell(s.mrr, "mrr"),
                _format_metric_cell(s.precision_at_k, "pct"),
            ]
        hybrid_s = cells.get((coll, MODE_HYBRID))
        row.append("yes (no anchors)" if hybrid_s and hybrid_s.fell_back_to_dense_only else "no")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    if auxiliary_stats:
        lines.append("## Auxiliary chunk statistics per method")
        lines.append("")
        lines.append(
            "| Collection | Chunks | Avg tokens | Median tokens | Chosen K (BIC) |"
        )
        lines.append("|---|---|---|---|---|")
        for coll in collections:
            stats = auxiliary_stats.get(coll, {})
            lines.append(
                f"| `{coll}` | "
                f"{stats.get('num_chunks', '-')} | "
                f"{stats.get('avg_tokens', '-')} | "
                f"{stats.get('median_tokens', '-')} | "
                f"{stats.get('chosen_k', '-')} |"
            )
        lines.append("")

    lines.append("## Per-cell summary")
    lines.append("")
    for s in summaries:
        lines.append(
            f"### `{s.collection}` - mode=`{s.mode}` "
            f"(requested=`{s.requested_mode}`)"
        )
        lines.append("")
        lines.append(f"- Total claims: {s.total_claims}")
        lines.append(f"- Hits: {s.hits}")
        lines.append(f"- Hit@{s.top_k}: {_format_metric_cell(s.hit_at_k, 'pct')}")
        lines.append(f"- MRR: {_format_metric_cell(s.mrr, 'mrr')}")
        lines.append(f"- P@{s.top_k}: {_format_metric_cell(s.precision_at_k, 'pct')}")
        if s.fell_back_to_dense_only:
            lines.append("- Hybrid mode fell back to dense_only "
                         "(no anchor_queries in payload).")
        lines.append("")

    out.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote comparison Markdown report to %s", out.resolve())
    return out


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def main() -> None:
    if sys.platform == "win32":
        for stream in (sys.stdout, sys.stderr):
            if hasattr(stream, "reconfigure"):
                try:
                    stream.reconfigure(encoding="utf-8", errors="replace")
                except (AttributeError, OSError, ValueError):
                    pass

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate retrieval vs SciFact test qrels (gold): "
            "each claim is a query; Hit@K, MRR, and Precision@K vs top-K."
        ),
    )
    parser.add_argument(
        "-d",
        "--db-path",
        default=QDRANT_DB_PATH,
        help=f"Qdrant storage path (default: {QDRANT_DB_PATH})",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Same N as semantic_chunker/baseline corpus cap — defines which "
            "evidence docs are treated as indexed for claim eligibility. "
            "Omit for full evidence-filtered corpus (default; SciFact → 300 claims)."
        ),
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Cap the number of claims to evaluate (random subsample after stable "
            f"sort; uses --seed, default {RANDOM_SEED}). Omit to evaluate every "
            "eligible claim with evidence in the indexed subset."
        ),
    )
    parser.add_argument(
        "-k",
        "--top-k",
        type=int,
        default=TOP_K,
        help=(
            f"Top-K retrieval cutoff for Hit@K, MRR, and Precision@K "
            f"(default: {TOP_K})"
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help=f"Random seed for claim sampling (default: {RANDOM_SEED})",
    )
    parser.add_argument(
        "--enriched-only",
        action="store_true",
        help="Skip baseline collection; report enriched Hit@K vs qrels only",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=DEFAULT_JSON_REPORT_PATH,
        help=(
            "Path for JSON export (summary_metrics + detailed_results; "
            f"default: {DEFAULT_JSON_REPORT_PATH})"
        ),
    )
    parser.add_argument(
        "--rerank-model",
        type=str,
        default=None,
        metavar="NAME",
        help=(
            "Optional sentence-transformers CrossEncoder model id "
            "(e.g. cross-encoder/ms-marco-MiniLM-L-6-v2). When set, enriched "
            "retrieval fetches --rerank-candidates rows, re-scores (claim, chunk) "
            "pairs, then keeps top-K for metrics."
        ),
    )
    parser.add_argument(
        "--rerank-candidates",
        type=int,
        default=40,
        metavar="N",
        help=(
            "Candidate pool size before cross-encoder re-ranking on the enriched "
            "path (default: 40). Ignored unless --rerank-model is set."
        ),
    )
    parser.add_argument(
        "--collections",
        nargs="+",
        default=None,
        metavar="NAME",
        help=(
            "Run the 4x2 ablation evaluator over one or more Qdrant collections. "
            "When provided, takes precedence over the legacy baseline-vs-enriched "
            "pipeline. Use --mode to control dense_only / hybrid / both."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=SUPPORTED_MODES,
        default=MODE_BOTH,
        help=(
            "Retrieval mode for the multi-collection ablation. dense_only uses "
            "vector-only search; hybrid combines dense + lexical MatchText on "
            "anchor_queries/keywords; 'both' runs each collection twice. "
            "(default: both)"
        ),
    )
    parser.add_argument(
        "--comparison-json",
        default=DEFAULT_COMPARISON_JSON,
        help=(
            "Output path for the multi-collection JSON report "
            f"(default: {DEFAULT_COMPARISON_JSON})"
        ),
    )
    parser.add_argument(
        "--comparison-md",
        default=DEFAULT_COMPARISON_MD,
        help=(
            "Output path for the multi-collection Markdown report "
            f"(default: {DEFAULT_COMPARISON_MD})"
        ),
    )

    args = parser.parse_args()

    if args.rerank_candidates < 1:
        parser.error("--rerank-candidates must be >= 1")

    if args.collections:
        # New multi-collection path (Step 4 of the ablation study).
        doc_limit = DOCUMENT_LIMIT if args.limit is None else args.limit
        summaries = run_ablation_evaluation(
            collections=list(args.collections),
            mode=args.mode,
            db_path=args.db_path,
            document_limit=doc_limit,
            sample_size=args.sample_size,
            top_k=args.top_k,
            random_seed=args.seed,
            rerank_model=args.rerank_model,
            rerank_candidates=args.rerank_candidates,
        )
        if not summaries:
            print("Ablation evaluation produced no results.")
            return

        metadata = {
            "db_path": args.db_path,
            "mode": args.mode,
            "top_k": args.top_k,
            "sample_size": args.sample_size,
            "rerank_model": args.rerank_model,
            "rerank_candidates": args.rerank_candidates if args.rerank_model else None,
        }
        write_comparison_json(summaries, args.comparison_json, extra_metadata=metadata)
        write_comparison_markdown(summaries, args.comparison_md, extra_metadata=metadata)

        # Console summary for quick triage.
        print()
        print(_SEPARATOR)
        print("  ABLATION SUMMARY (Hit@K / MRR / P@K)")
        print(_SEPARATOR)
        for s in summaries:
            print(
                f"  {s.collection:<28} mode={s.mode:<10} "
                f"Hit@{s.top_k}={_format_metric_cell(s.hit_at_k, 'pct'):>6}  "
                f"MRR={_format_metric_cell(s.mrr, 'mrr'):>6}  "
                f"P@{s.top_k}={_format_metric_cell(s.precision_at_k, 'pct'):>6}"
                + ("  (hybrid -> dense_only)" if s.fell_back_to_dense_only else "")
            )
        print(_SEPARATOR)
        return

    doc_limit = DOCUMENT_LIMIT if args.limit is None else args.limit
    results, compare_baseline_used = run_evaluation(
        db_path=args.db_path,
        document_limit=doc_limit,
        sample_size=args.sample_size,
        top_k=args.top_k,
        random_seed=args.seed,
        compare_baseline=not args.enriched_only,
        rerank_model=args.rerank_model,
        rerank_candidates=args.rerank_candidates,
    )
    enriched_label = (
        "Enriched (hybrid w-RRF + CrossEncoder)"
        if args.rerank_model
        else "Enriched (hybrid w-RRF)"
    )
    print_report(
        results,
        top_k=args.top_k,
        compare_baseline=compare_baseline_used,
        enriched_method_label=enriched_label,
        rerank_model=args.rerank_model,
        rerank_candidates=args.rerank_candidates,
    )
    EvaluationJsonExport(results, top_k=args.top_k).export_to_json(args.output)


if __name__ == "__main__":
    main()
