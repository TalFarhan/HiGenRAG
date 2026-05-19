"""
RAG Ablation Orchestrator – run the full 4x2 comparison pipeline end-to-end.

This script chains all stages of the ablation study so that one command
produces:

1. Raw chunks for each of the four chunking methods (Method 1: char-split,
   Method A: sentence-window, Method B: LLM-boundary, Method C: semantic-GMM).
2. Enriched chunks (topic_id, keywords, anchor_queries) for each method by
   running :func:`enrichment_layer.enrich_chunk_set`.
3. One Qdrant collection per method with the enriched payload.
4. The 4x2 evaluation matrix (dense_only x hybrid) saved as
   ``output/comparison_report.{json,md}``.

Each stage can be skipped with the corresponding ``--skip-*`` flag so a
re-run can pick up where the previous one left off. Skipping a stage
requires the artefacts from that stage to already exist on disk; missing
artefacts produce a clear error rather than a silent failure.

All comments and log messages in this file are in English so the codebase
stays consistent across the ablation suite.
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv

load_dotenv()

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from semantic_chunker import (
    DOCUMENT_LIMIT,
    EMBEDDING_MODEL_NAME,
    MAX_GLOBAL_TOPICS,
    MAX_TOKENS,
    MIN_GLOBAL_K,
    MIN_TOKENS,
    Chunk as SemanticChunk,
    _count_tokens,
    _json_default,
    load_corpus,
    split_document_into_sentences,
)
from baseline_pipeline import naive_chunk_documents
from sentence_window_chunker import (
    WINDOW_OVERLAP_SENTENCES,
    WINDOW_SIZE_SENTENCES,
    sentence_window_chunk_documents,
)
from llm_chunker import (
    DEFAULT_CACHE_PATH as LLM_CACHE_PATH,
    LLM_MODEL_NAME,
    LOOKBACK_SENTENCES,
    _LLMBoundaryJudge,
    _LLMResponseCache,
    llm_chunk_documents,
)
from semantic_chunker import (
    GMM_PCA_N_COMPONENTS,
    collect_core_sentences_per_topic,
    embed_sentence_texts,
    find_optimal_k_bic,
    fit_global_sentence_gmm,
    flatten_sentences_for_corpus,
    form_semantic_physical_chunks,
    reduce_embeddings_for_gmm,
)
from enrichment_layer import EnrichmentResult, enrich_chunk_set, write_anchors_payload
from indexer import (
    DEFAULT_PAYLOAD_FIELDS,
    DEFAULT_QDRANT_DB_PATH,
    index_chunks_to_qdrant,
)
from evaluation_framework import (
    DEFAULT_COMPARISON_JSON,
    DEFAULT_COMPARISON_MD,
    MODE_BOTH,
    SUPPORTED_MODES,
    run_ablation_evaluation,
    write_comparison_json,
    write_comparison_markdown,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Method registry
# ---------------------------------------------------------------------------


# Layout: ``output/ablation/{raw,enriched,anchors}/<method>.json``.
ABLATION_DIR = Path("output") / "ablation"
RAW_DIR = ABLATION_DIR / "raw_chunks"
ENRICHED_DIR = ABLATION_DIR / "enriched_chunks"
ANCHORS_DIR = ABLATION_DIR / "anchors"


METHODS: dict[str, dict[str, str]] = {
    "baseline_char": {
        "label": "Baseline (char split)",
        "collection": "baseline_char",
    },
    "sentence_window": {
        "label": "Sentence-window",
        "collection": "baseline_sentence_window",
    },
    "llm_boundary": {
        "label": "LLM-boundary",
        "collection": "llm_chunks",
    },
    "semantic_gmm": {
        "label": "Semantic-GMM (custom)",
        "collection": "enriched_chunks",
    },
}


def _raw_path(method: str) -> Path:
    return RAW_DIR / f"{method}.json"


def _enriched_path(method: str) -> Path:
    return ENRICHED_DIR / f"{method}.json"


def _anchors_path(method: str) -> Path:
    return ANCHORS_DIR / f"{method}.json"


# ---------------------------------------------------------------------------
# Stage 1: Corpus loading
# ---------------------------------------------------------------------------


def load_dataset_documents(
    dataset_name: str = "scifact",
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Load the evidence-filtered SciFact corpus (shared by every chunker)."""
    logger.info("Loading corpus once for every chunker (limit=%s)", limit)
    return load_corpus(dataset_name, limit=limit if limit is not None else DOCUMENT_LIMIT)


# ---------------------------------------------------------------------------
# Stage 2: Chunking
# ---------------------------------------------------------------------------


def _chunk_to_dict(chunk: Any) -> dict[str, Any]:
    """Normalise a chunk into a JSON-serialisable dict."""
    if isinstance(chunk, dict):
        return dict(chunk)
    return asdict(chunk)


def _save_chunks(path: Path, chunks: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(chunks, fh, ensure_ascii=False, indent=2, default=_json_default)
    logger.info("Saved %d chunks to %s", len(chunks), path)


def chunk_baseline_char(
    documents: list[dict[str, Any]],
    out_path: Path,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> list[dict[str, Any]]:
    """Method 1: fixed-size character splitter (existing baseline_pipeline).

    The naive baseline emits ``{chunk_id, doc_id, text}`` records; we
    normalise them into the canonical ``{chunk_id, doc_ids, text,
    token_count, topic_id, keywords}`` shape before saving.
    """
    raw = naive_chunk_documents(documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    out: list[dict[str, Any]] = []
    for c in raw:
        text = c["text"]
        out.append(
            {
                "chunk_id": c["chunk_id"],
                "doc_ids": [str(c["doc_id"])],
                "text": text,
                "token_count": _count_tokens(text),
                "topic_id": -1,
                "keywords": [],
            }
        )
    _save_chunks(out_path, out)
    return out


def chunk_sentence_window(
    documents: list[dict[str, Any]],
    out_path: Path,
    window_size: int = WINDOW_SIZE_SENTENCES,
    overlap: int = WINDOW_OVERLAP_SENTENCES,
    min_tokens: int = MIN_TOKENS,
    max_tokens: int = MAX_TOKENS,
) -> list[dict[str, Any]]:
    """Method A: fixed-size sentence-window chunker."""
    chunks = sentence_window_chunk_documents(
        documents,
        window_size=window_size,
        overlap=overlap,
        min_tokens=min_tokens,
        max_tokens=max_tokens,
    )
    out = [_chunk_to_dict(c) for c in chunks]
    _save_chunks(out_path, out)
    return out


def chunk_llm_boundary(
    documents: list[dict[str, Any]],
    out_path: Path,
    min_tokens: int = MIN_TOKENS,
    max_tokens: int = MAX_TOKENS,
    lookback: int = LOOKBACK_SENTENCES,
    cache_path: Path = LLM_CACHE_PATH,
    model_name: str = LLM_MODEL_NAME,
) -> list[dict[str, Any]]:
    """Method B: LLM-detected sentence boundaries with persistent cache."""
    cache = _LLMResponseCache(cache_path)
    judge = _LLMBoundaryJudge(cache=cache, model_name=model_name)
    try:
        chunks = llm_chunk_documents(
            documents,
            judge=judge,
            min_tokens=min_tokens,
            max_tokens=max_tokens,
            lookback=lookback,
        )
    finally:
        cache.save()
    out = [_chunk_to_dict(c) for c in chunks]
    _save_chunks(out_path, out)
    return out


def chunk_semantic_gmm(
    documents: list[dict[str, Any]],
    out_path: Path,
    min_tokens: int = MIN_TOKENS,
    max_tokens: int = MAX_TOKENS,
    embedding_model: Optional[SentenceTransformer] = None,
) -> list[dict[str, Any]]:
    """Method C: our custom semantic-GMM chunker (sentence-level boundaries).

    Runs the same sentence-level GMM that ``semantic_chunker.run_pipeline``
    uses, then materialises chunks via ``form_semantic_physical_chunks``.
    No anchor enrichment is performed here -- that happens uniformly in
    the next stage so every method passes through the same enrichment.
    """
    if embedding_model is None:
        embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    flat_sentences, _flat_doc_ids, sentences_per_doc = flatten_sentences_for_corpus(documents)
    embeddings = embed_sentence_texts(flat_sentences, model=embedding_model)
    if embeddings.shape[0] == 0:
        _save_chunks(out_path, [])
        return []
    reduced = reduce_embeddings_for_gmm(embeddings, n_components=GMM_PCA_N_COMPONENTS)
    optimal_k = find_optimal_k_bic(reduced, max_k=MAX_GLOBAL_TOPICS, min_k=MIN_GLOBAL_K)
    _gmm, labels = fit_global_sentence_gmm(reduced, optimal_k)
    topics_per_doc: list[list[int]] = []
    cursor = 0
    for sents in sentences_per_doc:
        topics_per_doc.append([int(labels[cursor + i]) for i in range(len(sents))])
        cursor += len(sents)
    chunks = form_semantic_physical_chunks(
        documents,
        sentences_per_doc,
        topics_per_doc,
        min_tokens=min_tokens,
        max_tokens=max_tokens,
    )
    out = [_chunk_to_dict(c) for c in chunks]
    _save_chunks(out_path, out)
    return out


CHUNKING_DISPATCH = {
    "baseline_char": chunk_baseline_char,
    "sentence_window": chunk_sentence_window,
    "llm_boundary": chunk_llm_boundary,
    "semantic_gmm": chunk_semantic_gmm,
}


def run_chunking_stage(
    documents: list[dict[str, Any]],
    methods: list[str],
    embedding_model: Optional[SentenceTransformer] = None,
) -> dict[str, list[dict[str, Any]]]:
    """Produce raw chunk JSON for every requested method."""
    chunks_by_method: dict[str, list[dict[str, Any]]] = {}
    for method in methods:
        path = _raw_path(method)
        logger.info("== Chunking stage: %s -> %s ==", method, path)
        if method == "baseline_char":
            chunks_by_method[method] = chunk_baseline_char(documents, path)
        elif method == "sentence_window":
            chunks_by_method[method] = chunk_sentence_window(documents, path)
        elif method == "llm_boundary":
            chunks_by_method[method] = chunk_llm_boundary(documents, path)
        elif method == "semantic_gmm":
            chunks_by_method[method] = chunk_semantic_gmm(
                documents, path, embedding_model=embedding_model,
            )
        else:
            raise ValueError(f"Unknown chunking method: {method!r}")
    return chunks_by_method


def load_existing_raw_chunks(methods: list[str]) -> dict[str, list[dict[str, Any]]]:
    """Read pre-computed raw chunks from disk (used when --skip-chunking is set)."""
    out: dict[str, list[dict[str, Any]]] = {}
    for method in methods:
        path = _raw_path(method)
        if not path.is_file():
            raise FileNotFoundError(
                f"--skip-chunking requested but {path} is missing; "
                f"re-run without the flag to regenerate."
            )
        with open(path, encoding="utf-8") as fh:
            out[method] = json.load(fh)
        logger.info("Loaded %d raw chunks for %s from %s", len(out[method]), method, path)
    return out


# ---------------------------------------------------------------------------
# Stage 3: Enrichment
# ---------------------------------------------------------------------------


def run_enrichment_stage(
    raw_chunks_by_method: dict[str, list[dict[str, Any]]],
    embedding_model: Optional[SentenceTransformer] = None,
    use_llm: bool = True,
    k_min: int = MIN_GLOBAL_K,
    k_max: int = MAX_GLOBAL_TOPICS,
) -> dict[str, EnrichmentResult]:
    """Apply :func:`enrich_chunk_set` to each method's raw chunks."""
    enriched_by_method: dict[str, EnrichmentResult] = {}
    if embedding_model is None:
        embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    for method, raw_chunks in raw_chunks_by_method.items():
        logger.info("== Enrichment stage: %s (%d chunks) ==", method, len(raw_chunks))
        result = enrich_chunk_set(
            raw_chunks,
            k_min=k_min,
            k_max=k_max,
            embedding_model=embedding_model,
            use_llm=use_llm,
        )
        enriched_path = _enriched_path(method)
        anchors_path = _anchors_path(method)

        _save_chunks(enriched_path, result.chunks)
        write_anchors_payload(
            anchors_path, result.optimal_k, result.topics_payload,
        )

        enriched_by_method[method] = result
    return enriched_by_method


def load_existing_enriched_chunks(
    methods: list[str],
) -> dict[str, list[dict[str, Any]]]:
    """Read pre-computed enriched chunks from disk (used when --skip-enrichment)."""
    out: dict[str, list[dict[str, Any]]] = {}
    for method in methods:
        path = _enriched_path(method)
        if not path.is_file():
            raise FileNotFoundError(
                f"--skip-enrichment requested but {path} is missing; "
                f"re-run without the flag to regenerate."
            )
        with open(path, encoding="utf-8") as fh:
            out[method] = json.load(fh)
        logger.info(
            "Loaded %d enriched chunks for %s from %s",
            len(out[method]),
            method,
            path,
        )
    return out


def load_existing_anchors_K(methods: list[str]) -> dict[str, int]:
    """Look up the BIC-selected K saved on disk for each method (for the report)."""
    out: dict[str, int] = {}
    for method in methods:
        path = _anchors_path(method)
        if not path.is_file():
            continue
        try:
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
            out[method] = int(data.get("k", 0))
        except (OSError, json.JSONDecodeError):
            continue
    return out


# ---------------------------------------------------------------------------
# Stage 4: Indexing
# ---------------------------------------------------------------------------


def run_indexing_stage(
    chunks_by_method: dict[str, list[dict[str, Any]]],
    db_path: str = DEFAULT_QDRANT_DB_PATH,
    embedding_model: Optional[SentenceTransformer] = None,
    payload_fields: Optional[list[str]] = None,
) -> dict[str, int]:
    """Index every method's enriched chunks into its dedicated collection.

    A single ``QdrantClient`` is shared across all collections because the
    local on-disk Qdrant only allows one writer per process at a time.
    """
    if embedding_model is None:
        embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    fields = payload_fields or list(DEFAULT_PAYLOAD_FIELDS)

    client = QdrantClient(path=db_path)
    counts: dict[str, int] = {}
    try:
        for method, chunks in chunks_by_method.items():
            spec = METHODS[method]
            collection = spec["collection"]
            logger.info(
                "== Indexing stage: %s -> Qdrant collection %r (%d chunks) ==",
                method,
                collection,
                len(chunks),
            )
            n = index_chunks_to_qdrant(
                chunks=chunks,
                collection_name=collection,
                with_payload_fields=fields,
                db_path=db_path,
                embedding_model=embedding_model,
                client=client,
                recreate=True,
            )
            counts[method] = n
    finally:
        client.close()
    return counts


# ---------------------------------------------------------------------------
# Stage 5: Evaluation
# ---------------------------------------------------------------------------


def run_evaluation_stage(
    methods: list[str],
    db_path: str = DEFAULT_QDRANT_DB_PATH,
    document_limit: int | None = None,
    sample_size: int | None = None,
    top_k: int = 3,
    mode: str = MODE_BOTH,
    random_seed: int = 42,
    rerank_model: str | None = None,
    rerank_candidates: int = 40,
) -> list:
    """Run the 4x2 ablation matrix over the configured methods."""
    collections = [METHODS[m]["collection"] for m in methods]
    summaries = run_ablation_evaluation(
        collections=collections,
        mode=mode,
        db_path=db_path,
        document_limit=document_limit,
        sample_size=sample_size,
        top_k=top_k,
        random_seed=random_seed,
        rerank_model=rerank_model,
        rerank_candidates=rerank_candidates,
    )
    return summaries


# ---------------------------------------------------------------------------
# Stage 6: Auxiliary statistics for the final report
# ---------------------------------------------------------------------------


def build_auxiliary_stats(
    methods: list[str],
    enriched_chunks_by_method: dict[str, list[dict[str, Any]]],
    chosen_k_by_method: dict[str, int],
) -> dict[str, dict[str, Any]]:
    """Per-method chunk statistics shown in the final report.

    Returns a dict keyed by Qdrant collection name (matching the matrix
    rows in the Markdown report) so the report writer can join on the
    same key.
    """
    out: dict[str, dict[str, Any]] = {}
    for method in methods:
        chunks = enriched_chunks_by_method.get(method, [])
        token_counts = [
            int(c.get("token_count") or _count_tokens(str(c.get("text", ""))))
            for c in chunks
        ]
        if token_counts:
            avg_tokens = round(sum(token_counts) / len(token_counts), 1)
            median_tokens = int(statistics.median(token_counts))
        else:
            avg_tokens = 0.0
            median_tokens = 0
        spec = METHODS[method]
        out[spec["collection"]] = {
            "num_chunks": len(chunks),
            "avg_tokens": avg_tokens,
            "median_tokens": median_tokens,
            "chosen_k": chosen_k_by_method.get(method, "-"),
        }
    return out


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------


def _validate_methods(methods: list[str]) -> list[str]:
    unknown = [m for m in methods if m not in METHODS]
    if unknown:
        raise ValueError(
            f"Unknown method(s): {unknown}; valid options: {list(METHODS.keys())}",
        )
    return list(methods)


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
            "End-to-end orchestrator for the RAG ablation study: chunk -> "
            "enrich -> index -> evaluate (4x2 matrix)."
        ),
    )
    parser.add_argument(
        "--dataset",
        default="scifact",
        help="BEIR dataset (default: scifact)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Evidence-document cap (omit for the full evidence-filtered corpus)",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=list(METHODS.keys()),
        help="Methods to include (default: all four)",
    )
    parser.add_argument(
        "--db-path",
        default=DEFAULT_QDRANT_DB_PATH,
        help=f"Qdrant on-disk path (default: {DEFAULT_QDRANT_DB_PATH})",
    )
    parser.add_argument(
        "--mode",
        choices=SUPPORTED_MODES,
        default=MODE_BOTH,
        help=f"Evaluation mode (default: {MODE_BOTH})",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Top-K retrieval cutoff (default: 3)",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Sample N claims for evaluation (default: all eligible)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for claim sampling (default: 42)",
    )
    parser.add_argument(
        "--no-llm-anchors",
        action="store_true",
        help="Skip Groq; use deterministic mock anchor queries",
    )
    parser.add_argument(
        "--rerank-model",
        default=None,
        help="Optional CrossEncoder model id for hybrid mode re-ranking",
    )
    parser.add_argument(
        "--rerank-candidates",
        type=int,
        default=40,
        help="Candidate pool size before cross-encoder re-rank (default: 40)",
    )
    parser.add_argument(
        "--comparison-json",
        default=DEFAULT_COMPARISON_JSON,
        help=f"Output JSON report path (default: {DEFAULT_COMPARISON_JSON})",
    )
    parser.add_argument(
        "--comparison-md",
        default=DEFAULT_COMPARISON_MD,
        help=f"Output Markdown report path (default: {DEFAULT_COMPARISON_MD})",
    )

    parser.add_argument("--skip-corpus", action="store_true",
                        help="Skip corpus load (only valid with --skip-chunking).")
    parser.add_argument("--skip-chunking", action="store_true",
                        help="Reuse existing raw_chunks/<method>.json.")
    parser.add_argument("--skip-enrichment", action="store_true",
                        help="Reuse existing enriched_chunks/<method>.json.")
    parser.add_argument("--skip-indexing", action="store_true",
                        help="Reuse existing Qdrant collections.")
    parser.add_argument("--skip-evaluation", action="store_true",
                        help="Stop before the 4x2 evaluation stage.")

    args = parser.parse_args()
    methods = _validate_methods(args.methods)

    if args.skip_corpus and not args.skip_chunking:
        parser.error("--skip-corpus requires --skip-chunking")

    # Shared resources -- loaded lazily so unrelated --skip-* flags do not
    # incur an embedding-model download.
    embedding_model: Optional[SentenceTransformer] = None

    def _get_embedding_model() -> SentenceTransformer:
        nonlocal embedding_model
        if embedding_model is None:
            embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        return embedding_model

    # ----- Stage 1+2: corpus + chunking ----------------------------------
    if args.skip_chunking:
        raw_chunks_by_method = load_existing_raw_chunks(methods)
    else:
        documents = load_dataset_documents(args.dataset, limit=args.limit)
        raw_chunks_by_method = run_chunking_stage(
            documents, methods, embedding_model=_get_embedding_model(),
        )

    # ----- Stage 3: enrichment -------------------------------------------
    if args.skip_enrichment:
        enriched_chunks_by_method = load_existing_enriched_chunks(methods)
        chosen_k_by_method = load_existing_anchors_K(methods)
    else:
        results_by_method = run_enrichment_stage(
            raw_chunks_by_method,
            embedding_model=_get_embedding_model(),
            use_llm=not args.no_llm_anchors,
        )
        enriched_chunks_by_method = {
            m: r.chunks for m, r in results_by_method.items()
        }
        chosen_k_by_method = {
            m: int(r.optimal_k) for m, r in results_by_method.items()
        }

    # ----- Stage 4: indexing ---------------------------------------------
    if args.skip_indexing:
        logger.info("Skipping Qdrant indexing per --skip-indexing")
    else:
        run_indexing_stage(
            enriched_chunks_by_method,
            db_path=args.db_path,
            embedding_model=_get_embedding_model(),
        )

    # ----- Stage 5: evaluation -------------------------------------------
    if args.skip_evaluation:
        logger.info("Skipping 4x2 evaluation per --skip-evaluation; pipeline done.")
        return

    summaries = run_evaluation_stage(
        methods=methods,
        db_path=args.db_path,
        document_limit=args.limit,
        sample_size=args.sample_size,
        top_k=args.top_k,
        mode=args.mode,
        random_seed=args.seed,
        rerank_model=args.rerank_model,
        rerank_candidates=args.rerank_candidates,
    )
    if not summaries:
        print("Ablation evaluation produced no results.")
        return

    metadata = {
        "dataset": args.dataset,
        "methods": methods,
        "mode": args.mode,
        "top_k": args.top_k,
        "sample_size": args.sample_size,
        "rerank_model": args.rerank_model,
        "rerank_candidates": args.rerank_candidates if args.rerank_model else None,
        "anchor_llm_model": (None if args.no_llm_anchors else LLM_MODEL_NAME),
    }
    auxiliary = build_auxiliary_stats(methods, enriched_chunks_by_method, chosen_k_by_method)

    write_comparison_json(summaries, args.comparison_json, extra_metadata=metadata)
    write_comparison_markdown(
        summaries,
        args.comparison_md,
        extra_metadata=metadata,
        auxiliary_stats=auxiliary,
    )

    print()
    print("=" * 72)
    print("  ABLATION COMPLETE")
    print("=" * 72)
    for s in summaries:
        print(
            f"  {s.collection:<28} mode={s.mode:<10} "
            f"Hit@{s.top_k}={s.hit_at_k * 100:6.1f}%  "
            f"MRR={s.mrr:6.3f}  "
            f"P@{s.top_k}={s.precision_at_k * 100:6.1f}%"
            + ("  (hybrid -> dense_only)" if s.fell_back_to_dense_only else "")
        )
    print("=" * 72)
    print(f"  JSON report: {Path(args.comparison_json).resolve()}")
    print(f"  Markdown   : {Path(args.comparison_md).resolve()}")


if __name__ == "__main__":
    main()
