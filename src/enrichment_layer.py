"""
Enrichment Layer – shared post-chunking pipeline for the RAG ablation study.

This module factors the topic discovery + anchor query generation logic out
of ``semantic_chunker.py`` and ``generative_threading.py`` so it can be
applied to ANY pre-formed list of chunks (character splitter, sentence
window, LLM boundary, or our custom semantic chunker).

Public API
----------
``enrich_chunk_set(chunks, k_min=3, k_max=20, ...) -> EnrichmentResult``
    Embed every chunk's constituent sentences, run PCA -> BIC -> GMM to
    discover global topics, derive a majority topic per chunk, extract core
    sentences per topic, generate one anchor query per topic via Groq, and
    associate anchor queries back to the chunks. Returns the enriched chunk
    dicts together with the topic anchor payload and BIC-selected K.

Every chunk dict returned by this layer is guaranteed to carry at least:
``chunk_id``, ``doc_ids``, ``text``, ``token_count``, ``topic_id``,
``keywords``, ``anchor_queries``.

Notes on input flexibility
--------------------------
The function accepts either dataclass instances exposing the same field
names as ``semantic_chunker.Chunk`` or plain dicts (e.g. the baseline
character chunks which use ``doc_id`` rather than ``doc_ids``). The result
is always a list of dicts so downstream stages can serialise with
``json.dumps`` directly.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from semantic_chunker import (
    EMBEDDING_DIM,
    EMBEDDING_MODEL_NAME,
    MAX_GLOBAL_TOPICS,
    MIN_GLOBAL_K,
    _count_tokens,
    collect_core_sentences_per_topic,
    enrich_chunks_with_bertopic,
    find_optimal_k_bic,
    fit_global_sentence_gmm,
    reduce_embeddings_for_gmm,
    split_document_into_sentences,
)
from generative_threading import (
    associate_queries_to_chunks,
    generate_global_anchor_queries,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public dataclass
# ---------------------------------------------------------------------------


class EnrichmentResult:
    """Container for the output of :func:`enrich_chunk_set`.

    Attributes
    ----------
    chunks : list of dict
        Chunk dicts with ``topic_id``, ``keywords``, ``anchor_queries``.
    optimal_k : int
        BIC-selected number of global topics.
    topics_payload : list of dict
        Per-topic ``{topic_id, core_sentences}`` records, ready to be
        persisted as ``global_topic_anchors.json``.
    anchor_queries : list of str
        One generated anchor query per topic (parallel to ``anchor_topic_ids``).
    anchor_topic_ids : list of int
        Topic ids aligned with ``anchor_queries``.
    """

    __slots__ = (
        "chunks",
        "optimal_k",
        "topics_payload",
        "anchor_queries",
        "anchor_topic_ids",
    )

    def __init__(
        self,
        chunks: list[dict[str, Any]],
        optimal_k: int,
        topics_payload: list[dict[str, Any]],
        anchor_queries: list[str],
        anchor_topic_ids: list[int],
    ) -> None:
        self.chunks = chunks
        self.optimal_k = optimal_k
        self.topics_payload = topics_payload
        self.anchor_queries = anchor_queries
        self.anchor_topic_ids = anchor_topic_ids


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _chunk_to_dict(chunk: Any) -> dict[str, Any]:
    """Normalise a chunk input (dataclass / dict) into a plain dict.

    Handles the baseline format (``doc_id`` string) by promoting it to a
    one-element ``doc_ids`` list so downstream stages can treat every
    method uniformly.
    """
    if isinstance(chunk, dict):
        d = dict(chunk)
    elif is_dataclass(chunk):
        d = asdict(chunk)
    else:
        raise TypeError(
            f"Unsupported chunk type for enrichment: {type(chunk).__name__}",
        )

    if "doc_ids" not in d:
        if "doc_id" in d:
            d["doc_ids"] = [str(d["doc_id"])]
        else:
            d["doc_ids"] = []
    d["doc_ids"] = [str(x) for x in (d.get("doc_ids") or [])]

    if "text" not in d:
        raise KeyError("Chunk dict missing required 'text' field")
    if "chunk_id" not in d:
        raise KeyError("Chunk dict missing required 'chunk_id' field")

    if "token_count" not in d or d["token_count"] is None:
        d["token_count"] = _count_tokens(str(d["text"]))

    # Initialise enrichment fields so callers can rely on them existing.
    d.setdefault("topic_id", -1)
    d.setdefault("keywords", [])
    d.setdefault("anchor_queries", [])
    return d


def _split_chunk_text_to_sentences(text: str) -> list[str]:
    """Split a chunk body back into sentences for topic discovery.

    Re-uses the canonical sentence splitter from ``semantic_chunker`` by
    wrapping the chunk text as a degenerate single-document dict so that
    the title-handling branch is bypassed (no title is prepended).
    """
    return split_document_into_sentences({"title": "", "text": text or ""})


def _majority_topic(labels: list[int]) -> int:
    """Return the majority topic id from a list of per-sentence labels."""
    if not labels:
        return -1
    counts: dict[int, int] = {}
    for lab in labels:
        counts[int(lab)] = counts.get(int(lab), 0) + 1
    max_count = max(counts.values())
    # Tie-breaking by first-occurrence order to keep results deterministic.
    for lab in labels:
        if counts[int(lab)] == max_count:
            return int(lab)
    return int(labels[0])


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def enrich_chunk_set(
    chunks: list[Any],
    k_min: int = MIN_GLOBAL_K,
    k_max: int = MAX_GLOBAL_TOPICS,
    embedding_model: Optional[SentenceTransformer] = None,
    use_llm: bool = True,
    pca_components: Optional[int] = None,
    batch_size: int = 64,
) -> EnrichmentResult:
    """Apply the full enrichment pipeline to any list of chunks.

    Pipeline
    --------
    1. Split each chunk's text into sentences, embed every sentence (plain
       text, no doc-id prefix).
    2. PCA-reduce the sentence embeddings.
    3. BIC-select the number of GMM components K in ``[k_min, k_max]``.
    4. Fit a global GMM and label every sentence with a topic id.
    5. Majority-vote per chunk to assign ``topic_id``.
    6. Collect core sentences (highest cosine sim to centroid) per topic.
    7. Generate exactly one global anchor query per topic via Groq.
    8. Associate anchor queries to chunks by cosine similarity.
    9. Run the supervised BERTopic enrichment to fill ``keywords`` per chunk.

    Parameters
    ----------
    chunks : list
        Input chunks (dataclass instances or dicts).
    k_min, k_max : int
        BIC search bounds for the number of global topics.
    embedding_model : SentenceTransformer or None
        Pre-loaded model; loaded on demand if omitted.
    use_llm : bool
        If False, mock anchor queries are used instead of Groq.
    pca_components : int or None
        Override the PCA component count; ``None`` uses the global default
        from :mod:`semantic_chunker`.
    batch_size : int
        Sentence embedding batch size.
    """
    normalised = [_chunk_to_dict(c) for c in chunks]
    if not normalised:
        logger.warning("enrich_chunk_set received an empty chunk list.")
        return EnrichmentResult(
            chunks=[],
            optimal_k=0,
            topics_payload=[],
            anchor_queries=[],
            anchor_topic_ids=[],
        )

    model = embedding_model or SentenceTransformer(EMBEDDING_MODEL_NAME)

    # 1. Sentence extraction per chunk -- keep parallel indices so we can
    #    fold labels back into chunk-level majority topics.
    sentences_per_chunk: list[list[str]] = []
    flat_sentences: list[str] = []
    chunk_idx_for_sentence: list[int] = []
    for i, ch in enumerate(normalised):
        sents = _split_chunk_text_to_sentences(str(ch["text"]))
        sentences_per_chunk.append(sents)
        for s in sents:
            flat_sentences.append(s)
            chunk_idx_for_sentence.append(i)

    if not flat_sentences:
        logger.warning(
            "enrich_chunk_set: no sentences extracted from %d chunks; "
            "returning chunks unchanged.",
            len(normalised),
        )
        return EnrichmentResult(
            chunks=normalised,
            optimal_k=0,
            topics_payload=[],
            anchor_queries=[],
            anchor_topic_ids=[],
        )

    # 2. Sentence embeddings (plain text only).
    logger.info(
        "Embedding %d sentences across %d chunks (batch_size=%d)",
        len(flat_sentences),
        len(normalised),
        batch_size,
    )
    embeddings = model.encode(
        flat_sentences,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    if embeddings.shape[0] == 0:
        return EnrichmentResult(
            chunks=normalised,
            optimal_k=0,
            topics_payload=[],
            anchor_queries=[],
            anchor_topic_ids=[],
        )

    # 3-4. PCA -> BIC -> GMM.
    reduced = (
        reduce_embeddings_for_gmm(embeddings)
        if pca_components is None
        else reduce_embeddings_for_gmm(embeddings, n_components=pca_components)
    )
    optimal_k = find_optimal_k_bic(reduced, max_k=k_max, min_k=k_min)
    _gmm, sentence_labels = fit_global_sentence_gmm(reduced, optimal_k)

    # 5. Majority-vote per chunk.
    labels_per_chunk: list[list[int]] = [[] for _ in normalised]
    for sent_idx, chunk_idx in enumerate(chunk_idx_for_sentence):
        labels_per_chunk[chunk_idx].append(int(sentence_labels[sent_idx]))
    for i, ch in enumerate(normalised):
        ch["topic_id"] = _majority_topic(labels_per_chunk[i])

    # 6. Core sentences per topic (uses original 768-d embeddings).
    topics_payload = collect_core_sentences_per_topic(
        embeddings, sentence_labels, flat_sentences, top_n=8,
    )

    # 7. Generate one anchor query per topic.
    anchor_queries, anchor_topic_ids, _k = generate_global_anchor_queries(
        topics_payload, use_llm=use_llm,
    )

    # 8. Associate anchor queries to chunks via cosine similarity.
    chunk_texts = [str(c["text"]) for c in normalised]
    associations = associate_queries_to_chunks(
        anchor_queries, anchor_topic_ids, chunk_texts, model,
    )
    for i, ch in enumerate(normalised):
        rows = associations.get(i, [])
        ch["anchor_queries"] = [str(r["text"]) for r in rows]

    # 9. BERTopic-driven keywords (no UMAP/HDBSCAN; uses chunk topic ids).
    #    We feed plain dict chunks through a lightweight adapter so we can
    #    reuse the existing helper which expects attribute access.
    _bertopic_enrich_dicts(normalised)

    logger.info(
        "Enrichment complete: K=%d topics, %d chunks tagged, %d anchor queries.",
        optimal_k,
        len(normalised),
        len(anchor_queries),
    )

    return EnrichmentResult(
        chunks=normalised,
        optimal_k=int(optimal_k),
        topics_payload=topics_payload,
        anchor_queries=list(anchor_queries),
        anchor_topic_ids=list(anchor_topic_ids),
    )


# ---------------------------------------------------------------------------
# Internal: BERTopic enrichment adapter for plain dicts
# ---------------------------------------------------------------------------


class _ChunkProxy:
    """Lightweight attribute-access wrapper around a chunk dict.

    The existing ``enrich_chunks_with_bertopic`` helper expects each element
    to expose ``text``, ``topic_id`` and ``keywords`` as attributes. We
    proxy attribute access onto the underlying dict and write the resulting
    keywords back through the proxy so the caller sees the changes.
    """

    __slots__ = ("_d",)

    def __init__(self, d: dict[str, Any]) -> None:
        self._d = d

    def __getattr__(self, name: str) -> Any:
        if name == "_d":
            raise AttributeError(name)
        return self._d[name]

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "_d":
            object.__setattr__(self, name, value)
        else:
            self._d[name] = value


def _bertopic_enrich_dicts(chunk_dicts: list[dict[str, Any]]) -> None:
    """Run the supervised BERTopic enrichment on a list of dict chunks.

    Chunks with a degenerate ``topic_id < 0`` (e.g. empty bodies that
    produced no sentences) are excluded from the BERTopic fit. The
    underlying helper short-circuits the whole call as soon as it sees a
    single negative label, so without this filter one degenerate chunk
    would silently leave every other chunk with empty ``keywords``.
    """
    valid_dicts = [d for d in chunk_dicts if int(d.get("topic_id", -1)) >= 0]
    if not valid_dicts:
        logger.warning(
            "BERTopic enrichment skipped: no chunks with a valid topic_id.",
        )
        return
    proxies = [_ChunkProxy(d) for d in valid_dicts]
    try:
        enrich_chunks_with_bertopic(proxies)
    except Exception as exc:
        logger.warning(
            "BERTopic enrichment failed (%s); chunks keep empty keywords.", exc,
        )


# ---------------------------------------------------------------------------
# Convenience persistence helpers (optional)
# ---------------------------------------------------------------------------


def write_anchors_payload(
    path: str | Path,
    optimal_k: int,
    topics_payload: list[dict[str, Any]],
    embedding_model_name: str = EMBEDDING_MODEL_NAME,
) -> None:
    """Persist the anchor payload in the same shape as the existing pipeline."""
    import json

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "embedding_model": embedding_model_name,
        "k": int(optimal_k),
        "topics": topics_payload,
    }
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    logger.info("Wrote enrichment anchors (K=%d) to %s", optimal_k, out)


__all__ = [
    "EnrichmentResult",
    "enrich_chunk_set",
    "write_anchors_payload",
    "EMBEDDING_DIM",
    "EMBEDDING_MODEL_NAME",
]
