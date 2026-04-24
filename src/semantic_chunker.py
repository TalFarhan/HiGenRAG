"""
Semantic Chunker – Task 1 (§3.0)

Loads a BEIR-compatible research corpus, segments documents into paragraphs,
and applies hierarchical topic-based chunking to produce semantically cohesive
segments of 256-512 tokens.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import ir_datasets
import numpy as np
import tiktoken
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

EMBEDDING_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
EMBEDDING_DIM = 768
MIN_TOKENS = 256
MAX_TOKENS = 512
DOCUMENT_LIMIT = 150

# Minimum paragraphs required for BERTopic to produce meaningful clusters
_BERTOPIC_MIN_DOCS = 10


@dataclass
class Chunk:
    chunk_id: str
    doc_ids: list[str]
    text: str
    token_count: int
    topic_id: int = -1
    keywords: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# 1. Corpus loading
# ---------------------------------------------------------------------------

def _get_evidence_doc_ids(dataset_name: str) -> set[str]:
    """Extract all doc_ids that have positive relevance in the test qrels."""
    canonical_name = dataset_name.split("/")[-1]
    dataset = ir_datasets.load(f"beir/{canonical_name}/test")
    evidence_ids: set[str] = set()
    for qrel in dataset.qrels_iter():
        if qrel.relevance > 0:
            evidence_ids.add(qrel.doc_id)
    return evidence_ids


def load_corpus(dataset_name: str, limit: int = DOCUMENT_LIMIT) -> list[dict]:
    """Load corpus filtered to documents that have evidence in the golden set.

    Steps:
      1. Load qrels from the test split and collect doc_ids with relevance > 0.
      2. Stream the corpus split and keep only documents whose ``_id`` appears
         in the evidence set.
      3. Stop once *limit* documents have been collected.

    Returns a list of dicts with keys ``doc_id``, ``title``, ``text``.
    """
    logger.info("Loading corpus: %s (evidence-filtered, limit=%d)", dataset_name, limit)

    evidence_doc_ids = _get_evidence_doc_ids(dataset_name)
    logger.info("Found %d unique doc_ids with evidence in qrels", len(evidence_doc_ids))

    canonical_name = dataset_name.split("/")[-1]
    corpus_dataset = ir_datasets.load(f"beir/{canonical_name}/test")

    documents: list[dict] = []
    for doc in corpus_dataset.docs_iter():
        if doc.doc_id not in evidence_doc_ids:
            continue
        documents.append(
            {
                "doc_id": doc.doc_id,
                "title": getattr(doc, "title", ""),
                "text": doc.text,
            }
        )
        if len(documents) >= limit:
            break

    logger.info("Loaded %d documents from %s (filtered by evidence)", len(documents), dataset_name)
    return documents


# ---------------------------------------------------------------------------
# 2. Paragraph segmentation
# ---------------------------------------------------------------------------

_PARAGRAPH_SPLIT_RE = re.compile(r"\n{2,}")


def segment_into_paragraphs(
    documents: list[dict],
    min_paragraph_len: int = 30,
) -> tuple[list[str], list[str]]:
    """Split each document's text into paragraph-level units.

    Returns
    -------
    paragraphs : list[str]
        Flat list of paragraphs across all documents.
    para_doc_ids : list[str]
        Parallel list mapping each paragraph to its source ``doc_id``.
    """
    paragraphs: list[str] = []
    para_doc_ids: list[str] = []

    for doc in documents:
        full_text = doc["text"]
        if doc["title"]:
            full_text = doc["title"] + "\n\n" + full_text

        parts = _PARAGRAPH_SPLIT_RE.split(full_text.strip())
        for part in parts:
            part = part.strip()
            if len(part) >= min_paragraph_len:
                paragraphs.append(part)
                para_doc_ids.append(doc["doc_id"])

    logger.info("Segmented into %d paragraphs", len(paragraphs))
    return paragraphs, para_doc_ids


# ---------------------------------------------------------------------------
# 3. Embedding
# ---------------------------------------------------------------------------

def embed_paragraphs(
    paragraphs: list[str],
    model: Optional[SentenceTransformer] = None,
    batch_size: int = 64,
) -> np.ndarray:
    """Produce 768-d vectors for each paragraph using all-mpnet-base-v2."""
    if model is None:
        model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    logger.info("Embedding %d paragraphs (batch_size=%d)", len(paragraphs), batch_size)
    embeddings: np.ndarray = model.encode(
        paragraphs,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    return embeddings


# ---------------------------------------------------------------------------
# 4. Global topic discovery
# ---------------------------------------------------------------------------

def discover_global_topics(
    embeddings: np.ndarray,
    paragraphs: list[str],
    nr_topics: Optional[int] = None,
) -> BERTopic:
    """First-pass BERTopic for document-level topic discovery.

    When the paragraph count is below the BERTopic minimum, a single-topic
    fallback is used so the downstream pipeline can still proceed.
    """
    if len(paragraphs) < _BERTOPIC_MIN_DOCS:
        logger.warning(
            "Only %d paragraphs — too few for BERTopic. "
            "Falling back to a single-topic assignment.",
            len(paragraphs),
        )
        return _single_topic_fallback()

    topic_model = BERTopic(
        embedding_model=EMBEDDING_MODEL_NAME,
        nr_topics=nr_topics,
        verbose=True,
    )
    topic_model.fit(paragraphs, embeddings)

    topic_info = topic_model.get_topic_info()
    logger.info("Discovered %d topics (incl. outlier -1)", len(topic_info))
    return topic_model


def _single_topic_fallback() -> BERTopic:
    """Return a minimally-fitted BERTopic model that assigns topic 0 to all docs."""
    model = BERTopic(embedding_model=EMBEDDING_MODEL_NAME, nr_topics=1)
    return model


# ---------------------------------------------------------------------------
# 5. Cluster assignment
# ---------------------------------------------------------------------------

def assign_paragraphs_to_clusters(
    paragraphs: list[str],
    para_doc_ids: list[str],
    topic_model: BERTopic,
    embeddings: np.ndarray,
) -> dict[int, list[tuple[str, str]]]:
    """Map each paragraph to its topic cluster.

    Returns a dict mapping ``topic_id`` -> list of (doc_id, paragraph_text).
    Outlier paragraphs (topic -1) are redistributed to the nearest valid
    topic using embedding cosine similarity.
    """
    if len(paragraphs) < _BERTOPIC_MIN_DOCS:
        cluster: list[tuple[str, str]] = list(zip(para_doc_ids, paragraphs))
        return {0: cluster}

    topics, _ = topic_model.transform(paragraphs, embeddings)

    clusters: dict[int, list[tuple[str, str]]] = {}
    outliers: list[int] = []

    for idx, topic_id in enumerate(topics):
        topic_id = int(topic_id)
        if topic_id == -1:
            outliers.append(idx)
            continue
        clusters.setdefault(topic_id, []).append(
            (para_doc_ids[idx], paragraphs[idx])
        )

    if outliers and clusters:
        _redistribute_outliers(
            outliers, paragraphs, para_doc_ids, embeddings, clusters, topic_model
        )
    elif outliers and not clusters:
        clusters[0] = [
            (para_doc_ids[i], paragraphs[i]) for i in outliers
        ]

    logger.info(
        "Assigned paragraphs to %d clusters (%d outliers redistributed)",
        len(clusters),
        len(outliers),
    )
    return clusters


def _redistribute_outliers(
    outlier_indices: list[int],
    paragraphs: list[str],
    para_doc_ids: list[str],
    embeddings: np.ndarray,
    clusters: dict[int, list[tuple[str, str]]],
    topic_model: BERTopic,
) -> None:
    """Assign outlier paragraphs to the nearest non-outlier topic centroid."""
    valid_topics = sorted(clusters.keys())
    centroids = []
    for tid in valid_topics:
        topic_embs = []
        for doc_id, text in clusters[tid]:
            idx = next(
                i for i, (d, p) in enumerate(zip(para_doc_ids, paragraphs))
                if d == doc_id and p == text
            )
            topic_embs.append(embeddings[idx])
        centroids.append(np.mean(topic_embs, axis=0))
    centroids_matrix = np.vstack(centroids)

    for oidx in outlier_indices:
        sims = embeddings[oidx] @ centroids_matrix.T
        best_cluster_pos = int(np.argmax(sims))
        best_topic = valid_topics[best_cluster_pos]
        clusters[best_topic].append((para_doc_ids[oidx], paragraphs[oidx]))


# ---------------------------------------------------------------------------
# 6. Chunk refinement (merge / split to 256-512 token bounds)
# ---------------------------------------------------------------------------

_TOKENIZER = tiktoken.get_encoding("cl100k_base")


def _count_tokens(text: str) -> int:
    return len(_TOKENIZER.encode(text))


def refine_chunks(
    clusters: dict[int, list[tuple[str, str]]],
    min_tokens: int = MIN_TOKENS,
    max_tokens: int = MAX_TOKENS,
) -> list[Chunk]:
    """Merge or split clustered paragraphs into token-bounded chunks.

    Greedy merge: accumulate paragraphs until adding another would exceed
    ``max_tokens``, then seal the chunk. If the accumulated text is below
    ``min_tokens`` and more paragraphs exist, keep merging. After
    exhausting the cluster, any oversized remainder is split at sentence
    boundaries.
    """
    all_chunks: list[Chunk] = []
    chunk_counter = 0

    for topic_id in sorted(clusters.keys()):
        items = clusters[topic_id]
        buffer_text = ""
        buffer_doc_ids: list[str] = []

        for doc_id, para in items:
            candidate = (buffer_text + "\n\n" + para).strip() if buffer_text else para
            candidate_tokens = _count_tokens(candidate)

            if candidate_tokens <= max_tokens:
                buffer_text = candidate
                if doc_id not in buffer_doc_ids:
                    buffer_doc_ids.append(doc_id)
            else:
                if buffer_text:
                    for chunk in _emit_chunks(
                        buffer_text, buffer_doc_ids, topic_id,
                        chunk_counter, min_tokens, max_tokens,
                    ):
                        all_chunks.append(chunk)
                        chunk_counter += 1

                buffer_text = para
                buffer_doc_ids = [doc_id]

        if buffer_text:
            for chunk in _emit_chunks(
                buffer_text, buffer_doc_ids, topic_id,
                chunk_counter, min_tokens, max_tokens,
            ):
                all_chunks.append(chunk)
                chunk_counter += 1

    logger.info("Produced %d refined chunks", len(all_chunks))
    return all_chunks


def _emit_chunks(
    text: str,
    doc_ids: list[str],
    topic_id: int,
    start_id: int,
    min_tokens: int,
    max_tokens: int,
) -> list[Chunk]:
    """Yield one or more Chunk objects, splitting oversized text if needed."""
    topic_id = int(topic_id)
    token_count = _count_tokens(text)

    if token_count <= max_tokens:
        return [
            Chunk(
                chunk_id=f"chunk-{start_id:05d}",
                doc_ids=list(doc_ids),
                text=text,
                token_count=token_count,
                topic_id=topic_id,
            )
        ]

    # Split on sentence boundaries for oversized text
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks: list[Chunk] = []
    buf = ""
    buf_tokens = 0
    cid = start_id

    for sent in sentences:
        sent_tokens = _count_tokens(sent)
        if buf and buf_tokens + sent_tokens > max_tokens:
            chunks.append(
                Chunk(
                    chunk_id=f"chunk-{cid:05d}",
                    doc_ids=list(doc_ids),
                    text=buf.strip(),
                    token_count=buf_tokens,
                    topic_id=topic_id,
                )
            )
            cid += 1
            buf = sent
            buf_tokens = sent_tokens
        else:
            buf = (buf + " " + sent).strip() if buf else sent
            buf_tokens = _count_tokens(buf)

    if buf:
        chunks.append(
            Chunk(
                chunk_id=f"chunk-{cid:05d}",
                doc_ids=list(doc_ids),
                text=buf.strip(),
                token_count=_count_tokens(buf.strip()),
                topic_id=topic_id,
            )
        )

    return chunks


# ---------------------------------------------------------------------------
# 7. Orchestrator
# ---------------------------------------------------------------------------

def _json_default(obj):
    """Fallback serializer for numpy scalar types that may leak into JSON output."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

def run_pipeline(
    dataset_name: str,
    min_tokens: int = MIN_TOKENS,
    max_tokens: int = MAX_TOKENS,
    output_path: Optional[str] = None,
    limit: Optional[int] = None,
) -> list[Chunk]:
    """End-to-end semantic chunking pipeline.

    Parameters
    ----------
    dataset_name : str
        BEIR dataset identifier (e.g. ``"scifact"``, ``"fiqa"``, ``"nfcorpus"``).
    min_tokens, max_tokens : int
        Token bounds for chunk refinement.
    output_path : str | None
        If provided, write serialized chunks to this JSON file.
    limit : int | None
        Max documents to load (evidence-filtered). Defaults to DOCUMENT_LIMIT (150).
    """
    doc_limit = limit if limit is not None else DOCUMENT_LIMIT
    documents = load_corpus(dataset_name, limit=doc_limit)
    paragraphs, para_doc_ids = segment_into_paragraphs(documents)
    embeddings = embed_paragraphs(paragraphs)

    topic_model = discover_global_topics(embeddings, paragraphs)
    clusters = assign_paragraphs_to_clusters(
        paragraphs, para_doc_ids, topic_model, embeddings
    )
    chunks = refine_chunks(clusters, min_tokens=min_tokens, max_tokens=max_tokens)

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(
                [asdict(c) for c in chunks],
                fh,
                ensure_ascii=False,
                indent=2,
                default=_json_default,
            )
        logger.info("Saved %d chunks to %s", len(chunks), out)

    return chunks


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Semantic Chunker – load a BEIR corpus and produce topic-based chunks."
    )
    parser.add_argument(
        "dataset",
        type=str,
        help="BEIR dataset name (e.g. scifact, fiqa, nfcorpus)",
    )
    parser.add_argument(
        "--min-tokens",
        type=int,
        default=MIN_TOKENS,
        help=f"Minimum tokens per chunk (default: {MIN_TOKENS})",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=MAX_TOKENS,
        help=f"Maximum tokens per chunk (default: {MAX_TOKENS})",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output JSON file path for serialized chunks",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N documents from the corpus (useful for quick trials)",
    )

    args = parser.parse_args()
    chunks = run_pipeline(
        dataset_name=args.dataset,
        min_tokens=args.min_tokens,
        max_tokens=args.max_tokens,
        output_path=args.output,
        limit=args.limit,
    )
    print(f"Pipeline complete – {len(chunks)} chunks produced.")


if __name__ == "__main__":
    main()
