"""
Granular Topic Modeler – Task 2 (§3.1)

Runs a secondary BERTopic pass on the chunks produced by Task 1,
extracts descriptive labels and keywords via c-TF-IDF, and enriches
each segment with topic metadata (labels, keywords, confidence scores).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np
from bertopic import BERTopic
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from semantic_chunker import Chunk, EMBEDDING_MODEL_NAME

logger = logging.getLogger(__name__)

_MIN_DOCS_FOR_BERTOPIC = 10
DEFAULT_TOP_N_KEYWORDS = 10


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class SubTopic:
    """A latent sub-topic discovered within or across chunks."""

    topic_id: int
    label: str
    keywords: list[str] = field(default_factory=list)
    confidence: float = 0.0


@dataclass
class EnrichedChunk:
    """A chunk enriched with granular topic metadata."""

    chunk_id: str
    doc_ids: list[str]
    text: str
    token_count: int
    topic_id: int
    keywords: list[str]
    subtopics: list[SubTopic] = field(default_factory=list)
    topic_label: str = ""
    confidence_score: float = 0.0


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_chunks(path: Path) -> list[Chunk]:
    """Deserialize chunks produced by Task 1 from a JSON file."""
    with open(path, encoding="utf-8") as fh:
        raw = json.load(fh)
    return [
        Chunk(
            chunk_id=item["chunk_id"],
            doc_ids=item["doc_ids"],
            text=item["text"],
            token_count=item["token_count"],
            topic_id=item.get("topic_id", -1),
            keywords=item.get("keywords", []),
        )
        for item in raw
    ]


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def embed_chunks(
    chunks: list[Chunk],
    model: Optional[SentenceTransformer] = None,
    batch_size: int = 32,
) -> np.ndarray:
    """Produce 768-d embeddings for every chunk text."""
    if model is None:
        model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    texts = [c.text for c in chunks]
    logger.info("Embedding %d chunks (batch_size=%d)", len(texts), batch_size)
    return model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
    )


# ---------------------------------------------------------------------------
# Secondary BERTopic pass
# ---------------------------------------------------------------------------

def fit_topic_model(
    chunks: list[Chunk],
    embeddings: np.ndarray,
) -> tuple[BERTopic, list[int], np.ndarray]:
    """Fit a granular BERTopic model on the chunk collection.

    Returns the fitted model, per-chunk topic assignments, and a
    probability matrix (docs × topics).
    """
    texts = [c.text for c in chunks]

    vectorizer_model = CountVectorizer(stop_words="english")

    if len(texts) < _MIN_DOCS_FOR_BERTOPIC:
        logger.warning(
            "Only %d chunks — too few for BERTopic; single-topic fallback.",
            len(texts),
        )
        model = BERTopic(
            embedding_model=EMBEDDING_MODEL_NAME,
            vectorizer_model=vectorizer_model,
            nr_topics=1,
        )
        return model, [0] * len(texts), np.ones((len(texts), 1))

    hdbscan_model = HDBSCAN(
        min_cluster_size=3,
        min_samples=2,
        prediction_data=True,
    )

    topic_model = BERTopic(
        embedding_model=EMBEDDING_MODEL_NAME,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        calculate_probabilities=True,
        verbose=True,
    )
    topics, probs = topic_model.fit_transform(texts, embeddings)

    if probs is None:
        probs = np.zeros((len(texts), 1))

    topic_info = topic_model.get_topic_info()
    logger.info(
        "Granular pass discovered %d topics (incl. outlier -1)",
        len(topic_info),
    )
    return topic_model, [int(t) for t in topics], np.atleast_2d(probs)


# ---------------------------------------------------------------------------
# Keyword extraction
# ---------------------------------------------------------------------------

def extract_keywords(
    topic_model: BERTopic,
    topic_id: int,
    top_n: int = DEFAULT_TOP_N_KEYWORDS,
) -> list[str]:
    """Extract top keywords for *topic_id* from BERTopic's c-TF-IDF."""
    if topic_id == -1:
        return []
    try:
        terms = topic_model.get_topic(topic_id)
        if not terms:
            return []
        return [word for word, _ in terms[:top_n]]
    except Exception:
        return []


def extract_chunk_keywords(
    chunks: list[Chunk],
    top_n: int = DEFAULT_TOP_N_KEYWORDS,
) -> list[list[str]]:
    """Per-chunk keyword extraction via TF-IDF over the whole collection.

    Provides chunk-specific keywords (as opposed to topic-level ones).
    """
    texts = [c.text for c in chunks]
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=5000,
        ngram_range=(1, 2),
    )
    tfidf = vectorizer.fit_transform(texts)
    features = vectorizer.get_feature_names_out()

    per_chunk: list[list[str]] = []
    for i in range(tfidf.shape[0]):
        row = tfidf[i].toarray().flatten()
        top_idx = row.argsort()[-top_n:][::-1]
        per_chunk.append([features[j] for j in top_idx if row[j] > 0])

    return per_chunk


# ---------------------------------------------------------------------------
# Sub-topic discovery per chunk
# ---------------------------------------------------------------------------

def _make_topic_label(topic_model: BERTopic, topic_id: int) -> str:
    """Derive a human-readable label from the top c-TF-IDF terms."""
    if topic_id == -1:
        return "outlier"
    try:
        terms = topic_model.get_topic(topic_id)
        if not terms:
            return f"topic_{topic_id}"
        return " | ".join(word for word, _ in terms[:4])
    except Exception:
        return f"topic_{topic_id}"


def _build_topic_id_to_col(topic_model: BERTopic) -> dict[int, int]:
    """Map BERTopic topic IDs to probability-matrix column indices."""
    topic_info = topic_model.get_topic_info()
    valid = sorted(
        int(row["Topic"])
        for _, row in topic_info.iterrows()
        if row["Topic"] != -1
    )
    return {tid: col for col, tid in enumerate(valid)}


def model_subtopics(
    chunk_text: str,
    topic_model: BERTopic,
    tid_to_col: dict[int, int],
    threshold: float = 0.01,
    top_n: int = DEFAULT_TOP_N_KEYWORDS,
) -> list[SubTopic]:
    """Discover sub-topics within a single chunk via approximate_distribution.

    Falls back to an empty list when the model is a fallback stub or
    approximate_distribution is unavailable.
    """
    try:
        distributions, _ = topic_model.approximate_distribution(
            [chunk_text], min_similarity=0.1,
        )
        dist = distributions[0]
    except Exception:
        return []

    col_to_tid = {col: tid for tid, col in tid_to_col.items()}

    subtopics: list[SubTopic] = []
    for col_idx in range(len(dist)):
        if dist[col_idx] <= threshold:
            continue
        tid = col_to_tid.get(col_idx)
        if tid is None:
            continue
        subtopics.append(
            SubTopic(
                topic_id=tid,
                label=_make_topic_label(topic_model, tid),
                keywords=extract_keywords(topic_model, tid, top_n),
                confidence=round(float(dist[col_idx]), 4),
            )
        )

    subtopics.sort(key=lambda s: s.confidence, reverse=True)
    return subtopics


# ---------------------------------------------------------------------------
# Enrichment
# ---------------------------------------------------------------------------

def enrich_chunk_metadata(
    chunk: Chunk,
    subtopics: list[SubTopic],
    chunk_keywords: list[str],
    primary_topic_id: int,
    topic_label: str,
    confidence: float,
) -> EnrichedChunk:
    """Attach topic labels, keywords, and confidence scores to a chunk."""
    return EnrichedChunk(
        chunk_id=chunk.chunk_id,
        doc_ids=chunk.doc_ids,
        text=chunk.text,
        token_count=chunk.token_count,
        topic_id=primary_topic_id,
        keywords=chunk_keywords,
        subtopics=subtopics,
        topic_label=topic_label,
        confidence_score=round(confidence, 4),
    )


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------

def _json_default(obj):
    """Handle numpy scalars that may leak into JSON output."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


# ---------------------------------------------------------------------------
# Pipeline orchestrator
# ---------------------------------------------------------------------------

def run_pipeline(
    input_path: str = "output/chunks.json",
    output_path: str = "output/enriched_chunks.json",
    top_n_keywords: int = DEFAULT_TOP_N_KEYWORDS,
) -> list[EnrichedChunk]:
    """End-to-end granular topic modeling and metadata enrichment.

    1. Load chunks from *input_path*.
    2. Embed all chunks with all-mpnet-base-v2.
    3. Fit a secondary BERTopic model on the chunk collection.
    4. Extract per-chunk TF-IDF keywords.
    5. Discover sub-topics within each chunk (approximate_distribution).
    6. Enrich every chunk with labels, keywords, and confidence scores.
    7. Persist to *output_path*.
    """
    inp = Path(input_path)
    chunks = load_chunks(inp)
    logger.info("Loaded %d chunks from %s", len(chunks), inp)

    embeddings = embed_chunks(chunks)

    topic_model, topics, probs = fit_topic_model(chunks, embeddings)

    chunk_kw_lists = extract_chunk_keywords(chunks, top_n_keywords)

    tid_to_col = _build_topic_id_to_col(topic_model)

    enriched: list[EnrichedChunk] = []
    for i, chunk in enumerate(chunks):
        assigned_topic = topics[i]

        if probs.ndim == 2 and probs.shape[1] > 0:
            col = tid_to_col.get(assigned_topic)
            if col is not None and col < probs.shape[1]:
                confidence = float(probs[i, col])
            else:
                confidence = float(np.max(probs[i])) if assigned_topic != -1 else 0.0
        else:
            confidence = 0.0

        subtopics = model_subtopics(
            chunk.text, topic_model, tid_to_col,
            top_n=top_n_keywords,
        )

        label = _make_topic_label(topic_model, assigned_topic)

        enriched.append(
            enrich_chunk_metadata(
                chunk=chunk,
                subtopics=subtopics,
                chunk_keywords=chunk_kw_lists[i],
                primary_topic_id=assigned_topic,
                topic_label=label,
                confidence=confidence,
            )
        )

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(
            [asdict(ec) for ec in enriched],
            fh,
            ensure_ascii=False,
            indent=2,
            default=_json_default,
        )
    logger.info("Saved %d enriched chunks to %s", len(enriched), out)
    return enriched


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Granular Topic Modeler – enrich chunks with sub-topic metadata.",
    )
    parser.add_argument(
        "-i", "--input",
        default="output/chunks.json",
        help="Input chunks JSON file (default: output/chunks.json)",
    )
    parser.add_argument(
        "-o", "--output",
        default="output/enriched_chunks.json",
        help="Output enriched chunks JSON (default: output/enriched_chunks.json)",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=DEFAULT_TOP_N_KEYWORDS,
        help=f"Number of keywords per topic/chunk (default: {DEFAULT_TOP_N_KEYWORDS})",
    )

    args = parser.parse_args()
    enriched = run_pipeline(
        input_path=args.input,
        output_path=args.output,
        top_n_keywords=args.top_n,
    )
    print(f"Pipeline complete – {len(enriched)} enriched chunks produced.")


if __name__ == "__main__":
    main()
