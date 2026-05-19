"""
Semantic Boundary Chunker (Method B) – local embedding-based topic boundaries.

This module previously called a Groq LLM after every sentence to ask whether
a topic boundary had occurred. That approach burned API tokens linearly with
corpus size and made the chunking stage the dominant cost of the pipeline.

The new implementation is fully local and deterministic:

* Sentences are encoded once per document with the shared
  ``sentence-transformers`` model (``all-mpnet-base-v2``, 768-d).
* For every candidate sentence ``s_k`` we compute the cosine similarity
  between its embedding and a small lookback context vector built from
  the last ``LOOKBACK_SENTENCES`` items of the current buffer. The
  context vector is the L2-normalised mean of the lookback embeddings.
* If that similarity falls **below** ``BOUNDARY_SIMILARITY_THRESHOLD`` we
  treat ``s_k`` as opening a new topic and flush the running buffer
  before appending it. The threshold acts as a topic-shift detector.
* The hard ``MAX_TOKENS`` cap is always honoured: if appending the next
  sentence would push the buffer above the limit we flush regardless of
  the similarity verdict.

Design constraints carried over from the LLM version
----------------------------------------------------
* Public entry points keep the names ``llm_chunk_documents`` and the
  helper classes that ``run_comparison.py`` already imports, so the
  surrounding orchestrator does not need to be re-wired.
* The Groq API is **no longer touched at all** by this stage – it is
  reserved exclusively for anchor-query generation in the enrichment
  layer. ``GROQ_API_KEY`` is therefore not required to chunk a corpus.
* The Chunk dataclass schema mirrors the rest of the pipeline so the
  enrichment, indexing, and evaluation stages keep working unchanged.

All inline comments and log messages in this module are in English to
keep the codebase consistent across the ablation suite.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from semantic_chunker import (
    EMBEDDING_DIM,
    EMBEDDING_MODEL_NAME,
    MAX_TOKENS,
    MIN_TOKENS,
    Chunk,
    _count_tokens,
    _json_default,
    load_corpus,
    split_document_into_sentences,
)

logger = logging.getLogger(__name__)

# Embedding model used for the cosine-similarity boundary check.
BOUNDARY_EMBEDDING_MODEL_NAME = EMBEDDING_MODEL_NAME

# A sentence opens a NEW topic when its cosine similarity to the lookback
# context vector drops below this value. The default was tuned to land
# close to the original LLM-judge decisions on the SciFact corpus while
# producing chunks that respect MIN_TOKENS / MAX_TOKENS.
BOUNDARY_SIMILARITY_THRESHOLD = 0.55

# Number of trailing sentences from the current buffer used to build the
# context vector against which a candidate sentence is compared.
LOOKBACK_SENTENCES = 4

# Below this many sentences in the buffer we keep accumulating without
# attempting a boundary decision (a single-sentence buffer is too sparse
# to form a stable context vector).
MIN_LOOKBACK_FOR_DECISION = 2

# Sentence batch size for the embedding step. Documents are usually small
# so a single batch is sufficient, but the parameter is exposed for very
# long documents and large corpora.
EMBEDDING_BATCH_SIZE = 64

# Public re-exports kept for backward compatibility with code paths that
# imported the old LLM-driven constants (e.g. ``run_comparison.py``).
DEFAULT_CACHE_PATH = Path("output/.embedding_chunker_cache.json")
LLM_MODEL_NAME = BOUNDARY_EMBEDDING_MODEL_NAME

# Fallback splitter for sentences that, on their own, already exceed
# MAX_TOKENS. Mirrors the constants used by sentence_window_chunker.py.
_FALLBACK_CHAR_PER_TOKEN = 4
_FALLBACK_OVERLAP_CHARS = 64


# ---------------------------------------------------------------------------
# Boundary detector (embedding-based topic-shift signal)
# ---------------------------------------------------------------------------


class _EmbeddingBoundaryDetector:
    """Compute embedding-based topic-shift signals between sentences.

    The detector wraps a sentence-transformer instance and applies cosine
    similarity between every candidate sentence and a context vector built
    from the trailing window of the current chunk. A boundary is declared
    when the similarity falls strictly below ``threshold``.

    A small in-memory cache stores the L2-normalised embeddings of the
    sentences seen during the current run so repeated lookups for the
    same sentence text are O(1). The cache is not persisted to disk:
    embeddings are cheap to recompute and the cache only protects against
    pathological duplicates inside a single corpus.
    """

    def __init__(
        self,
        model: Optional[SentenceTransformer] = None,
        model_name: str = BOUNDARY_EMBEDDING_MODEL_NAME,
        threshold: float = BOUNDARY_SIMILARITY_THRESHOLD,
        batch_size: int = EMBEDDING_BATCH_SIZE,
    ) -> None:
        self._model_name = model_name
        self._threshold = float(threshold)
        self._batch_size = int(batch_size)
        self._model: Optional[SentenceTransformer] = model
        # Cache mapping a raw sentence string to its L2-normalised vector.
        self._embedding_cache: dict[str, np.ndarray] = {}
        # Run-level stats so the orchestrator can log them after a run.
        self._comparisons = 0
        self._boundaries = 0

    # -- model lifecycle ----------------------------------------------------

    def _ensure_model(self) -> SentenceTransformer:
        """Load the embedding model on demand to keep imports cheap."""
        if self._model is None:
            logger.info(
                "Loading sentence-transformer for boundary detection: %s",
                self._model_name,
            )
            self._model = SentenceTransformer(self._model_name)
        return self._model

    # -- embedding helpers --------------------------------------------------

    def precompute_embeddings(self, sentences: list[str]) -> None:
        """Batch-encode and cache every sentence in *sentences*.

        Encoding the whole document up-front lets us reuse the same vector
        when the sentence is consulted both as a candidate AND later as
        part of a context window, which avoids quadratic re-encoding.
        """
        unseen = [s for s in sentences if s not in self._embedding_cache]
        if not unseen:
            return
        model = self._ensure_model()
        raw = model.encode(
            unseen,
            batch_size=self._batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        if raw.ndim == 1:
            # Defensive: encode() may return a 1-D array for a single input.
            raw = raw.reshape(1, -1)
        norms = np.linalg.norm(raw, axis=1, keepdims=True)
        norms = np.where(norms == 0.0, 1.0, norms)
        normalised = raw / norms
        for sent, vec in zip(unseen, normalised):
            self._embedding_cache[sent] = vec.astype(np.float64, copy=False)

    def _vector_for(self, sentence: str) -> np.ndarray:
        """Return the cached unit vector for *sentence* (encoding on miss)."""
        cached = self._embedding_cache.get(sentence)
        if cached is not None:
            return cached
        # Cache miss: encode this single sentence and store the result.
        self.precompute_embeddings([sentence])
        return self._embedding_cache[sentence]

    @staticmethod
    def _context_vector(vectors: list[np.ndarray]) -> Optional[np.ndarray]:
        """L2-normalised mean of a non-empty list of unit vectors."""
        if not vectors:
            return None
        stacked = np.stack(vectors, axis=0)
        mean_vec = stacked.mean(axis=0)
        norm = float(np.linalg.norm(mean_vec))
        if norm == 0.0:
            return None
        return mean_vec / norm

    # -- boundary decision --------------------------------------------------

    def is_boundary(
        self,
        context_sents: list[str],
        candidate: str,
    ) -> tuple[bool, float]:
        """Return ``(is_boundary, similarity)`` for *candidate* vs *context_sents*.

        ``is_boundary`` is True when the cosine similarity between the
        candidate and the lookback context vector is strictly below the
        configured threshold. The similarity score is returned alongside
        so the caller can log or trace decisions when needed.
        """
        self._comparisons += 1
        if not context_sents:
            # No context yet: cannot signal a boundary, keep accumulating.
            return False, 1.0
        context_vectors = [self._vector_for(s) for s in context_sents]
        ctx = self._context_vector(context_vectors)
        if ctx is None:
            return False, 1.0
        cand = self._vector_for(candidate)
        # Both vectors are L2-normalised so the dot product is the cosine.
        similarity = float(np.dot(ctx, cand))
        is_boundary = similarity < self._threshold
        if is_boundary:
            self._boundaries += 1
        return is_boundary, similarity

    # -- compatibility shim with the legacy ``judge.decide`` API -----------

    def decide(
        self,
        context_sents: list[str],
        candidate: str,
        buffer_token_count: int = 0,
    ) -> int:
        """Legacy adapter returning ``1`` for boundary, ``0`` otherwise.

        ``buffer_token_count`` is accepted for API compatibility with the
        previous LLM-based judge but is intentionally ignored: the new
        boundary signal is purely semantic.
        """
        del buffer_token_count  # parameter kept for backwards compatibility
        is_boundary, _ = self.is_boundary(context_sents, candidate)
        return 1 if is_boundary else 0

    # -- run-level stats ----------------------------------------------------

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "model": self._model_name,
            "threshold": self._threshold,
            "comparisons": self._comparisons,
            "boundaries": self._boundaries,
        }


# ---------------------------------------------------------------------------
# Legacy compatibility shims
# ---------------------------------------------------------------------------


class _LLMResponseCache:  # pragma: no cover - kept only for import safety
    """No-op cache shim kept for backwards compatibility with the previous
    LLM-based chunker. The embedding chunker does not need a persistent
    response cache (encoding is cheap and deterministic), so every method
    is a harmless stub. Callers that still construct/save this object
    continue to work without changes.
    """

    def __init__(self, path: Path | str = DEFAULT_CACHE_PATH) -> None:
        self._path = Path(path)

    @staticmethod
    def make_key(model: str, prompt: str) -> str:
        # Returning a deterministic placeholder keeps any caller happy.
        return f"{model}::{prompt}"

    def get(self, key: str) -> Optional[str]:  # noqa: D401 - stub
        return None

    def set(self, key: str, value: str) -> None:
        return None

    def save(self) -> None:
        # No state to persist for the embedding chunker.
        return None

    def __len__(self) -> int:
        return 0


def _LLMBoundaryJudge(  # noqa: N802 - backwards-compatible factory name
    cache: Optional[_LLMResponseCache] = None,
    model_name: str = BOUNDARY_EMBEDDING_MODEL_NAME,
    **_kwargs: Any,
) -> _EmbeddingBoundaryDetector:
    """Backwards-compatible factory that builds the embedding-based judge.

    The previous public API constructed an ``_LLMBoundaryJudge`` with a
    cache + model name. We keep the call site working by returning the
    new ``_EmbeddingBoundaryDetector`` and discarding cache/temperature/
    max_tokens kwargs that no longer have any meaning.
    """
    del cache  # not needed: embedding cache lives inside the detector
    return _EmbeddingBoundaryDetector(model_name=model_name)


# ---------------------------------------------------------------------------
# Sentence normalisation (shared helpers)
# ---------------------------------------------------------------------------


def _build_fallback_splitter(max_tokens: int) -> RecursiveCharacterTextSplitter:
    """Construct the safety-net splitter for sentences longer than max_tokens."""
    return RecursiveCharacterTextSplitter(
        chunk_size=max_tokens * _FALLBACK_CHAR_PER_TOKEN,
        chunk_overlap=_FALLBACK_OVERLAP_CHARS,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


def _split_oversized_sentence(
    sentence: str,
    max_tokens: int,
    fallback_splitter: RecursiveCharacterTextSplitter,
) -> list[str]:
    """Split a single oversize sentence into pieces below max_tokens."""
    pieces = fallback_splitter.split_text(sentence)
    safe_pieces: list[str] = []
    for piece in pieces:
        p = piece.strip()
        if not p:
            continue
        if _count_tokens(p) <= max_tokens:
            safe_pieces.append(p)
            continue
        # Last-resort greedy word grouping to enforce the hard ceiling.
        words = p.split()
        buf: list[str] = []
        for w in words:
            trial = " ".join(buf + [w])
            if buf and _count_tokens(trial) > max_tokens:
                safe_pieces.append(" ".join(buf))
                buf = [w]
            else:
                buf.append(w)
        if buf:
            safe_pieces.append(" ".join(buf))
    return safe_pieces


def _normalize_sentences(
    sentences: list[str],
    max_tokens: int,
    fallback_splitter: RecursiveCharacterTextSplitter,
) -> list[str]:
    """Return a list of sentences where every element fits below max_tokens."""
    normalized: list[str] = []
    for sent in sentences:
        s = sent.strip()
        if not s:
            continue
        if _count_tokens(s) > max_tokens:
            normalized.extend(_split_oversized_sentence(s, max_tokens, fallback_splitter))
        else:
            normalized.append(s)
    return normalized


# ---------------------------------------------------------------------------
# Core chunking algorithm
# ---------------------------------------------------------------------------


def _flush_buffer(
    doc_id: str,
    buffer: list[str],
    chunks: list[Chunk],
    chunk_counter: int,
) -> int:
    """Materialise the current sentence buffer as a Chunk and return next counter."""
    if not buffer:
        return chunk_counter
    text = "\n\n".join(buffer).strip()
    if not text:
        return chunk_counter
    chunks.append(
        Chunk(
            chunk_id=f"sem-{chunk_counter:05d}",
            doc_ids=[doc_id],
            text=text,
            token_count=_count_tokens(text),
            topic_id=-1,
            keywords=[],
        )
    )
    return chunk_counter + 1


def llm_chunk_documents(
    documents: list[dict[str, Any]],
    judge: _EmbeddingBoundaryDetector,
    min_tokens: int = MIN_TOKENS,
    max_tokens: int = MAX_TOKENS,
    lookback: int = LOOKBACK_SENTENCES,
    min_lookback_for_llm: int = MIN_LOOKBACK_FOR_DECISION,
) -> list[Chunk]:
    """Chunk every document using embedding-based topic-shift detection.

    The function name and signature are preserved to keep the orchestrator
    (``run_comparison.py``) working without any code change. ``judge`` is
    now an ``_EmbeddingBoundaryDetector`` rather than an LLM wrapper, but
    it exposes the same ``decide`` contract used by the previous code.

    Algorithm
    ---------
    For every document we:

    1. Split into sentences and replace any oversize sentence with safe
       sub-sentences so the hard ``max_tokens`` ceiling is enforceable.
    2. Pre-compute the embeddings of every (sub-)sentence in one batch.
    3. Walk the sentences in order while maintaining a running buffer.
       Before appending a candidate sentence we check two conditions:
         a. Hard cap – if joining the candidate would exceed
            ``max_tokens`` we flush the buffer and start a new chunk.
         b. Topic shift – once the buffer has at least
            ``min_lookback_for_llm`` sentences we compare the candidate to
            the last ``lookback`` sentences. If the cosine similarity is
            below the detector threshold we flush before appending so
            that the candidate opens the next chunk.
    """
    fallback_splitter = _build_fallback_splitter(max_tokens)
    chunks: list[Chunk] = []
    chunk_counter = 0

    for doc in documents:
        doc_id = str(doc["doc_id"])
        raw_sentences = split_document_into_sentences(doc)
        sentences = _normalize_sentences(raw_sentences, max_tokens, fallback_splitter)
        if not sentences:
            continue

        # Encode every sentence once so the boundary check is O(1) per call.
        judge.precompute_embeddings(sentences)

        buffer: list[str] = []
        buffer_token_count = 0
        for sent in sentences:
            sent_tokens = _count_tokens(sent)
            # Always honour the hard token cap before any semantic check.
            if buffer and buffer_token_count + sent_tokens > max_tokens:
                chunk_counter = _flush_buffer(doc_id, buffer, chunks, chunk_counter)
                buffer = []
                buffer_token_count = 0

            if len(buffer) >= min_lookback_for_llm:
                lookback_window = buffer[-lookback:]
                is_boundary, _ = judge.is_boundary(lookback_window, sent)
                if is_boundary:
                    chunk_counter = _flush_buffer(doc_id, buffer, chunks, chunk_counter)
                    buffer = []
                    buffer_token_count = 0

            buffer.append(sent)
            # Recompute the buffer token count from the joined text so the
            # value stays consistent with how the chunk text is finally
            # materialised (sentences joined by a blank line).
            buffer_token_count = _count_tokens("\n\n".join(buffer).strip())

        chunk_counter = _flush_buffer(doc_id, buffer, chunks, chunk_counter)

    under_min = sum(1 for c in chunks if c.token_count < min_tokens)
    logger.info(
        "Embedding-boundary chunking produced %d chunks (lookback=%d, "
        "%d below MIN_TOKENS=%d). Detector stats: %s",
        len(chunks),
        lookback,
        under_min,
        min_tokens,
        judge.stats,
    )
    return chunks


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------


def run_pipeline(
    dataset_name: str = "scifact",
    output_path: str = "output/llm_chunks.json",
    limit: Optional[int] = None,
    min_tokens: int = MIN_TOKENS,
    max_tokens: int = MAX_TOKENS,
    lookback: int = LOOKBACK_SENTENCES,
    threshold: float = BOUNDARY_SIMILARITY_THRESHOLD,
    model_name: str = BOUNDARY_EMBEDDING_MODEL_NAME,
    embedding_model: Optional[SentenceTransformer] = None,
    cache_path: str | Path = DEFAULT_CACHE_PATH,
) -> list[Chunk]:
    """Load corpus, run embedding-boundary chunking, and persist JSON.

    ``cache_path`` is accepted for backwards compatibility with the
    previous CLI but is unused (no LLM cache is required any more).
    """
    del cache_path  # legacy parameter kept for CLI compatibility
    documents = load_corpus(dataset_name, limit=limit)

    detector = _EmbeddingBoundaryDetector(
        model=embedding_model,
        model_name=model_name,
        threshold=threshold,
    )
    chunks = llm_chunk_documents(
        documents,
        judge=detector,
        min_tokens=min_tokens,
        max_tokens=max_tokens,
        lookback=lookback,
    )

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
        logger.info("Saved %d embedding-boundary chunks to %s", len(chunks), out)

    return chunks


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description=(
            "Embedding-based semantic boundary chunker (Method B) for the "
            "RAG ablation study. Replaces the previous Groq-LLM judge."
        ),
    )
    parser.add_argument(
        "dataset",
        nargs="?",
        default="scifact",
        help="BEIR dataset name (default: scifact)",
    )
    parser.add_argument(
        "-o", "--output",
        default="output/llm_chunks.json",
        help="Output JSON path (default: output/llm_chunks.json)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap evidence documents for quick trials; omit for full corpus",
    )
    parser.add_argument(
        "--min-tokens",
        type=int,
        default=MIN_TOKENS,
        help=f"Soft target minimum tokens per chunk (default: {MIN_TOKENS})",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=MAX_TOKENS,
        help=f"Hard maximum tokens per chunk (default: {MAX_TOKENS})",
    )
    parser.add_argument(
        "--lookback",
        type=int,
        default=LOOKBACK_SENTENCES,
        help=f"Sentences kept as lookback context (default: {LOOKBACK_SENTENCES})",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=BOUNDARY_SIMILARITY_THRESHOLD,
        help=(
            "Cosine-similarity threshold below which a topic shift is "
            f"declared (default: {BOUNDARY_SIMILARITY_THRESHOLD})"
        ),
    )
    parser.add_argument(
        "--model",
        default=BOUNDARY_EMBEDDING_MODEL_NAME,
        help=(
            "Sentence-transformer model used for boundary detection "
            f"(default: {BOUNDARY_EMBEDDING_MODEL_NAME})"
        ),
    )
    parser.add_argument(
        "--cache-path",
        default=str(DEFAULT_CACHE_PATH),
        help="(Deprecated) kept for CLI compatibility; ignored.",
    )

    args = parser.parse_args()
    chunks = run_pipeline(
        dataset_name=args.dataset,
        output_path=args.output,
        limit=args.limit,
        min_tokens=args.min_tokens,
        max_tokens=args.max_tokens,
        lookback=args.lookback,
        threshold=args.threshold,
        model_name=args.model,
        cache_path=args.cache_path,
    )
    print(
        f"Embedding-boundary pipeline complete - {len(chunks)} chunks produced."
    )


if __name__ == "__main__":
    main()


__all__ = [
    "BOUNDARY_EMBEDDING_MODEL_NAME",
    "BOUNDARY_SIMILARITY_THRESHOLD",
    "DEFAULT_CACHE_PATH",
    "EMBEDDING_BATCH_SIZE",
    "EMBEDDING_DIM",
    "LLM_MODEL_NAME",
    "LOOKBACK_SENTENCES",
    "MIN_LOOKBACK_FOR_DECISION",
    "_EmbeddingBoundaryDetector",
    "_LLMBoundaryJudge",
    "_LLMResponseCache",
    "llm_chunk_documents",
    "run_pipeline",
]
