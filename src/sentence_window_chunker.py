"""
Sentence-Window Chunker (Method A) – fixed-size sentence windows with overlap.

This is one of the four chunking strategies used in the RAG ablation study.
It produces chunks by sliding a fixed-size window over the ordered sentences
of every document, keeping a strict 2-3 sentence overlap between consecutive
windows so that no sentence is split mid-way. When a single sentence exceeds
MAX_TOKENS, a fallback character splitter is used so that the hard token
ceiling is always respected.

Public entry points
-------------------
* ``sentence_window_chunk_documents(documents, ...)``: pure function that
  turns a list of corpus documents into ``Chunk`` objects.
* ``run_pipeline(...)``: loads the SciFact evidence-filtered corpus, chunks
  it, and writes the result to ``output/sentence_window_chunks.json``.

Output format matches the existing semantic-chunker (``Chunk`` dataclass)
so the downstream enrichment / indexing layers can consume it uniformly.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter

sys.path.insert(0, str(Path(__file__).resolve().parent))
from semantic_chunker import (
    MAX_TOKENS,
    MIN_TOKENS,
    Chunk,
    _count_tokens,
    _json_default,
    load_corpus,
    split_document_into_sentences,
)

logger = logging.getLogger(__name__)

# Window geometry: target ~6 sentences per window with a strict 2-3 overlap.
# These numbers were tuned to land close to MIN_TOKENS=256 on average for
# SciFact while keeping every window strictly below MAX_TOKENS=512.
WINDOW_SIZE_SENTENCES = 6
WINDOW_OVERLAP_SENTENCES = 2

# Fallback splitter for sentences that, on their own, already blow past
# MAX_TOKENS. Characters are converted to tokens roughly at 4 chars/token,
# so 4 * MAX_TOKENS gives a safe character budget per piece.
_FALLBACK_CHAR_PER_TOKEN = 4
_FALLBACK_OVERLAP_CHARS = 64


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
    """Split a single sentence that exceeds max_tokens into smaller pieces.

    Uses the langchain RecursiveCharacterTextSplitter as the fallback. Any
    resulting piece that is still over the token ceiling is truncated by
    words as a last resort so the hard cap is never violated.
    """
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


def _normalize_sentences_for_windowing(
    sentences: list[str],
    max_tokens: int,
    fallback_splitter: RecursiveCharacterTextSplitter,
) -> list[str]:
    """Return a sentence list where every element is below max_tokens.

    Long sentences are replaced by their fallback-split pieces so the window
    builder can treat each unit atomically and never cut mid-sentence.
    """
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


def _window_text(window: list[str]) -> str:
    """Join window sentences using the same separator as the semantic chunker."""
    return "\n\n".join(window).strip()


def _trim_window_to_max_tokens(window: list[str], max_tokens: int) -> list[str]:
    """Drop trailing sentences until the joined window fits within max_tokens.

    This is only triggered if the configured window size (in sentences) would
    push the token count above the hard cap. Because every sentence is already
    guaranteed to be below max_tokens, at least one sentence always remains.
    """
    if not window:
        return window
    trimmed = list(window)
    while len(trimmed) > 1 and _count_tokens(_window_text(trimmed)) > max_tokens:
        trimmed.pop()
    return trimmed


def _build_windows_for_doc(
    sentences: list[str],
    window_size: int,
    overlap: int,
    max_tokens: int,
) -> list[list[str]]:
    """Slide a fixed-size sentence window with the configured overlap.

    The step between consecutive window starts is ``window_size - overlap`` so
    that ``overlap`` sentences are shared between adjacent windows. The last
    window of a document may be shorter if there are not enough sentences.
    """
    if not sentences:
        return []
    if window_size <= 0:
        raise ValueError("window_size must be >= 1")
    if overlap < 0 or overlap >= window_size:
        raise ValueError("overlap must satisfy 0 <= overlap < window_size")

    step = window_size - overlap
    windows: list[list[str]] = []
    n = len(sentences)
    start = 0
    last_emitted_end = -1

    while start < n:
        end = min(start + window_size, n)
        raw_window = sentences[start:end]
        window = _trim_window_to_max_tokens(raw_window, max_tokens)
        if not window:
            start += step
            continue
        emitted_end = start + len(window)
        # Avoid emitting a window that is fully contained in the previous one
        # after trimming caused it to shrink.
        if emitted_end > last_emitted_end:
            windows.append(window)
            last_emitted_end = emitted_end
        if end >= n:
            break
        start += step

    return windows


def sentence_window_chunk_documents(
    documents: list[dict[str, Any]],
    window_size: int = WINDOW_SIZE_SENTENCES,
    overlap: int = WINDOW_OVERLAP_SENTENCES,
    min_tokens: int = MIN_TOKENS,
    max_tokens: int = MAX_TOKENS,
) -> list[Chunk]:
    """Build sentence-window chunks for every document.

    Parameters
    ----------
    documents:
        Corpus documents as returned by ``semantic_chunker.load_corpus``
        (each item has ``doc_id``, ``title``, ``text``).
    window_size:
        Number of sentences per window (default 6 to aim near MIN_TOKENS).
    overlap:
        Number of sentences shared between adjacent windows (2 or 3).
    min_tokens / max_tokens:
        Soft/hard token bounds. ``min_tokens`` is informational only; chunks
        smaller than ``min_tokens`` are still produced (e.g. document tail).
        ``max_tokens`` is the hard upper bound and is always enforced via
        trimming or fallback splitting.
    """
    if not (2 <= overlap <= 3):
        logger.warning(
            "Overlap=%d outside recommended [2, 3]; behaviour preserved.", overlap,
        )

    fallback_splitter = _build_fallback_splitter(max_tokens)
    chunks: list[Chunk] = []
    chunk_counter = 0
    under_min = 0

    for doc in documents:
        doc_id = str(doc["doc_id"])
        raw_sentences = split_document_into_sentences(doc)
        sentences = _normalize_sentences_for_windowing(
            raw_sentences, max_tokens, fallback_splitter,
        )
        if not sentences:
            continue

        windows = _build_windows_for_doc(sentences, window_size, overlap, max_tokens)
        for window in windows:
            text = _window_text(window)
            if not text:
                continue
            token_count = _count_tokens(text)
            if token_count < min_tokens:
                under_min += 1
            chunks.append(
                Chunk(
                    chunk_id=f"sw-{chunk_counter:05d}",
                    doc_ids=[doc_id],
                    text=text,
                    token_count=token_count,
                    topic_id=-1,
                    keywords=[],
                )
            )
            chunk_counter += 1

    logger.info(
        "Sentence-window chunking produced %d chunks (window=%d, overlap=%d, "
        "%d below MIN_TOKENS=%d)",
        len(chunks),
        window_size,
        overlap,
        under_min,
        min_tokens,
    )
    return chunks


def run_pipeline(
    dataset_name: str = "scifact",
    output_path: str = "output/sentence_window_chunks.json",
    limit: Optional[int] = None,
    window_size: int = WINDOW_SIZE_SENTENCES,
    overlap: int = WINDOW_OVERLAP_SENTENCES,
    min_tokens: int = MIN_TOKENS,
    max_tokens: int = MAX_TOKENS,
) -> list[Chunk]:
    """Load corpus, build sentence-window chunks, and persist to JSON."""
    documents = load_corpus(dataset_name, limit=limit)
    chunks = sentence_window_chunk_documents(
        documents,
        window_size=window_size,
        overlap=overlap,
        min_tokens=min_tokens,
        max_tokens=max_tokens,
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
        logger.info("Saved %d sentence-window chunks to %s", len(chunks), out)

    return chunks


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Sentence-window chunker (Method A) for the RAG ablation study.",
    )
    parser.add_argument(
        "dataset",
        nargs="?",
        default="scifact",
        help="BEIR dataset name (default: scifact)",
    )
    parser.add_argument(
        "-o", "--output",
        default="output/sentence_window_chunks.json",
        help="Output JSON path (default: output/sentence_window_chunks.json)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap evidence documents for quick trials; omit for full corpus",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=WINDOW_SIZE_SENTENCES,
        help=f"Sentences per window (default: {WINDOW_SIZE_SENTENCES})",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=WINDOW_OVERLAP_SENTENCES,
        help=f"Sentence overlap between adjacent windows, 2 or 3 "
             f"(default: {WINDOW_OVERLAP_SENTENCES})",
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

    args = parser.parse_args()
    chunks = run_pipeline(
        dataset_name=args.dataset,
        output_path=args.output,
        limit=args.limit,
        window_size=args.window_size,
        overlap=args.overlap,
        min_tokens=args.min_tokens,
        max_tokens=args.max_tokens,
    )
    print(f"Sentence-window pipeline complete - {len(chunks)} chunks produced.")


if __name__ == "__main__":
    main()
