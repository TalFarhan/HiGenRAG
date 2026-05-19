"""
LLM-Boundary Chunker (Method B) – semantic boundaries via an LLM judge.

This chunker walks document sentences in order and asks an LLM, after each
new sentence, whether that sentence opens a new topic compared to a small
lookback window of recent sentences. Whenever the LLM answers "1" the
current chunk is closed and a new one is started. A hard MAX_TOKENS upper
bound is always enforced regardless of the LLM verdict, so chunks never
exceed the embedding-model context.

Design choices
--------------
* The LLM is invoked through LangChain ``ChatGroq`` with
  ``llama-3.3-70b-versatile`` (matches generative_threading.py).
* Lookback window: the last 4 sentences of the current buffer (or fewer if
  the buffer is shorter).
* Boundary prompt: strict ``0`` / ``1`` answer; any non-``1`` response is
  treated as "no boundary".
* Persistent disk cache (``output/.llm_chunker_cache.json``) keyed on a
  SHA-256 of ``model || prompt`` so repeated runs do not re-spend tokens.
  An in-memory LRU layer fronts the disk cache during a single run.
* Robust fallback: if the LLM is unreachable (no API key, network error,
  etc.), the chunker falls back to a deterministic heuristic so the
  pipeline can still produce comparable chunks for the ablation study.

Output format mirrors the existing ``Chunk`` dataclass used by the rest of
the pipeline so downstream enrichment / indexing can consume it uniformly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import sys
import threading
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv

load_dotenv()

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

# LLM configuration – kept aligned with generative_threading.py so the
# ablation study uses one Groq model across the project.
LLM_MODEL_NAME = "llama-3.3-70b-versatile"
LLM_TEMPERATURE = 0.0  # Boundary decisions must be deterministic.
LLM_MAX_TOKENS = 4

# Sentence-level configuration.
LOOKBACK_SENTENCES = 4
MIN_LOOKBACK_FOR_LLM = 2  # Below this we keep accumulating without calling the LLM.

# Persistent cache for LLM verdicts (relative to repo root).
DEFAULT_CACHE_PATH = Path("output/.llm_chunker_cache.json")

# Heuristic fallback when the LLM is unavailable: trigger a boundary whenever
# the buffer reaches this many tokens, mimicking a coarse topic boundary.
_HEURISTIC_BOUNDARY_TOKENS = int(MIN_TOKENS * 1.25)

# Fallback splitter for sentences that, on their own, already exceed
# MAX_TOKENS. Same heuristic as in sentence_window_chunker.py.
_FALLBACK_CHAR_PER_TOKEN = 4
_FALLBACK_OVERLAP_CHARS = 64


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------


class _LLMResponseCache:
    """Thread-safe disk-backed cache for LLM boundary verdicts.

    Keys are SHA-256 hashes of ``f"{model}||{prompt}"``; values are the raw
    string returned by the LLM (typically ``"0"`` or ``"1"``). Writes are
    flushed lazily but always before the chunker exits via ``save()``.
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = threading.Lock()
        self._cache: dict[str, str] = {}
        self._dirty = False
        self._load()

    def _load(self) -> None:
        if not self._path.is_file():
            return
        try:
            with open(self._path, encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, dict):
                self._cache = {str(k): str(v) for k, v in data.items()}
                logger.info(
                    "Loaded %d LLM cache entries from %s",
                    len(self._cache),
                    self._path,
                )
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Failed to load LLM cache at %s: %s", self._path, exc)
            self._cache = {}

    @staticmethod
    def make_key(model: str, prompt: str) -> str:
        """Return a stable cache key for a given (model, prompt) pair."""
        h = hashlib.sha256()
        h.update(model.encode("utf-8"))
        h.update(b"||")
        h.update(prompt.encode("utf-8"))
        return h.hexdigest()

    def get(self, key: str) -> Optional[str]:
        with self._lock:
            return self._cache.get(key)

    def set(self, key: str, value: str) -> None:
        with self._lock:
            self._cache[key] = value
            self._dirty = True

    def save(self) -> None:
        with self._lock:
            if not self._dirty:
                return
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = self._path.with_suffix(self._path.suffix + ".tmp")
            with open(tmp_path, "w", encoding="utf-8") as fh:
                json.dump(self._cache, fh, ensure_ascii=False, indent=2)
            tmp_path.replace(self._path)
            self._dirty = False
            logger.info("Saved %d LLM cache entries to %s", len(self._cache), self._path)

    def __len__(self) -> int:  # pragma: no cover - convenience only
        with self._lock:
            return len(self._cache)


# ---------------------------------------------------------------------------
# LLM judge
# ---------------------------------------------------------------------------

_PROMPT_TEMPLATE = (
    "You are a topic-boundary detector. You will see a short context "
    "(the last sentences of the current passage) and one new sentence.\n\n"
    "Task: decide whether the new sentence opens a NEW topic compared to "
    "the previous sentences.\n"
    "Answer strictly with a single character: '1' if it opens a new topic, "
    "or '0' if it continues the same topic. No words, no punctuation.\n\n"
    "CONTEXT (previous sentences):\n{context}\n\n"
    "NEW SENTENCE (sentence k+1):\n{candidate}\n\n"
    "Answer (0 or 1):"
)

_NUMERIC_RE = re.compile(r"[01]")


# Circuit breaker: after this many consecutive LLM call failures within a
# single run we stop hitting the API and rely on the heuristic for the rest
# of the corpus. The threshold is intentionally small because repeated 4xx
# responses (e.g. project-level model block) will never self-heal.
_LLM_FAILURE_CIRCUIT_BREAKER = 3


class _LLMBoundaryJudge:
    """Wrap a ChatGroq chain and apply the disk cache around every invocation."""

    def __init__(
        self,
        cache: _LLMResponseCache,
        model_name: str = LLM_MODEL_NAME,
        temperature: float = LLM_TEMPERATURE,
        max_tokens: int = LLM_MAX_TOKENS,
        failure_threshold: int = _LLM_FAILURE_CIRCUIT_BREAKER,
    ) -> None:
        self._cache = cache
        self._model_name = model_name
        self._chain: Any = None
        self._chain_init_failed = False
        self._calls = 0
        self._cache_hits = 0
        self._errors = 0
        self._consecutive_errors = 0
        self._circuit_open = False
        self._failure_threshold = max(1, int(failure_threshold))
        self._temperature = temperature
        self._max_tokens = max_tokens

    def _ensure_chain(self) -> Optional[Any]:
        """Lazy-init the LangChain Groq chain; cache failures so we stop retrying."""
        if self._circuit_open:
            return None
        if self._chain is not None or self._chain_init_failed:
            return self._chain
        if not os.environ.get("GROQ_API_KEY"):
            logger.warning(
                "GROQ_API_KEY not set; LLM chunker will use heuristic fallback for "
                "all cache-miss decisions.",
            )
            self._chain_init_failed = True
            return None
        try:
            from langchain_core.prompts import ChatPromptTemplate
            from langchain_groq import ChatGroq

            llm = ChatGroq(
                model=self._model_name,
                temperature=self._temperature,
                max_tokens=self._max_tokens,
            )
            prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        "You are a precise topic-boundary detector. "
                        "Answer with a single digit: 0 or 1. Nothing else.",
                    ),
                    ("human", "{prompt}"),
                ]
            )
            self._chain = prompt | llm
            logger.info("Groq LLM ready for boundary detection: %s", self._model_name)
        except Exception as exc:
            logger.warning(
                "LLM chain init failed: %s. Using heuristic fallback.", exc,
            )
            self._chain_init_failed = True
            self._chain = None
        return self._chain

    @staticmethod
    def _heuristic_decision(
        context_sents: list[str],
        candidate: str,
        buffer_token_count: int = 0,
    ) -> int:
        """Token-budget heuristic used when the LLM is unavailable.

        Triggers a boundary once the running BUFFER comfortably exceeds
        MIN_TOKENS so that fallback chunks still respect the target size band.
        ``buffer_token_count`` is the token count of the full current buffer
        (not just the lookback window); when it is unavailable the function
        falls back to the lookback-window-only estimate.
        """
        if buffer_token_count > 0:
            projected = buffer_token_count + _count_tokens(candidate)
        else:
            projected = _count_tokens("\n\n".join(context_sents + [candidate]).strip())
        if projected >= _HEURISTIC_BOUNDARY_TOKENS:
            return 1
        return 0

    @staticmethod
    def _parse_answer(text: str) -> Optional[int]:
        """Extract the first 0/1 digit from a raw LLM reply; return None on miss."""
        if text is None:
            return None
        m = _NUMERIC_RE.search(text)
        if not m:
            return None
        return int(m.group(0))

    def decide(
        self,
        context_sents: list[str],
        candidate: str,
        buffer_token_count: int = 0,
    ) -> int:
        """Return 1 if ``candidate`` opens a new topic, else 0.

        ``buffer_token_count`` is only used by the heuristic fallback and
        does NOT affect the cache key (so the LLM-derived answer is reused
        across runs even when buffer sizes shift slightly).
        """
        self._calls += 1
        context_block = "\n".join(f"- {s}" for s in context_sents)
        prompt = _PROMPT_TEMPLATE.format(context=context_block, candidate=candidate)

        cache_key = self._cache.make_key(self._model_name, prompt)
        cached = self._cache.get(cache_key)
        if cached is not None:
            self._cache_hits += 1
            parsed = self._parse_answer(cached)
            if parsed is not None:
                return parsed
            # Corrupted cache entry; fall through and re-query.

        chain = self._ensure_chain()
        if chain is None:
            decision = self._heuristic_decision(
                context_sents, candidate, buffer_token_count,
            )
            self._cache.set(cache_key, str(decision))
            return decision

        try:
            result = chain.invoke({"prompt": prompt})
            text = (
                result.content.strip() if hasattr(result, "content") else str(result).strip()
            )
        except Exception as exc:
            self._errors += 1
            self._consecutive_errors += 1
            if self._consecutive_errors >= self._failure_threshold:
                self._circuit_open = True
                logger.warning(
                    "LLM circuit breaker tripped after %d consecutive failures "
                    "(%s); remaining decisions will use the heuristic.",
                    self._consecutive_errors,
                    exc,
                )
            else:
                logger.warning("LLM boundary call failed (%s); using heuristic.", exc)
            decision = self._heuristic_decision(
                context_sents, candidate, buffer_token_count,
            )
            self._cache.set(cache_key, str(decision))
            return decision

        self._consecutive_errors = 0
        parsed = self._parse_answer(text)
        if parsed is None:
            logger.debug("Unparseable LLM reply %r; defaulting to 0.", text)
            parsed = 0
        self._cache.set(cache_key, str(parsed))
        return parsed

    @property
    def stats(self) -> dict[str, int]:
        return {
            "calls": self._calls,
            "cache_hits": self._cache_hits,
            "errors": self._errors,
            "circuit_open": int(self._circuit_open),
        }


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
        # Last-resort greedy word grouping.
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
            chunk_id=f"llm-{chunk_counter:05d}",
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
    judge: _LLMBoundaryJudge,
    min_tokens: int = MIN_TOKENS,
    max_tokens: int = MAX_TOKENS,
    lookback: int = LOOKBACK_SENTENCES,
    min_lookback_for_llm: int = MIN_LOOKBACK_FOR_LLM,
) -> list[Chunk]:
    """Chunk every document using LLM-detected sentence boundaries.

    Algorithm
    ---------
    For each document we maintain a sentence buffer. Sentences are appended
    one by one. After each append, two conditions are checked:

    1. **Hard cap**: if adding the next sentence would push the buffer above
       ``max_tokens``, we flush regardless of the LLM verdict.
    2. **LLM verdict**: once the buffer holds at least ``min_lookback_for_llm``
       sentences, we ask the LLM whether the new sentence opens a new topic.
       The lookback window is the last ``lookback`` sentences of the buffer
       (excluding the candidate itself). If the LLM says ``1`` we flush the
       buffer *before* adding the candidate so the new sentence starts the
       next chunk.
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

        buffer: list[str] = []
        buffer_token_count = 0
        for sent in sentences:
            sent_tokens = _count_tokens(sent)
            # Always honour the hard token cap before consulting the LLM.
            if buffer and buffer_token_count + sent_tokens > max_tokens:
                chunk_counter = _flush_buffer(doc_id, buffer, chunks, chunk_counter)
                buffer = []
                buffer_token_count = 0

            if len(buffer) >= min_lookback_for_llm:
                lookback_window = buffer[-lookback:]
                verdict = judge.decide(
                    lookback_window, sent, buffer_token_count=buffer_token_count,
                )
                if verdict == 1:
                    chunk_counter = _flush_buffer(doc_id, buffer, chunks, chunk_counter)
                    buffer = []
                    buffer_token_count = 0

            buffer.append(sent)
            # Recompute the buffer token count using the joined text to keep
            # the value consistent with how the chunk text is materialised
            # (avoids drift from per-sentence sums that ignore separators).
            buffer_token_count = _count_tokens("\n\n".join(buffer).strip())

        chunk_counter = _flush_buffer(doc_id, buffer, chunks, chunk_counter)

    under_min = sum(1 for c in chunks if c.token_count < min_tokens)
    logger.info(
        "LLM chunking produced %d chunks (lookback=%d, %d below MIN_TOKENS=%d). "
        "LLM stats: %s",
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
    cache_path: str | Path = DEFAULT_CACHE_PATH,
    model_name: str = LLM_MODEL_NAME,
) -> list[Chunk]:
    """Load corpus, run LLM-boundary chunking, persist JSON, and flush cache."""
    documents = load_corpus(dataset_name, limit=limit)

    cache = _LLMResponseCache(Path(cache_path))
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
        logger.info("Saved %d LLM chunks to %s", len(chunks), out)

    return chunks


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="LLM boundary chunker (Method B) for the RAG ablation study.",
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
        help=f"Sentences kept as lookback context for the LLM "
             f"(default: {LOOKBACK_SENTENCES})",
    )
    parser.add_argument(
        "--cache-path",
        default=str(DEFAULT_CACHE_PATH),
        help=f"Disk cache path for LLM verdicts (default: {DEFAULT_CACHE_PATH})",
    )
    parser.add_argument(
        "--model",
        default=LLM_MODEL_NAME,
        help=f"Groq model id (default: {LLM_MODEL_NAME})",
    )

    args = parser.parse_args()
    chunks = run_pipeline(
        dataset_name=args.dataset,
        output_path=args.output,
        limit=args.limit,
        min_tokens=args.min_tokens,
        max_tokens=args.max_tokens,
        lookback=args.lookback,
        cache_path=args.cache_path,
        model_name=args.model,
    )
    print(f"LLM-boundary pipeline complete - {len(chunks)} chunks produced.")


if __name__ == "__main__":
    main()
