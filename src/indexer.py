"""
Unified Qdrant Indexer for the RAG ablation study.

Exposes a single function, :func:`index_chunks_to_qdrant`, that turns any
list of chunk dicts into a fresh Qdrant collection. Every chunker in the
ablation matrix (character splitter, sentence window, LLM-boundary,
semantic-GMM) writes its chunks into its own collection through this
function so the evaluator can swap collections by name.

Design choices
--------------
* The dense vector is computed exclusively from ``chunk["text"]`` with
  ``sentence-transformers/all-mpnet-base-v2`` (768-d, cosine distance).
  Metadata fields are stored in the payload only -- they never bleed into
  the vector.
* The list of payload fields is configurable via ``with_payload_fields``.
  Mandatory fields (``chunk_id``, ``doc_ids``, ``text``) are always
  included. Optional fields (``anchor_queries``, ``keywords``, ``topic_id``,
  ``token_count``) are included only when present on the chunk.
* Full-text payload indexes are created for the lexical hybrid path when
  ``anchor_queries`` or ``keywords`` are part of the payload schema.
* Both ``doc_id`` (string) and ``doc_ids`` (list of strings) input shapes
  are accepted so the existing baseline chunks can be indexed without
  reformatting.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import uuid
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    TextIndexParams,
    TextIndexType,
    TokenizerType,
    VectorParams,
)
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from semantic_chunker import EMBEDDING_DIM, EMBEDDING_MODEL_NAME

logger = logging.getLogger(__name__)


DEFAULT_QDRANT_DB_PATH = "output/qdrant_db"
BATCH_UPSERT_SIZE = 64

# Collection names used by the 4x2 ablation matrix. Keeping them centralised
# guarantees consistency between the indexer, the orchestrator, and the
# evaluator.
COLLECTION_BASELINE_CHAR = "baseline_char"
COLLECTION_BASELINE_SENTENCE_WINDOW = "baseline_sentence_window"
COLLECTION_LLM_CHUNKS = "llm_chunks"
COLLECTION_ENRICHED_CHUNKS = "enriched_chunks"

ABLATION_COLLECTIONS: tuple[str, ...] = (
    COLLECTION_BASELINE_CHAR,
    COLLECTION_BASELINE_SENTENCE_WINDOW,
    COLLECTION_LLM_CHUNKS,
    COLLECTION_ENRICHED_CHUNKS,
)

# Default payload schema for every collection. Mandatory fields are always
# emitted; optional fields are dropped per-chunk if the value is missing.
DEFAULT_PAYLOAD_FIELDS: tuple[str, ...] = (
    "chunk_id",
    "doc_ids",
    "text",
    "token_count",
    "topic_id",
    "keywords",
    "anchor_queries",
)

_MANDATORY_FIELDS: frozenset[str] = frozenset({"chunk_id", "doc_ids", "text"})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _full_text_index_params() -> TextIndexParams:
    """Qdrant full-text payload index parameters (word tokenizer, min len 2)."""
    return TextIndexParams(
        type=TextIndexType.TEXT,
        tokenizer=TokenizerType.WORD,
        min_token_len=2,
        max_token_len=20,
        lowercase=True,
    )


def _normalise_doc_ids(chunk: dict[str, Any]) -> list[str]:
    """Return a list-of-strings representation of the chunk's doc ids.

    Accepts both ``doc_id`` (single string, e.g. baseline chunks) and
    ``doc_ids`` (list, e.g. semantic / LLM / sentence-window chunks).
    """
    if "doc_ids" in chunk and chunk["doc_ids"] is not None:
        raw = chunk["doc_ids"]
        if isinstance(raw, list):
            return [str(x) for x in raw]
        return [str(raw)]
    if "doc_id" in chunk and chunk["doc_id"] is not None:
        return [str(chunk["doc_id"])]
    return []


def _build_payload(
    chunk: dict[str, Any],
    fields: Sequence[str],
) -> dict[str, Any]:
    """Project a chunk dict into the payload schema for Qdrant.

    Only ``fields`` that are present on the chunk (or mandatory) appear in
    the resulting payload. ``doc_ids`` is always written as a list of
    strings and a convenience ``doc_id`` scalar is added when there is
    exactly one doc id (helpful for old code paths that filter on a
    single-valued field).
    """
    field_set = set(fields) | _MANDATORY_FIELDS
    payload: dict[str, Any] = {}

    if "chunk_id" not in chunk:
        raise KeyError(f"Chunk missing required 'chunk_id': {chunk}")
    if "text" not in chunk:
        raise KeyError(f"Chunk {chunk.get('chunk_id')!r} missing required 'text'")

    payload["chunk_id"] = str(chunk["chunk_id"])
    payload["text"] = str(chunk["text"])

    doc_ids = _normalise_doc_ids(chunk)
    payload["doc_ids"] = doc_ids
    if len(doc_ids) == 1:
        payload["doc_id"] = doc_ids[0]

    for name in field_set - {"chunk_id", "doc_ids", "text"}:
        if name not in chunk:
            continue
        value = chunk[name]
        if value is None:
            continue
        # Empty lists are kept (downstream code branches on truthiness),
        # but None scalars are skipped to keep the payload terse.
        payload[name] = value

    return payload


def _has_full_text_fields(fields: Sequence[str]) -> list[str]:
    """Return the subset of *fields* that should get a Qdrant text index."""
    candidates = {"anchor_queries", "keywords"}
    return [f for f in fields if f in candidates]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def index_chunks_to_qdrant(
    chunks: list[dict[str, Any]],
    collection_name: str,
    with_payload_fields: Sequence[str] = DEFAULT_PAYLOAD_FIELDS,
    db_path: str = DEFAULT_QDRANT_DB_PATH,
    embedding_model: Optional[SentenceTransformer] = None,
    batch_size: int = BATCH_UPSERT_SIZE,
    recreate: bool = True,
    client: Optional[QdrantClient] = None,
) -> int:
    """Embed and upsert *chunks* into Qdrant collection ``collection_name``.

    Parameters
    ----------
    chunks : list of dict
        Chunk dicts with at least ``chunk_id`` and ``text``. Either
        ``doc_id`` or ``doc_ids`` is accepted; both are normalised to a
        ``doc_ids`` list in the payload.
    collection_name : str
        Target Qdrant collection (created fresh by default).
    with_payload_fields : sequence of str
        Optional payload fields to include when present on a chunk. The
        mandatory fields (``chunk_id``, ``doc_ids``, ``text``) are always
        added on top of this list.
    db_path : str
        Local on-disk Qdrant path.
    embedding_model : SentenceTransformer or None
        Pre-loaded model; loaded on demand otherwise. Sharing one model
        across collections avoids reloading 400MB+ for every call.
    batch_size : int
        Upsert batch size (Qdrant). 64 keeps memory low.
    recreate : bool
        If True (default), drop the existing collection before writing.
    client : QdrantClient or None
        Re-use an existing connection. Local on-disk Qdrant only allows
        one client per process at a time, so the orchestrator passes a
        shared client when indexing multiple collections back-to-back.
    """
    if not chunks:
        logger.warning(
            "index_chunks_to_qdrant: empty chunk list for collection %r; skipping.",
            collection_name,
        )
        return 0

    owns_client = client is None
    if owns_client:
        client = QdrantClient(path=db_path)

    if embedding_model is None:
        embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    try:
        existing = {c.name for c in client.get_collections().collections}
        if recreate and collection_name in existing:
            client.delete_collection(collection_name)
            existing.discard(collection_name)
            logger.info("Dropped existing collection %r", collection_name)

        if collection_name not in existing:
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=EMBEDDING_DIM,
                    distance=Distance.COSINE,
                ),
            )
            logger.info(
                "Created collection %r (dim=%d, cosine)",
                collection_name,
                EMBEDDING_DIM,
            )

        # Full-text indexes for any lexical payload fields requested.
        _ensure_full_text_indexes(
            client, collection_name, _has_full_text_fields(with_payload_fields),
        )

        # Dense vectors from chunk body only.
        texts: list[str] = [str(c["text"]) for c in chunks]
        logger.info(
            "Embedding %d chunks for %r (batch_size=32)",
            len(texts),
            collection_name,
        )
        vectors: np.ndarray = embedding_model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True,
        )

        points: list[PointStruct] = []
        for i, chunk in enumerate(chunks):
            point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, str(chunk["chunk_id"])))
            payload = _build_payload(chunk, with_payload_fields)
            points.append(
                PointStruct(
                    id=point_id,
                    vector=vectors[i].tolist(),
                    payload=payload,
                )
            )

        total = 0
        for start in range(0, len(points), batch_size):
            batch = points[start : start + batch_size]
            client.upsert(collection_name=collection_name, points=batch)
            total += len(batch)
            logger.info(
                "Upserted %d/%d into %r",
                total,
                len(points),
                collection_name,
            )
        logger.info(
            "Indexing complete: %d points in collection %r", total, collection_name,
        )
        return total
    finally:
        if owns_client:
            client.close()


def _ensure_full_text_indexes(
    client: QdrantClient,
    collection_name: str,
    fields: Iterable[str],
) -> None:
    """Create full-text payload indexes for the listed payload fields."""
    params = _full_text_index_params()
    for field_name in fields:
        try:
            client.create_payload_index(
                collection_name=collection_name,
                field_name=field_name,
                field_schema=params,
            )
            logger.info(
                "Ensured full-text payload index on %r.%r",
                collection_name,
                field_name,
            )
        except Exception as exc:
            msg = str(exc).lower()
            if "already" in msg or "duplicate" in msg or "exists" in msg:
                logger.info(
                    "Full-text index on %r.%r already present",
                    collection_name,
                    field_name,
                )
            else:
                logger.warning(
                    "Could not create full-text index on %r.%r: %s",
                    collection_name,
                    field_name,
                    exc,
                )


# ---------------------------------------------------------------------------
# Convenience: load chunk JSON and index in one call
# ---------------------------------------------------------------------------


def load_chunks_json(path: str | Path) -> list[dict[str, Any]]:
    """Load a chunk JSON file emitted by any chunker in the ablation study."""
    p = Path(path)
    with open(p, encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list of chunk dicts at {p}")
    logger.info("Loaded %d chunks from %s", len(data), p)
    return data


def index_chunks_from_file(
    chunks_path: str | Path,
    collection_name: str,
    with_payload_fields: Sequence[str] = DEFAULT_PAYLOAD_FIELDS,
    db_path: str = DEFAULT_QDRANT_DB_PATH,
    embedding_model: Optional[SentenceTransformer] = None,
    batch_size: int = BATCH_UPSERT_SIZE,
    recreate: bool = True,
    client: Optional[QdrantClient] = None,
) -> int:
    """Wrapper that loads chunks from JSON and forwards to ``index_chunks_to_qdrant``."""
    chunks = load_chunks_json(chunks_path)
    return index_chunks_to_qdrant(
        chunks=chunks,
        collection_name=collection_name,
        with_payload_fields=with_payload_fields,
        db_path=db_path,
        embedding_model=embedding_model,
        batch_size=batch_size,
        recreate=recreate,
        client=client,
    )


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Index any chunk JSON file into a Qdrant collection.",
    )
    parser.add_argument(
        "-i", "--input",
        required=True,
        help="Path to the chunk JSON file (output of any chunker)",
    )
    parser.add_argument(
        "-c", "--collection",
        required=True,
        help=(
            "Target Qdrant collection name. Standard options: "
            + ", ".join(ABLATION_COLLECTIONS)
        ),
    )
    parser.add_argument(
        "-d", "--db-path",
        default=DEFAULT_QDRANT_DB_PATH,
        help=f"Qdrant on-disk storage path (default: {DEFAULT_QDRANT_DB_PATH})",
    )
    parser.add_argument(
        "--with-payload",
        nargs="*",
        default=list(DEFAULT_PAYLOAD_FIELDS),
        help="Optional payload fields to include if present on each chunk",
    )
    parser.add_argument(
        "--no-recreate",
        action="store_true",
        help="Do NOT drop the collection before ingesting (default: drop)",
    )

    args = parser.parse_args()
    count = index_chunks_from_file(
        chunks_path=args.input,
        collection_name=args.collection,
        with_payload_fields=args.with_payload,
        db_path=args.db_path,
        recreate=not args.no_recreate,
    )
    print(f"Indexed {count} chunks into collection '{args.collection}'.")


if __name__ == "__main__":
    main()
