"""
Baseline Pipeline – Task 5 (Research Comparison Baseline)

Builds a naive-chunking control group by loading 150 SciFact
documents (filtered by evidence in the golden set), splitting them with a fixed-size
RecursiveCharacterTextSplitter (500 chars / 50 overlap), embedding them
with all-mpnet-base-v2, and indexing the vectors in a dedicated
``baseline_chunks`` Qdrant collection.
"""

from __future__ import annotations

import logging
import sys
import uuid
from pathlib import Path

import ir_datasets
import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from semantic_chunker import EMBEDDING_DIM, EMBEDDING_MODEL_NAME

logger = logging.getLogger(__name__)

DATASET_NAME = "scifact"
DOCUMENT_LIMIT = 150
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
BASELINE_COLLECTION = "baseline_chunks"
QDRANT_DB_PATH = "output/qdrant_db"
BATCH_UPSERT_SIZE = 64


# ---------------------------------------------------------------------------
# 1. Corpus loading (mirrors semantic_chunker.load_corpus)
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


def load_corpus(dataset_name: str = DATASET_NAME, limit: int = DOCUMENT_LIMIT) -> list[dict]:
    """Load corpus filtered to documents that have evidence in the golden set.

    Steps:
      1. Load qrels from the test split and collect doc_ids with relevance > 0.
      2. Stream the corpus split and keep only documents whose ``_id`` appears
         in the evidence set.
      3. Stop once *limit* documents have been collected.
    """
    logger.info("Loading corpus: BeIR/%s (evidence-filtered, limit=%d)", dataset_name, limit)

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

    logger.info("Loaded %d documents from BeIR/%s (filtered by evidence)", len(documents), dataset_name)
    return documents


# ---------------------------------------------------------------------------
# 2. Naive chunking via RecursiveCharacterTextSplitter
# ---------------------------------------------------------------------------

def naive_chunk_documents(
    documents: list[dict],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[dict]:
    """Split every document with a fixed-size character splitter.

    Returns a flat list of dicts with ``chunk_id``, ``doc_id``, and ``text``.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    chunks: list[dict] = []
    for doc in documents:
        full_text = doc["text"]
        if doc["title"]:
            full_text = doc["title"] + "\n\n" + full_text

        pieces = splitter.split_text(full_text)
        for piece in pieces:
            chunks.append(
                {
                    "chunk_id": f"bl-{len(chunks):05d}",
                    "doc_id": doc["doc_id"],
                    "text": piece,
                }
            )

    logger.info(
        "Naive chunking produced %d chunks (size=%d, overlap=%d)",
        len(chunks), chunk_size, chunk_overlap,
    )
    return chunks


# ---------------------------------------------------------------------------
# 3. Embedding
# ---------------------------------------------------------------------------

def embed_chunks(
    chunks: list[dict],
    model: SentenceTransformer | None = None,
    batch_size: int = 32,
) -> np.ndarray:
    """Produce 768-d vectors for each chunk text using all-mpnet-base-v2."""
    if model is None:
        model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    texts = [c["text"] for c in chunks]
    logger.info("Embedding %d baseline chunks (batch_size=%d)", len(texts), batch_size)
    return model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
    )


# ---------------------------------------------------------------------------
# 4. Qdrant indexing
# ---------------------------------------------------------------------------

def index_in_qdrant(
    chunks: list[dict],
    vectors: np.ndarray,
    db_path: str = QDRANT_DB_PATH,
    collection_name: str = BASELINE_COLLECTION,
    batch_size: int = BATCH_UPSERT_SIZE,
) -> int:
    """Create the baseline collection and upsert all chunk vectors.

    Returns the total number of points upserted.
    """
    client = QdrantClient(path=db_path)

    existing = {c.name for c in client.get_collections().collections}
    if collection_name in existing:
        client.delete_collection(collection_name)
        logger.info("Deleted existing collection '%s'", collection_name)

    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
    )
    logger.info(
        "Created collection '%s' (dim=%d, cosine)", collection_name, EMBEDDING_DIM,
    )

    points: list[PointStruct] = []
    for i, chunk in enumerate(chunks):
        point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, chunk["chunk_id"]))
        points.append(
            PointStruct(
                id=point_id,
                vector=vectors[i].tolist(),
                payload={
                    "chunk_id": chunk["chunk_id"],
                    "doc_id": chunk["doc_id"],
                    "text": chunk["text"],
                },
            )
        )

    total_upserted = 0
    for start in range(0, len(points), batch_size):
        batch = points[start : start + batch_size]
        client.upsert(collection_name=collection_name, points=batch)
        total_upserted += len(batch)
        logger.info(
            "Upserted batch %d–%d (%d/%d)",
            start, start + len(batch) - 1, total_upserted, len(points),
        )

    logger.info(
        "Indexing complete — %d points in '%s'", total_upserted, collection_name,
    )
    client.close()
    return total_upserted


# ---------------------------------------------------------------------------
# 5. End-to-end pipeline
# ---------------------------------------------------------------------------

def run_pipeline() -> None:
    """Execute the full baseline pipeline: load → chunk → embed → index."""
    documents = load_corpus()
    chunks = naive_chunk_documents(documents)

    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    vectors = embed_chunks(chunks, model=model)

    count = index_in_qdrant(chunks, vectors)
    print(f"Baseline pipeline complete – {count} vectors indexed in '{BASELINE_COLLECTION}'.")


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    run_pipeline()
