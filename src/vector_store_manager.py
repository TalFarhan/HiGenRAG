"""
Vector Store Manager – Task 3 (§3.2)

Uses a local on-disk Qdrant instance (no server required), creates a collection
for the enriched chunks produced by Task 2, embeds chunk texts with
all-mpnet-base-v2 (768-d), and upserts them with full metadata payloads
(keywords, labels, confidence).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
)
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from semantic_chunker import EMBEDDING_DIM, EMBEDDING_MODEL_NAME

logger = logging.getLogger(__name__)

DEFAULT_COLLECTION = "enriched_chunks"
DEFAULT_QDRANT_DB_PATH = "output/qdrant_db"
BATCH_UPSERT_SIZE = 64


class VectorStoreManager:
    """Manages a Qdrant collection for enriched semantic chunks."""

    def __init__(
        self,
        db_path: str = DEFAULT_QDRANT_DB_PATH,
        collection_name: str = DEFAULT_COLLECTION,
        embedding_model: Optional[SentenceTransformer] = None,
    ) -> None:
        self.collection_name = collection_name
        self.client = QdrantClient(path=db_path)
        self.model = embedding_model or SentenceTransformer(EMBEDDING_MODEL_NAME)
        logger.info(
            "Opened local Qdrant store at %s (collection=%s)",
            db_path,
            collection_name,
        )

    # ------------------------------------------------------------------
    # Collection management
    # ------------------------------------------------------------------

    def ensure_collection(self, recreate: bool = False) -> None:
        """Create the collection if it does not already exist.

        When *recreate* is True the existing collection is deleted first,
        guaranteeing a clean state for a fresh ingest.
        """
        existing = {c.name for c in self.client.get_collections().collections}

        if recreate and self.collection_name in existing:
            self.client.delete_collection(self.collection_name)
            logger.info("Deleted existing collection '%s'", self.collection_name)
            existing.discard(self.collection_name)

        if self.collection_name not in existing:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=EMBEDDING_DIM,
                    distance=Distance.COSINE,
                ),
            )
            logger.info(
                "Created collection '%s' (dim=%d, cosine)",
                self.collection_name,
                EMBEDDING_DIM,
            )
        else:
            logger.info("Collection '%s' already exists — reusing", self.collection_name)

    # ------------------------------------------------------------------
    # Loading enriched chunks
    # ------------------------------------------------------------------

    @staticmethod
    def load_enriched_chunks(path: Path) -> list[dict]:
        """Load enriched-chunk dicts from the JSON produced by Task 2."""
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        logger.info("Loaded %d enriched chunks from %s", len(data), path)
        return data

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------

    def embed_texts(
        self,
        texts: list[str],
        batch_size: int = 32,
    ) -> np.ndarray:
        """Produce 768-d vectors for a list of texts."""
        logger.info("Embedding %d texts (batch_size=%d)", len(texts), batch_size)
        return self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
        )

    # ------------------------------------------------------------------
    # Payload construction
    # ------------------------------------------------------------------

    @staticmethod
    def _build_payload(chunk: dict) -> dict:
        """Construct a Qdrant payload from an enriched-chunk dict.

        Keeps all metadata produced by Task 2 so downstream stages
        (synthetic query generation, hybrid reranking) can filter on it.
        """
        return {
            "chunk_id": chunk["chunk_id"],
            "doc_ids": chunk.get("doc_ids", []),
            "text": chunk["text"],
            "token_count": chunk.get("token_count", 0),
            "topic_id": chunk.get("topic_id", -1),
            "keywords": chunk.get("keywords", []),
            "topic_label": chunk.get("topic_label", ""),
            "confidence_score": chunk.get("confidence_score", 0.0),
            "subtopics": chunk.get("subtopics", []),
        }

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def ingest_chunks(
        self,
        chunks: list[dict],
        batch_size: int = BATCH_UPSERT_SIZE,
    ) -> int:
        """Embed and upsert enriched chunks into the Qdrant collection.

        Returns the number of points successfully upserted.
        """
        texts = [c["text"] for c in chunks]
        vectors = self.embed_texts(texts)

        points: list[PointStruct] = []
        for i, chunk in enumerate(chunks):
            point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, chunk["chunk_id"]))
            points.append(
                PointStruct(
                    id=point_id,
                    vector=vectors[i].tolist(),
                    payload=self._build_payload(chunk),
                )
            )

        total_upserted = 0
        for start in range(0, len(points), batch_size):
            batch = points[start : start + batch_size]
            self.client.upsert(
                collection_name=self.collection_name,
                points=batch,
            )
            total_upserted += len(batch)
            logger.info(
                "Upserted batch %d–%d (%d/%d)",
                start,
                start + len(batch) - 1,
                total_upserted,
                len(points),
            )

        logger.info(
            "Ingestion complete — %d points in '%s'",
            total_upserted,
            self.collection_name,
        )
        return total_upserted

    # ------------------------------------------------------------------
    # Convenience: full pipeline
    # ------------------------------------------------------------------

    def run(
        self,
        input_path: str = "output/enriched_chunks.json",
        recreate: bool = False,
    ) -> int:
        """End-to-end: load → embed → upsert."""
        chunks = self.load_enriched_chunks(Path(input_path))
        self.ensure_collection(recreate=recreate)
        return self.ingest_chunks(chunks)


# -----------------------------------------------------------------------
# CLI entry-point
# -----------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Vector Store Manager – embed enriched chunks and index them in Qdrant.",
    )
    parser.add_argument(
        "-i", "--input",
        default="output/enriched_chunks.json",
        help="Path to the enriched chunks JSON (default: output/enriched_chunks.json)",
    )
    parser.add_argument(
        "-c", "--collection",
        default=DEFAULT_COLLECTION,
        help=f"Qdrant collection name (default: {DEFAULT_COLLECTION})",
    )
    parser.add_argument(
        "-d", "--db-path",
        default=DEFAULT_QDRANT_DB_PATH,
        help=f"Path for local Qdrant on-disk storage (default: {DEFAULT_QDRANT_DB_PATH})",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Drop and recreate the collection before ingesting",
    )

    args = parser.parse_args()

    manager = VectorStoreManager(
        db_path=args.db_path,
        collection_name=args.collection,
    )
    count = manager.run(input_path=args.input, recreate=args.recreate)
    print(f"Pipeline complete – {count} vectors indexed in collection '{args.collection}'.")


if __name__ == "__main__":
    main()
