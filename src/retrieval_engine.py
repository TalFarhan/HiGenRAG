"""
Retrieval Engine – Retrieval Stage

Connects to a local on-disk Qdrant instance, embeds user queries with
all-mpnet-base-v2 (768-d), and performs cosine-similarity vector search
to return the top-K most relevant enriched chunks.

Includes entity-aware boosting logic that prioritises specific entity
matches (keywords / subtopics) over broad parent-document hits for
queries that seek a particular entity ("Who is …", "What is …", etc.).
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from qdrant_client import QdrantClient
from qdrant_client.models import ScoredPoint
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from semantic_chunker import EMBEDDING_DIM, EMBEDDING_MODEL_NAME

logger = logging.getLogger(__name__)

DEFAULT_COLLECTION = "enriched_chunks"
DEFAULT_QDRANT_DB_PATH = "output/qdrant_db"
DEFAULT_TOP_K = 3

# ---------------------------------------------------------------------------
# Boost weights for entity-match prioritisation
# ---------------------------------------------------------------------------
# When the query targets a specific entity (e.g. "Who is BC1 RNA?"), chunks
# whose keywords or subtopic labels overlap with the extracted entity receive
# an additive score boost.  This prevents broad, high-level parent documents
# from eclipsing the narrow chunk that actually answers the question.

BOOST_CONTENT_MATCH_WHO = 0.15
BOOST_KEYWORD_OVERLAP = 0.10

_ENTITY_QUERY_PATTERNS = re.compile(
    r"^(?:who\s+is|what\s+is|what\s+are|who\s+are|define|describe|explain)\s+",
    re.IGNORECASE,
)

_OVER_FETCH_FACTOR = 3


@dataclass
class RetrievalResult:
    """A single retrieval hit with its similarity score and metadata."""

    rank: int
    score: float
    chunk_id: str
    text: str
    doc_ids: list[str] = field(default_factory=list)
    token_count: int = 0
    topic_id: int = -1
    topic_label: str = ""
    keywords: list[str] = field(default_factory=list)
    confidence_score: float = 0.0
    subtopics: list[dict] = field(default_factory=list)


class RetrievalEngine:
    """Vector-search retrieval over enriched chunks stored in Qdrant."""

    def __init__(
        self,
        db_path: str = DEFAULT_QDRANT_DB_PATH,
        collection_name: str = DEFAULT_COLLECTION,
        embedding_model: Optional[SentenceTransformer] = None,
    ) -> None:
        self.collection_name = collection_name
        self.client = QdrantClient(path=db_path)
        self.model = embedding_model or SentenceTransformer(EMBEDDING_MODEL_NAME)

        self._validate_collection()
        logger.info(
            "RetrievalEngine ready (db=%s, collection=%s)",
            db_path,
            collection_name,
        )

    def _validate_collection(self) -> None:
        """Verify the target collection exists and report its point count."""
        existing = {c.name for c in self.client.get_collections().collections}
        if self.collection_name not in existing:
            raise ValueError(
                f"Collection '{self.collection_name}' not found in Qdrant. "
                f"Available: {sorted(existing) or '(none)'}. "
                "Run the ingestion pipeline (vector_store_manager.py) first."
            )
        info = self.client.get_collection(self.collection_name)
        logger.info(
            "Collection '%s' contains %d points (dim=%s)",
            self.collection_name,
            info.points_count,
            info.config.params.vectors.size,
        )

    # ------------------------------------------------------------------
    # Entity-aware boosting helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_entity_terms(query: str) -> list[str]:
        """Return lowercased entity tokens if *query* is an entity-seeking question.

        Strips the leading interrogative prefix (e.g. "Who is", "What are")
        and splits the remainder into individual tokens that are at least 2
        characters long.  Returns an empty list when the query does not match
        any known entity-seeking pattern.
        """
        match = _ENTITY_QUERY_PATTERNS.match(query)
        if not match:
            return []
        remainder = query[match.end():].rstrip("?., ").lower()
        return [tok for tok in remainder.split() if len(tok) >= 2]

    @staticmethod
    def _compute_boost(
        entity_terms: list[str],
        keywords: list[str],
        subtopics: list[dict],
    ) -> float:
        """Compute an additive score boost based on entity-keyword overlap.

        Two independent signals are combined:
          1. BOOST_CONTENT_MATCH_WHO – applied when *any* entity term appears
             in the chunk's top-level keyword list.
          2. BOOST_KEYWORD_OVERLAP  – applied when *any* entity term appears
             in the keywords of at least one subtopic.

        Both boosts are binary (applied once, not per-match) so the maximum
        possible boost is BOOST_CONTENT_MATCH_WHO + BOOST_KEYWORD_OVERLAP.
        """
        if not entity_terms:
            return 0.0

        entity_set = set(entity_terms)
        boost = 0.0

        kw_lower = {kw.lower() for kw in keywords}
        if entity_set & kw_lower:
            boost += BOOST_CONTENT_MATCH_WHO

        for st in subtopics:
            st_kw = {k.lower() for k in st.get("keywords", [])}
            if entity_set & st_kw:
                boost += BOOST_KEYWORD_OVERLAP
                break

        return boost

    # ------------------------------------------------------------------
    # Core search
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        score_threshold: Optional[float] = None,
    ) -> list[RetrievalResult]:
        """Embed *query* and retrieve the top-K nearest chunks.

        When the query matches an entity-seeking pattern the engine
        over-fetches candidates and applies additive boosting so that
        chunks whose keywords overlap with the target entity are
        promoted above broader parent-document hits.

        Parameters
        ----------
        query : str
            Free-text search query.
        top_k : int
            Number of results to return (default 3).
        score_threshold : float | None
            If set, discard hits below this cosine-similarity score.

        Returns
        -------
        list[RetrievalResult]
            Ranked results with full payload metadata.
        """
        query_vector = self.model.encode(query, convert_to_numpy=True).tolist()

        entity_terms = self._extract_entity_terms(query)
        fetch_limit = top_k * _OVER_FETCH_FACTOR if entity_terms else top_k

        hits: list[ScoredPoint] = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=fetch_limit,
            score_threshold=score_threshold,
        ).points

        logger.info(
            "Query returned %d hits (fetch_limit=%d, threshold=%s, entity_terms=%s)",
            len(hits),
            fetch_limit,
            score_threshold,
            entity_terms or "(none)",
        )

        scored_results: list[tuple[float, RetrievalResult]] = []
        for hit in hits:
            payload = hit.payload or {}
            base_score = float(hit.score)

            keywords = payload.get("keywords", [])
            subtopics = payload.get("subtopics", [])
            boost = self._compute_boost(entity_terms, keywords, subtopics)
            effective_score = base_score + boost

            scored_results.append((
                effective_score,
                RetrievalResult(
                    rank=0,
                    score=round(effective_score, 4),
                    chunk_id=payload.get("chunk_id", ""),
                    text=payload.get("text", ""),
                    doc_ids=payload.get("doc_ids", []),
                    token_count=payload.get("token_count", 0),
                    topic_id=payload.get("topic_id", -1),
                    topic_label=payload.get("topic_label", ""),
                    keywords=keywords,
                    confidence_score=payload.get("confidence_score", 0.0),
                    subtopics=subtopics,
                ),
            ))

        scored_results.sort(key=lambda pair: pair[0], reverse=True)

        results: list[RetrievalResult] = []
        for rank, (_, result) in enumerate(scored_results[:top_k], start=1):
            result.rank = rank
            results.append(result)

        return results


# -----------------------------------------------------------------------
# Pretty-print helpers
# -----------------------------------------------------------------------

_SEPARATOR = "\u2500" * 72


def format_results(results: list[RetrievalResult], query: str) -> str:
    """Render retrieval results as a human-readable string."""
    if not results:
        return f'No results found for query: "{query}"'

    lines: list[str] = [
        "",
        _SEPARATOR,
        f'  Query: "{query}"',
        f"  Results: {len(results)}",
        _SEPARATOR,
    ]

    for r in results:
        keywords_str = ", ".join(r.keywords[:8]) if r.keywords else "(none)"
        snippet = textwrap.shorten(r.text, width=300, placeholder=" ...")

        lines.append(f"\n  #{r.rank}  Score: {r.score:.4f}   Chunk: {r.chunk_id}")
        lines.append(f"  Topic: {r.topic_label}  (confidence {r.confidence_score:.2f})")
        lines.append(f"  Keywords: {keywords_str}")
        lines.append(f"  Tokens: {r.token_count}   Docs: {r.doc_ids}")
        lines.append(f"  Text:\n{textwrap.indent(snippet, '    ')}")
        lines.append(f"  {_SEPARATOR}")

    return "\n".join(lines)


# -----------------------------------------------------------------------
# CLI entry-point
# -----------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Retrieval Engine \u2013 semantic search over enriched chunks in Qdrant.",
    )
    parser.add_argument(
        "query",
        type=str,
        help="Free-text search query",
    )
    parser.add_argument(
        "-k", "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"Number of results to return (default: {DEFAULT_TOP_K})",
    )
    parser.add_argument(
        "-d", "--db-path",
        default=DEFAULT_QDRANT_DB_PATH,
        help=f"Path to local Qdrant on-disk storage (default: {DEFAULT_QDRANT_DB_PATH})",
    )
    parser.add_argument(
        "-c", "--collection",
        default=DEFAULT_COLLECTION,
        help=f"Qdrant collection name (default: {DEFAULT_COLLECTION})",
    )
    parser.add_argument(
        "-t", "--threshold",
        type=float,
        default=None,
        help="Minimum cosine-similarity score threshold (default: none)",
    )

    args = parser.parse_args()

    engine = RetrievalEngine(
        db_path=args.db_path,
        collection_name=args.collection,
    )

    results = engine.search(
        query=args.query,
        top_k=args.top_k,
        score_threshold=args.threshold,
    )

    print(format_results(results, args.query))


if __name__ == "__main__":
    main()
