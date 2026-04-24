"""
Generative Threading & Query Unification – Tasks 3-4 (§3.2, §3.3)

For every enriched chunk, generates synthetic queries (Factual, Comparative,
Causal) via ChatGroq (Llama-3 70B) with a deterministic mock fallback when
the API is unreachable, embeds them with all-mpnet-base-v2, clusters the
query vectors per chunk using a Gaussian Mixture Model (optimal K via BIC),
computes centroid vectors per cluster, and persists the result.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.mixture import GaussianMixture

sys.path.insert(0, str(Path(__file__).resolve().parent))
from semantic_chunker import EMBEDDING_MODEL_NAME, EMBEDDING_DIM

logger = logging.getLogger(__name__)

QUERY_TYPOLOGIES = ("factual", "comparative", "causal")
MAX_GMM_K = 6
MIN_QUERIES_FOR_GMM = 3

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class SyntheticQuery:
    """A single synthetic query generated for a sub-topic."""

    text: str
    typology: str
    subtopic_id: int


@dataclass
class QueryCluster:
    """A GMM cluster of synthetic queries with its centroid vector."""

    cluster_id: int
    centroid: list[float]
    query_texts: list[str]


@dataclass
class ThreadedChunk:
    """An enriched chunk augmented with synthetic queries and centroid anchors."""

    chunk_id: str
    doc_ids: list[str]
    text: str
    token_count: int
    topic_id: int
    keywords: list[str]
    subtopics: list[dict]
    topic_label: str
    confidence_score: float
    synthetic_queries: list[dict] = field(default_factory=list)
    query_clusters: list[dict] = field(default_factory=list)
    optimal_k: int = 1


# ---------------------------------------------------------------------------
# LLM-based query generation (ChatGroq via LangChain)
# ---------------------------------------------------------------------------

def _build_llm_chain():
    """Lazily construct a LangChain chain for synthetic query generation.

    Imports are deferred so the module loads even when langchain / langchain-groq
    are not installed (the mock path does not need them).
    """
    from langchain_groq import ChatGroq
    from langchain_core.prompts import ChatPromptTemplate

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.7,
        max_tokens=256,
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a research-query generator. Given a sub-topic "
                "description and its keywords, produce exactly ONE concise "
                "search query of the requested typology. "
                "Return ONLY the query text, no numbering or explanation.",
            ),
            (
                "human",
                "Sub-topic label: {label}\n"
                "Keywords: {keywords}\n"
                "Typology: {typology}\n\n"
                "Generate the query:",
            ),
        ]
    )

    return prompt | llm


def generate_queries_llm(
    subtopic: dict,
    chain,
) -> list[SyntheticQuery]:
    """Generate one query per typology using ChatGroq (Llama-3 70B)."""
    label = subtopic.get("label", "unknown")
    keywords = ", ".join(subtopic.get("keywords", []))
    topic_id = subtopic.get("topic_id", -1)
    queries: list[SyntheticQuery] = []

    for typology in QUERY_TYPOLOGIES:
        try:
            result = chain.invoke(
                {"label": label, "keywords": keywords, "typology": typology}
            )
            text = result.content.strip() if hasattr(result, "content") else str(result).strip()
            queries.append(SyntheticQuery(text=text, typology=typology, subtopic_id=topic_id))
        except Exception as exc:
            logger.warning("LLM query generation failed for %s/%s: %s", label, typology, exc)
            queries.append(
                SyntheticQuery(
                    text=_mock_single_query(label, keywords, typology),
                    typology=typology,
                    subtopic_id=topic_id,
                )
            )
    return queries


# ---------------------------------------------------------------------------
# Mock (fallback) query generation
# ---------------------------------------------------------------------------

def _mock_single_query(label: str, keywords_str: str, typology: str) -> str:
    """Produce a deterministic mock query for a single typology."""
    kw = keywords_str.split(", ")[:3]
    kw_fragment = " and ".join(kw) if kw else label

    templates = {
        "factual": f"What are the key characteristics of {kw_fragment}?",
        "comparative": f"How does {kw[0] if kw else label} compare to related concepts in {label}?",
        "causal": f"What causes {kw[0] if kw else label} to influence outcomes in {label}?",
    }
    return templates.get(typology, f"Describe {kw_fragment} in the context of {label}")


def generate_queries_mock(subtopic: dict) -> list[SyntheticQuery]:
    """Generate 5-6 deterministic synthetic queries without an LLM.

    Produces the standard 3 typology queries plus 2-3 hybrid variants
    to ensure enough vectors for meaningful GMM clustering.
    """
    label = subtopic.get("label", "unknown")
    kw = subtopic.get("keywords", [])
    kw_str = ", ".join(kw[:5])
    topic_id = subtopic.get("topic_id", -1)

    kw_fragment = " and ".join(kw[:3]) if kw else label
    first_kw = kw[0] if kw else label
    second_kw = kw[1] if len(kw) > 1 else label

    queries = [
        SyntheticQuery(
            text=f"What are the key characteristics of {kw_fragment}?",
            typology="factual",
            subtopic_id=topic_id,
        ),
        SyntheticQuery(
            text=f"How does {first_kw} compare to {second_kw} in the context of {label}?",
            typology="comparative",
            subtopic_id=topic_id,
        ),
        SyntheticQuery(
            text=f"What causes {first_kw} to influence outcomes related to {label}?",
            typology="causal",
            subtopic_id=topic_id,
        ),
        SyntheticQuery(
            text=f"What evidence supports the relationship between {first_kw} and {second_kw}?",
            typology="factual",
            subtopic_id=topic_id,
        ),
        SyntheticQuery(
            text=f"How do changes in {first_kw} affect {label} processes?",
            typology="causal",
            subtopic_id=topic_id,
        ),
    ]

    if len(kw) > 2:
        queries.append(
            SyntheticQuery(
                text=f"Compare the roles of {first_kw}, {second_kw}, and {kw[2]} in {label}.",
                typology="comparative",
                subtopic_id=topic_id,
            )
        )

    return queries


# ---------------------------------------------------------------------------
# Unified query generation dispatcher
# ---------------------------------------------------------------------------

def generate_synthetic_queries(
    subtopics: list[dict],
    use_llm: bool = True,
) -> list[SyntheticQuery]:
    """Generate synthetic queries for all sub-topics of a chunk.

    When *use_llm* is True, calls ChatGroq (Llama-3 70B) via LangChain.
    If the LLM chain fails to initialise or every single invocation
    errors out, the deterministic mock generator is used as a silent
    fallback so the pipeline never crashes on an API outage.
    """
    if use_llm:
        try:
            chain = _build_llm_chain()
            logger.info("Groq LLM chain ready — using LLM generation")
            all_queries: list[SyntheticQuery] = []
            for st in subtopics:
                all_queries.extend(generate_queries_llm(st, chain))
            return all_queries
        except Exception as exc:
            logger.warning(
                "LLM chain initialisation failed — falling back to mock generator: %s",
                exc,
            )

    all_queries = []
    for st in subtopics:
        all_queries.extend(generate_queries_mock(st))
    return all_queries


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def embed_queries(
    queries: list[SyntheticQuery],
    model: Optional[SentenceTransformer] = None,
    batch_size: int = 64,
) -> np.ndarray:
    """Produce 768-d vectors for the synthetic query texts."""
    if model is None:
        model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    texts = [q.text for q in queries]
    return model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
    )


# ---------------------------------------------------------------------------
# GMM clustering with BIC-based K selection  (§3.3)
# ---------------------------------------------------------------------------

def find_optimal_k(
    vectors: np.ndarray,
    max_k: int = MAX_GMM_K,
) -> int:
    """Select the optimal number of clusters via BIC minimisation.

    Iterates K from 1 to min(max_k, n_samples) and returns the K
    that yields the lowest BIC score.
    """
    n = vectors.shape[0]
    upper = min(max_k, n)
    if upper <= 1:
        return 1

    best_k = 1
    best_bic = np.inf

    for k in range(1, upper + 1):
        try:
            gmm = GaussianMixture(
                n_components=k,
                covariance_type="full",
                random_state=42,
                max_iter=200,
            )
            gmm.fit(vectors)
            bic = gmm.bic(vectors)
            logger.debug("K=%d  BIC=%.2f", k, bic)
            if bic < best_bic:
                best_bic = bic
                best_k = k
        except Exception as exc:
            logger.debug("GMM fit failed for K=%d: %s", k, exc)
            continue

    return best_k


def cluster_queries_gmm(
    vectors: np.ndarray,
    max_k: int = MAX_GMM_K,
) -> tuple[np.ndarray, int]:
    """Cluster query vectors using a Gaussian Mixture Model.

    Returns (labels_array, optimal_k).
    """
    n = vectors.shape[0]
    if n < MIN_QUERIES_FOR_GMM:
        return np.zeros(n, dtype=int), 1

    optimal_k = find_optimal_k(vectors, max_k)
    logger.info("Optimal K=%d (from %d query vectors)", optimal_k, n)

    gmm = GaussianMixture(
        n_components=optimal_k,
        covariance_type="full",
        random_state=42,
        max_iter=200,
    )
    labels = gmm.fit_predict(vectors)
    return labels, optimal_k


# ---------------------------------------------------------------------------
# Centroid computation
# ---------------------------------------------------------------------------

def compute_centroids(
    vectors: np.ndarray,
    labels: np.ndarray,
) -> dict[int, np.ndarray]:
    """Compute mean vector (centroid) for each cluster label."""
    unique_labels = np.unique(labels)
    centroids: dict[int, np.ndarray] = {}
    for lbl in unique_labels:
        mask = labels == lbl
        centroids[int(lbl)] = vectors[mask].mean(axis=0)
    return centroids


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

def load_enriched_chunks(path: Path) -> list[dict]:
    """Deserialize enriched chunks produced by Task 2."""
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    logger.info("Loaded %d enriched chunks from %s", len(data), path)
    return data


# ---------------------------------------------------------------------------
# JSON serialization helpers
# ---------------------------------------------------------------------------

def _json_default(obj):
    """Handle numpy types that may leak into JSON output."""
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
    input_path: str = "output/enriched_chunks.json",
    output_path: str = "output/threaded_chunks.json",
    use_llm: bool = True,
    max_k: int = MAX_GMM_K,
) -> list[ThreadedChunk]:
    """End-to-end generative threading and query unification.

    1. Load enriched chunks.
    2. For each chunk, generate synthetic queries per sub-topic.
    3. Embed queries with all-mpnet-base-v2.
    4. Cluster query vectors with GMM (optimal K via BIC).
    5. Compute centroid vectors per cluster.
    6. Persist threaded chunks to *output_path*.
    """
    chunks = load_enriched_chunks(Path(input_path))

    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    logger.info("Loaded embedding model: %s", EMBEDDING_MODEL_NAME)

    threaded: list[ThreadedChunk] = []

    for idx, chunk in enumerate(chunks):
        chunk_id = chunk["chunk_id"]
        subtopics = chunk.get("subtopics", [])

        # -- §3.2  Generative threading --------------------------------
        queries = generate_synthetic_queries(subtopics, use_llm=use_llm)

        if not queries:
            logger.warning("Chunk %s: no queries generated (no subtopics?)", chunk_id)
            threaded.append(
                ThreadedChunk(
                    chunk_id=chunk_id,
                    doc_ids=chunk.get("doc_ids", []),
                    text=chunk["text"],
                    token_count=chunk.get("token_count", 0),
                    topic_id=chunk.get("topic_id", -1),
                    keywords=chunk.get("keywords", []),
                    subtopics=subtopics,
                    topic_label=chunk.get("topic_label", ""),
                    confidence_score=chunk.get("confidence_score", 0.0),
                )
            )
            continue

        vectors = embed_queries(queries, model=embedding_model)

        # -- §3.3  Generative query unification ------------------------
        labels, optimal_k = cluster_queries_gmm(vectors, max_k=max_k)
        centroids = compute_centroids(vectors, labels)

        cluster_dicts: list[dict] = []
        for cid, centroid_vec in centroids.items():
            member_texts = [
                queries[i].text for i in range(len(queries)) if labels[i] == cid
            ]
            cluster_dicts.append(
                asdict(
                    QueryCluster(
                        cluster_id=cid,
                        centroid=centroid_vec.tolist(),
                        query_texts=member_texts,
                    )
                )
            )

        query_dicts = [asdict(q) for q in queries]

        threaded.append(
            ThreadedChunk(
                chunk_id=chunk_id,
                doc_ids=chunk.get("doc_ids", []),
                text=chunk["text"],
                token_count=chunk.get("token_count", 0),
                topic_id=chunk.get("topic_id", -1),
                keywords=chunk.get("keywords", []),
                subtopics=subtopics,
                topic_label=chunk.get("topic_label", ""),
                confidence_score=chunk.get("confidence_score", 0.0),
                synthetic_queries=query_dicts,
                query_clusters=cluster_dicts,
                optimal_k=optimal_k,
            )
        )

        if (idx + 1) % 10 == 0 or idx == len(chunks) - 1:
            logger.info(
                "Processed %d / %d chunks (current K=%d, queries=%d)",
                idx + 1,
                len(chunks),
                optimal_k,
                len(queries),
            )

    # -- Persist -------------------------------------------------------
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(
            [asdict(tc) for tc in threaded],
            fh,
            ensure_ascii=False,
            indent=2,
            default=_json_default,
        )
    logger.info("Saved %d threaded chunks to %s", len(threaded), out)
    return threaded


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description=(
            "Generative Threading & Query Unification – "
            "generate synthetic queries, cluster with GMM, compute centroids."
        ),
    )
    parser.add_argument(
        "-i", "--input",
        default="output/enriched_chunks.json",
        help="Input enriched chunks JSON (default: output/enriched_chunks.json)",
    )
    parser.add_argument(
        "-o", "--output",
        default="output/threaded_chunks.json",
        help="Output threaded chunks JSON (default: output/threaded_chunks.json)",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Force mock query generation (skip Groq LLM)",
    )
    parser.add_argument(
        "--max-k",
        type=int,
        default=MAX_GMM_K,
        help=f"Maximum K for GMM BIC search (default: {MAX_GMM_K})",
    )

    args = parser.parse_args()

    result = run_pipeline(
        input_path=args.input,
        output_path=args.output,
        use_llm=not args.no_llm,
        max_k=args.max_k,
    )
    print(f"Pipeline complete – {len(result)} threaded chunks produced.")


if __name__ == "__main__":
    main()
