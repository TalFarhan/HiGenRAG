"""
Generative threading – one anchor query per global GMM topic and flexible
top-K association to chunks via cosine similarity.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import textwrap
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv

load_dotenv()

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, str(Path(__file__).resolve().parent))
from semantic_chunker import EMBEDDING_MODEL_NAME

logger = logging.getLogger(__name__)

BASE_K = 3
SIMILARITY_TOLERANCE = 0.05
MIN_SIMILARITY_THRESHOLD = 0.60


@dataclass
class ThreadedChunk:
    """Enriched chunk with globally associated anchor queries."""

    chunk_id: str
    doc_ids: list[str]
    text: str
    token_count: int
    topic_id: int
    keywords: list[str]
    subtopics: list[dict[str, Any]]
    topic_label: str
    confidence_score: float
    synthetic_queries: list[dict[str, Any]] = field(default_factory=list)
    query_clusters: list[dict[str, Any]] = field(default_factory=list)
    optimal_k: int = 1


def _build_anchor_llm_chain():
    """Construct LangChain chain for a single global anchor query per topic."""
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_groq import ChatGroq

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.35,
        max_tokens=160,
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You write one highly representative search query for a fixed global "
                "research topic. The query must be specific, retrieval-friendly, and "
                "grounded in the given source sentences. Return ONLY the query text, "
                "no quotes or numbering.",
            ),
            (
                "human",
                "Global topic id: {topic_id}\n"
                "Representative sentences from this topic cluster:\n{core_block}\n\n"
                "Generate exactly ONE anchor query:",
            ),
        ]
    )
    return prompt | llm


def generate_anchor_query_llm(
    topic_id: int,
    core_sentences: list[str],
    chain: Any,
) -> str:
    """Ask the LLM for one anchor query for this global topic."""
    block = "\n".join(f"- {s}" for s in core_sentences[:12])
    try:
        result = chain.invoke(
            {
                "topic_id": topic_id,
                "core_block": block,
            }
        )
        text = result.content.strip() if hasattr(result, "content") else str(result).strip()
        return _normalize_single_line(text)
    except Exception as exc:
        logger.warning("LLM anchor generation failed for topic %s: %s", topic_id, exc)
        return _mock_anchor_query(topic_id, core_sentences)


def _normalize_single_line(text: str) -> str:
    t = " ".join(text.split())
    t = re.sub(r"^[\"']|[\"']$", "", t).strip()
    return t


def _mock_anchor_query(topic_id: int, core_sentences: list[str]) -> str:
    """Deterministic anchor when the API is unavailable."""
    hint = core_sentences[0] if core_sentences else "biomedical evidence"
    short = textwrap.shorten(hint, width=120, placeholder=" …")
    return f"What findings relate to: {short} (global topic {topic_id})?"


def generate_global_anchor_queries(
    topics_payload: list[dict[str, Any]],
    use_llm: bool = True,
) -> tuple[list[str], list[int], int]:
    """Produce exactly one anchor string per global topic (length K)."""
    chain: Any = None
    if use_llm:
        try:
            chain = _build_anchor_llm_chain()
            logger.info("Groq LLM ready for global anchor queries")
        except Exception as exc:
            logger.warning("LLM chain init failed; using mock anchors: %s", exc)
            chain = None

    ordered = sorted(topics_payload, key=lambda x: int(x["topic_id"]))
    texts: list[str] = []
    topic_ids: list[int] = []
    for entry in ordered:
        tid = int(entry["topic_id"])
        topic_ids.append(tid)
        core = list(entry.get("core_sentences") or [])
        if chain is not None:
            texts.append(generate_anchor_query_llm(tid, core, chain))
        else:
            texts.append(_mock_anchor_query(tid, core))
    return texts, topic_ids, len(texts)


def load_global_topic_anchors(path: Path) -> tuple[int, list[dict[str, Any]]]:
    """Load global_topic_anchors.json from the semantic chunker stage."""
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    k = int(data.get("k", 0))
    topics = list(data.get("topics") or [])
    logger.info("Loaded global anchors: K=%d from %s", k, path)
    return k, topics


def embed_texts(
    texts: list[str],
    model: SentenceTransformer,
    batch_size: int = 64,
) -> np.ndarray:
    """Encode texts with the shared sentence-transformer model."""
    if not texts:
        return np.zeros((0, model.get_sentence_embedding_dimension()), dtype=np.float64)
    return model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
    )


def flexible_top_k_chunk_indices(similarities: np.ndarray) -> list[int]:
    """Rank chunks by similarity; extend ties past BASE_K; drop below min threshold."""
    n = int(similarities.shape[0])
    if n == 0:
        return []
    order = np.argsort(-similarities)
    scores = similarities[order]
    selected_order_idx: list[int] = []
    base = min(BASE_K, n)
    for i in range(base):
        selected_order_idx.append(int(order[i]))
    j = base
    while j < n and base > 0:
        if float(scores[j]) >= float(scores[base - 1]) - SIMILARITY_TOLERANCE:
            selected_order_idx.append(int(order[j]))
            j += 1
        else:
            break
    out: list[int] = []
    for idx in selected_order_idx:
        if float(similarities[idx]) >= MIN_SIMILARITY_THRESHOLD:
            out.append(idx)
    return out


def associate_queries_to_chunks(
    anchor_queries: list[str],
    anchor_topic_ids: list[int],
    chunk_texts: list[str],
    embedding_model: SentenceTransformer,
) -> dict[int, list[dict[str, Any]]]:
    """Map each chunk index to anchor queries selected by flexible top-K per query."""
    q_emb = embed_texts(anchor_queries, embedding_model)
    c_emb = embed_texts(chunk_texts, embedding_model)
    if q_emb.shape[0] == 0 or c_emb.shape[0] == 0:
        return {}

    sims = cosine_similarity(q_emb, c_emb)
    chunk_to_queries: dict[int, list[dict[str, Any]]] = {i: [] for i in range(len(chunk_texts))}

    for q_idx, query_text in enumerate(anchor_queries):
        tid = int(anchor_topic_ids[q_idx])
        row = sims[q_idx]
        pick = flexible_top_k_chunk_indices(row)
        for cidx in pick:
            chunk_to_queries[cidx].append(
                {
                    "text": query_text,
                    "typology": "anchor",
                    "subtopic_id": tid,
                }
            )
    return chunk_to_queries


def load_enriched_chunks(path: Path) -> list[dict[str, Any]]:
    """Deserialize enriched chunks produced by the granular topic modeler."""
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    logger.info("Loaded %d enriched chunks from %s", len(data), path)
    return data


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def default_anchors_path(enriched_path: Path) -> Path:
    """Place global_topic_anchors.json next to the enriched chunks directory."""
    return enriched_path.parent / "global_topic_anchors.json"


def run_pipeline(
    input_path: str = "output/enriched_chunks.json",
    output_path: str = "output/threaded_chunks.json",
    anchors_path: Optional[str] = None,
    use_llm: bool = True,
) -> list[ThreadedChunk]:
    """Generate K global anchors, associate them to chunks, and persist JSON."""
    inp = Path(input_path)
    chunks = load_enriched_chunks(inp)

    anchor_file = Path(anchors_path) if anchors_path else default_anchors_path(inp)
    if not anchor_file.is_file():
        raise FileNotFoundError(
            f"Global topic anchors not found: {anchor_file}. "
            "Run semantic_chunker with output so global_topic_anchors.json is created."
        )

    global_k, topics_payload = load_global_topic_anchors(anchor_file)
    anchor_queries, anchor_topic_ids, k = generate_global_anchor_queries(
        topics_payload, use_llm=use_llm
    )
    if k != global_k:
        logger.warning("Anchor count %d differs from file K=%d", k, global_k)

    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    chunk_texts = [str(c.get("text", "")) for c in chunks]
    associations = associate_queries_to_chunks(
        anchor_queries, anchor_topic_ids, chunk_texts, embedding_model
    )

    threaded: list[ThreadedChunk] = []
    for idx, chunk in enumerate(chunks):
        extra = associations.get(idx, [])
        existing = list(chunk.get("synthetic_queries") or [])
        merged = existing + extra

        threaded.append(
            ThreadedChunk(
                chunk_id=str(chunk["chunk_id"]),
                doc_ids=list(chunk.get("doc_ids", [])),
                text=str(chunk.get("text", "")),
                token_count=int(chunk.get("token_count", 0)),
                topic_id=int(chunk.get("topic_id", -1)),
                keywords=list(chunk.get("keywords", [])),
                subtopics=list(chunk.get("subtopics", [])),
                topic_label=str(chunk.get("topic_label", "")),
                confidence_score=float(chunk.get("confidence_score", 0.0)),
                synthetic_queries=merged,
                query_clusters=[],
                optimal_k=int(k),
            )
        )

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


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Global anchor queries and cosine-based chunk association.",
    )
    parser.add_argument(
        "-i",
        "--input",
        default="output/enriched_chunks.json",
        help="Input enriched chunks JSON",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="output/threaded_chunks.json",
        help="Output threaded chunks JSON",
    )
    parser.add_argument(
        "--anchors",
        default=None,
        help="Path to global_topic_anchors.json (default: beside enriched input)",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Skip Groq; use deterministic mock anchor queries",
    )

    args = parser.parse_args()
    result = run_pipeline(
        input_path=args.input,
        output_path=args.output,
        anchors_path=args.anchors,
        use_llm=not args.no_llm,
    )
    print(f"Pipeline complete – {len(result)} threaded chunks produced.")


if __name__ == "__main__":
    main()
