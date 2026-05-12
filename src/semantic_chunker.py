"""
Semantic Chunker – global sentence-level topic discovery and chunk formation.

Loads a BEIR-compatible corpus, splits documents into sentences, embeds each
sentence as plain text (no document-id prefix), applies PCA before fitting a
global Gaussian mixture model (BIC-selected K), then forms token-bounded
chunks with semantic splits on topic boundaries.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from collections import Counter
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional

import ir_datasets
import numpy as np
import tiktoken
from bertopic import BERTopic
from bertopic.dimensionality import BaseDimensionalityReduction
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.mixture import GaussianMixture

logger = logging.getLogger(__name__)

EMBEDDING_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
EMBEDDING_DIM = 768  # kept for downstream vector stores using the same model
MIN_TOKENS = 256
MAX_TOKENS = 512
MAX_GLOBAL_TOPICS = 20
MIN_GLOBAL_K = 3
DOCUMENT_LIMIT = 150
# PCA reduces sentence embeddings before GMM to mitigate high-dimensional mixing issues.
GMM_PCA_N_COMPONENTS = 15

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_TOKENIZER = tiktoken.get_encoding("cl100k_base")


@dataclass
class Chunk:
    chunk_id: str
    doc_ids: list[str]
    text: str
    token_count: int
    topic_id: int = -1
    keywords: list[str] = field(default_factory=list)


def _count_tokens(text: str) -> int:
    return len(_TOKENIZER.encode(text))


def _get_evidence_doc_ids(dataset_name: str) -> set[str]:
    """Collect doc_ids that have positive relevance in the test qrels."""
    canonical_name = dataset_name.split("/")[-1]
    dataset = ir_datasets.load(f"beir/{canonical_name}/test")
    evidence_ids: set[str] = set()
    for qrel in dataset.qrels_iter():
        if qrel.relevance > 0:
            evidence_ids.add(qrel.doc_id)
    return evidence_ids


def load_corpus(dataset_name: str, limit: int = DOCUMENT_LIMIT) -> list[dict[str, Any]]:
    """Load corpus filtered to documents with evidence in the golden qrels."""
    logger.info("Loading corpus: %s (evidence-filtered, limit=%d)", dataset_name, limit)

    evidence_doc_ids = _get_evidence_doc_ids(dataset_name)
    logger.info("Found %d unique doc_ids with evidence in qrels", len(evidence_doc_ids))

    canonical_name = dataset_name.split("/")[-1]
    corpus_dataset = ir_datasets.load(f"beir/{canonical_name}/test")

    documents: list[dict[str, Any]] = []
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

    logger.info("Loaded %d documents from %s (filtered by evidence)", len(documents), dataset_name)
    return documents


def split_document_into_sentences(doc: dict[str, Any]) -> list[str]:
    """Split one document into ordered sentences (title prepended to body like paragraph flow)."""
    body = (doc.get("text") or "").strip()
    title = (doc.get("title") or "").strip()
    full_text = f"{title}\n\n{body}" if title else body
    if not full_text:
        return []
    parts = _SENTENCE_SPLIT_RE.split(full_text)
    out: list[str] = []
    for p in parts:
        s = p.strip()
        if s:
            out.append(s)
    return out


def flatten_sentences_for_corpus(documents: list[dict[str, Any]]) -> tuple[list[str], list[str], list[list[str]]]:
    """Return flat sentence strings for embedding, parallel doc_ids, and per-doc sentence lists.

    Sentences are plain text only; doc_id is not concatenated (metadata lives in chunk / store).
    """
    flat_sentence_texts: list[str] = []
    flat_doc_ids: list[str] = []
    sentences_per_doc: list[list[str]] = []

    for doc in documents:
        doc_id = str(doc["doc_id"])
        sents = split_document_into_sentences(doc)
        sentences_per_doc.append(sents)
        for s in sents:
            flat_sentence_texts.append(s)
            flat_doc_ids.append(doc_id)

    logger.info("Extracted %d sentences across %d documents", len(flat_sentence_texts), len(documents))
    return flat_sentence_texts, flat_doc_ids, sentences_per_doc


def embed_sentence_texts(
    sentence_texts: list[str],
    model: Optional[SentenceTransformer] = None,
    batch_size: int = 64,
) -> np.ndarray:
    """Encode sentence strings with the shared embedding model (clean text only)."""
    if model is None:
        model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    logger.info("Embedding %d sentences (batch_size=%d)", len(sentence_texts), batch_size)
    if not sentence_texts:
        return np.zeros((0, EMBEDDING_DIM), dtype=np.float64)
    arr: np.ndarray = model.encode(
        sentence_texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    return arr


def reduce_embeddings_for_gmm(
    embeddings: np.ndarray,
    n_components: int = GMM_PCA_N_COMPONENTS,
) -> np.ndarray:
    """Apply PCA so GMM runs in a lower-dimensional subspace."""
    n, d = int(embeddings.shape[0]), int(embeddings.shape[1])
    if n == 0:
        return embeddings
    k = min(int(n_components), n, d)
    if k < 1:
        k = 1
    pca = PCA(n_components=k, random_state=42)
    reduced = pca.fit_transform(embeddings)
    logger.info(
        "PCA for GMM: %d samples, %d -> %d dimensions (explained variance ratio sum=%.4f)",
        n,
        d,
        k,
        float(np.sum(pca.explained_variance_ratio_)),
    )
    return reduced


def find_optimal_k_bic(embeddings: np.ndarray, max_k: int, min_k: int = MIN_GLOBAL_K) -> int:
    """Pick mixture component count K in [lower..upper] with lowest BIC (diagonal covariance).

    Expects *embeddings* in the same space used for GMM (typically PCA-reduced).
    """
    n = int(embeddings.shape[0])
    if n <= 0:
        return 1
    upper = min(int(max_k), n)
    lower = max(1, min(min_k, upper))
    if upper < lower:
        return 1

    logger.info(
        "BIC search: min_k=%d max_k=%d n=%d covariance_type=diag",
        min_k,
        max_k,
        n,
    )

    best_k = 1
    best_bic = float("inf")

    for k in range(lower, upper + 1):
        try:
            gmm = GaussianMixture(
                n_components=k,
                covariance_type="diag",
                random_state=42,
                max_iter=200,
            )
            gmm.fit(embeddings)
            bic = float(gmm.bic(embeddings))
            if bic < best_bic:
                best_bic = bic
                best_k = k
        except Exception as exc:
            logger.debug("GMM fit failed for K=%d: %s", k, exc)
            continue

    logger.info(
        "BIC-optimal global K=%d (n=%d, min_k=%d, cap=%d, cov=diag)",
        best_k,
        n,
        min_k,
        max_k,
    )
    return best_k


def fit_global_sentence_gmm(
    embeddings: np.ndarray,
    k: int,
) -> tuple[GaussianMixture, np.ndarray]:
    """Fit a single global GMM and return the model and per-sample labels."""
    n = int(embeddings.shape[0])
    if n == 0:
        raise ValueError("Cannot fit GMM on zero sentences.")
    k_use = max(1, min(int(k), n))
    gmm = GaussianMixture(
        n_components=k_use,
        covariance_type="diag",
        random_state=42,
        max_iter=200,
    )
    labels = gmm.fit_predict(embeddings)
    return gmm, labels.astype(int)


def _labels_for_sentences_per_doc(
    sentences_per_doc: list[list[str]],
    flat_labels: np.ndarray,
) -> list[list[int]]:
    """Map flat label order back to per-document sentence topic lists."""
    topics_per_doc: list[list[int]] = []
    idx = 0
    for sents in sentences_per_doc:
        row: list[int] = []
        for _ in sents:
            row.append(int(flat_labels[idx]))
            idx += 1
        topics_per_doc.append(row)
    return topics_per_doc


def _primary_topic(buffer_topics: list[int]) -> int:
    """Majority topic in the buffer; ties broken by earliest sentence order."""
    if not buffer_topics:
        return -1
    counts = Counter(buffer_topics)
    max_c = max(counts.values())
    for t in buffer_topics:
        if counts[t] == max_c:
            return int(t)
    return int(buffer_topics[0])


def _split_text_to_max_tokens(text: str, max_tokens: int) -> list[str]:
    """Greedy word grouping so each piece is at most max_tokens (for oversized sentences)."""
    words = text.split()
    if not words:
        return []
    pieces: list[str] = []
    buf: list[str] = []
    for w in words:
        trial = " ".join(buf + [w])
        if buf and _count_tokens(trial) > max_tokens:
            pieces.append(" ".join(buf))
            buf = [w]
        else:
            buf.append(w)
    if buf:
        pieces.append(" ".join(buf))
    return pieces


def form_semantic_physical_chunks(
    documents: list[dict[str, Any]],
    sentences_per_doc: list[list[str]],
    topics_per_doc: list[list[int]],
    min_tokens: int,
    max_tokens: int,
) -> list[Chunk]:
    """Merge adjacent sentences per document with token bounds and global-topic boundaries."""
    logger.debug("Chunking with min_tokens=%d max_tokens=%d", min_tokens, max_tokens)
    all_chunks: list[Chunk] = []
    chunk_counter = 0

    for doc, sents, topics in zip(documents, sentences_per_doc, topics_per_doc, strict=True):
        doc_id = str(doc["doc_id"])
        buffer_sents: list[str] = []
        buffer_topics: list[int] = []

        def flush_buffer() -> None:
            nonlocal chunk_counter
            if not buffer_sents:
                return
            text = "\n\n".join(buffer_sents).strip()
            tid = int(_primary_topic(buffer_topics))
            all_chunks.append(
                Chunk(
                    chunk_id=f"chunk-{chunk_counter:05d}",
                    doc_ids=[doc_id],
                    text=text,
                    token_count=_count_tokens(text),
                    topic_id=tid,
                )
            )
            chunk_counter += 1
            buffer_sents.clear()
            buffer_topics.clear()

        for sent, topic in zip(sents, topics, strict=True):
            pieces = _split_text_to_max_tokens(sent, max_tokens) if _count_tokens(sent) > max_tokens else [sent]
            for piece in pieces:
                t = int(topic)
                if buffer_sents and t != _primary_topic(buffer_topics):
                    flush_buffer()

                candidate = "\n\n".join(buffer_sents + [piece]).strip() if buffer_sents else piece
                if buffer_sents and _count_tokens(candidate) > max_tokens:
                    flush_buffer()
                    candidate = piece

                buffer_sents.append(piece)
                buffer_topics.append(t)

        flush_buffer()

    logger.info("Produced %d chunks (sentence-ordered, global-topic splits)", len(all_chunks))
    return all_chunks


def enrich_chunks_with_bertopic(chunks: list[Chunk], top_n_words: int = 5) -> None:
    """Populate ``Chunk.keywords`` with c-TF-IDF terms from BERTopic guided by GMM topics.

    For each chunk, the GMM-derived primary topic is ``Chunk.topic_id`` (majority sentence
    label inside the chunk). BERTopic is fit in supervised mode so ``y`` fixes topic
    membership and only the class-based c-TF-IDF representation runs (no UMAP/HDBSCAN
    discovery). Embeddings are a one-hot of ``y`` so no extra sentence-transformer pass
    is required beyond the main pipeline.
    """
    if not chunks:
        return

    texts = [c.text for c in chunks]
    topics = [int(c.topic_id) for c in chunks]

    if any(t < 0 for t in topics):
        logger.warning("Skipping BERTopic enrichment: negative topic_id in chunks.")
        return

    y = np.asarray(topics, dtype=int)
    uniq = np.unique(y)
    label_to_col = {int(v): i for i, v in enumerate(uniq)}
    n = len(y)
    k = len(uniq)
    embeddings = np.zeros((n, k), dtype=np.float64)
    for i, lab in enumerate(y):
        embeddings[i, label_to_col[int(lab)]] = 1.0

    vectorizer_model = CountVectorizer(stop_words="english")
    topic_model = BERTopic(
        umap_model=BaseDimensionalityReduction(),
        hdbscan_model=LogisticRegression(max_iter=5000, random_state=42),
        vectorizer_model=vectorizer_model,
        calculate_probabilities=False,
    )
    topic_model.fit_transform(texts, embeddings=embeddings, y=topics)

    mappings = topic_model.topic_mapper_.get_mappings()

    for chunk in chunks:
        primary_topic = int(chunk.topic_id)
        bertopic_tid = int(mappings.get(primary_topic, primary_topic))
        topic_words = topic_model.get_topic(bertopic_tid) or []
        out: list[str] = []
        for word, _score in topic_words:
            w = (word or "").strip()
            if w and w not in out:
                out.append(w)
            if len(out) >= top_n_words:
                break
        chunk.keywords = out


def collect_core_sentences_per_topic(
    sentence_embeddings: np.ndarray,
    sentence_labels: np.ndarray,
    flat_sentences: list[str],
    top_n: int = 8,
) -> list[dict[str, Any]]:
    """For each GMM component, pick sentences with highest cosine similarity to the cluster centroid."""
    if sentence_embeddings.shape[0] == 0:
        return []

    unique_topics = sorted(int(x) for x in np.unique(sentence_labels))
    topics_payload: list[dict[str, Any]] = []

    centroids = []
    for tid in unique_topics:
        mask = sentence_labels == tid
        emb_k = sentence_embeddings[mask]
        c = emb_k.mean(axis=0, keepdims=True)
        centroids.append((tid, c, emb_k, np.where(mask)[0]))

    for tid, centroid_row, emb_k, global_indices in centroids:
        sims = cosine_similarity(emb_k, centroid_row).ravel()
        order = np.argsort(-sims)
        pick = order[: min(top_n, len(order))]
        core = [flat_sentences[int(global_indices[j])] for j in pick]
        topics_payload.append({"topic_id": tid, "core_sentences": core})

    topics_payload.sort(key=lambda x: int(x["topic_id"]))
    return topics_payload


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def write_global_topic_anchors(
    path: Path,
    k: int,
    topics: list[dict[str, Any]],
) -> None:
    """Persist core sentences per global topic for downstream anchor-query generation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "embedding_model": EMBEDDING_MODEL_NAME,
        "k": int(k),
        "topics": topics,
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    logger.info("Wrote global topic anchors (%d topics) to %s", k, path)


def run_pipeline(
    dataset_name: str,
    min_tokens: int = MIN_TOKENS,
    max_tokens: int = MAX_TOKENS,
    output_path: Optional[str] = None,
    limit: Optional[int] = None,
    anchors_output_path: Optional[str] = None,
    min_global_k: int = MIN_GLOBAL_K,
) -> list[Chunk]:
    """Run corpus load, global sentence GMM, and semantic–physical chunking."""
    doc_limit = limit if limit is not None else DOCUMENT_LIMIT
    documents = load_corpus(dataset_name, limit=doc_limit)

    flat_sentence_texts, _flat_doc_ids, sentences_per_doc = flatten_sentences_for_corpus(documents)
    flat_sentences: list[str] = []
    for sents in sentences_per_doc:
        flat_sentences.extend(sents)

    embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    embeddings = embed_sentence_texts(flat_sentence_texts, model=embed_model)

    if embeddings.shape[0] == 0:
        logger.warning("No sentences found; writing empty chunk list.")
        chunks: list[Chunk] = []
    else:
        reduced = reduce_embeddings_for_gmm(embeddings, n_components=GMM_PCA_N_COMPONENTS)
        optimal_k = find_optimal_k_bic(
            reduced, MAX_GLOBAL_TOPICS, min_k=min_global_k
        )
        _gmm, labels = fit_global_sentence_gmm(reduced, optimal_k)
        topics_per_doc = _labels_for_sentences_per_doc(sentences_per_doc, labels)
        chunks = form_semantic_physical_chunks(
            documents,
            sentences_per_doc,
            topics_per_doc,
            min_tokens=min_tokens,
            max_tokens=max_tokens,
        )

        topics_payload = collect_core_sentences_per_topic(
            embeddings, labels, flat_sentences, top_n=8
        )

        anchor_path: Optional[Path] = None
        if anchors_output_path:
            anchor_path = Path(anchors_output_path)
        elif output_path:
            anchor_path = Path(output_path).parent / "global_topic_anchors.json"
        if anchor_path is not None:
            write_global_topic_anchors(anchor_path, int(optimal_k), topics_payload)

        enrich_chunks_with_bertopic(chunks)

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
        logger.info("Saved %d chunks to %s", len(chunks), out)

    return chunks


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Semantic chunker – global sentence GMM and token-bounded chunks.",
    )
    parser.add_argument(
        "dataset",
        type=str,
        help="BEIR dataset name (e.g. scifact, fiqa, nfcorpus)",
    )
    parser.add_argument(
        "--min-tokens",
        type=int,
        default=MIN_TOKENS,
        help=f"Minimum target tokens per chunk (default: {MIN_TOKENS})",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=MAX_TOKENS,
        help=f"Maximum tokens per chunk (default: {MAX_TOKENS})",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output JSON file path for serialized chunks",
    )
    parser.add_argument(
        "--anchors-output",
        type=str,
        default=None,
        help="Path for global_topic_anchors.json (default: beside chunks output)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N evidence documents (quick trials)",
    )
    parser.add_argument(
        "--min-global-k",
        type=int,
        default=MIN_GLOBAL_K,
        help="Minimum number of global topics for GMM to discover (default: %(default)s).",
    )

    args = parser.parse_args()
    chunks = run_pipeline(
        dataset_name=args.dataset,
        min_tokens=args.min_tokens,
        max_tokens=args.max_tokens,
        output_path=args.output,
        limit=args.limit,
        anchors_output_path=args.anchors_output,
        min_global_k=args.min_global_k,
    )
    print(f"Pipeline complete – {len(chunks)} chunks produced.")


if __name__ == "__main__":
    main()
