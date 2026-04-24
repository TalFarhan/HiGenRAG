# Implementation Plan

> Cross-cutting directive: All code comments inside functions must be written in **English only**.

---

## Task 1 – Semantic Chunking (§3.0)

**Goal**  
Load a structured research corpus from a BEIR-compatible dataset (e.g. SciFact, FiQA, NFCorpus) via `datasets` / `ir_datasets`, segment each document into paragraph-level units, and apply hierarchical topic-based chunking to produce semantically cohesive segments of 256–512 tokens. The dataset name is accepted as a runtime parameter.

**Required Libraries**  
| Package | Purpose |
|---------|---------|
| `datasets` | Load BEIR / Hugging Face research corpora |
| `ir_datasets` | Alternative loader for BEIR benchmarks |
| `sentence-transformers` | Embedding with `all-mpnet-base-v2` (768d) |
| `bertopic` | Global & granular topic discovery |
| `umap-learn` | Dimensionality reduction (BERTopic internal) |
| `hdbscan` | Density-based clustering (BERTopic internal) |
| `tiktoken` or `transformers` | Token counting for chunk-size enforcement |

**Key Functions / Skills**  
- `load_corpus(dataset_name: str) -> list[dict]` – loads the corpus split from a BEIR-compatible dataset and returns a list of document dicts (`doc_id`, `title`, `text`).
- `segment_into_paragraphs(documents: list[dict]) -> list[str]` – splits each document's text into paragraph-level units.
- `embed_paragraphs(paragraphs: list[str]) -> np.ndarray` – produces 768d vectors via `all-mpnet-base-v2`.
- `discover_global_topics(embeddings, paragraphs) -> BERTopicModel` – first-pass BERTopic for document-level topics.
- `assign_paragraphs_to_clusters(paragraphs, topic_model) -> dict[int, list[str]]` – maps each paragraph to its topic cluster.
- `refine_chunks(clusters: dict, min_tokens=256, max_tokens=512) -> list[Chunk]` – merges/splits clusters into final token-bounded chunks.

---

## Task 2 – Topic Modeling & Extraction (§3.1)

**Goal**  
Run a secondary (granular) BERTopic pass on each chunk produced in Task 1 to identify latent sub-topics, extract descriptive labels and keywords, and attach metadata (topic labels, keywords, confidence scores) to every segment.

**Required Libraries**  
| Package | Purpose |
|---------|---------|
| `bertopic` | Granular topic modeling per chunk |
| `sentence-transformers` | Shared embedding model |
| `scikit-learn` | c-TF-IDF computation utilities |

**Key Functions / Skills**  
- `model_subtopics(chunk: Chunk) -> list[SubTopic]` – second-pass BERTopic within a single chunk; returns sub-topics with labels.
- `extract_keywords(subtopic: SubTopic, top_n=10) -> list[str]` – c-TF-IDF-based keyword extraction per sub-topic.
- `enrich_chunk_metadata(chunk: Chunk, subtopics: list[SubTopic]) -> EnrichedChunk` – attaches topic labels, keywords, and confidence scores to the chunk.

---

## Task 3 – Generative Threading (§3.2)

**Goal**  
For every sub-topic discovered in Task 2, generate exactly 3 synthetic queries (Factual, Comparative, Causal) using a local LLM, embed them with `all-mpnet-base-v2`, and index the resulting vectors in a local Qdrant instance to create multiple retrieval entry-points per segment.

**Required Libraries**  
| Package | Purpose |
|---------|---------|
| `ollama` (Python SDK) | Local LLM inference (Llama-3-8B / Qwen1.5) |
| `sentence-transformers` | Query embedding |
| `qdrant-client` | Vector indexing & hybrid search |

**Key Functions / Skills**  
- `generate_synthetic_queries(subtopic: SubTopic, typologies=["factual","comparative","causal"]) -> list[str]` – prompts the local LLM to produce 3 queries per sub-topic.
- `embed_queries(queries: list[str]) -> np.ndarray` – produces 768d vectors for synthetic queries.
- `index_vectors_in_qdrant(vectors, metadata, collection: str) -> None` – upserts query vectors + metadata into Qdrant.

---

## Task 4 – Generative Query Unification (§3.3)

**Goal**  
Cluster the synthetic query vectors into semantically homogeneous threads using a Gaussian Mixture Model, identify the optimal number of clusters (K) per chunk via BIC, compute centroid vectors for each thread, and replace per-query entries in Qdrant with distilled centroid anchors.

**Required Libraries**  
| Package | Purpose |
|---------|---------|
| `scikit-learn` | `GaussianMixture` for GMM clustering + BIC |
| `numpy` | Centroid (mean vector) calculation |
| `qdrant-client` | Centroid upsert & old-vector cleanup |
| `kneed` (optional) | Inflection-point / elbow detection |

**Key Functions / Skills**  
- `cluster_queries_gmm(vectors: np.ndarray, max_k: int) -> GMMResult` – fits GMM with adaptive K selected by BIC.
- `find_inflection_point(bic_scores: list[float]) -> int` – determines optimal K using elbow/inflection heuristic.
- `compute_centroids(vectors: np.ndarray, labels: np.ndarray) -> np.ndarray` – mean vector per cluster.
- `unify_qdrant_index(collection: str, centroids: np.ndarray, metadata) -> None` – replaces individual query vectors with centroid anchors in the Qdrant collection.
