Methodology: Architecture
3.0 Semantic Chunking
Input Specification: The system loads a structured research corpus from a BEIR-compatible dataset (e.g. SciFact, FiQA, NFCorpus) via Hugging Face / `ir_datasets`. Each document in the corpus is treated as a raw text unit and segmented into paragraphs as the base textual unit. Target chunk size is 256-512 tokens per segment, providing sufficient semantic context for topic discovery without diluting latent signals. The dataset name is provided as a runtime parameter, enabling reproducible evaluation across multiple benchmarks.
Embedding Model: All vector representations throughout the pipeline (topic modeling, query anchoring, centroid calculation) are produced using `sentence-transformers/all-mpnet-base-v2` (768-dimensional). The higher dimensionality provides finer semantic resolution critical for downstream GMM clustering and centroid accuracy, while remaining fully local with no external API dependency.
The partitioning process is executed using a hierarchical, topic-based approach aimed at generating text segments (chunks) with high semantic cohesion. The process consists of the following stages:
Global Topic Discovery: Performing document-level topic modeling using BERTopic to identify dominant topics and their distribution across the textual space. BERTopic is selected for its modular architecture -- the separation of UMAP (dimensionality reduction), HDBSCAN (clustering), and c-TF-IDF (topic representation) allows independent fine-tuning at each layer, which is essential for the hierarchical two-pass modeling required by this pipeline.
Clustering & Segmentation: Partitioning document content into clusters based on the identified topics. Each paragraph-level text unit is assigned to the most relevant cluster while maintaining semantic continuity between adjacent sentences.
Chunk Refinement: Consolidating and merging clusters into final segments within the 256-512 token target range. This stage ensures that chunks are not merely collections of similar sentences but continuous, self-contained units of meaning.
Granular Topic Modeling (Secondary): For each generated chunk, a second round of BERTopic modeling is applied. This granular analysis facilitates the extraction of specific keywords and tags for each segment, enhancing retrieval precision in later stages.
3.1 Topic Modeling & Extraction
Following chunk creation, a secondary BERTopic pass is deployed for an in-depth analysis of the internal content within each segment. The objective is to uncover and label "latent topics" embedded in the text. Leveraging BERTopic's representation tuning capabilities, each discovered sub-topic is assigned descriptive labels derived from c-TF-IDF term rankings.
The working assumption is that even a single semantic chunk may contain multiple sub-topics or varied contexts. Identifying these early allows for metadata enrichment -- attaching topic labels, keywords, and confidence scores to each segment -- significantly improving retrieval precision and enabling context-aware retrieval.
3.2 Generative Threading
In this stage, the system performs generative enrichment for each identified sub-topic. This process aims to bridge the semantic gap between the document's language (passive information) and the user's query language (active querying):
LLM Provider: Synthetic query generation is performed by a local Nano-LLM (Llama-3-8B-Instruct or Qwen1.5) served via Ollama or vLLM. Running inference locally eliminates external API dependency and cost overhead when generating queries at scale across thousands of sub-topics.
Multi-dimensional Synthetic Query Generation: For each sub-topic, the LLM generates exactly 3 hypothetical queries -- one per logical typology -- representing different retrieval perspectives:
Factual: Data and detail retrieval.
Comparative: Examining relationships between entities or concepts.
Causal: Investigating cause-and-effect relationships and processes.
Limiting generation to 3 queries per sub-topic prevents semantic noise that would impair the GMM's ability to find an optimal Inflection Point during the unification stage (3.3).
Vector Anchoring and Indexing: Each synthetic query is converted into a 768-dimensional vector using `all-mpnet-base-v2` and indexed in Qdrant (deployed locally). Qdrant is selected for its native support of hybrid search (dense + sparse vectors) and advanced metadata filtering, which are required by the downstream Hybrid Reranking stages. This creates multiple "points of entry" for each textual segment, thereby increasing the probability of finding semantic proximity to a user's real-time query.
3.3 Generative Query Unification
This stage focuses on optimizing and streamlining the synthetic query space into distilled representations to prevent redundancy and noise in the vector index. The process is based on the following mechanisms:
Clustering & Threading: Queries are grouped into homogeneous "threads" based on:
Semantic Similarity: Grouping queries with overlapping meanings in the 768-dimensional embedding space produced by `all-mpnet-base-v2`.
Gaussian Mixture Model (GMM): Probabilistic clustering that allows flexibility in cluster structures, assuming each thread represents a unique semantic distribution.
Inflection Point Threshold: Identifying the optimal distance threshold for unification (similar to the Elbow Method) to prevent information loss while avoiding over-aggregation.
Centroid Calculation: For each generated group, a mean vector (Centroid) is calculated in the 768-dimensional space to serve as the "semantic essence," acting as the final anchoring point in the Qdrant index.
Adaptive K-Parameter Optimization: Unlike a static approach, the selection of the number of clusters (K) is performed adaptively for each chunk. The system utilizes quality metrics (such as the Bayesian Information Criterion - BIC) to determine the optimal number of topics, ensuring a granularity level customized to the complexity of each segment's content.

## Clarifications
### Session 2026-04-18
- Q: Embedding model for all vector operations? -> A: `sentence-transformers/all-mpnet-base-v2` (768d, local). Higher dimensionality provides finer semantic resolution for GMM clustering and centroid accuracy.
- Q: Topic modeling engine (BERTopic vs Top2Vec)? -> A: BERTopic. Modular architecture (UMAP + HDBSCAN + c-TF-IDF) allows independent fine-tuning at each layer for hierarchical two-pass modeling.
- Q: LLM provider for synthetic query generation? -> A: Local Nano-LLM (Llama-3-8B-Instruct or Qwen1.5) via Ollama/vLLM. 3 queries per sub-topic (one per typology: Factual, Comparative, Causal).
- Q: Vector store for embedding persistence and retrieval? -> A: Qdrant (local deployment). Native hybrid search (dense + sparse) and metadata filtering for downstream Hybrid Reranking.
- Q: Input format and chunk size bounds? -> A: Structured research corpus from BEIR/Hugging Face (e.g. SciFact, FiQA, NFCorpus), paragraphs as base unit, target 256-512 tokens per chunk.
