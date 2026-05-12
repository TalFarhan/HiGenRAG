"""Trace one SciFact example through all pipeline stages.

Usage examples:
  python scripts/trace_one_example.py --query-id 49
  python scripts/trace_one_example.py --query-text "ADAR1 binds to Dicer to cleave pre-miRNA."
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
import textwrap
from pathlib import Path
from typing import Optional


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
CSV_PATH = ROOT / "full_data_dump" / "scifact_qa_pairs.csv"
CHUNKS_PATH = ROOT / "output" / "chunks.json"
ENRICHED_PATH = ROOT / "output" / "enriched_chunks.json"
THREADED_PATH = ROOT / "output" / "threaded_chunks.json"

QDRANT_DB_PATH = str(ROOT / "output" / "qdrant_db")
BASELINE_COLLECTION = "baseline_chunks"
ENRICHED_COLLECTION = "enriched_chunks"

SEPARATOR = "=" * 88
SUB_SEPARATOR = "-" * 88


def _read_json(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _load_example(
    csv_path: Path,
    query_id: Optional[str],
    query_text: Optional[str],
) -> dict:
    with open(csv_path, encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))

    if not rows:
        raise ValueError(f"No rows found in CSV: {csv_path}")

    if query_id is not None:
        for row in rows:
            if str(row.get("Query_ID", "")).strip() == str(query_id).strip():
                return row
        raise ValueError(f"Query_ID={query_id} not found in {csv_path}")

    if query_text is not None:
        needle = query_text.strip().lower()
        for row in rows:
            if row.get("Claim_Text", "").strip().lower() == needle:
                return row
        raise ValueError("Claim text not found in CSV")

    return rows[0]


def _safe_load_stage(path: Path) -> list[dict]:
    if not path.exists():
        return []
    try:
        return _read_json(path)
    except Exception:
        return []


def _print_header(example: dict) -> None:
    print()
    print(SEPARATOR)
    print("TRACE ONE EXAMPLE - END TO END")
    print(SEPARATOR)
    print(f"Query_ID         : {example.get('Query_ID', '')}")
    print(f"Evidence_Doc_ID  : {example.get('Evidence_Doc_ID', '')}")
    print("Claim_Text       :")
    print(textwrap.fill(example.get("Claim_Text", ""), width=84, initial_indent="  ", subsequent_indent="  "))
    print()


def _norm_id(value: object) -> str:
    """Normalize IDs so int/str formatting differences do not break matching."""
    if value is None:
        return ""
    text = str(value).strip()
    if not text:
        return ""
    digits_only = re.sub(r"\D", "", text)
    if digits_only:
        return str(int(digits_only))
    return text.lower()


def _contains_id(chunk: dict, target_doc_id: str) -> bool:
    """Support doc_ids/doc_id variants and mixed int/string storage."""
    chunk_ids = chunk.get("doc_ids")
    if isinstance(chunk_ids, list):
        normalized = {_norm_id(v) for v in chunk_ids}
        return _norm_id(target_doc_id) in normalized
    single_id = chunk.get("doc_id")
    return _norm_id(single_id) == _norm_id(target_doc_id)


def _lexical_overlap_ratio(a: str, b: str) -> float:
    """Compute simple token overlap for corpus-mismatch diagnostics."""
    toks_a = set(re.findall(r"[a-z0-9]+", a.lower()))
    toks_b = set(re.findall(r"[a-z0-9]+", b.lower()))
    if not toks_a or not toks_b:
        return 0.0
    inter = len(toks_a & toks_b)
    return inter / max(1, min(len(toks_a), len(toks_b)))


def _chunk_signature(chunk: dict) -> str:
    """Stable signature that disambiguates repeated chunk_id values across runs."""
    chunk_id = str(chunk.get("chunk_id", ""))
    text = str(chunk.get("text", ""))
    text_hash = hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]
    return f"{chunk_id}:{text_hash}"


def _print_stage_chunks(example: dict, chunks: list[dict]) -> list[dict]:
    doc_id = str(example.get("Evidence_Doc_ID", "")).strip()
    evidence_text = str(example.get("Evidence_Text", "")).strip()
    matches = [c for c in chunks if _contains_id(c, doc_id)]
    text_matches: list[dict] = []
    if not matches and evidence_text:
        text_matches = [
            c for c in chunks
            if _lexical_overlap_ratio(evidence_text, str(c.get("text", ""))) >= 0.25
        ]

    print(SUB_SEPARATOR)
    print("STAGE 1 - semantic_chunker output/chunks.json")
    print(SUB_SEPARATOR)
    if not chunks:
        print("chunks.json not found or unreadable.")
        print()
        return []

    print(f"Total chunks in file       : {len(chunks)}")
    print(f"Chunks containing doc_id   : {len(matches)}")
    if not matches and text_matches:
        print(f"Fallback lexical matches   : {len(text_matches)} (possible corpus/ID mismatch)")
    if not matches and not text_matches:
        print("No ID or lexical matches found; chunks.json may come from a different run/corpus.")
    for c in matches[:5]:
        snippet = textwrap.shorten(c.get("text", ""), width=180, placeholder=" ...")
        print(
            f"- {c.get('chunk_id')} | tokens={c.get('token_count')} | "
            f"topic_id={c.get('topic_id')} | docs={c.get('doc_ids')}"
        )
        print(f"  text: {snippet}")
    if not matches:
        for c in text_matches[:3]:
            snippet = textwrap.shorten(c.get("text", ""), width=180, placeholder=" ...")
            print(
                f"- lexical_hit {c.get('chunk_id')} | tokens={c.get('token_count')} | "
                f"topic_id={c.get('topic_id')} | docs={c.get('doc_ids', c.get('doc_id'))}"
            )
            print(f"  text: {snippet}")
    print()
    return matches or text_matches


def _print_stage_enriched(stage1_matches: list[dict], enriched: list[dict]) -> list[dict]:
    ids = {c.get("chunk_id") for c in stage1_matches}
    signatures = {_chunk_signature(c) for c in stage1_matches}
    matches = [
        c for c in enriched
        if c.get("chunk_id") in ids and _chunk_signature(c) in signatures
    ]

    print(SUB_SEPARATOR)
    print("STAGE 2 - granular_topic_modeler output/enriched_chunks.json")
    print(SUB_SEPARATOR)
    if not enriched:
        print("enriched_chunks.json not found or unreadable.")
        print()
        return []

    print(f"Total enriched chunks      : {len(enriched)}")
    print(f"Enriched matches by id     : {len(matches)}")
    for c in matches[:5]:
        keywords = ", ".join(c.get("keywords", [])[:8]) or "(none)"
        print(
            f"- {c.get('chunk_id')} | topic_label={c.get('topic_label', '')} | "
            f"confidence={c.get('confidence_score', 0.0):.4f}"
        )
        print(f"  keywords: {keywords}")
        print(f"  subtopics: {len(c.get('subtopics', []))}")
    print()
    return matches


def _print_stage_threaded(enriched_matches: list[dict], threaded: list[dict]) -> None:
    ids = {c.get("chunk_id") for c in enriched_matches}
    signatures = {_chunk_signature(c) for c in enriched_matches}
    matches = [
        c for c in threaded
        if c.get("chunk_id") in ids and _chunk_signature(c) in signatures
    ]

    print(SUB_SEPARATOR)
    print("STAGE 3/4 - generative_threading output/threaded_chunks.json")
    print(SUB_SEPARATOR)
    if not threaded:
        print("threaded_chunks.json not found or unreadable.")
        print()
        return

    print(f"Total threaded chunks      : {len(threaded)}")
    print(f"Threaded matches by id     : {len(matches)}")
    for c in matches[:3]:
        queries = c.get("synthetic_queries", [])
        clusters = c.get("query_clusters", [])
        print(
            f"- {c.get('chunk_id')} | synthetic_queries={len(queries)} | "
            f"clusters={len(clusters)} | optimal_k={c.get('optimal_k', 1)}"
        )
        for q in queries[:3]:
            print(f"  q[{q.get('typology', '')}]: {q.get('text', '')}")
    print()


def _print_retrieval(example: dict, top_k: int) -> None:
    print(SUB_SEPARATOR)
    print("STAGE 5 - retrieval comparison from Qdrant")
    print(SUB_SEPARATOR)

    claim = example.get("Claim_Text", "")
    if not claim:
        print("No claim text available, skipping retrieval.")
        print()
        return

    try:
        from qdrant_client import QdrantClient
        from sentence_transformers import SentenceTransformer

        from src.retrieval_engine import RetrievalEngine
        from src.semantic_chunker import EMBEDDING_MODEL_NAME
    except Exception as exc:
        print(f"Retrieval dependencies are unavailable: {exc}")
        print()
        return

    try:
        model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        client = QdrantClient(path=QDRANT_DB_PATH)
    except Exception as exc:
        print(f"Could not open embedding model or Qdrant DB: {exc}")
        print()
        return

    # Baseline retrieval
    try:
        qvec = model.encode(claim, convert_to_numpy=True).tolist()
        hits = client.query_points(
            collection_name=BASELINE_COLLECTION,
            query=qvec,
            limit=top_k,
        ).points
        print("Baseline top-k:")
        for i, hit in enumerate(hits, start=1):
            payload = hit.payload or {}
            did = payload.get("doc_id", "")
            cid = payload.get("chunk_id", "")
            print(f"  #{i} score={float(hit.score):.4f} doc_id={did} chunk_id={cid}")
    except Exception as exc:
        print(f"Baseline retrieval unavailable: {exc}")
    finally:
        # Important: local Qdrant path backend is single-writer/single-client.
        # Close baseline client before opening RetrievalEngine on same db path.
        client.close()

    # Enriched retrieval
    engine = None
    try:
        engine = RetrievalEngine(
            db_path=QDRANT_DB_PATH,
            collection_name=ENRICHED_COLLECTION,
        )
        results = engine.search(claim, top_k=top_k)
        print("Enriched top-k:")
        for r in results:
            print(
                f"  #{r.rank} score={r.score:.4f} chunk_id={r.chunk_id} "
                f"doc_ids={r.doc_ids} topic={r.topic_label}"
            )
    except Exception as exc:
        print(f"Enriched retrieval unavailable: {exc}")
    finally:
        if engine is not None and hasattr(engine, "client"):
            engine.client.close()

    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Trace one SciFact example across chunks, enrichment, threading, and retrieval.",
    )
    parser.add_argument("--query-id", type=str, default=None, help="Query_ID from scifact_qa_pairs.csv")
    parser.add_argument("--query-text", type=str, default=None, help="Exact Claim_Text from scifact_qa_pairs.csv")
    parser.add_argument("-k", "--top-k", type=int, default=3, help="Top-K retrieval results to print")
    args = parser.parse_args()

    example = _load_example(CSV_PATH, args.query_id, args.query_text)
    chunks = _safe_load_stage(CHUNKS_PATH)
    enriched = _safe_load_stage(ENRICHED_PATH)
    threaded = _safe_load_stage(THREADED_PATH)

    _print_header(example)
    stage1_matches = _print_stage_chunks(example, chunks)
    enriched_matches = _print_stage_enriched(stage1_matches, enriched)
    _print_stage_threaded(enriched_matches, threaded)
    _print_retrieval(example, top_k=args.top_k)


if __name__ == "__main__":
    main()
