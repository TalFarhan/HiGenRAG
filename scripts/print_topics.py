"""Print global GMM topics (anchors + per-topic keywords from chunks) for sanity checks.

Optional: summarize granular BERTopic primary labels from enriched_chunks.json.

Usage:
  python scripts/print_topics.py
  python scripts/print_topics.py --anchors output/global_topic_anchors.json --chunks output/chunks.json
  python scripts/print_topics.py --enriched output/enriched_chunks.json --max-subtopics 3
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
from collections import Counter, defaultdict
from pathlib import Path


def _stdout_utf8() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def load_json(path: Path) -> dict | list:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def print_global_topics(anchors_path: Path, chunks_path: Path, preview_sentences: int) -> None:
    anchors = load_json(anchors_path)
    chunks: list[dict] = load_json(chunks_path)

    k = int(anchors.get("k", 0))
    topics = anchors.get("topics") or []
    print("=" * 80)
    print("GLOBAL GMM TOPICS (global_topic_anchors.json)")
    print("=" * 80)
    print(f"embedding_model : {anchors.get('embedding_model', '')}")
    print(f"k (topics)      : {k}")
    print(f"topics[] length : {len(topics)}")
    print()

    tid_to_keywords: dict[int, list[str]] = {}
    tid_counts: Counter[int] = Counter()
    for c in chunks:
        tid = int(c.get("topic_id", -1))
        if tid < 0:
            continue
        tid_counts[tid] += 1
        if tid not in tid_to_keywords and c.get("keywords"):
            tid_to_keywords[tid] = list(c["keywords"])

    for row in sorted(topics, key=lambda r: int(r.get("topic_id", -1))):
        tid = int(row.get("topic_id", -1))
        cores = row.get("core_sentences") or []
        n_chunks = tid_counts.get(tid, 0)
        kws = tid_to_keywords.get(tid, [])
        print("-" * 80)
        print(f"topic_id={tid}   chunks={n_chunks}")
        print(f"  keywords (from chunks.json, same for all chunks with this topic_id): {kws}")
        print(f"  core_sentences ({len(cores)} total, show first {preview_sentences}):")
        for s in cores[:preview_sentences]:
            line = (s or "").replace("\n", " ").strip()
            print(textwrap.fill(f"    • {line}", width=100, subsequent_indent="      "))
    print("-" * 80)
    print()
    orphan_chunks = sum(1 for c in chunks if int(c.get("topic_id", -1)) not in {int(t.get("topic_id")) for t in topics})
    anchor_ids = {int(t.get("topic_id")) for t in topics}
    missing_in_chunks = sorted(anchor_ids - set(tid_counts.elements()))
    if missing_in_chunks:
        print(f"Note: no chunks for topic_ids (only in anchors): {missing_in_chunks}")
    if orphan_chunks:
        print(f"Note: chunks whose topic_id not in anchors list: {orphan_chunks} (check data consistency)")


def print_enriched_summary(enriched_path: Path, max_subtopics: int) -> None:
    data: list[dict] = load_json(enriched_path)
    print("=" * 80)
    print("GRANULAR BERTopic (enriched_chunks.json) — primary topic_id + label")
    print("=" * 80)
    print(f"total chunks: {len(data)}")
    by_primary: dict[int, list[str]] = defaultdict(list)
    counts: Counter[int] = Counter()
    for c in data:
        tid = int(c.get("topic_id", -1))
        counts[tid] += 1
        label = (c.get("topic_label") or "").strip()
        if label and label not in by_primary[tid]:
            by_primary[tid].append(label)

    for tid, n in counts.most_common():
        labels = by_primary.get(tid, [])
        label_preview = labels[0] if labels else "(no label)"
        print("-" * 80)
        print(f"topic_id={tid}   count={n}   topic_label sample: {label_preview}")
        print(f"  chunk keywords (per-chunk TF-IDF, first chunk in this group):")
        ex = next((c for c in data if int(c.get("topic_id", -1)) == tid), None)
        if ex:
            print(f"    {ex.get('keywords', [])}")
            subs = ex.get("subtopics") or []
            print(f"  subtopics (top {max_subtopics} by confidence):")
            for st in subs[:max_subtopics]:
                print(
                    f"    id={st.get('topic_id')} conf={st.get('confidence')} "
                    f"label={st.get('label')} kw={st.get('keywords')}"
                )
    print("-" * 80)
    print()


def main() -> None:
    _stdout_utf8()
    root = Path(__file__).resolve().parent.parent
    p = argparse.ArgumentParser(description="Print topic structure for pipeline sanity checks.")
    p.add_argument(
        "--anchors",
        type=Path,
        default=root / "output" / "global_topic_anchors.json",
        help="Path to global_topic_anchors.json",
    )
    p.add_argument(
        "--chunks",
        type=Path,
        default=root / "output" / "chunks.json",
        help="Path to chunks.json",
    )
    p.add_argument(
        "--enriched",
        type=Path,
        default=None,
        help="If set, also print granular BERTopic summary from this enriched_chunks.json",
    )
    p.add_argument(
        "--preview-sentences",
        type=int,
        default=3,
        help="How many core_sentences to print per global topic",
    )
    p.add_argument(
        "--max-subtopics",
        type=int,
        default=3,
        help="How many subtopics to show per granular topic group",
    )
    args = p.parse_args()

    if not args.anchors.is_file():
        sys.exit(f"Missing anchors file: {args.anchors}")
    if not args.chunks.is_file():
        sys.exit(f"Missing chunks file: {args.chunks}")

    print_global_topics(args.anchors, args.chunks, args.preview_sentences)

    if args.enriched is not None:
        if not args.enriched.is_file():
            sys.exit(f"Missing enriched file: {args.enriched}")
        print_enriched_summary(args.enriched, args.max_subtopics)


if __name__ == "__main__":
    main()
