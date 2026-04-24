"""Quick Exploratory Data Analysis on the semantic chunking output."""

import json
import random
import statistics
import sys
import textwrap
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

CHUNKS_PATH = Path(__file__).resolve().parent.parent / "output" / "chunks.json"

SEPARATOR = "=" * 72
SUB_SEPARATOR = "-" * 72


def load_chunks(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def print_token_stats(chunks: list[dict]) -> None:
    token_counts = [c["token_count"] for c in chunks]
    min_t = min(token_counts)
    max_t = max(token_counts)
    mean_t = statistics.mean(token_counts)
    median_t = statistics.median(token_counts)
    stdev_t = statistics.stdev(token_counts) if len(token_counts) > 1 else 0.0

    in_range = sum(1 for t in token_counts if 256 <= t <= 512)
    pct_in_range = in_range / len(token_counts) * 100

    print(SEPARATOR)
    print("  TOKEN-COUNT STATISTICS")
    print(SEPARATOR)
    print(f"  Min tokens       : {min_t}")
    print(f"  Max tokens       : {max_t}")
    print(f"  Mean tokens      : {mean_t:.1f}")
    print(f"  Median tokens    : {median_t:.1f}")
    print(f"  Std-dev          : {stdev_t:.1f}")
    print(f"  In [256-512]     : {in_range}/{len(token_counts)}  ({pct_in_range:.1f}%)")
    print()


def print_topic_distribution(chunks: list[dict]) -> None:
    topic_counts: dict[int, int] = {}
    for c in chunks:
        tid = c["topic_id"]
        topic_counts[tid] = topic_counts.get(tid, 0) + 1

    print(SEPARATOR)
    print("  TOPIC DISTRIBUTION")
    print(SEPARATOR)
    for tid in sorted(topic_counts):
        cnt = topic_counts[tid]
        bar = "#" * cnt
        print(f"  Topic {tid:>3d} : {cnt:>4d}  {bar}")
    print()


def print_sample_chunks(chunks: list[dict], n: int = 2) -> None:
    samples = random.sample(chunks, min(n, len(chunks)))

    print(SEPARATOR)
    print(f"  {n} RANDOM SAMPLE CHUNKS")
    print(SEPARATOR)
    for i, chunk in enumerate(samples, 1):
        wrapped_text = textwrap.fill(chunk["text"], width=80)
        print(f"\n  [{i}]  {chunk['chunk_id']}")
        print(SUB_SEPARATOR)
        print(f"  Topic ID     : {chunk['topic_id']}")
        print(f"  Token count  : {chunk['token_count']}")
        print(f"  Doc IDs      : {', '.join(chunk['doc_ids'])}")
        print(f"  Keywords     : {', '.join(chunk['keywords']) or '(none)'}")
        print()
        print(wrapped_text)
        print()


def main() -> None:
    chunks = load_chunks(CHUNKS_PATH)

    print()
    print(SEPARATOR)
    print(f"  CHUNKS EDA  —  {CHUNKS_PATH.name}")
    print(f"  Total chunks : {len(chunks)}")
    print(SEPARATOR)
    print()

    print_token_stats(chunks)
    print_topic_distribution(chunks)
    print_sample_chunks(chunks, n=2)


if __name__ == "__main__":
    main()
