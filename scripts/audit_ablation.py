"""Quick audit of the on-disk ablation artefacts.

Reports raw chunk count, enriched chunk count, average / median tokens,
BIC-selected K, presence of anchor_queries / keywords in payloads, and
which Qdrant collections exist locally. Intended as a one-shot diagnostic
during the spec-compliance review.
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

METHODS = ("baseline_char", "sentence_window", "llm_boundary", "semantic_gmm")
COLLECTION_MAP = {
    "baseline_char": "baseline_char",
    "sentence_window": "baseline_sentence_window",
    "llm_boundary": "llm_chunks",
    "semantic_gmm": "enriched_chunks",
}


def _load_json(path: Path) -> list | dict | None:
    if not path.is_file():
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def main() -> None:
    print(f"{'Method':<18} {'Raw':>5} {'Enr':>5} {'K':>3} {'AvgTok':>7} {'MedTok':>7} {'anchors':>9} {'kw':>6}")
    print("-" * 72)
    for m in METHODS:
        raw = _load_json(Path(f"output/ablation/raw_chunks/{m}.json")) or []
        enr = _load_json(Path(f"output/ablation/enriched_chunks/{m}.json")) or []
        anchors = _load_json(Path(f"output/ablation/anchors/{m}.json")) or {}
        tokens = [int(c.get("token_count") or 0) for c in enr]
        avg = round(sum(tokens) / len(tokens), 1) if tokens else 0
        med = int(statistics.median(tokens)) if tokens else 0
        has_anchors = sum(1 for c in enr if c.get("anchor_queries"))
        has_kw = sum(1 for c in enr if c.get("keywords"))
        k = anchors.get("k", "-") if isinstance(anchors, dict) else "-"
        print(f"{m:<18} {len(raw):>5} {len(enr):>5} {k:>3} {avg:>7} {med:>7} {has_anchors:>9} {has_kw:>6}")


if __name__ == "__main__":
    main()
