"""Quick EDA on enriched chunks produced by the granular topic modeler (Task 2)."""

import json
import random
import sys
import textwrap
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ENRICHED_PATH = (
    Path(__file__).resolve().parent.parent / "output" / "enriched_chunks.json"
)

SEP = "=" * 72
SUB_SEP = "-" * 72


def load_enriched_chunks(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def first_n_lines(text: str, n: int = 3) -> str:
    lines = [ln for ln in text.splitlines() if ln.strip()]
    return "\n".join(lines[:n])


def print_chunk_detail(idx: int, chunk: dict) -> None:
    print(f"\n  [{idx}]  Chunk ID: {chunk['chunk_id']}")
    print(SUB_SEP)

    keywords = chunk.get("keywords", [])
    print(f"  Keywords       : {', '.join(keywords) if keywords else '(none)'}")

    topic_label = chunk.get("topic_label")
    if topic_label:
        print(f"  Topic label    : {topic_label}")

    confidence = chunk.get("confidence_score")
    if confidence is not None:
        print(f"  Confidence     : {confidence:.4f}")

    subtopics = chunk.get("subtopics", [])
    if subtopics:
        print(f"  Sub-topics ({len(subtopics)}):")
        for st in subtopics:
            st_kw = ", ".join(st.get("keywords", [])[:5])
            st_conf = st.get("confidence")
            conf_str = f"  (conf {st_conf:.4f})" if st_conf is not None else ""
            print(f"    - [{st.get('topic_id', '?')}] {st.get('label', '')}"
                  f"{conf_str}")
            print(f"      top kw: {st_kw}")

    preview = first_n_lines(chunk.get("text", ""), n=3)
    wrapped = textwrap.fill(preview, width=80, initial_indent="    ",
                            subsequent_indent="    ")
    print(f"\n  Text preview:")
    print(wrapped)
    print()


def main() -> None:
    chunks = load_enriched_chunks(ENRICHED_PATH)

    print()
    print(SEP)
    print(f"  ENRICHED CHUNKS EDA  —  {ENRICHED_PATH.name}")
    print(SEP)
    print(f"  Total enriched chunks : {len(chunks)}")
    print(SEP)
    print()

    samples = random.sample(chunks, min(2, len(chunks)))
    print(SEP)
    print("  2 RANDOM ENRICHED CHUNKS")
    print(SEP)
    for i, chunk in enumerate(samples, 1):
        print_chunk_detail(i, chunk)

    print(SEP)
    print("  Done.")
    print(SEP)


if __name__ == "__main__":
    main()
