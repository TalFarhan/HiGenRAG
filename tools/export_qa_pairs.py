"""
Export QA pairs from the BeIR SciFact test set to a flat CSV file.

Each row maps a single query to a single relevant document (one-to-many:
one query can appear on multiple rows if it has several relevant documents).
"""

import csv
import os
import ir_datasets


OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "full_data_dump")
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "scifact_qa_pairs.csv")

DATASET_ID = "beir/scifact/test"


def load_dataset():
    """Load the SciFact test split via its canonical name to avoid KeyError."""
    ds = ir_datasets.load(DATASET_ID)
    return ds


def build_corpus_lookup(dataset):
    """Return a dict mapping doc_id -> full document text."""
    corpus = {}
    for doc in dataset.docs_iter():
        corpus[doc.doc_id] = doc.text
    return corpus


def build_query_lookup(dataset):
    """Return a dict mapping query_id -> query text."""
    queries = {}
    for query in dataset.queries_iter():
        queries[query.query_id] = query.text
    return queries


def collect_qa_pairs(dataset, queries, corpus):
    """
    Iterate over qrels and yield one row per positive-relevance judgement.

    Rows contain: Query_ID, Claim_Text, Evidence_Doc_ID, Evidence_Text.
    """
    for qrel in dataset.qrels_iter():
        if qrel.relevance <= 0:
            continue

        query_text = queries.get(qrel.query_id, "")
        doc_text = corpus.get(qrel.doc_id, "")

        yield {
            "Query_ID": qrel.query_id,
            "Claim_Text": query_text,
            "Evidence_Doc_ID": qrel.doc_id,
            "Evidence_Text": doc_text,
        }


def export_to_csv(rows, path):
    """Write QA-pair rows to a UTF-8 CSV file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)

    fieldnames = ["Query_ID", "Claim_Text", "Evidence_Doc_ID", "Evidence_Text"]

    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        count = 0
        for row in rows:
            writer.writerow(row)
            count += 1

    return count


def main():
    print(f"Loading dataset: {DATASET_ID} ...")
    ds = load_dataset()

    print("Building corpus lookup ...")
    corpus = build_corpus_lookup(ds)
    print(f"  -> {len(corpus):,} documents indexed.")

    print("Building query lookup ...")
    queries = build_query_lookup(ds)
    print(f"  -> {len(queries):,} queries indexed.")

    print("Collecting positive-relevance QA pairs ...")
    rows = collect_qa_pairs(ds, queries, corpus)

    print(f"Writing CSV to {OUTPUT_PATH} ...")
    total = export_to_csv(rows, OUTPUT_PATH)
    print(f"Done. {total:,} QA-pair rows written.")


if __name__ == "__main__":
    main()
