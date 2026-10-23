
import json
import math
import csv
from pathlib import Path

from rank_bm25 import BM25Okapi


DATA_DIR = Path("data") / "scifact"
RESULTS_DIR = Path("results")
TOP_K = 10


def tokenize(text):
    """Simple, consistent lowercase whitespace tokenization."""
    return text.lower().split()


def load_jsonl(path):
    """Read a JSON Lines file into a list of dictionaries."""
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def recall_at_k(ranked_ids, relevant_ids, k):
    relevant = set(relevant_ids)
    if not relevant:
        return 0.0
    hits = len(set(ranked_ids[:k]) & relevant)
    return hits / len(relevant)


def reciprocal_rank(ranked_ids, relevant_ids):
    relevant = set(relevant_ids)
    for rank, doc_id in enumerate(ranked_ids, start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranked_ids, relevance_scores, k):
    """Compute nDCG using graded relevance scores."""
    def gain(score):
        return (2 ** score) - 1

    dcg = sum(
        gain(relevance_scores.get(doc_id, 0)) / math.log2(rank + 1)
        for rank, doc_id in enumerate(ranked_ids[:k], start=1)
    )

    ideal_scores = sorted(relevance_scores.values(), reverse=True)[:k]
    ideal_dcg = sum(
        gain(score) / math.log2(rank + 1)
        for rank, score in enumerate(ideal_scores, start=1)
    )

    return dcg / ideal_dcg if ideal_dcg > 0 else 0.0


def main():
    print("RAGBench — SciFact BM25 baseline")

    corpus_rows = load_jsonl(DATA_DIR / "corpus.jsonl")
    query_rows = load_jsonl(DATA_DIR / "queries.jsonl")
    qrels_rows = load_jsonl(DATA_DIR / "qrels_test.jsonl")

    documents = {
        row["id"]: f'{row["title"]} {row["text"]}'.strip()
        for row in corpus_rows
    }

    queries = {row["id"]: row["text"] for row in query_rows}

    qrels = {
        row["query_id"]: {
            doc_id: int(score)
            for doc_id, score in row["relevant_documents"].items()
            if int(score) > 0
        }
        for row in qrels_rows
    }

    # Only evaluate queries with at least one positive relevance label.
    evaluable_queries = {
        query_id: queries[query_id]
        for query_id in qrels
        if query_id in queries
        and any(score > 0 for score in qrels[query_id].values())
    }

    if not documents or not evaluable_queries:
        raise ValueError(
            "No documents or evaluable queries found. "
            "Check the files in data/scifact/."
        )

    doc_ids = list(documents.keys())
    tokenized_docs = [
        tokenize(documents[doc_id]) for doc_id in doc_ids
    ]

    print(f"Documents: {len(doc_ids)}")
    print(f"Queries to evaluate: {len(evaluable_queries)}")
    print("Building BM25 index...")

    bm25 = BM25Okapi(tokenized_docs)
    query_results = []

    for query_id, query_text in evaluable_queries.items():
        scores = bm25.get_scores(tokenize(query_text))

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda index: scores[index],
            reverse=True,
        )

        ranked_ids = [
            doc_ids[index] for index in ranked_indices[:TOP_K]
        ]

        relevance_scores = qrels[query_id]
        relevant_ids = {
            doc_id
            for doc_id, score in relevance_scores.items()
            if score > 0
        }

        query_results.append({
            "query_id": query_id,
            "query": query_text,
            "relevant_document_count": len(relevant_ids),
            "recall_at_3": recall_at_k(ranked_ids, relevant_ids, 3),
            "recall_at_10": recall_at_k(ranked_ids, relevant_ids, 10),
            "mrr": reciprocal_rank(ranked_ids, relevant_ids),
            "ndcg_at_10": ndcg_at_k(
                ranked_ids, relevance_scores, 10
            ),
            "top_1_document_id": ranked_ids[0] if ranked_ids else "",
            "top_1_is_relevant": (
                bool(ranked_ids) and ranked_ids[0] in relevant_ids
            ),
        })

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    query_file = RESULTS_DIR / "scifact_bm25_query_results.csv"
    with query_file.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file, fieldnames=list(query_results[0].keys())
        )
        writer.writeheader()
        writer.writerows(query_results)

    metric_names = [
        "recall_at_3",
        "recall_at_10",
        "mrr",
        "ndcg_at_10",
    ]

    summary = {
        "method": "BM25",
        "corpus_documents": len(doc_ids),
        "evaluated_queries": len(query_results),
    }

    for metric in metric_names:
        summary[metric] = sum(
            row[metric] for row in query_results
        ) / len(query_results)

    summary_file = RESULTS_DIR / "scifact_bm25_summary.csv"
    with summary_file.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file, fieldnames=list(summary.keys())
        )
        writer.writeheader()
        writer.writerow(summary)

    print("\nBM25 baseline results")
    print("-" * 36)
    for metric in metric_names:
        print(f"{metric}: {summary[metric]:.4f}")

    print(f"\nSaved per-query results: {query_file}")
    print(f"Saved summary: {summary_file}")


if __name__ == "__main__":
    main()
