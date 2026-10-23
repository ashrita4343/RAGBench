
import csv
import json
import math
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

DATA_DIR = Path("data") / "scifact"
CACHE_DIR = Path("cache")
RESULTS_DIR = Path("results")

TOP_K = 10
ALPHA = 0.75


def load_jsonl(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def save_csv(path, rows):
    if not rows:
        raise ValueError(f"No rows to save: {path}")
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def tokenize(text):
    return text.lower().split()


def normalize(scores):
    low = scores.min()
    high = scores.max()
    if high == low:
        return np.zeros_like(scores, dtype=float)
    return (scores - low) / (high - low)


def stable_rank(scores):
    return sorted(
        range(len(scores)),
        key=lambda index: scores[index],
        reverse=True,
    )


def recall_at_k(ranked_ids, relevant_ids, k):
    if not relevant_ids:
        return 0.0
    return len(set(ranked_ids[:k]) & relevant_ids) / len(relevant_ids)


def reciprocal_rank(ranked_ids, relevant_ids):
    for rank, doc_id in enumerate(ranked_ids, start=1):
        if doc_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranked_ids, relevance, k):
    def gain(score):
        return (2 ** score) - 1

    dcg = sum(
        gain(relevance.get(doc_id, 0)) / math.log2(rank + 1)
        for rank, doc_id in enumerate(ranked_ids[:k], start=1)
    )
    ideal = sorted(relevance.values(), reverse=True)[:k]
    ideal_dcg = sum(
        gain(score) / math.log2(rank + 1)
        for rank, score in enumerate(ideal, start=1)
    )
    return dcg / ideal_dcg if ideal_dcg else 0.0


def evaluate(ranked_ids, relevance):
    relevant_ids = {
        doc_id for doc_id, score in relevance.items() if score > 0
    }
    return {
        "recall_at_3": recall_at_k(ranked_ids, relevant_ids, 3),
        "recall_at_10": recall_at_k(ranked_ids, relevant_ids, 10),
        "mrr": reciprocal_rank(ranked_ids, relevant_ids),
        "ndcg_at_10": ndcg_at_k(ranked_ids, relevance, 10),
    }


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    corpus = load_jsonl(DATA_DIR / "corpus.jsonl")
    queries_rows = load_jsonl(DATA_DIR / "queries.jsonl")
    qrels_rows = load_jsonl(DATA_DIR / "qrels_final_test.jsonl")

    doc_ids = [str(row["id"]) for row in corpus]
    documents = [
        f'{row["title"]} {row["text"]}'.strip()
        for row in corpus
    ]
    queries = {
        str(row["id"]): row["text"]
        for row in queries_rows
    }

    doc_embeddings = np.load(
        CACHE_DIR / "scifact_document_embeddings.npy"
    )
    query_embeddings = np.load(
        CACHE_DIR / "scifact_query_embeddings.npy"
    )

    cached_doc_ids = json.loads(
        (CACHE_DIR / "scifact_document_ids.json").read_text(
            encoding="utf-8"
        )
    )
    cached_query_ids = json.loads(
        (CACHE_DIR / "scifact_query_ids.json").read_text(
            encoding="utf-8"
        )
    )

    if [str(value) for value in cached_doc_ids] != doc_ids:
        raise ValueError(
            "Cached document IDs do not match corpus order. "
            "Run python embed_cache.py again."
        )

    if len(doc_embeddings) != len(doc_ids):
        raise ValueError("Document embedding count mismatch.")

    if len(query_embeddings) != len(cached_query_ids):
        raise ValueError("Query embedding count mismatch.")

    query_embedding_by_id = {
        str(query_id): query_embeddings[index]
        for index, query_id in enumerate(cached_query_ids)
    }

    relevance_by_query = {
        str(row["query_id"]): {
            str(doc_id): int(score)
            for doc_id, score in row["relevant_documents"].items()
            if int(score) > 0
        }
        for row in qrels_rows
    }

    test_queries = {
        query_id: relevance
        for query_id, relevance in relevance_by_query.items()
        if query_id in queries
        and query_id in query_embedding_by_id
        and relevance
    }

    if not test_queries:
        raise ValueError("No usable final-test queries found.")

    print("Building BM25 index...")
    bm25 = BM25Okapi([tokenize(doc) for doc in documents])

    per_method = {
        "BM25": [],
        "Semantic": [],
        "Hybrid": [],
    }
    per_query_rows = []

    print(f"Evaluating {len(test_queries)} shared final-test queries...")
    print(f"Hybrid alpha: {ALPHA}")

    for query_id, relevance in test_queries.items():
        query_text = queries[query_id]

        bm25_scores = np.asarray(
            bm25.get_scores(tokenize(query_text)),
            dtype=float,
        )
        semantic_scores = (
            doc_embeddings @ query_embedding_by_id[query_id]
        )

        # BM25-only and semantic-only use their raw scores.
        rankings = {
            "BM25": [
                doc_ids[index]
                for index in stable_rank(bm25_scores)[:TOP_K]
            ],
            "Semantic": [
                doc_ids[index]
                for index in stable_rank(semantic_scores)[:TOP_K]
            ],
        }

        # Same score-fusion formula as scifact_hybrid.py.
        hybrid_scores = (
            (1.0 - ALPHA) * normalize(bm25_scores)
            + ALPHA * normalize(semantic_scores)
        )
        rankings["Hybrid"] = [
            doc_ids[index]
            for index in stable_rank(hybrid_scores)[:TOP_K]
        ]

        query_record = {
            "query_id": query_id,
            "query": query_text,
        }

        for method, ranked_ids in rankings.items():
            metrics = evaluate(ranked_ids, relevance)
            per_method[method].append(metrics)

            for metric, value in metrics.items():
                query_record[f"{method.lower()}_{metric}"] = value

        per_query_rows.append(query_record)

    summary_rows = []
    for method, metric_rows in per_method.items():
        row = {
            "method": method,
            "queries": len(metric_rows),
            "alpha": ALPHA if method == "Hybrid" else "",
        }
        for metric in (
            "recall_at_3",
            "recall_at_10",
            "mrr",
            "ndcg_at_10",
        ):
            row[metric] = float(np.mean([
                result[metric] for result in metric_rows
            ]))
        summary_rows.append(row)

    save_csv(
        RESULTS_DIR / "scifact_same_split_comparison.csv",
        summary_rows,
    )
    save_csv(
        RESULTS_DIR / "scifact_same_split_query_results.csv",
        per_query_rows,
    )

    print("\nSAME-SPLIT FINAL-TEST COMPARISON")
    print(f"Queries evaluated for every method: {len(test_queries)}")
    print(f"{'Method':<12} {'R@3':>8} {'R@10':>8} {'MRR':>8} {'nDCG@10':>10}")

    for row in summary_rows:
        print(
            f"{row['method']:<12} "
            f"{row['recall_at_3']:>8.4f} "
            f"{row['recall_at_10']:>8.4f} "
            f"{row['mrr']:>8.4f} "
            f"{row['ndcg_at_10']:>10.4f}"
        )

    print("\nSaved:")
    print("results/scifact_same_split_comparison.csv")
    print("results/scifact_same_split_query_results.csv")


if __name__ == "__main__":
    main()
