
import csv
import json
import math
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

DATA_DIR = Path("data") / "scifact"
CACHE_DIR = Path("cache")
RESULTS_DIR = Path("results")

ALPHAS = (0.0, 0.25, 0.50, 0.75, 1.0)
TOP_K = 10


def load_jsonl(path):
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def tokenize(text):
    return text.lower().split()


def reciprocal_rank(ranked_ids, relevant_ids):
    relevant_ids = set(relevant_ids)
    for rank, doc_id in enumerate(ranked_ids, start=1):
        if doc_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def recall_at_k(ranked_ids, relevant_ids, k):
    relevant_ids = set(relevant_ids)
    if not relevant_ids:
        return 0.0
    return len(set(ranked_ids[:k]) & relevant_ids) / len(relevant_ids)


def ndcg_at_k(ranked_ids, relevance_scores, k):
    def gain(score):
        return (2 ** score) - 1

    dcg = sum(
        gain(relevance_scores.get(doc_id, 0))
        / math.log2(rank + 1)
        for rank, doc_id in enumerate(ranked_ids[:k], start=1)
    )

    ideal = sorted(relevance_scores.values(), reverse=True)[:k]
    ideal_dcg = sum(
        gain(score) / math.log2(rank + 1)
        for rank, score in enumerate(ideal, start=1)
    )

    return dcg / ideal_dcg if ideal_dcg else 0.0


def stable_rank(scores):
    return sorted(
        range(len(scores)),
        key=lambda index: scores[index],
        reverse=True,
    )


def normalize(scores):
    low = scores.min()
    high = scores.max()

    if high == low:
        return np.zeros_like(scores, dtype=float)

    return (scores - low) / (high - low)


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    corpus = load_jsonl(DATA_DIR / "corpus.jsonl")
    queries_rows = load_jsonl(DATA_DIR / "queries.jsonl")
    qrels_rows = load_jsonl(DATA_DIR / "qrels_test.jsonl")

    doc_ids = [str(row["id"]) for row in corpus]
    documents = [
        f'{row["title"]} {row["text"]}'.strip()
        for row in corpus
    ]

    queries = {
        str(row["id"]): row["text"]
        for row in queries_rows
    }

    # Load cached embeddings and their matching IDs.
    doc_embeddings = np.load(
        CACHE_DIR / "scifact_document_embeddings.npy"
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
    query_embeddings = np.load(
        CACHE_DIR / "scifact_query_embeddings.npy"
    )

    # Guard against accidentally using stale or misaligned caches.
    if [str(x) for x in cached_doc_ids] != doc_ids:
        raise ValueError(
            "Cached document IDs do not match the current corpus. "
            "Run python embed_cache.py again."
        )

    if len(cached_query_ids) != len(query_embeddings):
        raise ValueError("Cached query IDs and embeddings do not match.")

    query_embedding_by_id = {
        str(query_id): query_embeddings[index]
        for index, query_id in enumerate(cached_query_ids)
    }

    qrels = {
        str(row["query_id"]): {
            str(doc_id): int(score)
            for doc_id, score in row["relevant_documents"].items()
            if int(score) > 0
        }
        for row in qrels_rows
    }

    evaluable = {
        query_id: relevance
        for query_id, relevance in qrels.items()
        if query_id in queries
        and query_id in query_embedding_by_id
        and relevance
    }

    if len(doc_ids) != len(doc_embeddings):
        raise ValueError(
            "Cached document embeddings do not match the corpus. "
            "Run python embed_cache.py again."
        )

    print("RAGBench — Cached SciFact hybrid experiment")
    print(f"Documents: {len(doc_ids)}")
    print(f"Queries: {len(evaluable)}")
    print(f"Alpha values: {ALPHAS}")

    print("\nBuilding BM25 index...")
    bm25 = BM25Okapi([tokenize(doc) for doc in documents])

    summary_rows = []
    query_result_rows = []

    for alpha in ALPHAS:
        print(f"\nEvaluating alpha={alpha:.2f}...")
        metric_rows = []

        for query_id, relevance in evaluable.items():
            query_text = queries[query_id]

            bm25_scores = np.asarray(
                bm25.get_scores(tokenize(query_text)),
                dtype=float,
            )
            semantic_scores = (
                doc_embeddings @ query_embedding_by_id[query_id]
            )

            if alpha == 0.0:
                scores = bm25_scores
            elif alpha == 1.0:
                scores = semantic_scores
            else:
                scores = (
                    (1.0 - alpha) * normalize(bm25_scores)
                    + alpha * normalize(semantic_scores)
                )

            ranked_indices = stable_rank(scores)
            ranked_ids = [
                doc_ids[index] for index in ranked_indices[:TOP_K]
            ]

            relevant_ids = set(relevance)
            metrics = {
                "recall_at_3": recall_at_k(ranked_ids, relevant_ids, 3),
                "recall_at_10": recall_at_k(ranked_ids, relevant_ids, 10),
                "mrr": reciprocal_rank(ranked_ids, relevant_ids),
                "ndcg_at_10": ndcg_at_k(ranked_ids, relevance, 10),
            }
            metric_rows.append(metrics)

            query_result_rows.append({
                "alpha": alpha,
                "query_id": query_id,
                "query": query_text,
                "top_1_document": ranked_ids[0] if ranked_ids else "",
                "top_3_documents": "|".join(ranked_ids[:3]),
                **metrics,
            })

        summary = {
            "alpha": alpha,
            "queries": len(metric_rows),
        }

        for metric in (
            "recall_at_3", "recall_at_10", "mrr", "ndcg_at_10"
        ):
            summary[metric] = float(
                np.mean([row[metric] for row in metric_rows])
            )

        summary_rows.append(summary)

        print(
            f"Recall@3={summary['recall_at_3']:.4f} | "
            f"Recall@10={summary['recall_at_10']:.4f} | "
            f"MRR={summary['mrr']:.4f} | "
            f"nDCG@10={summary['ndcg_at_10']:.4f}"
        )

    def save_csv(path, rows):
        with path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    save_csv(
        RESULTS_DIR / "scifact_hybrid_alpha_summary.csv",
        summary_rows,
    )
    save_csv(
        RESULTS_DIR / "scifact_hybrid_query_results.csv",
        query_result_rows,
    )

    print("\nExperiment complete.")
    print("Saved results/scifact_hybrid_alpha_summary.csv")
    print("Saved results/scifact_hybrid_query_results.csv")
    print("Cached embeddings were reused; no model encoding was needed.")


if __name__ == "__main__":
    main()
