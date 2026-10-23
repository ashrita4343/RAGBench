
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
PRIMARY_METRIC = "ndcg_at_10"


def load_jsonl(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def save_csv(path, rows):
    if not rows:
        raise ValueError(f"No rows to save: {path}")
    with path.open("w", newline="", encoding="utf-8") as file:
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
    # Stable sorting makes tied scores deterministic.
    return sorted(
        range(len(scores)),
        key=lambda index: scores[index],
        reverse=True,
    )


def recall_at_k(ranked_ids, relevant_ids, k):
    relevant_ids = set(relevant_ids)
    if not relevant_ids:
        return 0.0
    return len(set(ranked_ids[:k]) & relevant_ids) / len(relevant_ids)


def reciprocal_rank(ranked_ids, relevant_ids):
    relevant_ids = set(relevant_ids)
    for rank, doc_id in enumerate(ranked_ids, start=1):
        if doc_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranked_ids, relevance_scores, k):
    def gain(score):
        return (2 ** score) - 1

    dcg = sum(
        gain(relevance_scores.get(doc_id, 0))
        / math.log2(rank + 1)
        for rank, doc_id in enumerate(ranked_ids[:k], start=1)
    )

    ideal_scores = sorted(relevance_scores.values(), reverse=True)[:k]
    ideal_dcg = sum(
        gain(score) / math.log2(rank + 1)
        for rank, score in enumerate(ideal_scores, start=1)
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


def load_qrels(filename):
    rows = load_jsonl(DATA_DIR / filename)
    return {
        str(row["query_id"]): {
            str(doc_id): int(score)
            for doc_id, score in row["relevant_documents"].items()
            if int(score) > 0
        }
        for row in rows
    }


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    corpus = load_jsonl(DATA_DIR / "corpus.jsonl")
    query_rows = load_jsonl(DATA_DIR / "queries.jsonl")

    doc_ids = [str(row["id"]) for row in corpus]
    documents = [
        f'{row["title"]} {row["text"]}'.strip()
        for row in corpus
    ]
    queries = {str(row["id"]): row["text"] for row in query_rows}

    # Load cached embeddings and verify document order.
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

    if [str(x) for x in cached_doc_ids] != doc_ids:
        raise ValueError(
            "Cached document IDs do not match the corpus. "
            "Run python embed_cache.py again."
        )

    if len(doc_embeddings) != len(doc_ids):
        raise ValueError("Document embedding count mismatch.")

    if len(query_embeddings) != len(cached_query_ids):
        raise ValueError("Query embedding count mismatch.")

    query_embedding_by_id = {
        str(qid): query_embeddings[i]
        for i, qid in enumerate(cached_query_ids)
    }

    validation_qrels = load_qrels("qrels_validation.jsonl")
    test_qrels = load_qrels("qrels_final_test.jsonl")

    def usable_qrels(qrels):
        return {
            qid: relevance
            for qid, relevance in qrels.items()
            if qid in queries
            and qid in query_embedding_by_id
            and relevance
        }

    validation_qrels = usable_qrels(validation_qrels)
    test_qrels = usable_qrels(test_qrels)

    if not validation_qrels or not test_qrels:
        raise ValueError(
            "Validation or final-test queries are missing. "
            "Run python prepare_splits.py first."
        )

    overlap = set(validation_qrels) & set(test_qrels)
    if overlap:
        raise ValueError("Validation and test query IDs overlap.")

    print("RAGBench — Validation-based alpha tuning")
    print(f"Documents: {len(doc_ids)}")
    print(f"Validation queries: {len(validation_qrels)}")
    print(f"Final-test queries: {len(test_qrels)}")
    print(f"Alpha candidates: {ALPHAS}")

    print("\nBuilding BM25 index...")
    bm25 = BM25Okapi([tokenize(doc) for doc in documents])

    def score_query(query_id):
        query_text = queries[query_id]
        bm25_scores = np.asarray(
            bm25.get_scores(tokenize(query_text)),
            dtype=float,
        )
        semantic_scores = (
            doc_embeddings @ query_embedding_by_id[query_id]
        )
        return bm25_scores, semantic_scores

    def rank_for_alpha(bm25_scores, semantic_scores, alpha):
        if alpha == 0.0:
            scores = bm25_scores
        elif alpha == 1.0:
            scores = semantic_scores
        else:
            scores = (
                (1.0 - alpha) * normalize(bm25_scores)
                + alpha * normalize(semantic_scores)
            )

        indices = stable_rank(scores)
        return [doc_ids[i] for i in indices[:TOP_K]]

    # --------------------------------------------------
    # Phase 1: tune alpha on validation queries only.
    # --------------------------------------------------
    validation_rows = []

    print("\nVALIDATION RESULTS")

    for alpha in ALPHAS:
        per_query_metrics = []

        for query_id, relevance in validation_qrels.items():
            bm25_scores, semantic_scores = score_query(query_id)
            ranked_ids = rank_for_alpha(
                bm25_scores, semantic_scores, alpha
            )
            per_query_metrics.append(evaluate(ranked_ids, relevance))

        row = {
            "alpha": alpha,
            "queries": len(per_query_metrics),
        }
        for metric in (
            "recall_at_3",
            "recall_at_10",
            "mrr",
            "ndcg_at_10",
        ):
            row[metric] = float(np.mean([
                item[metric] for item in per_query_metrics
            ]))

        validation_rows.append(row)
        print(
            f"alpha={alpha:.2f} | "
            f"Recall@10={row['recall_at_10']:.4f} | "
            f"MRR={row['mrr']:.4f} | "
            f"nDCG@10={row['ndcg_at_10']:.4f}"
        )

    # Select by the predeclared primary metric.
    # If tied, prefer the smaller alpha.
    best_row = max(
        validation_rows,
        key=lambda row: (row[PRIMARY_METRIC], -row["alpha"]),
    )
    best_alpha = best_row["alpha"]

    save_csv(
        RESULTS_DIR / "scifact_validation_alpha_summary.csv",
        validation_rows,
    )

    print(f"\nSelected alpha from validation: {best_alpha:.2f}")
    print(f"Selection metric: {PRIMARY_METRIC}")

    # --------------------------------------------------
    # Phase 2: evaluate the selected alpha on final test.
    # Do not select alpha using these results.
    # --------------------------------------------------
    test_metrics = []
    test_query_rows = []

    for query_id, relevance in test_qrels.items():
        bm25_scores, semantic_scores = score_query(query_id)
        ranked_ids = rank_for_alpha(
            bm25_scores, semantic_scores, best_alpha
        )
        metrics = evaluate(ranked_ids, relevance)
        test_metrics.append(metrics)

        test_query_rows.append({
            "query_id": query_id,
            "alpha_selected_on_validation": best_alpha,
            **metrics,
        })

    test_summary = {
        "selected_alpha": best_alpha,
        "selection_metric": PRIMARY_METRIC,
        "evaluated_queries": len(test_metrics),
    }

    for metric in (
        "recall_at_3",
        "recall_at_10",
        "mrr",
        "ndcg_at_10",
    ):
        test_summary[metric] = float(np.mean([
            row[metric] for row in test_metrics
        ]))

    save_csv(
        RESULTS_DIR / "scifact_final_test_summary.csv",
        [test_summary],
    )
    save_csv(
        RESULTS_DIR / "scifact_final_test_query_results.csv",
        test_query_rows,
    )

    print("\nFINAL-TEST RESULTS")
    print(f"Selected alpha: {best_alpha:.2f}")
    print(f"Queries: {len(test_metrics)}")
    for metric in (
        "recall_at_3",
        "recall_at_10",
        "mrr",
        "ndcg_at_10",
    ):
        print(f"{metric}: {test_summary[metric]:.4f}")

    print("\nSaved files:")
    print("results/scifact_validation_alpha_summary.csv")
    print("results/scifact_final_test_summary.csv")
    print("results/scifact_final_test_query_results.csv")
    print("\nExperiment complete.")


if __name__ == "__main__":
    main()
