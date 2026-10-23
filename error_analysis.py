
import csv
import json
import math
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

DATA_DIR = Path("data") / "scifact"
CACHE_DIR = Path("cache")
RESULTS_DIR = Path("results")

ALPHA = 0.75
TOP_K = 10
TEXT_PREVIEW_LENGTH = 300


def load_jsonl(path):
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def tokenize(text):
    return text.lower().split()


def normalize(scores):
    low = scores.min()
    high = scores.max()
    if high == low:
        return np.zeros_like(scores, dtype=float)
    return (scores - low) / (high - low)


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


def preview(text):
    text = " ".join(str(text).split())
    if len(text) > TEXT_PREVIEW_LENGTH:
        return text[:TEXT_PREVIEW_LENGTH] + "..."
    return text


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

    doc_by_id = {
        str(row["id"]): row
        for row in corpus
    }

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

    if [str(x) for x in cached_doc_ids] != doc_ids:
        raise ValueError(
            "Cached document IDs do not match the corpus. "
            "Run python embed_cache.py again."
        )

    if len(doc_embeddings) != len(doc_ids):
        raise ValueError(
            "Document embedding count does not match corpus."
        )

    if len(query_embeddings) != len(cached_query_ids):
        raise ValueError(
            "Query embedding count does not match cached query IDs."
        )

    query_embedding_by_id = {
        str(qid): query_embeddings[index]
        for index, qid in enumerate(cached_query_ids)
    }

    qrels = {
        str(row["query_id"]): {
            str(doc_id): int(score)
            for doc_id, score in row["relevant_documents"].items()
            if int(score) > 0
        }
        for row in qrels_rows
    }

    test_queries = {
        qid: relevance
        for qid, relevance in qrels.items()
        if qid in queries
        and qid in query_embedding_by_id
        and relevance
    }

    print("Building BM25 index...")
    bm25 = BM25Okapi([tokenize(doc) for doc in documents])

    successes = []
    failures = []
    metric_rows = []

    print(f"Analyzing {len(test_queries)} final-test queries...")
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

        scores = (
            (1.0 - ALPHA) * normalize(bm25_scores)
            + ALPHA * normalize(semantic_scores)
        )

        ranking = sorted(
            range(len(scores)),
            key=lambda index: scores[index],
            reverse=True,
        )

        ranked_ids = [doc_ids[index] for index in ranking[:TOP_K]]
        relevant_ids = set(relevance)

        # Record the rank of every relevant document, if retrieved.
        relevant_ranks = []
        for relevant_id in sorted(relevant_ids):
            if relevant_id in ranked_ids:
                relevant_ranks.append(
                    f"{relevant_id}:rank{ranked_ids.index(relevant_id) + 1}"
                )
            else:
                relevant_ranks.append(f"{relevant_id}:not_in_top_{TOP_K}")

        # Include titles and text previews for the retrieved documents.
        retrieved_details = []
        for rank, doc_id in enumerate(ranked_ids, start=1):
            document = doc_by_id[doc_id]
            retrieved_details.append(
                f"Rank {rank} | ID: {doc_id} | "
                f"Title: {document.get('title', '')} | "
                f"Text: {preview(document.get('text', ''))}"
            )

        relevant_details = []
        for relevant_id in sorted(relevant_ids):
            document = doc_by_id.get(relevant_id, {})
            relevant_details.append(
                f"ID: {relevant_id} | "
                f"Title: {document.get('title', '[title unavailable]')} | "
                f"Text: {preview(document.get('text', '[text unavailable]'))}"
            )

        hit_at_1 = bool(ranked_ids and ranked_ids[0] in relevant_ids)
        hit_at_3 = bool(set(ranked_ids[:3]) & relevant_ids)

        metrics = {
            "recall_at_3": recall_at_k(ranked_ids, relevant_ids, 3),
            "recall_at_10": recall_at_k(ranked_ids, relevant_ids, 10),
            "mrr": reciprocal_rank(ranked_ids, relevant_ids),
            "ndcg_at_10": ndcg_at_k(ranked_ids, relevance, 10),
        }
        metric_rows.append(metrics)

        record = {
            "query_id": query_id,
            "query": query_text,
            "relevant_document_ids": "|".join(sorted(relevant_ids)),
            "relevant_document_ranks": " || ".join(relevant_ranks),
            "relevant_document_details": " || ".join(relevant_details),
            "top_1_document_id": ranked_ids[0] if ranked_ids else "",
            "top_1_document_title": (
                doc_by_id[ranked_ids[0]].get("title", "")
                if ranked_ids else ""
            ),
            "top_3_document_ids": "|".join(ranked_ids[:3]),
            "top_3_document_titles": " || ".join(
                doc_by_id[doc_id].get("title", "")
                for doc_id in ranked_ids[:3]
            ),
            "retrieved_top_10_details": " || ".join(retrieved_details),
            "hit_at_1": hit_at_1,
            "hit_at_3": hit_at_3,
            **metrics,
        }

        if hit_at_3:
            successes.append(record)
        else:
            failures.append(record)

    fields = [
        "query_id",
        "query",
        "relevant_document_ids",
        "relevant_document_ranks",
        "relevant_document_details",
        "top_1_document_id",
        "top_1_document_title",
        "top_3_document_ids",
        "top_3_document_titles",
        "retrieved_top_10_details",
        "hit_at_1",
        "hit_at_3",
        "recall_at_3",
        "recall_at_10",
        "mrr",
        "ndcg_at_10",
    ]

    def save_csv(path, rows):
        with path.open("w", newline="", encoding="utf-8-sig") as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    save_csv(RESULTS_DIR / "error_analysis_failures.csv", failures)
    save_csv(RESULTS_DIR / "error_analysis_successes.csv", successes)

    print("\nError analysis summary")
    print(f"Queries analyzed: {len(test_queries)}")
    print(f"Top-3 successes: {len(successes)}")
    print(f"Top-3 misses: {len(failures)}")

    if metric_rows:
        print("\nMetrics recomputed by this script:")
        for metric in (
            "recall_at_3",
            "recall_at_10",
            "mrr",
            "ndcg_at_10",
        ):
            print(
                f"{metric}: "
                f"{np.mean([row[metric] for row in metric_rows]):.4f}"
            )

    print("\nSample failures:")
    for record in failures[:5]:
        print(f"\nQuery {record['query_id']}: {record['query']}")
        print(f"Relevant documents: {record['relevant_document_details']}")
        print(f"Relevant document ranks: {record['relevant_document_ranks']}")
        print(f"Retrieved top 3: {record['top_3_document_titles']}")

    print("\nSaved:")
    print("results/error_analysis_failures.csv")
    print("results/error_analysis_successes.csv")


if __name__ == "__main__":
    main()
