
import json
import math
import csv
import time
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


DATA_DIR = Path("data") / "scifact"
RESULTS_DIR = Path("results")
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 10
BATCH_SIZE = 32


def load_jsonl(path):
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def recall_at_k(ranked_ids, relevant_ids, k):
    relevant = set(relevant_ids)
    if not relevant:
        return 0.0
    return len(set(ranked_ids[:k]) & relevant) / len(relevant)


def reciprocal_rank(ranked_ids, relevant_ids):
    relevant = set(relevant_ids)
    for rank, doc_id in enumerate(ranked_ids, start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranked_ids, relevance_scores, k):
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
    print("RAGBench — SciFact semantic retrieval baseline")

    corpus_rows = load_jsonl(DATA_DIR / "corpus.jsonl")
    query_rows = load_jsonl(DATA_DIR / "queries.jsonl")
    qrels_rows = load_jsonl(DATA_DIR / "qrels_test.jsonl")

    documents = {
        row["id"]: f'{row["title"]}. {row["text"]}'.strip()
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

    evaluable_queries = {
        query_id: queries[query_id]
        for query_id in qrels
        if query_id in queries and qrels[query_id]
    }

    if not documents or not evaluable_queries:
        raise ValueError("Dataset is empty or no queries can be evaluated.")

    doc_ids = list(documents.keys())
    doc_texts = [documents[doc_id] for doc_id in doc_ids]
    query_ids = list(evaluable_queries.keys())
    query_texts = [evaluable_queries[qid] for qid in query_ids]

    print(f"Documents: {len(doc_texts)}")
    print(f"Queries to evaluate: {len(query_texts)}")
    print(f"Embedding model: {MODEL_NAME}")
    print("Loading model and encoding documents...")

    model = SentenceTransformer(MODEL_NAME)

    start = time.perf_counter()
    doc_embeddings = model.encode(
        doc_texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    document_encoding_seconds = time.perf_counter() - start

    print("Encoding queries...")
    start = time.perf_counter()
    query_embeddings = model.encode(
        query_texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    query_encoding_seconds = time.perf_counter() - start

    print("Ranking documents for each query...")
    start = time.perf_counter()

    query_rows_out = []

    for i, query_id in enumerate(query_ids):
        # Normalized embeddings make dot product equal cosine similarity.
        scores = doc_embeddings @ query_embeddings[i]
        ranked_indices = np.argsort(-scores)[:TOP_K]
        ranked_ids = [doc_ids[index] for index in ranked_indices]

        relevance_scores = qrels[query_id]
        relevant_ids = set(relevance_scores)

        query_rows_out.append({
            "query_id": query_id,
            "query": evaluable_queries[query_id],
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

    ranking_seconds = time.perf_counter() - start

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    query_file = RESULTS_DIR / "scifact_semantic_query_results.csv"
    with query_file.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file, fieldnames=list(query_rows_out[0].keys())
        )
        writer.writeheader()
        writer.writerows(query_rows_out)

    metric_names = [
        "recall_at_3",
        "recall_at_10",
        "mrr",
        "ndcg_at_10",
    ]

    summary = {
        "method": "Semantic",
        "model": MODEL_NAME,
        "corpus_documents": len(doc_ids),
        "evaluated_queries": len(query_rows_out),
        "document_encoding_seconds": round(document_encoding_seconds, 3),
        "query_encoding_seconds": round(query_encoding_seconds, 3),
        "ranking_seconds": round(ranking_seconds, 3),
    }

    for metric in metric_names:
        summary[metric] = sum(
            row[metric] for row in query_rows_out
        ) / len(query_rows_out)

    summary_file = RESULTS_DIR / "scifact_semantic_summary.csv"
    with summary_file.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)

    print("\nSemantic retrieval results")
    print("-" * 36)
    for metric in metric_names:
        print(f"{metric}: {summary[metric]:.4f}")

    print(f"\nDocument encoding time: {document_encoding_seconds:.2f}s")
    print(f"Query encoding time: {query_encoding_seconds:.2f}s")
    print(f"Ranking time: {ranking_seconds:.2f}s")
    print(f"\nSaved query results: {query_file}")
    print(f"Saved summary: {summary_file}")


if __name__ == "__main__":
    main()
