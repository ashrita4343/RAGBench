
from pathlib import Path

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


# --------------------------------------------------
# 1. Project paths
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# --------------------------------------------------
# 2. Small, labelled document collection
# --------------------------------------------------

DOCUMENTS = [
    {
        "id": "d1",
        "text": (
            "Class imbalance occurs when one class has far fewer examples "
            "than another. Precision, recall, F1 and PR-AUC can help "
            "evaluate classifiers on imbalanced datasets."
        ),
    },
    {
        "id": "d2",
        "text": (
            "Data leakage happens when training uses information that "
            "would not be available at prediction time. Split the data "
            "before fitting preprocessing steps."
        ),
    },
    {
        "id": "d3",
        "text": (
            "BM25 is a keyword retrieval algorithm. It ranks documents "
            "using query-term frequency, inverse document frequency "
            "and document length."
        ),
    },
    {
        "id": "d4",
        "text": (
            "Semantic search encodes text into vector embeddings and "
            "retrieves passages based on similarity in meaning."
        ),
    },
    {
        "id": "d5",
        "text": (
            "Retrieval-augmented generation retrieves relevant evidence "
            "and supplies it to a language model to help generate "
            "grounded answers."
        ),
    },
    {
        "id": "d6",
        "text": (
            "A confusion matrix records true positives, true negatives, "
            "false positives and false negatives for a classifier."
        ),
    },
    {
        "id": "d7",
        "text": (
            "SMOTE creates synthetic minority-class examples to help "
            "address class imbalance. Apply it only to training data "
            "after splitting the dataset."
        ),
    },
    {
        "id": "d8",
        "text": (
            "Sentence Transformers provides pretrained models that "
            "convert sentences and passages into dense vector embeddings."
        ),
    },
    {
        "id": "d9",
        "text": (
            "A train-test split estimates how a model performs on unseen "
            "data. Keep the test set separate from model fitting."
        ),
    },
    {
        "id": "d10",
        "text": (
            "Information retrieval metrics include Recall at k and "
            "Mean Reciprocal Rank. They evaluate whether relevant "
            "documents appear near the top of search results."
        ),
    },
]


# --------------------------------------------------
# 3. Questions and known relevant documents
# --------------------------------------------------

QUERIES = [
    {
        "query": "How should I evaluate a rare fraud class?",
        "relevant_ids": {"d1"},
    },
    {
        "query": "How can I avoid using future information during training?",
        "relevant_ids": {"d2"},
    },
    {
        "query": "Which algorithm ranks text using query words?",
        "relevant_ids": {"d3"},
    },
    {
        "query": "How do I find passages with similar meanings?",
        "relevant_ids": {"d4"},
    },
    {
        "query": "How can a language model answer using retrieved evidence?",
        "relevant_ids": {"d5"},
    },
    {
        "query": "How do I count false alarms and missed positive cases?",
        "relevant_ids": {"d6"},
    },
    {
        "query": "How can I create synthetic examples for the minority class?",
        "relevant_ids": {"d7"},
    },
    {
        "query": "Which library creates vector representations of sentences?",
        "relevant_ids": {"d8"},
    },
    {
        "query": "How do I evaluate performance on data never used for training?",
        "relevant_ids": {"d9"},
    },
    {
        "query": "How can I measure whether search returns relevant results near the top?",
        "relevant_ids": {"d10"},
    },
]


# --------------------------------------------------
# 4. Text processing and retrieval metrics
# --------------------------------------------------

def tokenize(text):
    """Simple baseline tokenizer for keyword retrieval."""
    return text.lower().split()


def recall_at_k(ranked_ids, relevant_ids, k=3):
    """Fraction of relevant documents found in the top k."""
    if not relevant_ids:
        return 0.0

    retrieved = set(ranked_ids[:k])
    return len(retrieved & relevant_ids) / len(relevant_ids)


def reciprocal_rank(ranked_ids, relevant_ids):
    """Reciprocal rank of the first relevant result."""
    for rank, doc_id in enumerate(ranked_ids, start=1):
        if doc_id in relevant_ids:
            return 1.0 / rank

    return 0.0


# --------------------------------------------------
# 5. BM25 keyword retrieval
# --------------------------------------------------

def build_bm25(documents):
    tokenized_docs = [tokenize(doc["text"]) for doc in documents]
    return BM25Okapi(tokenized_docs)


def search_bm25(query, model, documents, top_k=3):
    scores = model.get_scores(tokenize(query))
    ranked_indices = np.argsort(-scores, kind="stable")[:top_k]

    return [
        {
            "id": documents[int(i)]["id"],
            "score": float(scores[int(i)]),
        }
        for i in ranked_indices
    ]


# --------------------------------------------------
# 6. Hugging Face embedding-based semantic retrieval
# --------------------------------------------------

def build_semantic_model(documents):
    # This pretrained model is hosted on Hugging Face.
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    texts = [doc["text"] for doc in documents]
    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    return model, embeddings


def search_semantic(query, model, embeddings, documents, top_k=3):
    query_embedding = model.encode(
        [query],
        normalize_embeddings=True,
    )[0]

    # Normalized vectors let us use dot product as cosine similarity.
    scores = embeddings @ query_embedding
    ranked_indices = np.argsort(-scores, kind="stable")[:top_k]

    return [
        {
            "id": documents[int(i)]["id"],
            "score": float(scores[int(i)]),
        }
        for i in ranked_indices
    ]


# --------------------------------------------------
# 7. Run the benchmark
# --------------------------------------------------

def main():
    print("RAGBench: retrieval evaluation")
    print(f"Documents: {len(DOCUMENTS)}")
    print(f"Queries: {len(QUERIES)}")

    bm25_model = build_bm25(DOCUMENTS)
    semantic_model, embeddings = build_semantic_model(DOCUMENTS)

    query_rows = []

    for item in QUERIES:
        query = item["query"]
        relevant_ids = item["relevant_ids"]

        bm25_results = search_bm25(
            query, bm25_model, DOCUMENTS, top_k=3
        )
        semantic_results = search_semantic(
            query, semantic_model, embeddings, DOCUMENTS, top_k=3
        )

        for method, results in [
            ("BM25", bm25_results),
            ("Semantic", semantic_results),
        ]:
            ranked_ids = [result["id"] for result in results]

            query_rows.append(
                {
                    "query": query,
                    "method": method,
                    "relevant_ids": ",".join(sorted(relevant_ids)),
                    "top_3_ids": ",".join(ranked_ids),
                    "recall_at_3": recall_at_k(
                        ranked_ids, relevant_ids, k=3
                    ),
                    "reciprocal_rank": reciprocal_rank(
                        ranked_ids, relevant_ids
                    ),
                    "top_result_score": results[0]["score"],
                }
            )

    details = pd.DataFrame(query_rows)

    summary = (
        details.groupby("method")
        .agg(
            recall_at_3=("recall_at_3", "mean"),
            mrr=("reciprocal_rank", "mean"),
        )
        .reset_index()
    )

    details.to_csv(RESULTS_DIR / "query_results.csv", index=False)
    summary.to_csv(RESULTS_DIR / "summary.csv", index=False)

    print("\nBenchmark summary:")
    print(summary.to_string(index=False))

    print(f"\nDetailed results: {RESULTS_DIR / 'query_results.csv'}")
    print(f"Summary results:  {RESULTS_DIR / 'summary.csv'}")
    print("\nNote: this small dataset is a smoke test, not scientific evidence.")


if __name__ == "__main__":
    main()
