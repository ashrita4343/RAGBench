
from datasets import load_dataset


def load_scifact():
    """Load SciFact documents, test queries, and relevance judgments."""

    corpus_data = load_dataset("BeIR/scifact", "corpus")["corpus"]
    query_data = load_dataset("BeIR/scifact", "queries")["queries"]
    qrels_data = load_dataset("BeIR/scifact-qrels")["test"]

    # Store documents by their official document IDs.
    corpus = {
        str(row["_id"]): {
            "title": row["title"],
            "text": row["text"],
        }
        for row in corpus_data
    }

    # Store queries by their official query IDs.
    all_queries = {
        str(row["_id"]): row["text"]
        for row in query_data
    }

    # Store relevance judgments: query ID -> document ID -> score.
    qrels = {}

    for row in qrels_data:
        query_id = str(row["query-id"])
        document_id = str(row["corpus-id"])
        score = int(row["score"])

        qrels.setdefault(query_id, {})[document_id] = score

    # Evaluate only queries with official test relevance judgments.
    queries = {
        query_id: all_queries[query_id]
        for query_id in qrels
        if query_id in all_queries
    }

    # Keep only judgments for queries and documents that exist.
    qrels = {
        query_id: {
            document_id: score
            for document_id, score in judgments.items()
            if document_id in corpus
        }
        for query_id, judgments in qrels.items()
        if query_id in queries
    }

    return corpus, queries, qrels


if __name__ == "__main__":
    corpus, queries, qrels = load_scifact()

    print("SciFact data loaded successfully")
    print("Documents:", len(corpus))
    print("Evaluable test queries:", len(queries))
    print("Queries with relevance judgments:", len(qrels))
    print("Positive relevance judgments:",
          sum(
              score > 0
              for judgments in qrels.values()
              for score in judgments.values()
          ))

    sample_query_id = next(iter(queries))
    print("\nSample query ID:", sample_query_id)
    print("Sample query:", queries[sample_query_id])
    print("Relevant documents:", qrels[sample_query_id])

import json
from pathlib import Path


def export_scifact():
    corpus, queries, qrels = load_scifact()

    output_dir = Path("data") / "scifact"
    output_dir.mkdir(parents=True, exist_ok=True)

    files = {
        "corpus.jsonl": (
            {"id": doc_id, **document}
            for doc_id, document in corpus.items()
        ),
        "queries.jsonl": (
            {"id": query_id, "text": text}
            for query_id, text in queries.items()
        ),
        "qrels_test.jsonl": (
            {
                "query_id": query_id,
                "relevant_documents": judgments,
            }
            for query_id, judgments in qrels.items()
        ),
    }

    for filename, rows in files.items():
        path = output_dir / filename

        with path.open("w", encoding="utf-8") as file:
            for row in rows:
                file.write(json.dumps(row, ensure_ascii=False) + "\n")

        print(f"Saved {path}")


if __name__ == "__main__":
    export_scifact()
