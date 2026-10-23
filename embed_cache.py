
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

DATA_DIR = Path("data") / "scifact"
CACHE_DIR = Path("cache")
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_jsonl(path):
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def main():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    corpus = load_jsonl(DATA_DIR / "corpus.jsonl")
    queries = load_jsonl(DATA_DIR / "queries.jsonl")

    documents = [
        f'{row["title"]} {row["text"]}'.strip()
        for row in corpus
    ]
    query_texts = [row["text"] for row in queries]

    model = SentenceTransformer(MODEL_NAME)

    print("Encoding documents...")
    document_embeddings = model.encode(
        documents,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    print("Encoding queries...")
    query_embeddings = model.encode(
        query_texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    np.save(CACHE_DIR / "scifact_document_embeddings.npy",
            document_embeddings)
    np.save(CACHE_DIR / "scifact_query_embeddings.npy",
            query_embeddings)

    with (CACHE_DIR / "scifact_document_ids.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump([str(row["id"]) for row in corpus], file)

    with (CACHE_DIR / "scifact_query_ids.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump([str(row["id"]) for row in queries], file)

    print("\nEmbedding cache saved in the cache/ folder.")
    print("Run this script again only when the dataset or model changes.")


if __name__ == "__main__":
    main()
