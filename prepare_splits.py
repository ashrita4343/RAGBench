
import json
import random
from pathlib import Path

DATA_DIR = Path("data") / "scifact"
SEED = 42
VALIDATION_FRACTION = 0.5


def load_jsonl(path):
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def save_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    qrels_path = DATA_DIR / "qrels_test.jsonl"
    qrels = load_jsonl(qrels_path)

    # Keep only queries that have positive relevance judgments.
    qrels = [
        row for row in qrels
        if any(int(score) > 0
               for score in row["relevant_documents"].values())
    ]

    random.Random(SEED).shuffle(qrels)

    split_index = int(len(qrels) * VALIDATION_FRACTION)
    validation = qrels[:split_index]
    test = qrels[split_index:]

    save_jsonl(DATA_DIR / "qrels_validation.jsonl", validation)
    save_jsonl(DATA_DIR / "qrels_final_test.jsonl", test)

    print(f"Total queries: {len(qrels)}")
    print(f"Validation queries: {len(validation)}")
    print(f"Final test queries: {len(test)}")
    print(f"Random seed: {SEED}")
    print("Saved validation and final-test query splits.")


if __name__ == "__main__":
    main()
