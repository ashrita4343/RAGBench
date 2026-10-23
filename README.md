# RAGBench — Retrieval-Augmented Generation Research Framework

RAGBench is a Python research project for evaluating and comparing information retrieval methods on the SciFact scientific claim verification dataset.

The project investigates how keyword-based retrieval, dense semantic retrieval, and hybrid score fusion affect the ranking of relevant scientific documents.

## Objectives

- Implement a BM25 keyword retrieval baseline.
- Implement dense semantic retrieval using sentence embeddings.
- Combine lexical and semantic scores using a tunable weighting parameter.
- Evaluate retrieval quality with standard information retrieval metrics.
- Run controlled experiments and save query-level results for analysis.

## Methods

### 1. BM25 retrieval

Uses the `rank-bm25` library to rank scientific documents based on lexical matching.

### 2. Semantic retrieval

Uses `sentence-transformers/all-MiniLM-L6-v2` to encode documents and queries into dense vector embeddings. Cosine similarity is computed using normalized embeddings.

### 3. Hybrid retrieval

Combines normalized BM25 and semantic scores:

`hybrid_score = (1 - alpha) * bm25_score + alpha * semantic_score`

The parameter `alpha` controls the relative contribution of semantic retrieval. An alpha of 0 represents BM25-only retrieval, while an alpha of 1 represents semantic-only retrieval.

## Dataset

This project uses SciFact, a scientific claim verification dataset, through Hugging Face's `BeIR/scifact` and `BeIR/scifact-qrels` datasets.

The current local experiment uses:

- **5,183** corpus documents
- **300** evaluable test queries
- **339** positive relevance judgments

The evaluation uses the available relevance judgments; documents without a positive judgment are treated as nonrelevant for these metrics.

## Preliminary Results

The following results are from the current hybrid alpha sweep on 300 SciFact test queries.

| Semantic weight (alpha) | Recall@3 | Recall@10 | MRR | nDCG@10 |
|---:|---:|---:|---:|---:|
| 0.00 | 0.5694 | 0.6862 | 0.5242 | 0.5597 |
| 0.25 | 0.6201 | 0.7170 | 0.5752 | 0.6056 |
| 0.50 | 0.6551 | 0.7796 | 0.6239 | 0.6573 |
| 0.75 | 0.6874 | 0.7972 | 0.6522 | 0.6828 |
| 1.00 | 0.6603 | 0.7833 | 0.6047 | 0.6451 |

The strongest result among the tested hybrid settings was alpha = 0.75. These findings are preliminary: alpha was selected using the test queries, so a separate validation split and a held-out final evaluation are needed before making a final performance claim.

## Evaluation Metrics

- **Recall@k:** Fraction of judged-relevant documents retrieved in the top k, averaged over queries.
- **MRR (Mean Reciprocal Rank):** Measures how highly the first relevant document is ranked.
- **nDCG@10:** Measures ranking quality in the top 10 using graded relevance judgments.

## Project Structure

```text
RAGBench/
├── ragbench.py
├── ragbench_backup.py
├── scifact_data.py
├── scifact_benchmark.py
├── scifact_semantic.py
├── scifact_hybrid.py
├── embed_cache.py
├── tests/
│   └── test_ragbench.py
├── data/       # Generated locally; not tracked by Git
├── cache/      # Cached embeddings; not tracked by Git
└── results/    # Experiment outputs; not tracked by Git
```

## Setup

Python 3.13 was used during development.

Create and activate a virtual environment on Windows:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the dependencies:

```powershell
python -m pip install --upgrade pip
pip install datasets sentence-transformers rank-bm25 numpy pandas pytest
```

## Running the Experiments

Download and export the SciFact data:

```powershell
python scifact_data.py
```

Run the BM25 baseline:

```powershell
python scifact_benchmark.py
```

Run semantic retrieval:

```powershell
python scifact_semantic.py
```

Generate cached embeddings:

```powershell
python embed_cache.py
```

Run the hybrid alpha sweep:

```powershell
python scifact_hybrid.py
```

The scripts write experiment summaries and per-query results to the local `results/` directory. The embedding cache is stored in `cache/` and can be regenerated when the dataset or model changes.

## Current Limitations and Future Work

- Select hyperparameters using validation data rather than the final test set.
- Standardize ranking and evaluation code across all retrieval methods.
- Add automated tests for retrieval metrics and score fusion.
- Analyze failed queries and compare relevant-document ranks.
- Report runtime and memory requirements consistently.
- Add confidence intervals and repeated experiments where appropriate.
- Explore evidence-grounded generation as a future extension.

## Skills Demonstrated

Python · NLP · Sentence Embeddings · Information Retrieval · BM25 · Hybrid Search · Ranking Metrics · Experimental Evaluation · Dataset Processing · Reproducible Research

## Disclaimer

RAGBench is an educational research project. Reported results are specific to the current implementation, dataset split, model, and evaluation protocol; they should not be interpreted as evidence of general superiority across retrieval tasks.
