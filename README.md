# RAGBench: Retrieval-Augmented Generation Research Framework

RAGBench is a Python research project for evaluating and comparing information retrieval methods on the SciFact scientific claim verification dataset.

The project investigates how keyword-based retrieval, dense semantic retrieval, and hybrid score fusion affect the ranking of relevant scientific documents.

## Objectives

- Implement a BM25 keyword retrieval baseline.
- Implement dense semantic retrieval using sentence embeddings.
- Combine lexical and semantic scores using a tunable weighting parameter.
- Evaluate retrieval quality using standard information retrieval metrics.
- Tune hyperparameters on validation data and evaluate retrieval performance.
- Analyze successful and failed queries to understand retrieval errors.

## Methods

### 1. BM25 Retrieval

Uses the `rank-bm25` library to rank scientific documents based on lexical matching.

### 2. Semantic Retrieval

Uses `sentence-transformers/all-MiniLM-L6-v2` to encode documents and queries into dense vector embeddings. Similarity is calculated using dot products of normalized embeddings, equivalent to cosine similarity.

### 3. Hybrid Retrieval

Combines normalized BM25 and semantic scores:

```text
hybrid_score = (1 - alpha) * normalized_bm25_score
             + alpha * normalized_semantic_score
```

The parameter `alpha` controls the relative contribution of semantic retrieval:

- `alpha = 0.0`: BM25-only retrieval.
- `alpha = 1.0`: Semantic-only retrieval.
- Intermediate values: Hybrid retrieval.

## Dataset

This project uses the SciFact scientific claim verification dataset from Hugging Face's `BeIR/scifact` and associated relevance judgments.

The experiment uses:

- **5,183 corpus documents**
- **300 evaluable queries**
- **339 positive relevance judgments**

The labeled queries were divided into:

- **150 validation queries** for selecting the hybrid weight.
- **150 final-test queries** for evaluating the selected configuration.

The experiment uses available positive relevance judgments. Documents without a positive judgment are treated as nonrelevant for these metrics.

**Evaluation caveat:** The 150-query final-test split was created after earlier experiments had already examined results on the full set of 300 queries. It should therefore be considered a development evaluation rather than a completely untouched holdout. A future rigorous evaluation should use a newly established holdout that remains unseen during experimentation.

## Validation-Based Hyperparameter Tuning

The hybrid weight `alpha` was selected from the following grid:

`0.00, 0.25, 0.50, 0.75, 1.00`

The selection criterion was **nDCG@10 on the validation set**, with ties favoring the smaller alpha.

| Alpha | Validation Recall@10 | Validation MRR | Validation nDCG@10 |
|---:|---:|---:|---:|
| 0.00 | 0.6528 | 0.5030 | 0.5351 |
| 0.25 | 0.6678 | 0.5420 | 0.5679 |
| 0.50 | 0.7494 | 0.5857 | 0.6203 |
| 0.75 | 0.7747 | 0.6132 | 0.6487 |
| 1.00 | 0.7527 | 0.5988 | 0.6340 |

The selected configuration was **alpha = 0.75**, which achieved the highest validation nDCG@10 among the tested settings.

## Final-Test Results

The selected hybrid configuration was evaluated on the 150-query final-test split.

| Metric | Final-test result |
|---|---:|
| Recall@3 | 0.7427 |
| Recall@10 | 0.8198 |
| MRR | 0.6911 |
| nDCG@10 | 0.7168 |

These results describe this implementation and evaluation setup only. Because the final-test split was created after earlier experiments on the complete query set, the results are not an independent estimate from a pristine holdout.

## Same-Split Retrieval Comparison

BM25, semantic retrieval, and hybrid retrieval were evaluated on the same 150 final-test queries.

| Method | Recall@3 | Recall@10 | MRR | nDCG@10 |
|---|---:|---:|---:|---:|
| BM25 | 0.6000 | 0.7196 | 0.5455 | 0.5843 |
| Semantic | 0.6771 | 0.8140 | 0.6106 | 0.6562 |
| Hybrid (alpha = 0.75) | 0.7427 | 0.8198 | 0.6911 | 0.7168 |

Hybrid retrieval achieved the highest value across all four metrics in this experiment. The results suggest that combining lexical matching with dense semantic similarity improved retrieval quality for this evaluation setup.

**Limitation:** The final-test split was created after earlier experiments examined results on the full 300-query set. This comparison uses the same queries for all three methods, but the split is not an untouched holdout. Independent evaluation on a newly established holdout is needed to support stronger generalization claims.

Generated comparison files:

- `results/scifact_same_split_comparison.csv`
- `results/scifact_same_split_query_results.csv`


## Error Analysis

The `error_analysis.py` script recomputes rankings for the final-test queries and generates detailed query-level CSV reports.

Across 150 queries:

- **114 queries (76%)** had at least one relevant document in the top three.
- **36 queries (24%)** had no relevant document in the top three.

The reports include query text, relevant document IDs