# RAGBench Experiment Summary

## Research Objective

Evaluate keyword-based, semantic, and hybrid retrieval methods on the SciFact scientific claim verification dataset.

## Dataset

- Corpus documents: 5,183
- Evaluable queries: 300
- Positive relevance judgments: 339
- Validation queries for hybrid tuning: 150
- Final-test queries: 150

**Evaluation caveat:** The validation/final-test split was created after earlier experiments had already examined results on the full 300-query set. The final-test results should therefore be treated as development results, not as an independent estimate from an untouched holdout.

## Methods

1. **BM25:** Keyword-based retrieval.
2. **Semantic retrieval:** Dense embeddings from `sentence-transformers/all-MiniLM-L6-v2`.
3. **Hybrid retrieval:** Fusion of normalized BM25 and semantic scores.

The SciFact hybrid benchmark selected `alpha = 0.75` by validation nDCG@10.

## Results

### Original 300-query baseline experiments

| Method | Recall@3 | Recall@10 | MRR | nDCG@10 |
|---|---:|---:|---:|---:|
| BM25 | 0.5694 | 0.6862 | 0.5242 | 0.5597 |
| Semantic | 0.6564 | 0.7883 | 0.6068 | 0.6484 |

These baseline results were measured over 300 queries.

### Validation-tuned hybrid: 150-query final-test split

| Metric | Result |
|---|---:|
| Selected alpha | 0.75 |
| Recall@3 | 0.7427 |
| Recall@10 | 0.8198 |
| MRR | 0.6911 |
| nDCG@10 | 0.7168 |

**Important:** The two tables use different evaluation sets. Do not interpret the values as a controlled, same-split comparison between methods.

## Error Analysis

On the 150-query final-test split:

- Queries with a relevant document in the top 3: 114 (76%).
- Queries without a relevant document in the top 3: 36 (24%).

The error-analysis script records relevant-document ranks, retrieved titles, text previews, and per-query metrics.

## Automated Testing

The project has 10 passing automated tests covering:

- Text tokenization.
- Recall@k.
- Mean Reciprocal Rank.
- nDCG behavior.
- Hybrid retrieval ranking and equal-score handling.

## Reproducibility

The repository includes scripts for dataset preparation, embedding caching, baseline evaluation, hybrid tuning, and error analysis. Dataset files, cached embeddings, and generated result CSVs are stored locally rather than tracked in Git.

## Limitations and Future Work

- Evaluate all retrieval methods on the same untouched holdout.
- Standardize metrics and score-fusion implementations.
- Add confidence intervals and repeated experiments.
- Investigate difficult queries and improve retrieval quality.
- Explore evidence-grounded answer generation as a future extension.

## Conclusion

RAGBench provides an experimental framework for studying lexical, semantic, and hybrid information retrieval. Its current results demonstrate the implementation of retrieval benchmarking, validation-based parameter selection, query-level error analysis, and automated testing. Further controlled evaluation is needed before making general performance claims.
