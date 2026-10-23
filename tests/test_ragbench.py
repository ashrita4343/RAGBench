
import numpy as np
import pytest

from ragbench import (
    recall_at_k,
    reciprocal_rank,
    search_hybrid,
    tokenize,
)


def ndcg_at_k(ranked_ids, relevance, k):
    """Calculate normalized discounted cumulative gain."""
    def gain(score):
        return (2**score) - 1

    dcg = sum(
        gain(relevance.get(doc_id, 0)) / np.log2(rank + 1)
        for rank, doc_id in enumerate(ranked_ids[:k], start=1)
    )

    ideal_scores = sorted(relevance.values(), reverse=True)[:k]
    ideal_dcg = sum(
        gain(score) / np.log2(rank + 1)
        for rank, score in enumerate(ideal_scores, start=1)
    )

    return dcg / ideal_dcg if ideal_dcg else 0.0


def test_tokenize_lowercases_words():
    assert tokenize("Semantic Search") == ["semantic", "search"]


def test_recall_at_k_finds_relevant_document():
    ranked = ["d2", "d1", "d3"]
    relevant = {"d1"}

    assert recall_at_k(ranked, relevant, k=3) == 1.0


def test_recall_at_k_misses_relevant_document():
    ranked = ["d2", "d3", "d4"]
    relevant = {"d1"}

    assert recall_at_k(ranked, relevant, k=3) == 0.0


def test_reciprocal_rank():
    ranked = ["d2", "d3", "d1"]
    relevant = {"d1"}

    assert reciprocal_rank(ranked, relevant) == pytest.approx(1 / 3)


def test_reciprocal_rank_when_not_found():
    ranked = ["d2", "d3", "d4"]
    relevant = {"d1"}

    assert reciprocal_rank(ranked, relevant) == 0.0


def test_ndcg_perfect_ranking():
    ranked = ["d1", "d2", "d3"]
    relevance = {"d1": 3, "d2": 2, "d3": 1}

    assert ndcg_at_k(ranked, relevance, 3) == pytest.approx(1.0)


def test_ndcg_zero_when_no_relevant_documents():
    ranked = ["d1", "d2", "d3"]
    relevance = {}

    assert ndcg_at_k(ranked, relevance, 3) == 0.0


def test_ndcg_rewards_better_ranking():
    relevance = {"d1": 2, "d2": 1, "d3": 0}

    good_ranking = ["d1", "d2", "d3"]
    poor_ranking = ["d3", "d2", "d1"]

    assert ndcg_at_k(good_ranking, relevance, 3) > ndcg_at_k(
        poor_ranking, relevance, 3
    )


class FakeBM25:
    def get_scores(self, tokens):
        return np.array([10.0, 0.0, 5.0])


class FakeSemanticModel:
    def encode(self, texts, normalize_embeddings=True):
        return np.array([[1.0, 0.0]])


def test_hybrid_search_returns_top_k_in_rank_order():
    documents = [
        {"id": "d1", "text": "alpha"},
        {"id": "d2", "text": "beta"},
        {"id": "d3", "text": "gamma"},
    ]

    # Semantic similarities will be [0.0, 1.0, 0.5].
    embeddings = np.array([
        [0.0, 1.0],
        [1.0, 0.0],
        [0.5, 0.5],
    ])

    results = search_hybrid(
        query="test query",
        bm25_model=FakeBM25(),
        semantic_model=FakeSemanticModel(),
        embeddings=embeddings,
        documents=documents,
        top_k=2,
        alpha=1.0,
    )

    assert [item["id"] for item in results] == ["d1", "d3"]
    assert results[0]["score"] >= results[1]["score"]


def test_hybrid_search_handles_equal_scores():
    documents = [
        {"id": "d1", "text": "alpha"},
        {"id": "d2", "text": "beta"},
        {"id": "d3", "text": "gamma"},
    ]

    results = search_hybrid(
        query="test query",
        bm25_model=FakeBM25(),
        semantic_model=FakeSemanticModel(),
        embeddings=np.array([
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
        ]),
        documents=documents,
        top_k=3,
        alpha=0.0,
    )

    assert [item["id"] for item in results] == ["d1", "d2", "d3"]
    assert all(item["score"] == pytest.approx(0.0) for item in results)
