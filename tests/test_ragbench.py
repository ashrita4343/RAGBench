
import pytest

from ragbench import recall_at_k, reciprocal_rank, tokenize


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
