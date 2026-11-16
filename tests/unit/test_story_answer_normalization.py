"""Regression checks for complete-answer story scoring."""

from benchmarks.story_comprehension_410m.evaluation import normalize_answer


def test_normalize_answer_preserves_complete_multiword_answer():
    assert normalize_answer("  New York! ") == "new york"
    assert normalize_answer("The answer is New York.") == "the answer is new york"
