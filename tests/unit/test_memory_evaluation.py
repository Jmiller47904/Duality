"""Unit checks for paired memory-evaluation helpers."""

from benchmarks.memory_evaluation.cases import CASES, CATEGORIES
from benchmarks.memory_evaluation.run import (
    cluster_bootstrap_ci,
    format_history,
    format_question,
    paired_cluster_bootstrap_ci,
    validate_cases,
)


def test_builtin_cases_cover_all_categories_and_have_unique_answers():
    validate_cases(CASES)
    assert {case["category"] for case in CASES} == set(CATEGORIES)
    assert len({case["case_id"] for case in CASES}) == len(CASES)
    assert all(case["answer"] in case["options"] for case in CASES)


def test_question_is_shared_and_history_format_contains_every_session():
    case = CASES[0]
    assert "Question:" in format_question(case)
    assert "Possible answers:" in format_question(case)
    history = format_history(case)
    assert "Session 1:" in history
    assert "Session 2:" in history
    assert case["sessions"][0] in history


def test_paired_bootstrap_is_zero_for_identical_conditions():
    cases = CASES[:6]
    values = {str(case["case_id"]): float(index % 2) for index, case in enumerate(cases)}
    result = paired_cluster_bootstrap_ci(cases, values, values, seed=7, replicates=200)
    assert result["mean"] == 0.0
    assert result["ci95"] == [0.0, 0.0]


def test_accuracy_interval_is_bounded_and_reproducible():
    cases = CASES[:8]
    values = {str(case["case_id"]): float(index % 2) for index, case in enumerate(cases)}
    first = cluster_bootstrap_ci(cases, values, seed=19, replicates=500)
    second = cluster_bootstrap_ci(cases, values, seed=19, replicates=500)
    assert first == second
    assert 0.0 <= first["ci95"][0] <= first["mean"] <= first["ci95"][1] <= 1.0
