"""Acceptance checks for the deterministic native memory benchmark."""

import pytest

from benchmarks.native_memory_ablation import run_benchmark


def test_native_memory_ablation_reports_measurable_costs_and_behavior() -> None:
    result = run_benchmark(seed=29, sequence_length=8, warmup=0, iterations=1)

    assert result["benchmark"] == "native-memory-ablation-v1"
    assert result["memory_off"]["median_forward_ms"] > 0.0
    assert result["memory_on"]["median_forward_ms"] > 0.0
    assert result["comparison"]["latency_ratio"] > 0.0
    assert result["comparison"]["extra_parameter_bytes"] > 0
    assert result["memory_on"]["retained_state_bytes"] > 0
    assert result["memory_on"]["active_slots_after_primer"] > 0
    assert result["comparison"]["continuation_logit_mean_abs_delta"] > 0.0


@pytest.mark.parametrize(
    "arguments, message",
    [
        ({"sequence_length": 3}, "sequence_length"),
        ({"sequence_length": 65}, "sequence_length"),
        ({"warmup": -1}, "warmup"),
        ({"iterations": 0}, "iterations"),
    ],
)
def test_native_memory_ablation_rejects_invalid_bounds(arguments, message) -> None:
    with pytest.raises(ValueError, match=message):
        run_benchmark(**arguments)
