"""Acceptance checks for the deterministic native memory benchmark."""

import pytest
import torch

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


def test_native_memory_ablation_restores_caller_rng_state() -> None:
    torch.manual_seed(1234)
    expected_next = torch.rand(1)
    torch.manual_seed(1234)
    initial_state = torch.random.get_rng_state().clone()

    run_benchmark(seed=29, sequence_length=4, warmup=0, iterations=1)

    assert torch.equal(torch.random.get_rng_state(), initial_state)
    assert torch.equal(torch.rand(1), expected_next)


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
