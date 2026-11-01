"""Regression checks for the standalone continuity smoke evaluation."""

from pathlib import Path

from benchmarks.continuity_smoke import run_evaluation


def test_continuity_smoke_measures_replay_provenance_conflicts_and_isolation(
    tmp_path: Path,
) -> None:
    result = run_evaluation(tmp_path / "first")
    repeated = run_evaluation(tmp_path / "repeated")
    assert repeated == result

    assert result["schema_version"] == 1
    assert result["seed"] == 17
    assert "consciousness" not in result["scope"]
    measurements = result["measurements"]
    assert measurements["continuation_exact_after_reopen"] is True
    assert measurements["continuation_max_abs_error"] == 0.0
    assert measurements["snapshot_provenance_verified"] is True
    assert measurements["contradiction_candidate_count"] == 1
    assert measurements["identity_a_reset_deleted_rows"] == {
        "memory": 1,
        "memory_claim": 2,
        "model_state": 1,
        "self_model": 1,
        "model_snapshot": 0,
    }
    assert measurements["identity_a_memory_count_after_reset"] == 0
    assert measurements["identity_b_memory_count_after_reset"] == 1
    assert measurements["identity_b_claim_count_after_reset"] == 1
    assert measurements["identity_b_model_state_retained"] is True
    assert measurements["identity_a_snapshot_retained"] is True
