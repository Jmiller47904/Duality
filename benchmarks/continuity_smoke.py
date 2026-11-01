"""Small deterministic checks for durable continuity store behavior.

This smoke evaluation uses an untrained CPU memory layer and a temporary SQLite
database. It measures persistence invariants, not recall quality, model quality,
intelligence, consciousness, or production performance.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from duality_lm import (  # noqa: E402
    ClaimRecord,
    DualityMemory,
    MemoryRecord,
    PersistentMemoryStore,
    SelfModel,
)


SEED = 17


def run_evaluation(work_dir: Path) -> Dict[str, Any]:
    """Run repeatable replay, provenance, contradiction, and isolation checks."""

    torch.manual_seed(SEED)
    layer = DualityMemory(d_model=8, slots=4).eval()
    prefix = torch.linspace(-1.0, 1.0, steps=24).reshape(1, 3, 8)
    continuation = torch.linspace(0.75, -0.75, steps=16).reshape(1, 2, 8)
    with torch.inference_mode():
        _, initial_state = layer(prefix)
        expected_output, _ = layer(continuation, initial_state)

    work_dir.mkdir(parents=True, exist_ok=True)
    database = work_dir / "continuity-smoke.sqlite3"
    with PersistentMemoryStore(database) as store:
        store.save_snapshot(
            "identity-a",
            "seeded-prefix",
            (initial_state,),
            {"evaluation": "continuity-smoke-v1", "seed": str(SEED)},
        )
        store.save_model_state("identity-a", (initial_state,))
        store.save_model_state("identity-b", (initial_state,))
        store.remember(
            "identity-a",
            MemoryRecord(
                id="memory-a",
                content="fixture observation for identity a",
                vector=[1.0, 0.0],
                resonance_vector=[1.0, 0.0],
            ),
        )
        store.remember(
            "identity-b",
            MemoryRecord(
                id="memory-b",
                content="fixture observation for identity b",
                vector=[0.0, 1.0],
                resonance_vector=[0.0, 1.0],
            ),
        )
        store.save_self_model(SelfModel(identity="identity-a", goals=["fixture goal a"]))
        store.save_self_model(SelfModel(identity="identity-b", goals=["fixture goal b"]))
        for claim_id, value, source in (
            ("claim-a-1", "active", "fixture-source-1"),
            ("claim-a-2", "inactive", "fixture-source-2"),
        ):
            store.record_claim(
                "identity-a",
                ClaimRecord(
                    id=claim_id,
                    subject="fixture/service",
                    predicate="status",
                    value=value,
                    provenance={"source": source},
                ),
            )
        store.record_claim(
            "identity-b",
            ClaimRecord(
                id="claim-b-1",
                subject="fixture/service",
                predicate="status",
                value="unknown",
                provenance={"source": "fixture-source-b"},
            ),
        )
        contradiction_candidates = len(store.find_contradictions("identity-a"))

    with PersistentMemoryStore(database) as reopened:
        restored = reopened.load_snapshot("identity-a", "seeded-prefix")[0]
        with torch.inference_mode():
            actual_output, _ = layer(continuation, restored)
        replay_exact = torch.equal(expected_output, actual_output)
        replay_max_abs_error = float((expected_output - actual_output).abs().max().item())
        snapshot_provenance_verified = (
            reopened.list_snapshots("identity-a")[0]["provenance_integrity"] == "verified"
        )

        deleted = reopened.reset_identity("identity-a")
        identity_a_memory_count = len(reopened.recall("identity-a", [1.0, 0.0]))
        identity_b_memory_count = len(reopened.recall("identity-b", [0.0, 1.0]))
        identity_b_claim_count = len(reopened.list_claims("identity-b"))
        identity_b_model_state_retained = reopened.load_model_state("identity-b") is not None
        identity_a_snapshot_retained = (
            len(reopened.list_snapshots("identity-a")) == 1
            and reopened.load_snapshot("identity-a", "seeded-prefix") is not None
        )

    return {
        "schema_version": 1,
        "seed": SEED,
        "measurements": {
            "continuation_exact_after_reopen": replay_exact,
            "continuation_max_abs_error": replay_max_abs_error,
            "snapshot_provenance_verified": snapshot_provenance_verified,
            "contradiction_candidate_count": contradiction_candidates,
            "identity_a_reset_deleted_rows": deleted,
            "identity_a_memory_count_after_reset": identity_a_memory_count,
            "identity_b_memory_count_after_reset": identity_b_memory_count,
            "identity_b_claim_count_after_reset": identity_b_claim_count,
            "identity_b_model_state_retained": identity_b_model_state_retained,
            "identity_a_snapshot_retained": identity_a_snapshot_retained,
        },
        "scope": "functional persistence invariants on an untrained tiny layer",
    }


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="duality-continuity-smoke-") as directory:
        result = run_evaluation(Path(directory))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
