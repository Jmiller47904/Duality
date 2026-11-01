"""Acceptance checks for the sourced contradiction-candidate ledger."""

import math
from pathlib import Path

import pytest

from duality_lm import ClaimRecord, PersistentMemoryStore


def test_preserves_provenance_and_surfaces_differing_values(tmp_path: Path) -> None:
    with PersistentMemoryStore(tmp_path / "claims.sqlite3") as store:
        first = ClaimRecord(
            id="claim-a",
            subject="service/api",
            predicate="status",
            value={"healthy": True, "region": "east"},
            provenance={"source": "probe-a", "observed_at": "2026-11-01T12:00:00Z"},
            confidence=0.8,
        )
        equivalent = ClaimRecord(
            id="claim-b",
            subject="service/api",
            predicate="status",
            value={"region": "east", "healthy": True},
            provenance={"source": "probe-b"},
        )
        differing = ClaimRecord(
            id="claim-c",
            subject="service/api",
            predicate="status",
            value={"healthy": False, "region": "east"},
            provenance={"source": "probe-c"},
        )
        for claim in (first, equivalent, differing):
            store.record_claim("experiment-a", claim)

        claims = store.list_claims("experiment-a")
        assert claims[0].provenance["source"] == "probe-a"
        candidates = store.find_contradictions("experiment-a")
        assert [(item.left_id, item.right_id) for item in candidates] == [
            ("claim-a", "claim-c"),
            ("claim-b", "claim-c"),
        ]
        assert store.find_contradictions("other") == []


def test_claim_validation_is_strict_and_ids_are_identity_bound(tmp_path: Path) -> None:
    with PersistentMemoryStore(tmp_path / "claims.sqlite3") as store:
        claim = ClaimRecord(
            id="shared-id",
            subject="model",
            predicate="score",
            value=0.5,
            provenance={"run": "seed-17"},
        )
        store.record_claim("experiment-a", claim)
        with pytest.raises(ValueError, match="another identity"):
            store.record_claim("experiment-b", claim)
        with pytest.raises(ValueError, match="strict JSON"):
            store.record_claim(
                "experiment-a",
                ClaimRecord(
                    subject="model",
                    predicate="score",
                    value=math.nan,
                    provenance={"run": "seed-18"},
                ),
            )
        with pytest.raises(ValueError, match="provenance"):
            store.record_claim(
                "experiment-a",
                ClaimRecord(subject="model", predicate="score", value=0.6, provenance={}),
            )


def test_identity_reset_removes_claims_without_cross_identity_effects(tmp_path: Path) -> None:
    with PersistentMemoryStore(tmp_path / "claims.sqlite3") as store:
        for identity in ("experiment-a", "experiment-b"):
            store.record_claim(
                identity,
                ClaimRecord(
                    subject="model",
                    predicate="version",
                    value=1,
                    provenance={"source": identity},
                ),
            )
        deleted = store.reset_identity("experiment-a")
        assert deleted["memory_claim"] == 1
        assert store.list_claims("experiment-a") == []
        assert len(store.list_claims("experiment-b")) == 1
