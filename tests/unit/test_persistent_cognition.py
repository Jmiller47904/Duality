"""Tests for durable memory, self-modeling, and bounded reflection."""

import json
import math
from pathlib import Path

import pytest
import torch

from duality_lm import (
    ByteTokenizer,
    ContinuityEngine,
    DualityLM,
    DualityLMConfig,
    MemoryKind,
    MemoryRecord,
    PersistentMemoryStore,
    ReflectionResult,
    SelfModel,
    measure_metastate,
)
from duality_lm.memory import MemoryState


def tiny_model() -> DualityLM:
    tokenizer = ByteTokenizer()
    config = DualityLMConfig.tiny(vocab_size=tokenizer.vocab_size, max_seq_len=64)
    return DualityLM(config).eval()


def test_model_memory_survives_process_boundary(tmp_path: Path) -> None:
    model = tiny_model()
    tokens = torch.tensor([[1, 3, 4, 5]], dtype=torch.long)
    state = model(tokens).memory_state
    assert state is not None

    database = tmp_path / "continuity.sqlite3"
    with PersistentMemoryStore(database) as store:
        store.save_model_state("duality-test", state)

    with PersistentMemoryStore(database) as reopened:
        restored = reopened.load_model_state("duality-test")

    assert restored is not None
    assert len(restored) == len(state)
    assert restored[2] is not None
    assert state[2] is not None
    assert torch.equal(restored[2].attractors, state[2].attractors)
    assert torch.equal(restored[2].strengths, state[2].strengths)
    assert torch.equal(restored[2].timestep, state[2].timestep)


def test_resonance_recall_prefers_matching_memory(tmp_path: Path) -> None:
    with PersistentMemoryStore(tmp_path / "memory.sqlite3") as store:
        matching = MemoryRecord(
            content="persistent identity",
            vector=[1.0, 0.0, 0.0],
            resonance_vector=[1.0, 0.0, 0.0],
            importance=0.8,
        )
        unrelated = MemoryRecord(
            content="unrelated observation",
            vector=[0.0, 1.0, 0.0],
            resonance_vector=[0.0, 1.0, 0.0],
            importance=1.0,
        )
        store.remember("duality-test", matching)
        store.remember("duality-test", unrelated)

        recalled = store.recall("duality-test", [1.0, 0.0, 0.0], limit=2)

        assert recalled[0].id == matching.id
        assert recalled[0].score > recalled[1].score
        assert recalled[0].activation_count == 1


def test_recall_rejects_vector_dimension_mismatch(tmp_path: Path) -> None:
    with PersistentMemoryStore(tmp_path / "memory.sqlite3") as store:
        store.remember(
            "duality-test",
            MemoryRecord(
                content="three dimensions",
                vector=[1.0, 0.0, 0.0],
                resonance_vector=[1.0, 0.0, 0.0],
            ),
        )
        with pytest.raises(ValueError, match="same dimension"):
            store.recall("duality-test", [1.0, 0.0])


def test_recall_ranks_large_finite_vectors_without_overflow(tmp_path: Path) -> None:
    query = [1e308, 1e308]
    matching = MemoryRecord(
        content="matching large vector",
        vector=query,
        resonance_vector=query,
        importance=0.5,
    )
    orthogonal = MemoryRecord(
        content="orthogonal large vector",
        vector=[1e308, -1e308],
        resonance_vector=[1e308, -1e308],
        importance=1.0,
    )

    with PersistentMemoryStore(tmp_path / "memory.sqlite3") as store:
        store.remember("duality-test", matching)
        store.remember("duality-test", orthogonal)

        recalled = store.recall(
            "duality-test", query, limit=2, now=matching.last_activated
        )

        assert recalled[0].id == matching.id
        assert all(math.isfinite(record.score) for record in recalled)


@pytest.mark.parametrize(
    "decay_rate",
    [float("nan"), float("inf"), float("-inf"), -0.01],
)
def test_remember_rejects_nonfinite_or_negative_decay_rate(
    tmp_path: Path, decay_rate: float
) -> None:
    with PersistentMemoryStore(tmp_path / "memory.sqlite3") as store:
        record = MemoryRecord(
            content="invalid decay rate",
            vector=[1.0, 0.0],
            resonance_vector=[1.0, 0.0],
            decay_rate=decay_rate,
        )

        with pytest.raises(ValueError, match="decay_rate must be finite and >= 0"):
            store.remember("duality-test", record)

        assert store.recall("duality-test", [1.0, 0.0]) == []


@pytest.mark.parametrize(
    "decay_rate",
    [float("nan"), float("inf"), float("-inf"), -0.01],
)
def test_reflection_cycle_rejects_invalid_decay_rate_before_writing(
    tmp_path: Path, decay_rate: float
) -> None:
    record = MemoryRecord(
        content="invalid reflection decay",
        vector=[1.0, 0.0],
        resonance_vector=[1.0, 0.0],
        decay_rate=decay_rate,
    )
    with PersistentMemoryStore(tmp_path / "invalid-reflection-decay.sqlite3") as store:
        with pytest.raises(ValueError, match="decay_rate must be finite and >= 0"):
            store.save_reflection_cycle(
                "identity", record, SelfModel(identity="identity")
            )

        for table in ("memory", "self_model", "model_state"):
            count = store.connection.execute(
                f"SELECT COUNT(*) FROM {table}"
            ).fetchone()[0]
            assert count == 0


def test_memory_id_cannot_move_between_identities(tmp_path: Path) -> None:
    memory_id = "shared-record-id"
    original = MemoryRecord(
        id=memory_id,
        content="identity a memory",
        vector=[1.0, 0.0],
        resonance_vector=[1.0, 0.0],
    )
    replacement = MemoryRecord(
        id=memory_id,
        content="identity b replacement",
        vector=[1.0, 0.0],
        resonance_vector=[1.0, 0.0],
    )

    with PersistentMemoryStore(tmp_path / "memory.sqlite3") as store:
        store.remember("identity-a", original)
        with pytest.raises(ValueError, match="belongs to another identity"):
            store.remember("identity-b", replacement)

        recalled_a = store.recall("identity-a", [1.0, 0.0])
        recalled_b = store.recall("identity-b", [1.0, 0.0])
        assert [record.content for record in recalled_a] == ["identity a memory"]
        assert recalled_b == []


def test_metastate_is_normalized() -> None:
    model = tiny_model()
    state = model(torch.tensor([[1, 8, 9, 10]], dtype=torch.long)).memory_state
    meta = measure_metastate(state)
    assert meta.active_slots > 0
    assert meta.total_slots == model.config.memory_slots
    assert 0.0 <= meta.confidence <= 1.0
    assert 0.0 <= meta.uncertainty <= 1.0
    assert 0.0 <= meta.stability <= 1.0
    assert 0.0 <= meta.occupancy <= 1.0
    assert 0.0 <= meta.diversity <= 1.0


def test_reflection_cycle_updates_self_memory(tmp_path: Path) -> None:
    with PersistentMemoryStore(tmp_path / "memory.sqlite3") as store:
        self_model = SelfModel(
            identity="duality-test",
            goals=["maintain coherent identity"],
            unresolved_questions=["what changed since the last cycle?"],
            limitations=["subjective experience cannot be inferred from behavior alone"],
        )
        store.save_self_model(self_model)
        engine = ContinuityEngine(store, "duality-test")
        engine.observe("initial observation", [1.0, 0.0, 0.0], importance=0.7)

        def reason(context):  # type: ignore[no-untyped-def]
            assert context.focus == "what changed since the last cycle?"
            assert context.recalled_memories
            return ReflectionResult(
                content="The prior observation is still consistent with my active goal.",
                vector=[1.0, 0.0, 0.0],
                resolved_question=context.focus,
                new_question="what evidence would falsify this conclusion?",
            )

        result = engine.reflect_once([1.0, 0.0, 0.0], reason)
        updated = store.load_self_model("duality-test")
        self_memories = store.recall(
            "duality-test",
            [1.0, 0.0, 0.0],
            kinds=[MemoryKind.SELF],
        )

        assert result.content == updated.last_reflection
        assert updated.cycle_count == 1
        assert "what changed since the last cycle?" not in updated.unresolved_questions
        assert "what evidence would falsify this conclusion?" in updated.unresolved_questions
        assert self_memories[0].content == result.content


@pytest.mark.parametrize("belief", [float("nan"), float("inf"), float("-inf")])
def test_save_self_model_rejects_nonfinite_beliefs_without_overwriting(
    tmp_path: Path, belief: float
) -> None:
    with PersistentMemoryStore(tmp_path / "self-model-json.sqlite3") as store:
        store.save_self_model(
            SelfModel(identity="a", goals=["existing"], beliefs={"claim": 0.75})
        )
        invalid = SelfModel(
            identity="a",
            goals=["invalid replacement"],
            beliefs={"claim": belief},
            updated_at=123.0,
        )

        with pytest.raises(ValueError, match="Out of range float"):
            store.save_self_model(invalid)

        assert invalid.updated_at == 123.0
        persisted = store.load_self_model("a")
        assert persisted.goals == ["existing"]
        assert persisted.beliefs == {"claim": 0.75}


def test_reflection_outputs_roll_back_together_when_checkpoint_write_fails(tmp_path: Path) -> None:
    import sqlite3

    with PersistentMemoryStore(tmp_path / "reflection-transaction.sqlite3") as store:
        engine = ContinuityEngine(store, "duality-test")

        def reason(context):  # type: ignore[no-untyped-def]
            return ReflectionResult(content="one measured reflection", vector=[1.0, 0.0])

        store.connection.execute(
            """
            CREATE TRIGGER reject_checkpoint BEFORE INSERT ON model_state
            BEGIN SELECT RAISE(ABORT, 'checkpoint write rejected'); END;
            """
        )
        store.connection.commit()

        with pytest.raises(sqlite3.IntegrityError, match="checkpoint write rejected"):
            engine.reflect_once([1.0, 0.0], reason, model_state=(None,))

        assert store.recall(
            "duality-test", [1.0, 0.0], kinds=[MemoryKind.SELF]
        ) == []
        assert store.load_self_model("duality-test").cycle_count == 0
        assert store.load_model_state("duality-test") is None


def test_invalid_reflection_checkpoint_cannot_partially_persist(tmp_path: Path) -> None:
    with PersistentMemoryStore(tmp_path / "invalid-reflection-state.sqlite3") as store:
        engine = ContinuityEngine(store, "duality-test")

        def reason(context):  # type: ignore[no-untyped-def]
            return ReflectionResult(content="one measured reflection", vector=[1.0, 0.0])

        malformed_state = MemoryState(
            attractors=torch.zeros(1, 2),
            strengths=torch.zeros(1),
            timestep=torch.zeros(1),
        )
        with pytest.raises(ValueError, match="attractors must contain vectors"):
            engine.reflect_once([1.0, 0.0], reason, model_state=(malformed_state,))

        assert store.recall(
            "duality-test", [1.0, 0.0], kinds=[MemoryKind.SELF]
        ) == []
        assert store.load_self_model("duality-test").cycle_count == 0
        assert store.load_model_state("duality-test") is None


def test_named_snapshot_replays_memory_and_preserves_checkpoint(tmp_path):
    from duality_lm import DualityMemory

    torch.manual_seed(17)
    memory = DualityMemory(8, slots=4).eval()
    _, before = memory(torch.randn(1, 3, 8))
    continuation = torch.randn(1, 2, 8)
    expected, _ = memory(continuation, before)
    path = tmp_path / "snapshots.sqlite3"
    with PersistentMemoryStore(path) as store:
        digest = store.save_snapshot("a", "before", (None, before), {"seed": "17"})
        store.save_model_state("a", (None, None))
        assert store.list_snapshots("a")[0]["sha256"] == digest
        assert store.list_snapshots("a")[0]["provenance"] == {"seed": "17"}
    with PersistentMemoryStore(path) as store:
        restored = store.load_snapshot("a", "before")
        actual, _ = memory(continuation, restored[1])
        assert torch.equal(expected, actual)
        assert store.load_model_state("a") == (None, None)
        restored[1].strengths.zero_()
        assert torch.equal(store.load_snapshot("a", "before")[1].strengths, before.strengths)
        assert store.list_snapshots("other") == []
        with pytest.raises(KeyError):
            store.load_snapshot("other", "before")


def test_active_model_state_detects_payload_corruption(tmp_path):
    with PersistentMemoryStore(tmp_path / "active-state.sqlite3") as store:
        store.save_model_state("a", (None,))
        checksum = store.connection.execute(
            "SELECT payload_sha256 FROM model_state WHERE identity = ?", ("a",)
        ).fetchone()[0]
        assert len(checksum) == 64

        store.connection.execute(
            "UPDATE model_state SET payload = ? WHERE identity = ?",
            ('{"version":1,"layers":[null,null]}', "a"),
        )
        store.connection.commit()

        with pytest.raises(ValueError, match="model state checksum"):
            store.load_model_state("a")


def test_legacy_active_model_state_remains_readable_after_migration(tmp_path):
    import sqlite3

    database = tmp_path / "legacy-active-state.sqlite3"
    payload = json.dumps(
        {"version": 1, "layers": [None]}, sort_keys=True, separators=(",", ":")
    )
    with sqlite3.connect(database) as connection:
        connection.execute(
            "CREATE TABLE model_state ("
            "identity TEXT PRIMARY KEY, payload TEXT NOT NULL, updated_at REAL NOT NULL)"
        )
        connection.execute(
            "INSERT INTO model_state VALUES (?, ?, ?)", ("legacy", payload, 1.0)
        )

    with PersistentMemoryStore(database) as store:
        row = store.connection.execute(
            "SELECT payload_sha256 FROM model_state WHERE identity = ?", ("legacy",)
        ).fetchone()
        assert row["payload_sha256"] is None
        assert store.load_model_state("legacy") == (None,)

        store.save_model_state("legacy", (None,))
        assert store.connection.execute(
            "SELECT payload_sha256 FROM model_state WHERE identity = ?", ("legacy",)
        ).fetchone()["payload_sha256"]


def test_snapshot_is_immutable_and_detects_corruption(tmp_path):
    import sqlite3

    with PersistentMemoryStore(tmp_path / "snapshots.sqlite3") as store:
        digest = store.save_snapshot("a", "v1", (None,))
        assert store.save_snapshot("b", "v1", (None,)) == digest
        with pytest.raises(sqlite3.IntegrityError):
            store.save_snapshot("a", "v1", ())
        assert store.load_snapshot("a", "v1") == (None,)
        store.connection.execute("UPDATE model_snapshot SET payload = '{}' WHERE identity = 'a'")
        store.connection.commit()
        with pytest.raises(ValueError, match="checksum"):
            store.load_snapshot("a", "v1")
        with pytest.raises(ValueError):
            store.save_snapshot("a", "", ())
        with pytest.raises(ValueError):
            store.save_snapshot("a", "bad-metadata", (), {"seed": 1})


def test_snapshot_detects_provenance_tampering(tmp_path):
    with PersistentMemoryStore(tmp_path / "snapshot-provenance.sqlite3") as store:
        store.save_snapshot("a", "v1", (None,), {"checkpoint": "weights-1"})
        assert store.list_snapshots("a")[0]["provenance_integrity"] == "verified"

        store.connection.execute(
            "UPDATE model_snapshot SET provenance = ? WHERE identity = ? AND label = ?",
            ('{"checkpoint":"weights-2"}', "a", "v1"),
        )
        store.connection.commit()

        with pytest.raises(ValueError, match="provenance checksum"):
            store.list_snapshots("a")
        with pytest.raises(ValueError, match="provenance checksum"):
            store.load_snapshot("a", "v1")


def test_legacy_snapshot_migration_marks_provenance_unverified(tmp_path):
    import hashlib
    import sqlite3

    database = tmp_path / "legacy-snapshot.sqlite3"
    payload = json.dumps(
        {"version": 1, "layers": [None]}, sort_keys=True, separators=(",", ":")
    )
    with sqlite3.connect(database) as connection:
        connection.execute(
            "CREATE TABLE model_snapshot ("
            "identity TEXT NOT NULL, label TEXT NOT NULL, payload TEXT NOT NULL, "
            "sha256 TEXT NOT NULL, provenance TEXT NOT NULL, created_at REAL NOT NULL, "
            "PRIMARY KEY (identity, label))"
        )
        connection.execute(
            "INSERT INTO model_snapshot VALUES (?, ?, ?, ?, ?, ?)",
            (
                "legacy",
                "before-migration",
                payload,
                hashlib.sha256(payload.encode("utf-8")).hexdigest(),
                '{"seed":"legacy"}',
                1.0,
            ),
        )

    with PersistentMemoryStore(database) as store:
        legacy = store.list_snapshots("legacy")[0]
        assert legacy["provenance"] == {"seed": "legacy"}
        assert legacy["provenance_integrity"] == "legacy_unverified"
        assert store.load_snapshot("legacy", "before-migration") == (None,)

        store.save_snapshot("legacy", "after-migration", (None,), {"seed": "new"})
        assert store.list_snapshots("legacy")[1]["provenance_integrity"] == "verified"


def test_identity_reset_is_scoped_and_preserves_snapshots_by_default(tmp_path):
    with PersistentMemoryStore(tmp_path / "reset.sqlite3") as store:
        for identity in ("a", "b"):
            store.remember(
                identity,
                MemoryRecord(
                    content=f"memory for {identity}",
                    vector=[1.0, 0.0],
                    resonance_vector=[1.0, 0.0],
                ),
            )
            store.save_model_state(identity, (None,))
            store.save_self_model(SelfModel(identity=identity, goals=["persist"]))
            store.save_snapshot(identity, "baseline", (None,), {"identity": identity})

        deleted = store.reset_identity("a")

        assert deleted == {
            "memory": 1,
            "model_state": 1,
            "self_model": 1,
            "model_snapshot": 0,
        }
        assert store.recall("a", [1.0, 0.0]) == []
        assert store.load_model_state("a") is None
        reset_self_model = store.load_self_model("a")
        assert reset_self_model.identity == "a"
        assert reset_self_model.goals == []
        assert reset_self_model.cycle_count == 0
        assert [snapshot["label"] for snapshot in store.list_snapshots("a")] == ["baseline"]

        assert [record.content for record in store.recall("b", [1.0, 0.0])] == ["memory for b"]
        assert store.load_model_state("b") == (None,)
        assert store.load_self_model("b").goals == ["persist"]
        assert [snapshot["label"] for snapshot in store.list_snapshots("b")] == ["baseline"]

        snapshot_deleted = store.reset_identity("a", include_snapshots=True)
        assert snapshot_deleted == {
            "memory": 0,
            "model_state": 0,
            "self_model": 0,
            "model_snapshot": 1,
        }
        assert store.list_snapshots("a") == []

        with pytest.raises(ValueError, match="identity cannot be empty"):
            store.reset_identity("  ")


@pytest.mark.parametrize("identity", ["", "  ", None])
def test_active_state_save_rejects_blank_identity_before_writing(tmp_path, identity):
    with PersistentMemoryStore(tmp_path / "invalid-state-identity.sqlite3") as store:
        with pytest.raises(ValueError, match="identity"):
            store.save_model_state(identity, (None,))

        count = store.connection.execute("SELECT COUNT(*) FROM model_state").fetchone()[0]
        assert count == 0


@pytest.mark.parametrize("identity", ["", "  ", None])
def test_self_model_save_rejects_blank_identity_before_writing(tmp_path, identity):
    model = SelfModel(identity=identity, goals=["must not persist"])
    original_updated_at = model.updated_at
    with PersistentMemoryStore(tmp_path / "invalid-self-identity.sqlite3") as store:
        with pytest.raises(ValueError, match="identity"):
            store.save_self_model(model)

        count = store.connection.execute("SELECT COUNT(*) FROM self_model").fetchone()[0]
        assert count == 0
        assert model.updated_at == original_updated_at


@pytest.mark.parametrize("identity", ["", "  ", None])
def test_identity_scoped_reads_reject_invalid_identity(tmp_path, identity):
    with PersistentMemoryStore(tmp_path / "invalid-read-identity.sqlite3") as store:
        with pytest.raises(ValueError, match="identity"):
            store.recall(identity, [1.0, 0.0])
        with pytest.raises(ValueError, match="identity"):
            store.load_model_state(identity)
        with pytest.raises(ValueError, match="identity"):
            store.load_self_model(identity)
        with pytest.raises(ValueError, match="identity"):
            store.list_snapshots(identity)
        with pytest.raises(ValueError, match="identity"):
            store.load_snapshot(identity, "baseline")


@pytest.mark.parametrize(
    "payload, message",
    [
        ({"version": 1, "layers": {}}, "layers must be a list"),
        (
            {
                "version": 1,
                "layers": [
                    {
                        "dtype": "int64",
                        "attractors": [[[1.0]]],
                        "strengths": [[1.0]],
                        "timestep": [1.0],
                    }
                ],
            },
            "unsupported tensor dtype",
        ),
        (
            {
                "version": 1,
                "layers": [
                    {
                        "dtype": "float32",
                        "attractors": [[[1.0]], [[2.0]]],
                        "strengths": [[1.0]],
                        "timestep": [1.0, 2.0],
                    }
                ],
            },
            "batch dimensions do not match",
        ),
        (
            {
                "version": 1,
                "layers": [
                    {
                        "dtype": "float32",
                        "attractors": [[[1.0], [2.0, 3.0]]],
                        "strengths": [[1.0, 1.0]],
                        "timestep": [1.0],
                    }
                ],
            },
            "invalid attractor vectors",
        ),
    ],
)
def test_persistent_state_rejects_invalid_tensor_metadata(payload, message):
    with pytest.raises(ValueError, match=message):
        PersistentMemoryStore._decode_state(json.dumps(payload))


def test_persistent_state_rejects_nonfinite_json_constants():
    serialized = (
        '{"version":1,"layers":[{"dtype":"float32",'
        '"attractors":[[[NaN]]],"strengths":[[1]],"timestep":[1]}]}'
    )
    with pytest.raises(ValueError, match="invalid numeric constant"):
        PersistentMemoryStore._decode_state(serialized)
