"""Tests for durable memory, self-modeling, and bounded reflection."""

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
