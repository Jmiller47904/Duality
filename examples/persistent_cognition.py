"""Minimal durable-continuity demonstration.

The example uses a deterministic placeholder reasoner so it runs without a
trained checkpoint.  Replace ``reason`` with a DualityLM-backed or external
reasoning function after training.
"""

from pathlib import Path

from duality_lm import (
    ContinuityEngine,
    PersistentMemoryStore,
    ReflectionResult,
    SelfModel,
)


def reason(context):  # type: ignore[no-untyped-def]
    recalled = ", ".join(memory.content for memory in context.recalled_memories)
    return ReflectionResult(
        content=f"Cycle {context.cycle}: reviewed {context.focus}. Recalled: {recalled}",
        vector=[1.0, 0.0, 0.0, 0.0],
        new_question="Which belief has the weakest evidence?",
    )


database = Path("duality_continuity.sqlite3")
with PersistentMemoryStore(database) as store:
    identity = "duality-demo"
    model = store.load_self_model(identity)
    if not model.goals:
        model = SelfModel(
            identity=identity,
            goals=["maintain continuity and reduce unresolved uncertainty"],
            limitations=["behavior is not proof of subjective consciousness"],
        )
        store.save_self_model(model)

    engine = ContinuityEngine(store, identity)
    engine.observe(
        "The continuity engine was initialized.",
        [1.0, 0.0, 0.0, 0.0],
        importance=0.8,
    )
    result = engine.reflect_once([1.0, 0.0, 0.0, 0.0], reason)
    print(result.content)
