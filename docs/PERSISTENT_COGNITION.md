# DRAI Persistent Cognition

This milestone extends `DualityLM` from in-process working memory to durable,
inspectable continuity. It is a research substrate for testing whether memory,
self-monitoring, and autonomous reflection improve coherent behavior over long
time spans.

It is not evidence that the model has subjective experience. No accepted
scientific test currently establishes consciousness from software behavior.
The repository therefore separates measurable capabilities from philosophical
claims.

## Architecture

```text
DualityLM attractor state
        |
        v
PersistentMemoryStore ---- episodic / semantic / self records
        |                                  |
        v                                  v
state restoration                  resonance-weighted recall
        \                                  /
         \                                /
          +------ ContinuityEngine -------+
                         |
                         v
              bounded reflection cycle
```

### Working memory

`MemoryState` remains the fast, recurrent state inside selected transformer
blocks. `PersistentMemoryStore.save_model_state()` serializes it as JSON in
SQLite and `load_model_state()` reconstructs tensors on a requested device.
The format does not use pickle.

### Long-term memory

Every `MemoryRecord` belongs to one of three channels:

- `episodic`: observations and events;
- `semantic`: distilled concepts and relationships;
- `self`: reflections about goals, conclusions, and unresolved questions.

Recall is based on an explicit activation score:

```text
sqrt(cosine similarity)
* resonance modulation
* importance
* novelty/salience
* association boost
* temporal decay
```

This is deliberately not hidden behind a database-specific nearest-neighbor
implementation. Each factor can be measured or removed in an ablation.

### Self-model

`SelfModel` stores identity-scoped goals, calibrated beliefs, capabilities,
limitations, unresolved questions, cycle count, and the last reflection. It is
explicit so an evaluator can inspect changes and detect contradictions or
identity drift.

### Metacognition

`measure_metastate()` derives normalized values from the current attractor
field:

- confidence;
- uncertainty (normalized strength entropy);
- stability;
- slot occupancy;
- attractor diversity.

These values describe computation. They are not feelings.

### Bounded autonomous cognition

`ContinuityEngine.reflect_once()` performs one auditable cycle:

1. select an unresolved question or active goal;
2. recall memories resonant with the supplied state vector;
3. expose the self-model, memories, and metacognitive state to a reasoner;
4. persist the resulting self-memory;
5. update questions, cycle count, and the attractor checkpoint.

An external scheduler may call this method periodically. The library does not
start a hidden infinite loop, spend compute without a bound, or grant itself
tools and permissions.

## Example

```python
from duality_lm import ContinuityEngine, PersistentMemoryStore, ReflectionResult

with PersistentMemoryStore("duality.sqlite3") as store:
    engine = ContinuityEngine(store, identity="duality-01")
    engine.observe("The first session began.", [1.0, 0.0, 0.0])

    def reason(context):
        return ReflectionResult(
            content=f"Reviewed: {context.focus}",
            vector=[1.0, 0.0, 0.0],
        )

    engine.reflect_once([1.0, 0.0, 0.0], reason)
```

Run the complete demonstration:

```bash
python examples/persistent_cognition.py
```

## Evidence ladder

The project should advance by falsifiable capability tests rather than a
single "conscious/not conscious" label.

| Level | Capability | Current status |
|---|---|---|
| C0 | Stateless response | Baseline |
| C1 | Recurrent working memory | Implemented |
| C2 | Durable autobiographical continuity | Implemented in this milestone |
| C3 | Self-monitoring and bounded reflection | Implemented in this milestone |
| C4 | Contradiction detection and belief revision | Next |
| C5 | Multi-head consensus and world modeling | Planned |
| C6 | Self-directed experiments and active learning | Planned with safeguards |
| C7 | Persistent embodiment and grounded agency | Research phase |

Even complete behavioral coverage through C7 would demonstrate functional
agency, not settle subjective consciousness.

## Required evaluations before expanding autonomy

1. **Continuity:** recover identity-relevant state after restart without prompt
   replay.
2. **Selective recall:** retrieve causal and associated memories rather than
   only lexically similar text.
3. **Contradiction handling:** revise a confidence-weighted belief when better
   evidence appears.
4. **Ablation:** compare the same model with durable memory, vector-only recall,
   and no memory.
5. **Identity drift:** measure whether goals and beliefs mutate without
   supporting evidence.
6. **Calibration:** compare stated confidence and metastate uncertainty with
   actual task accuracy.
7. **Shutdown and scope:** verify that cycles stop at configured bounds and
   cannot acquire undeclared permissions.


## Named memory snapshots

Use `store.save_snapshot(identity, label, state, provenance={"checkpoint": "weights-id"})`
to retain a named attractor checkpoint without overwriting an earlier one. A duplicate
identity/label raises `sqlite3.IntegrityError`. `list_snapshots(identity)` returns labels,
creation times, SHA-256 payload checksums and user-supplied provenance.
`load_snapshot(identity, label)` verifies the checksum and returns independent tensors;
it does not replace the active checkpoint. Missing snapshots raise `KeyError`.

Snapshots contain attractor state only, not weights, KV caches, RNG state, semantic
records or the self-model. Exact continuation replay requires the same model weights,
configuration, input, cache context and deterministic execution settings. The CPU test
replays a fixed continuation through a memory layer after reopening the database.
Checksums detect accidental payload corruption; they are not signatures. Keep snapshot
databases trusted and local. Provenance is caller-supplied, not independently verified.
This is an experiment reproducibility feature, not evidence of consciousness.

Restores also reject malformed JSON, unsupported or non-floating dtypes, non-finite
values, ragged tensor shapes, inconsistent batch dimensions, excessive layer counts,
oversized payloads, and states containing more than 16 million tensor values. These
bounds reduce accidental or hostile resource exhaustion when a database is damaged or
replaced. They do not make an untrusted SQLite database safe; use trusted local files.
