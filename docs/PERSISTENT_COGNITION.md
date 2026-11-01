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

Every operation on identity-scoped data, including reads, writes, and reset,
requires a non-empty, non-whitespace identity key. This applies to active model
state, self-models, memories, reflection cycles, and named snapshots. Identity
values are preserved as supplied; the check does not silently normalize or
merge keys.

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

Memory inputs are validated before storage in both `remember()` and the atomic
`save_reflection_cycle()` path. In particular, `decay_rate` must be finite and
non-negative, so neither write path can introduce non-finite recall scores.
Cosine similarity rescales finite vectors before its dot product and norm
calculations, preventing intermediate overflow for large finite values.

This is deliberately not hidden behind a database-specific nearest-neighbor
implementation. Each factor can be measured or removed in an ablation.

### Self-model

`SelfModel` stores identity-scoped goals, calibrated beliefs, capabilities,
limitations, unresolved questions, cycle count, and the last reflection. It is
explicit so an evaluator can inspect changes and detect contradictions or
identity drift.

Memory record IDs are identity-bound. Reusing an existing record ID for a
different identity is rejected transactionally, so a caller cannot silently move
or overwrite another identity's stored memory.

Self-model saves use strict JSON serialization. Non-finite values such as
`NaN` and infinity are rejected before the stored row or caller's `updated_at`
is changed, so an invalid replacement cannot silently replace the last valid
self-model. Reflection-cycle persistence applies the same strict JSON rule.

`reset_identity(identity)` removes that identity's long-term memories, sourced claims,
active model state, and self-model in one SQLite transaction, returning a row count for each
store. It leaves named snapshots intact by default so experiment evidence is not
silently destroyed. Pass `include_snapshots=True` only when those checkpoints must
also be deleted. Other identities are never affected. This is a reset of stored
functional state, not a claim about identity or consciousness.

### Sourced claims and contradiction candidates

`ClaimRecord` stores an immutable subject, predicate, JSON value, confidence,
and nonempty caller-supplied provenance. `record_claim()` preserves each
observation rather than overwriting an earlier value. `list_claims()` keeps the
ledger identity-scoped, and `find_contradictions()` returns pairs whose subject
and predicate match but whose canonical JSON values differ.

These pairs are review candidates. The store does not determine which source is
correct, infer logical negation, reconcile units or time periods, or revise the
self-model. Equivalent JSON objects with a different key order are treated as
the same value. This provides measurable provenance and conflict behavior; it
does not demonstrate understanding, truth, identity, or consciousness.

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
5. update questions, cycle count, and the optional attractor checkpoint.

After the reasoner returns, the reflection self-memory, revised self-model, and
optional checkpoint are validated and written in one SQLite transaction. A
database write failure rolls back those output rows together. Recall activation
counts are updated when memories are selected, before the reasoner runs; they
remain a separate record of the recall attempt.

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

### Reproducible persistence smoke evaluation

Run `python benchmarks/continuity_smoke.py` to emit a small JSON result for
snapshot replay after closing and reopening SQLite, provenance checksum status,
contradiction-candidate count, and identity-scoped reset row counts and
isolation. The fixture is seeded, uses an untrained tiny CPU memory layer, and
does not download weights or datasets. The focused regression check is
`pytest tests/unit/test_continuity_smoke.py`.

This is a persistence and reset regression evaluation only. It does not measure
recall quality, task performance, model continuity quality, or subjective
experience. Its deterministic values describe this fixed synthetic fixture;
they are not evidence of general model improvement.


## Named memory snapshots

Use `store.save_snapshot(identity, label, state, provenance={"checkpoint": "weights-id"})`
to retain a named attractor checkpoint without overwriting an earlier one. A duplicate
identity/label raises `sqlite3.IntegrityError`. `list_snapshots(identity)` returns labels,
creation times, SHA-256 payload checksums, user-supplied provenance, and whether the
provenance checksum is verified. `load_snapshot(identity, label)` verifies the payload
and any available provenance checksum, then returns independent tensors; it does not
replace the active checkpoint. Missing snapshots raise `KeyError`.

Snapshots contain attractor state only, not weights, KV caches, RNG state, semantic
records or the self-model. Exact continuation replay requires the same model weights,
configuration, input, cache context and deterministic execution settings. The CPU test
replays a fixed continuation through a memory layer after reopening the database.
New snapshots checksum canonical provenance JSON as well as the state payload, so
editing provenance without updating its checksum is detected by listing or loading.
Snapshots created before this migration remain loadable and are marked
`legacy_unverified` because no provenance checksum was recorded. Checksums detect
accidental corruption; they are not signatures and do not establish the truth of
caller-supplied provenance. Keep snapshot databases trusted and local. This is an
experiment reproducibility feature, not evidence of consciousness.

Restores also reject malformed JSON, unsupported or non-floating dtypes, non-finite
values, ragged tensor shapes, inconsistent batch dimensions, excessive layer counts,
oversized payloads, and states containing more than 16 million tensor values. These
bounds reduce accidental or hostile resource exhaustion when a database is damaged or
replaced. They do not make an untrusted SQLite database safe; use trusted local files.
