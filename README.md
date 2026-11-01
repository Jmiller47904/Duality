# DRAI: Dynamic Resonance AI

**DRAI introduces persistent internal state into transformer models using a lightweight, fully differentiable mechanism based on attractor dynamics.**  
The goal is practical: improve stability, coherence, and short-term memory during inference without modifying pre-training, without external memory systems, and without non-standard operations.

Everything demonstrated here is reproducible, measurable, and implemented entirely with standard PyTorch and transformer components.

---

## Why This Exists

Modern transformers handle “memory” by repeatedly recomputing or externally retrieving context:

- **Context windows:** brute-force, quadratic compute  
- **RAG:** external documents, external orchestration  
- **Agent frameworks:** Python-level simulation of working memory  
- **Chain-of-thought:** text-based scaffolding, not internal state  

All of these operate *outside* the model.  
A transformer itself has **no persistent internal state** across forward passes.

**DRAI fills that gap.**

---

## Overview

DRAI introduces a small, persistent attractor field inside the model.  
This field is interpreted by specific attention heads at chosen layers, producing a structured, evolving “cloud” of state that:

- persists across tokens  
- conditions future computation  
- supports lightweight working-memory behaviour  
- adds <1% compute overhead  

The mechanism is dynamic, but the implementation is simple: a static vector, a projection step, and a controlled influence path into existing attention circuitry.

---

## Quick Example

```python
from transformers import AutoModelForCausalLM
from src.drai import apply_drai_v1, get_v1_conservative_config

model = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-410m")

# Add internal state
config = get_v1_conservative_config()
model = apply_drai_v1(model, config=config)

# Use normally
outputs = model.generate(inputs)
```

No training.  
No fine-tuning.  
Your model now carries persistent internal state during inference.

## Native DualityLM

The repository now also includes `duality_lm`, a trainable decoder-only model
that makes the Duality idea part of the model architecture rather than only a
post-hoc patch. It provides causal self-attention, explicit attractor memory in
selected blocks, prompt-prefill plus KV-cache generation, and a dependency-free
byte tokenizer for bootstrap experiments.

```python
import torch
from duality_lm import ByteTokenizer, DualityLM, DualityLMConfig

tokenizer = ByteTokenizer()
config = DualityLMConfig.tiny(vocab_size=tokenizer.vocab_size)
model = DualityLM(config)
prompt = torch.tensor([tokenizer.encode("Duality remembers")])
generated = model.generate(prompt, max_new_tokens=32, do_sample=False)
print(tokenizer.decode(generated[0].tolist()))
```

This native model is an untrained research foundation until it is fitted to a
corpus. See [`docs/DUALITY_LM.md`](docs/DUALITY_LM.md) for the architecture,
training command, state-passing API, and current limitations.

## Persistent Cognition

`duality_lm` now includes an experimental continuity layer that can preserve
attractor tensors across process restarts, store episodic/semantic/self memory,
perform resonance-weighted recall, maintain an inspectable self-model, measure
metacognitive state, retain sourced claims, surface contradiction candidates,
and execute one bounded reflection cycle at a time. A
reflection's self-memory, updated self-model, and optional attractor checkpoint
are committed together; a database write failure rolls those outputs back. Recall
activation counts remain a separate record of which memories were selected.
Identity resets likewise delete selected records in one SQLite transaction; if a
delete fails, earlier deletes in that reset roll back together.

```python
from duality_lm import ContinuityEngine, PersistentMemoryStore

with PersistentMemoryStore("duality.sqlite3") as store:
    engine = ContinuityEngine(store, identity="duality-01")
    engine.observe("The first session began.", [1.0, 0.0, 0.0])
```

See [`docs/PERSISTENT_COGNITION.md`](docs/PERSISTENT_COGNITION.md) for the API,
architecture, evidence ladder, and evaluation plan. These mechanisms implement
functional continuity and self-monitoring; they do not prove subjective
consciousness.

Run `python benchmarks/continuity_smoke.py` for a small CPU-only, JSON-producing
check of snapshot replay after reopening SQLite, provenance verification,
contradiction candidates, and identity-scoped reset/isolation. It uses an
untrained tiny memory layer and synthetic fixtures; it measures persistence
invariants, not model quality or evidence of consciousness. Run its regression
check with `pytest tests/unit/test_continuity_smoke.py`.

---

## What You Get

DRAI enables several measurable computational primitives:

- **Persistent state across tokens**  
- **Self-conditioning** across layers  
- **Compositional binding** inside the residual stream  
- **Goal and context persistence** without external memory  
- **Stable multi-step reasoning at small scales**

These emerge from standard transformer operations, not architectural hacks.

---

## Results

**Story Comprehension (Pythia-410M)**  
- Historical baseline: **85.0%** (51/60)
- Historical DRAI V1: **88.3%** (53/60)
- Historical difference: **+3.3 percentage points**
- Runtime overhead depends on hardware, sequence length, and configuration. Use
  `python benchmarks/native_memory_ablation.py` for a same-host memory-on/off
  latency and memory-cost measurement; no universal overhead bound is claimed.

These figures came from one small synthetic story run whose generator was seeded,
but whose model decoding used unseeded sampling. They are historical observations,
not a reproducible result or evidence of general improvement. The current story
benchmark uses greedy decoding and a fixed initialization seed; its new scores
must be measured before making a comparison. See
[`benchmarks/story_comprehension_410m/README.md`](benchmarks/story_comprehension_410m/README.md).

---

## Reproduce in 5–10 Minutes

```bash
git clone https://github.com/HalcyonAIR/Duality.git
cd Duality
pip install -r requirements.txt
bash scripts/run_reproduction_410m.sh
```

Runs on CPU.

---

## How It Works

### 1. The Attractor Field (Seed)
A small, persistent vector is injected at a designated layer.  
Its job is not to store information directly — it provides a stable reference frame.

### 2. Head-Specific Interpretation (Cascade)
Different attention heads interpret the field differently:

- head 1 → lexical cues  
- head 12 → syntactic structure  
- head 28 → semantic relations  
- head 45 → task context  
*(actual mappings vary by model, this is representative)*

The diversity of interpretations produces a rich, multi-dimensional internal state.

### 3. The Cloud (State)
Across layers, these interpretations accumulate into a **dynamic attractor cloud** inside the residual stream.

This cloud:  
- persists across tokens  
- shapes the next-step computation  
- adapts as the sequence unfolds  

The static field is the anchor; the cloud is the working memory.

---

## Mathematics (Intuition Only)

A simplified formulation:

```
F(q; M_A) = θ(‖S‖) · π_M_A(q̂)
```

Where:

- **π_M_A** — soft projection onto an attractor manifold  
- **θ** — burn-in / gating function  
- **M_A** — evolving attractor state  

For the full derivation, see:  
`docs/DRAI_V1_INVARIANT.md`

---

## Configurations

### Conservative (recommended for 400M–1B)

```python
config = get_v1_conservative_config()
```

- max_attractors: 16  
- max_influence_scale: 0.15  
- burn_in_threshold: 50 tokens  
- layer_mode: "mid"

### Standard (recommended for 7B+)

```python
from src.drai import get_v1_standard_config
config = get_v1_standard_config()
```

- max_attractors: 64  
- max_influence_scale: 0.5  
- multi-layer integration

See `src/drai/config.py` for details.

---

## Benchmarks

| Model           | Task                 | Baseline | DRAI V1 | Δ     | Status       |
|----------------|----------------------|----------|---------|-------|---------------|
| pythia-410m    | Story comprehension  | 85.0%*   | 88.3%*  | +3.3 pp* | Historical single run |
| pythia-70m     | Story comprehension  | 0.0%     | 0.0%    | 0.0%  | Too small     |

\* Historical scores used unseeded sampling; the deterministic benchmark protocol
has not been rerun, so these figures are not a result for the current protocol.

### Predictions (based on scaling behaviour)

*These are informed expectations, not claims.*

- **7B class:** +8–12% on memory-heavy tasks  
- **70B class:** +15–25% with additional emergent structure  

---

## Repository Structure

```
src/drai/            # Core implementation
tests/               # 52 passing tests
benchmarks/          # Reproducible tasks
examples/            # Quickstart scripts
docs/                # Theory and mathematical foundations
```

Key documents:

- **THE_CLOUD_MECHANISM.md** — explains seed → cascade → cloud  
- **COMPUTATIONAL_SIGNATURES.md** — measurable behaviours & order parameters  
- **DRAI_V1_INVARIANT.md** — invariant equation and scale properties  

---

## Roadmap

### 7B+ Validation (In Progress)
- Multi-layer coupling  
- Longer-range state persistence  
- Contradiction-handling behaviour  
- Compositional binding at scale  

### Integration Targets
- GPT-NeoX (pythia family) — ✔️  
- LLaMA / LLaMA2 — in progress  
- Mistral — planned  
- Qwen — planned  

We actively welcome community experiments.

---

## FAQ

**How is this different from RAG?**  
RAG is external memory. DRAI is internal state.

**How is this different from long context windows?**  
Long context recomputes everything. DRAI maintains an evolving internal state.

**Does this work with fine-tuned models?**  
Yes — it is fully post-hoc.

**Does this require retraining?**  
No. It attaches cleanly to pretrained models.

**Is this a cognitive architecture?**  
No. This project focuses strictly on engineering persistent internal state.

---

## Citation

If you use this project:

```
@software{drai2025,
  title={DRAI: Dynamic Resonance AI},
  author={Halcyon AI Research},
  year={2025},
  url={https://github.com/HalcyonAIR/Duality}
}
```

---

## License

Apache 2.0 – see `LICENSE` for details.

---

## Contact

- Issues: https://github.com/HalcyonAIR/Duality/issues  
- Discussions: https://github.com/HalcyonAIR/Duality/discussions  

**Built by Halcyon AI Research**  
Internal state for transformers — finally.
