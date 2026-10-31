# Story Comprehension Benchmark - Pythia-410M

This exploratory benchmark measures exact-match answers on one small synthetic
story set. It is not a production-readiness test or a general model-quality claim.

## Historical result (pre-determinism fix)

```
Baseline (no DRAI):  85.0% accuracy (51/60 correct)
V1 DRAI:             88.3% accuracy (53/60 correct)
Delta:               +3.3 percentage points
```

These values are retained as historical observations. The earlier script used
stochastic sampling without seeding the model-generation RNG; the story generator
seed alone did not make the scores reproducible. They are not the expected output
for the current script and should not be used as evidence of a general improvement.

## What This Tests

- **Task**: Answer factual questions about short stories
- **Model**: EleutherAI/pythia-410m (~410M parameters)
- **Dataset**: 20 synthetic stories with 3 questions each (60 total)
- **Metric**: Exact-match accuracy

## Quick Start

```bash
# Install dependencies
pip install torch transformers

# Run benchmark (takes ~5-10 minutes on CPU)
python run.py
```

## What Happens

1. **Generate test stories** (20 stories, 60 questions)
2. **Test baseline** (pythia-410m without DRAI)
3. **Test V1 DRAI** (pythia-410m with DRAI V1)
4. **Compare results** on this synthetic task; no improvement is assumed

## Expected Output

```
================================
FINAL COMPARISON
================================

Model                          Accuracy        Correct
----------------------------------------------------------------------
Baseline (no DRAI)                 current run output
V1 DRAI (experimental)             current run output
----------------------------------------------------------------------

Delta: computed from the current run

V1 Attractor Statistics:
  - Layers with DRAI: 1
  - Total active attractors: 16
  - Total strength: 429.89
  - Burn-in complete: YES
  - Timesteps processed: 1200

====================================
VERDICT
====================================
Task-specific comparison only; not a generalization result
```

## Configuration

V1 uses conservative hyperparameters for pythia-410m:

```python
max_attractors = 16               # Gentle memory capacity
theta_match = 0.8                 # Conservative (only strong matches)
alpha_update = 0.05               # Slow, stable updates
lambda_decay = 0.995              # Slow decay (long memory)
max_influence_scale = 0.15        # Gentle influence (15% max)
burn_in_threshold = 50.0          # ~10 tokens typical
layer_mode = "mid"                # Only layer 12 (middle layer)
```

## Files

- `run.py` - Main benchmark script
- `story_generator.py` - Synthetic story generation
- `README.md` - This file

## Current evaluation protocol

The current script uses greedy decoding (`do_sample=False`), seeds model
initialization, and uses seed 42 for synthetic story generation. Greedy decoding
removes token-sampling randomness; results can still vary with library versions
or hardware. The updated protocol has not been run to refresh the historical
scores above.

```bash
python run.py
```

Results are saved to `results/comparison.json` when the benchmark is run.

## Why This Matters

This measures one narrow behavioral task. A single score does not establish
generalization, causality, or production readiness. Repeat the comparison with
published seeds, model/library versions, and additional held-out tasks before
making broader performance claims.

**Next**: Add held-out memory and continuity tasks with explicit seeds and report
per-run results rather than extrapolating from this benchmark.
