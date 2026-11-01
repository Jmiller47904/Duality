# Native DualityLM

`DualityLM` is the first trainable decoder-only model in this repository. It
keeps the original DRAI V1 integration intact for augmenting pretrained
Hugging Face models, while adding a native architecture that can be trained
from an explicit configuration.

## What is new

Each selected decoder block contains:

1. pre-normalized causal self-attention;
2. a bounded attractor cloud with configurable slots;
3. a gated memory residual read from that cloud; and
4. a gated feed-forward sublayer.

The memory state is explicit rather than hidden in module parameters. A call
returns the next state, and a later call can pass it back alongside the
attention KV cache. This supports both teacher-forced training and
prompt-prefill plus token-by-token generation.

For inference that keeps state inside one model instance, pass
`persistent_memory=True`. `return_memory_state=False` hides the state from the
call result while still retaining it for the next persistent call. Use
`reset_memory()` to clear that retained state. These options control output
visibility and in-process persistence independently.

The state update is detached from autograd. That gives the model a recurrent
working-memory signal without retaining an unbounded computation graph across
an entire conversation. The query and read projections, along with the
influence gate, remain trainable.

## Quickstart

Install the project with its development dependencies:

```bash
python -m pip install -e ".[dev]"
```

Run the native smoke example:

```bash
python examples/duality_lm_toy.py
```

The example prints the logits shape, active-memory statistics, and a short
untrained greedy generation. Random output is expected until the model has
been trained.

## Minimal API

```python
import torch

from duality_lm import ByteTokenizer, DualityLM, DualityLMConfig

tokenizer = ByteTokenizer()
config = DualityLMConfig.tiny(vocab_size=tokenizer.vocab_size)
model = DualityLM(config)

prompt = torch.tensor([tokenizer.encode("Duality remembers")])
output = model(prompt, use_cache=True)

next_output = model(
    torch.tensor([[tokenizer.eos_token_id]]),
    memory_state=output.memory_state,
    past_key_values=output.past_key_values,
    use_cache=True,
)

generated = model.generate(prompt, max_new_tokens=32, do_sample=False)
print(tokenizer.decode(generated[0].tolist()))
```

For a plain-text bootstrap corpus, use:

```bash
python scripts/train_duality_lm.py \
  --text-file corpus.txt \
  --steps 500 \
  --output checkpoints/duality_lm.pt
```

The default model is deliberately a CPU-friendly starting point. Increase
`d_model`, `n_layers`, `n_heads`, `max_seq_len`, and the corpus size for a
serious experiment. The byte tokenizer is dependency-free and robust, but a
larger run should eventually replace it with a learned subword tokenizer.

## Memory placement and diagnostics

By default, memory is attached to the middle transformer block:

```python
config = DualityLMConfig.small(
    vocab_size=tokenizer.vocab_size,
    memory_layers=(2, 3),
)
```

Set `memory_layers=()` to build the same parameterized decoder without the
memory pathway for a baseline or ablation run.

After a forward pass:

```python
print(model.memory_stats(output.memory_state))
```

The diagnostics expose active slots, total strength, mean strength, and the
number of tokens processed by each memory-enabled block. These values are
observability signals, not a claim that the model has human-like memory or
reasoning.

### CPU memory ablation

Run a deterministic tiny-model comparison without downloading a checkpoint:

```bash
python benchmarks/native_memory_ablation.py --iterations 7
```

The JSON report compares median memory-on/off forward latency, parameter bytes,
retained state bytes, active slots, and the mean absolute logit change between a
continued state and a reset state. Both models share the same non-memory weights.
Latency varies by host; compare runs only on the same machine. An optional
`--max-latency-ratio` turns a locally established ratio into a regression gate.
The callable benchmark restores PyTorch's CPU RNG state and thread count on
completion or failure, so it does not perturb later experiments in the same
process.

This benchmark uses an untrained tiny model. A nonzero continuation delta shows
that retained state changed computation; it does not show better recall,
accuracy, reasoning, identity, or consciousness.

## Scope of this milestone

This is a trainable research foundation, not a competitive pretrained LLM.
The repository still needs a curated corpus, tokenizer training, longer
training runs, evaluation against a baseline of the same parameter count, and
ablation studies that compare memory-enabled and memory-disabled models.
