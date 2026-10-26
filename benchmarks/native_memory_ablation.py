"""Deterministic CPU measurements for the native Duality memory pathway.

This benchmark measures implementation behavior on an untrained tiny model. It
does not measure recall quality, intelligence, consciousness, or production
throughput. Latency results are meaningful only when compared on the same host.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from typing import Any, Dict, Iterable, Optional

import torch
from torch import Tensor

from duality_lm import DualityLM, DualityLMConfig


def _tensor_bytes(tensors: Iterable[Tensor]) -> int:
    return sum(tensor.numel() * tensor.element_size() for tensor in tensors)


def _parameter_bytes(model: DualityLM) -> int:
    return _tensor_bytes(model.parameters())


def _memory_state_bytes(state: Optional[tuple]) -> int:
    if state is None:
        return 0
    tensors = []
    for layer in state:
        if layer is not None:
            tensors.extend((layer.attractors, layer.strengths, layer.timestep))
    return _tensor_bytes(tensors)


def _median_forward_ms(model: DualityLM, tokens: Tensor, warmup: int, iterations: int) -> float:
    with torch.inference_mode():
        for _ in range(warmup):
            model(tokens)
        timings = []
        for _ in range(iterations):
            started = time.perf_counter()
            model(tokens)
            timings.append((time.perf_counter() - started) * 1000.0)
    return statistics.median(timings)


def _copy_shared_weights(source: DualityLM, target: DualityLM) -> None:
    """Give the ablation pair identical weights outside the memory module."""

    source_state = source.state_dict()
    target_state = target.state_dict()
    for name, value in source_state.items():
        if name in target_state and target_state[name].shape == value.shape:
            target_state[name] = value.detach().clone()
    target.load_state_dict(target_state)


def run_benchmark(
    *, seed: int = 17, sequence_length: int = 16, warmup: int = 2, iterations: int = 7
) -> Dict[str, Any]:
    """Return JSON-friendly memory-on/off CPU measurements."""

    if sequence_length < 4 or sequence_length > 64:
        raise ValueError("sequence_length must be between 4 and 64")
    if warmup < 0 or iterations <= 0:
        raise ValueError("warmup must be >= 0 and iterations must be > 0")

    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        torch.manual_seed(seed)
        memory_config = DualityLMConfig.tiny(max_seq_len=64, memory_layers=(2,))
        memory_model = DualityLM(memory_config).eval()
        baseline_config = DualityLMConfig.tiny(max_seq_len=64, memory_layers=())
        baseline_model = DualityLM(baseline_config).eval()
        _copy_shared_weights(memory_model, baseline_model)

        generator = torch.Generator().manual_seed(seed + 1)
        tokens = torch.randint(
            3,
            memory_config.vocab_size,
            (1, sequence_length),
            generator=generator,
        )
        baseline_ms = _median_forward_ms(baseline_model, tokens, warmup, iterations)
        memory_ms = _median_forward_ms(memory_model, tokens, warmup, iterations)

        split = sequence_length // 2
        with torch.inference_mode():
            primer = memory_model(tokens[:, :split])
            continued = memory_model(tokens[:, split:], memory_state=primer.memory_state)
            reset = memory_model(tokens[:, split:])
        assert primer.memory_state is not None
        assert continued.memory_state is not None
        continuation_delta = float((continued.logits - reset.logits).abs().mean().item())
        active_slots = sum(
            int((layer.strengths > 1e-3).sum().item())
            for layer in primer.memory_state
            if layer is not None
        )

        baseline_parameters = _parameter_bytes(baseline_model)
        memory_parameters = _parameter_bytes(memory_model)
        state_bytes = _memory_state_bytes(primer.memory_state)
        return {
            "benchmark": "native-memory-ablation-v1",
            "seed": seed,
            "sequence_length": sequence_length,
            "warmup_iterations": warmup,
            "measured_iterations": iterations,
            "memory_off": {
                "median_forward_ms": baseline_ms,
                "parameter_bytes": baseline_parameters,
            },
            "memory_on": {
                "median_forward_ms": memory_ms,
                "parameter_bytes": memory_parameters,
                "retained_state_bytes": state_bytes,
                "active_slots_after_primer": active_slots,
            },
            "comparison": {
                "latency_ratio": memory_ms / baseline_ms,
                "extra_parameter_bytes": memory_parameters - baseline_parameters,
                "continuation_logit_mean_abs_delta": continuation_delta,
            },
            "interpretation": (
                "Untrained-model implementation measurements; no accuracy, recall, "
                "reasoning, or consciousness claim. Compare latency only on the same host."
            ),
        }
    finally:
        torch.set_num_threads(previous_threads)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--sequence-length", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=7)
    parser.add_argument(
        "--max-latency-ratio",
        type=float,
        help="Optional same-host regression budget; exits nonzero when exceeded.",
    )
    args = parser.parse_args()
    result = run_benchmark(
        seed=args.seed,
        sequence_length=args.sequence_length,
        warmup=args.warmup,
        iterations=args.iterations,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if (
        args.max_latency_ratio is not None
        and result["comparison"]["latency_ratio"] > args.max_latency_ratio
    ):
        raise SystemExit("memory-on latency exceeded the requested same-host budget")


if __name__ == "__main__":
    main()
