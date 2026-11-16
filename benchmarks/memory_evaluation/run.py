"""Paired three-condition evaluation of DRAI V1 conversation memory."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn.functional as F

from src.drai import (
    DRAIV1ResonanceLayer,
    apply_v1,
    get_v1_conservative,
    get_v1_state_size_bytes,
    reset_v1_state,
)

try:
    from .cases import CASES, CATEGORIES
except ImportError:
    from cases import CASES, CATEGORIES


CONDITIONS = ("no_drai", "drai_state", "full_history")
_STATE_BUFFER_NAMES = (
    "attractor_vectors",
    "attractor_strengths",
    "attractor_last_used",
    "timestep",
)


def validate_cases(cases: Sequence[Mapping[str, object]]) -> None:
    """Reject malformed or ambiguous cases before loading a large model."""

    seen = set()
    for case in cases:
        case_id = str(case.get("case_id", ""))
        if not case_id or case_id in seen:
            raise ValueError("case_id values must be present and unique")
        seen.add(case_id)
        if str(case.get("category")) not in CATEGORIES:
            raise ValueError(f"unsupported category in {case_id}")
        sessions = case.get("sessions")
        options = case.get("options")
        if not isinstance(sessions, list) or not sessions or not all(
            isinstance(value, str) and value.strip() for value in sessions
        ):
            raise ValueError(f"{case_id} must contain non-empty text sessions")
        if not isinstance(options, list) or len(options) < 2 or not all(
            isinstance(value, str) and value.strip() for value in options
        ):
            raise ValueError(f"{case_id} must contain at least two answer options")
        if len(set(options)) != len(options):
            raise ValueError(f"{case_id} has duplicate answer options")
        if str(case.get("answer")) not in options:
            raise ValueError(f"{case_id} answer must match one answer option")
        if not str(case.get("question", "")).strip():
            raise ValueError(f"{case_id} has an empty question")


def load_cases(path: Optional[Path] = None) -> List[Dict[str, object]]:
    """Load built-in probes or normalized JSONL cases."""

    if path is None:
        cases = [dict(case) for case in CASES]
    else:
        cases = []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if line.strip():
                    try:
                        cases.append(json.loads(line))
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"invalid JSONL on line {line_number}") from exc
    validate_cases(cases)
    return cases


def format_question(case: Mapping[str, object]) -> str:
    options = case["options"]
    option_text = "\n".join(
        f"{index + 1}. {option}" for index, option in enumerate(options)
    )
    return (
        f"Question: {case['question']}\n"
        f"Possible answers:\n{option_text}\n"
        "Answer:"
    )


def format_history(case: Mapping[str, object]) -> str:
    sessions = case["sessions"]
    body = "\n\n".join(
        f"Session {index + 1}:\n{session}" for index, session in enumerate(sessions)
    )
    return "Prior conversation:\n" + body


def token_ids(tokenizer, text: str) -> List[int]:
    return tokenizer.encode(text, add_special_tokens=False)


def score_candidate(model, tokenizer, prefix: str, candidate: str,
                    device: torch.device) -> float:
    """Score a complete option causally, without exposing its future tokens."""

    prefix_tokens = token_ids(tokenizer, prefix)
    candidate_tokens = token_ids(tokenizer, candidate)
    if not prefix_tokens or not candidate_tokens:
        raise ValueError("prompt and candidate must both tokenize to at least one token")

    input_ids = torch.tensor([prefix_tokens], dtype=torch.long, device=device)
    max_positions = (
        getattr(model.config, "max_position_embeddings", None)
        or getattr(model.config, "n_positions", None)
    )
    if max_positions is not None and len(prefix_tokens) + len(candidate_tokens) > max_positions:
        raise ValueError(
            f"prompt length {len(prefix_tokens) + len(candidate_tokens)} "
            f"exceeds model context limit {max_positions}"
        )

    token_log_probs = []
    with torch.inference_mode():
        output = model(input_ids=input_ids, use_cache=True)
        past_key_values = output.past_key_values
        next_logits = output.logits[0, -1].float()

        for index, target_token in enumerate(candidate_tokens):
            token_log_probs.append(
                float(F.log_softmax(next_logits, dim=-1)[target_token].item())
            )
            if index + 1 < len(candidate_tokens):
                next_input = torch.tensor(
                    [[target_token]], dtype=torch.long, device=device
                )
                output = model(
                    input_ids=next_input,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                past_key_values = output.past_key_values
                next_logits = output.logits[0, -1].float()

    return _mean(token_log_probs)


def capture_v1_state(model) -> List[Tuple[torch.nn.Module, Dict[str, torch.Tensor]]]:
    snapshot = []
    for module in model.modules():
        if isinstance(module, DRAIV1ResonanceLayer):
            snapshot.append(
                (
                    module,
                    {
                        name: getattr(module, name).detach().clone()
                        for name in _STATE_BUFFER_NAMES
                    },
                )
            )
    if not snapshot:
        raise ValueError("the supplied model has no DRAI V1 state")
    return snapshot


@torch.no_grad()
def restore_v1_state(snapshot) -> None:
    for module, buffers in snapshot:
        for name, value in buffers.items():
            getattr(module, name).copy_(value)


def score_options(model, tokenizer, prefix: str, options: Sequence[str],
                  device: torch.device, snapshot=None) -> List[float]:
    scores = []
    for option in options:
        if snapshot is not None:
            restore_v1_state(snapshot)
        scores.append(score_candidate(model, tokenizer, prefix, option, device))
    if snapshot is not None:
        restore_v1_state(snapshot)
    return scores


def _mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    fraction = position - lower
    return float(ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction)


def cluster_bootstrap_ci(
    cases: Sequence[Mapping[str, object]],
    values: Mapping[str, float],
    seed: int = 42,
    replicates: int = 5000,
) -> Dict[str, object]:
    """Percentile interval resampling whole conversations, not individual turns."""

    grouped: Dict[str, List[float]] = {}
    for case in cases:
        key = str(case.get("conversation_id", case["case_id"]))
        grouped.setdefault(key, []).append(float(values[str(case["case_id"])]))
    cluster_values = [_mean(group) for group in grouped.values()]
    point = _mean(cluster_values)
    if not cluster_values:
        return {
            "mean": 0.0,
            "ci95": [0.0, 0.0],
            "n_conversations": 0,
            "n_cases": 0,
        }
    rng = random.Random(seed)
    samples = [
        _mean([rng.choice(cluster_values) for _ in cluster_values])
        for _ in range(max(1, replicates))
    ]
    return {
        "mean": point,
        "ci95": [_percentile(samples, 0.025), _percentile(samples, 0.975)],
        "n_conversations": len(cluster_values),
        "n_cases": len(cases),
    }


def paired_cluster_bootstrap_ci(
    cases: Sequence[Mapping[str, object]],
    left: Mapping[str, float],
    right: Mapping[str, float],
    seed: int = 42,
    replicates: int = 5000,
) -> Dict[str, object]:
    """Interval for paired left-minus-right differences by conversation."""

    differences = {
        str(case["case_id"]): float(left[str(case["case_id"])])
        - float(right[str(case["case_id"])])
        for case in cases
    }
    return cluster_bootstrap_ci(cases, differences, seed, replicates)


def _latency_summary(records: Sequence[Mapping[str, object]]) -> Dict[str, float]:
    total = [float(record["total_ms"]) for record in records]
    context = [float(record["context_ms"]) for record in records]
    query = [float(record["query_ms"]) for record in records]
    return {
        "mean_total_ms": _mean(total),
        "median_total_ms": float(statistics.median(total)) if total else 0.0,
        "mean_context_ms": _mean(context),
        "mean_query_scoring_ms": _mean(query),
        "mean_context_tokens": _mean([float(r["context_tokens"]) for r in records]),
        "mean_query_tokens": _mean([float(r["query_tokens"]) for r in records]),
        "mean_state_bytes": _mean([float(r["state_bytes"]) for r in records]),
        "max_state_bytes": max((int(r["state_bytes"]) for r in records), default=0),
    }


def summarize(
    cases: Sequence[Mapping[str, object]],
    results: Mapping[str, Sequence[Mapping[str, object]]],
    seed: int = 42,
    replicates: int = 5000,
) -> Dict[str, object]:
    categories = ("overall", *CATEGORIES)
    summary: Dict[str, object] = {}
    rows_by_condition = {
        condition: {str(row["case_id"]): row for row in rows}
        for condition, rows in results.items()
    }
    for category in categories:
        subset = [
            case for case in cases
            if category == "overall" or case["category"] == category
        ]
        section: Dict[str, object] = {"conditions": {}, "paired_deltas": {}}
        accuracies = {}
        for condition in CONDITIONS:
            rows = rows_by_condition[condition]
            values = {
                str(case["case_id"]): float(rows[str(case["case_id"])]["correct"])
                for case in subset
            }
            accuracies[condition] = values
            section["conditions"][condition] = {
                "accuracy": cluster_bootstrap_ci(subset, values, seed, replicates),
                **_latency_summary([rows[str(case["case_id"])] for case in subset]),
            }
        section["paired_deltas"]["drai_minus_no_drai"] = paired_cluster_bootstrap_ci(
            subset, accuracies["drai_state"], accuracies["no_drai"], seed, replicates
        )
        section["paired_deltas"]["full_history_minus_no_drai"] = paired_cluster_bootstrap_ci(
            subset, accuracies["full_history"], accuracies["no_drai"], seed, replicates
        )
        section["paired_deltas"]["drai_minus_full_history"] = paired_cluster_bootstrap_ci(
            subset, accuracies["drai_state"], accuracies["full_history"], seed, replicates
        )
        summary[category] = section
    return summary


def _score_static_case(model, tokenizer, case, prefix, device):
    query = format_question(case)
    scoring_prefix = prefix + ("\n\n" if prefix else "") + query + "\n"
    query_tokens = len(token_ids(tokenizer, query + "\n"))
    context_tokens = len(token_ids(tokenizer, prefix)) if prefix else 0
    started = time.perf_counter()
    scores = score_options(model, tokenizer, scoring_prefix, case["options"], device)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    predicted_index = max(range(len(scores)), key=scores.__getitem__)
    predicted = case["options"][predicted_index]
    return {
        "case_id": case["case_id"],
        "conversation_id": case.get("conversation_id", case["case_id"]),
        "category": case["category"],
        "expected": case["answer"],
        "predicted": predicted,
        "correct": predicted == case["answer"],
        "candidate_log_likelihoods": scores,
        "total_ms": elapsed_ms,
        "context_ms": 0.0,
        "query_ms": elapsed_ms,
        "context_tokens": context_tokens,
        "query_tokens": query_tokens,
        "state_bytes": 0,
    }


def _prime_drai_state(model, tokenizer, case, device):
    context_tokens = 0
    started = time.perf_counter()
    for index, session in enumerate(case["sessions"]):
        prefix = "Prior conversation:\n" if index == 0 else "\n\n"
        text = prefix + f"Session {index + 1}:\n{session}"
        ids = token_ids(tokenizer, text)
        context_tokens += len(ids)
        input_ids = torch.tensor([ids], dtype=torch.long, device=device)
        with torch.inference_mode():
            model(input_ids=input_ids)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return elapsed_ms, context_tokens


def _score_drai_case(model, tokenizer, case, device):
    reset_v1_state(model)
    context_ms, context_tokens = _prime_drai_state(model, tokenizer, case, device)
    state_bytes = get_v1_state_size_bytes(model)
    snapshot = capture_v1_state(model)
    query = format_question(case) + "\n"
    query_tokens = len(token_ids(tokenizer, query))
    started = time.perf_counter()
    scores = score_options(
        model, tokenizer, query, case["options"], device, snapshot=snapshot
    )
    query_ms = (time.perf_counter() - started) * 1000.0
    predicted_index = max(range(len(scores)), key=scores.__getitem__)
    predicted = case["options"][predicted_index]
    return {
        "case_id": case["case_id"],
        "conversation_id": case.get("conversation_id", case["case_id"]),
        "category": case["category"],
        "expected": case["answer"],
        "predicted": predicted,
        "correct": predicted == case["answer"],
        "candidate_log_likelihoods": scores,
        "total_ms": context_ms + query_ms,
        "context_ms": context_ms,
        "query_ms": query_ms,
        "context_tokens": context_tokens,
        "query_tokens": query_tokens,
        "state_bytes": state_bytes,
    }


def _load_model(model_name: str, revision: Optional[str], device: torch.device):
    from transformers import AutoModelForCausalLM

    kwargs = {"torch_dtype": torch.float32}
    if revision:
        kwargs["revision"] = revision
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    resolved_revision = getattr(model.config, "_commit_hash", None) or revision
    model.to(device)
    model.eval()
    return model, resolved_revision


def parameter_fingerprint(model) -> str:
    """Hash model parameters to prove the baseline and DRAI arms start equally."""

    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        value = parameter.detach().to(device="cpu").contiguous().view(torch.uint8)
        digest.update(name.encode("utf-8"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def run_benchmark(
    model_name: str = "EleutherAI/pythia-410m",
    revision: Optional[str] = None,
    cases: Sequence[Mapping[str, object]] = CASES,
    device_name: str = "cpu",
    seed: int = 42,
    bootstrap_replicates: int = 5000,
) -> Dict[str, object]:
    from transformers import AutoTokenizer

    validate_cases(cases)
    torch.manual_seed(seed)
    device = torch.device(device_name)
    baseline, resolved_revision = _load_model(model_name, revision, device)
    baseline_fingerprint = parameter_fingerprint(baseline)
    tokenizer_kwargs = {"revision": resolved_revision} if resolved_revision else {}
    tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)
    if tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    static_results: Dict[str, List[Dict[str, object]]] = {
        "no_drai": [],
        "full_history": [],
    }
    for case in cases:
        static_results["no_drai"].append(
            _score_static_case(baseline, tokenizer, case, "", device)
        )
        static_results["full_history"].append(
            _score_static_case(
                baseline, tokenizer, case, format_history(case), device
            )
        )
    del baseline
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    drai_model, drai_revision = _load_model(model_name, resolved_revision, device)
    drai_fingerprint = parameter_fingerprint(drai_model)
    if drai_fingerprint != baseline_fingerprint:
        raise RuntimeError("baseline and DRAI models do not have identical initial weights")
    torch.manual_seed(seed)
    apply_v1(drai_model, config=get_v1_conservative())
    drai_results = [
        _score_drai_case(drai_model, tokenizer, case, device)
        for case in cases
    ]
    results = {
        "no_drai": static_results["no_drai"],
        "drai_state": drai_results,
        "full_history": static_results["full_history"],
    }
    return {
        "schema_version": 1,
        "model": {
            "id": model_name,
            "revision": drai_revision or resolved_revision or revision,
            "initial_parameter_sha256": baseline_fingerprint,
            "device": str(device),
            "dtype": "float32",
        },
        "protocol": {
            "seed": seed,
            "bootstrap_replicates": bootstrap_replicates,
            "bootstrap": "95% percentile interval; resample conversation clusters",
            "score": "mean token log likelihood of each complete answer option",
            "state_reset": "DRAI state reset before every independent conversation",
            "candidate_isolation": "restore post-context state before each option",
            "case_count": len(cases),
            "category_counts": {
                category: sum(case["category"] == category for case in cases)
                for category in CATEGORIES
            },
        },
        "summary": summarize(cases, results, seed, bootstrap_replicates),
        "results": {condition: list(rows) for condition, rows in results.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="EleutherAI/pythia-410m")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--cases", type=Path, default=None, help="normalized JSONL case file")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap-replicates", type=int, default=5000)
    parser.add_argument(
        "--max-per-category",
        type=int,
        default=0,
        help="optional pilot cap; zero uses all cases",
    )
    args = parser.parse_args()

    cases = load_cases(args.cases)
    if args.max_per_category > 0:
        cases = [
            case
            for category in CATEGORIES
            for case in [
                case for case in cases if case["category"] == category
            ][:args.max_per_category]
        ]

    report = run_benchmark(
        model_name=args.model,
        revision=args.revision,
        cases=cases,
        device_name=args.device,
        seed=args.seed,
        bootstrap_replicates=args.bootstrap_replicates,
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
