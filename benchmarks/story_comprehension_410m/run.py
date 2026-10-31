#!/usr/bin/env python3
"""Run a small exploratory story-comprehension check for DRAI V1.

The synthetic task is a narrow behavioral measure. Its score is not evidence
of general model quality or production readiness.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM

# Local story generator
from story_generator import StoryGenerator
from benchmarks.story_comprehension_410m.evaluation import (
    EVALUATION_SEED,
    generate_answer,
    seed_model_initialization,
)

# DRAI V1
from src.drai import apply_v1, get_v1_conservative, get_v1_stats


def evaluate_model(model, tokenizer, stories, model_name="Model"):
    """Evaluate model on simple stories."""
    print(f"\n{'='*70}")
    print(f"Evaluating: {model_name}")
    print(f"{'='*70}")

    results = []
    total_questions = 0

    for i, story in enumerate(stories, 1):
        print(f"\n[Story {i}/{len(stories)}]", end=" ")

        for q_idx, question in enumerate(story.questions):
            total_questions += 1

            # Create prompt
            prompt = f"{story.text}\n\nQuestion: {question['question']}\nAnswer:"

            generated = generate_answer(model, tokenizer, prompt)

            # Check correctness
            correct = generated.lower() == question['answer'].lower()
            results.append({
                "story_idx": i,
                "question": question['question'],
                "expected": question['answer'],
                "generated": generated,
                "correct": correct
            })

            if correct:
                print("✓", end="")
            else:
                print("✗", end="")

    # Calculate accuracy
    correct_count = sum(1 for r in results if r["correct"])
    accuracy = correct_count / total_questions

    print(f"\n\n{'='*70}")
    print(f"RESULTS: {model_name}")
    print(f"{'='*70}")
    print(f"  Correct: {correct_count}/{total_questions}")
    print(f"  Accuracy: {accuracy:.1%}")
    print(f"{'='*70}\n")

    return {
        "model_name": model_name,
        "accuracy": accuracy,
        "correct": correct_count,
        "total": total_questions,
        "results": results
    }


def main():
    print("\n" + "="*70)
    print("DRAI V1 EXPLORATORY STORY EVALUATION")
    print("="*70)
    print("\nThis measures exact-match answers on one small synthetic dataset.")
    print("\n" + "="*70)

    # Configuration
    model_name = "EleutherAI/pythia-410m"
    num_stories = 20  # Reasonable test set

    # Generate test stories
    print(f"\n[1/4] Generating {num_stories} test stories...")
    generator = StoryGenerator(seed=EVALUATION_SEED)
    stories = generator.generate_dataset(num_stories=num_stories)
    total_questions = sum(len(story.questions) for story in stories)
    print(f"  ✓ Generated {num_stories} stories ({total_questions} questions)")

    # Test 1: Baseline (no DRAI)
    print(f"\n[2/4] Testing BASELINE (no DRAI)...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    seed_model_initialization(torch)
    model_baseline = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float32,
        device_map="cpu",
    )

    baseline_results = evaluate_model(
        model_baseline, tokenizer, stories,
        model_name="Baseline (no DRAI)"
    )

    # Clean up
    del model_baseline
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Test 2: V1 DRAI
    print(f"\n[3/4] Testing V1 DRAI...")
    print(f"  Configuration:")

    config = get_v1_conservative()
    print(f"    - Burn-in threshold: {config.hyperparameters.burn_in_threshold}")
    print(f"    - Burn-in mode: {config.hyperparameters.burn_in_mode}")
    print(f"    - Max influence scale: {config.hyperparameters.max_influence_scale}")
    print(f"    - Layer mode: {config.layer_mode}")
    print(f"    - Theta match: {config.hyperparameters.theta_match}")

    # Reuse the fixed initialization seed for any layers added by the DRAI
    # wrapper; greedy decoding itself consumes no sampling RNG.
    seed_model_initialization(torch)
    model_v1 = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float32,
        device_map="cpu",
    )

    model_v1 = apply_v1(model_v1, config=config)

    v1_results = evaluate_model(
        model_v1, tokenizer, stories,
        model_name="V1 DRAI (burn-in=50.0)"
    )

    # Get V1 statistics
    v1_stats = get_v1_stats(model_v1)

    # Clean up
    del model_v1
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Compare results
    print(f"\n[4/4] FINAL COMPARISON")
    print(f"{'='*70}")
    print(f"\n{'Model':<30} {'Accuracy':>12} {'Correct':>15}")
    print(f"{'-'*70}")
    print(f"{'Baseline (no DRAI)':<30} {baseline_results['accuracy']:>11.1%} "
          f"{baseline_results['correct']:>6}/{baseline_results['total']:<6}")
    print(f"{'V1 DRAI (experimental)':<30} {v1_results['accuracy']:>11.1%} "
          f"{v1_results['correct']:>6}/{v1_results['total']:<6}")
    print(f"{'-'*70}")

    # Calculate delta
    delta = v1_results['accuracy'] - baseline_results['accuracy']
    delta_pct = delta * 100

    print(f"\nDelta: {delta:+.1%} ({delta_pct:+.1f} percentage points)")

    # V1 Statistics
    print(f"\nV1 Attractor Statistics:")
    print(f"  - Layers with DRAI: {v1_stats['num_drai_layers']}")
    print(f"  - Total active attractors: {v1_stats['total_active']}")
    print(f"  - Total strength: {v1_stats['total_strength']:.2f}")

    if len(v1_stats['layers']) > 0:
        layer_stats = v1_stats['layers'][0]
        print(f"  - Burn-in complete: {'YES' if not layer_stats['burn_in_active'] else 'NO'}")
        print(f"  - Timesteps processed: {int(layer_stats['timestep'])}")

    # Verdict
    print(f"\n{'='*70}")
    print(f"VERDICT")
    print(f"{'='*70}")

    if delta >= -0.05:  # Within 5 percentage points on this task only
        print(f"Within five percentage points of baseline on this task (Δ = {delta:+.1%}).")
        print("This single synthetic task does not establish generalization or production readiness.")
        verdict = "WITHIN_TASK_TOLERANCE"
    elif delta >= -0.15:  # 5-15% degradation
        print(f"⚠️  PARTIAL SUCCESS: Mild degradation (Δ = {delta:+.1%})")
        print(f"\nV1 works but needs tuning:")
        print(f"  - Consider increasing burn-in threshold")
        print(f"  - Consider lowering max_influence_scale")
        print(f"  - Consider higher theta_match")
        verdict = "PARTIAL"
    else:  # >15% degradation
        print(f"❌ FAILURE: Significant degradation (Δ = {delta:+.1%})")
        print(f"\nV1 still has issues. Investigate:")
        print(f"  - Burn-in threshold too low?")
        print(f"  - Attractors still learning bad patterns?")
        print(f"  - Field vector causing interference?")
        verdict = "FAILURE"

    print(f"{'='*70}\n")

    # Save results
    results_file = Path(__file__).parent / "results" / "v1_final" / "comparison.json"
    results_file.parent.mkdir(parents=True, exist_ok=True)

    with open(results_file, 'w') as f:
        json.dump({
            "baseline": baseline_results,
            "v1_drai": v1_results,
            "v1_stats": v1_stats,
            "delta": delta,
            "verdict": verdict,
            "config": {
                "burn_in_threshold": config.hyperparameters.burn_in_threshold,
                "burn_in_mode": config.hyperparameters.burn_in_mode,
                "max_influence_scale": config.hyperparameters.max_influence_scale,
                "theta_match": config.hyperparameters.theta_match,
                "layer_mode": config.layer_mode,
            }
        }, f, indent=2)

    print(f"Results saved to: {results_file}\n")

    return verdict == "WITHIN_TASK_TOLERANCE"


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
