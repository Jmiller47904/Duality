"""
High-level API for applying DRAI V1 to transformer models.

This module exposes the public entry point:

    apply_v1(model_or_name, config=None, verbose=False)

and the statistics interface:

    get_v1_stats(model)

Everything else provides architecture detection, loading,
double-injection protection, and consistent stats formatting.
"""

from typing import Optional, Union, Dict, Any, List
from transformers import (
    PreTrainedModel,
    AutoModelForCausalLM,
)

from .resonance_layer_v1 import DRAIV1ResonanceLayer

from .config import (
    V1Config,
    get_v1_conservative,
    get_v1_standard,
)

# Corrected to match global naming
from .neox_integration_v1 import (
    apply_v1_neox_injection,
    DRAIV1NeoXAttention,
)


# ============================================================================
# Internal utilities
# ============================================================================

def _load_model_from_name(name: str, **kwargs) -> PreTrainedModel:
    """
    Internal helper for when users call:

        apply_v1("EleutherAI/pythia-410m")

    This keeps model loading self-contained.
    """
    try:
        return AutoModelForCausalLM.from_pretrained(name, **kwargs)
    except Exception as e:
        raise RuntimeError(
            f"DRAI V1 could not load model '{name}'. "
            f"Pass an already-instantiated PreTrainedModel instead. "
            f"HF Error: {e}"
        )


def _get_layers_for_stats(model: PreTrainedModel) -> List:
    """
    Identify the correct transformer block list.

    Supported families:
      - GPT-NeoX / Pythia: model.gpt_neox.layers
      - HF architectures exposing model.model.layers
      - Generic models with .layers
    """
    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        return model.gpt_neox.layers

    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers

    if hasattr(model, "layers"):
        return model.layers

    raise NotImplementedError(
        "DRAI V1 does not recognize this model architecture. "
        "Supported: GPT-NeoX, Pythia, or models exposing `.layers`."
    )


def _validate_not_already_injected(model: PreTrainedModel):
    """Prevent accidental double-patching."""
    if getattr(model, "_drai_v1_applied", False):
        raise RuntimeError(
            "DRAI V1 has already been applied to this model. "
            "Re-applying would corrupt attention modules."
        )


def _mark_injected(model: PreTrainedModel):
    """Mark model as patched."""
    setattr(model, "_drai_v1_applied", True)


# ============================================================================
# Public API: apply_v1
# ============================================================================

def apply_v1(
    model: Union[PreTrainedModel, str],
    config: Optional[V1Config] = None,
    verbose: bool = False,
    **kwargs,
) -> PreTrainedModel:
    """
    Apply DRAI V1 to a transformer model in-place.

    Args:
        model:
            Either a PreTrainedModel instance, or the name/path of a
            HuggingFace model to be automatically loaded.

        config:
            Optional V1Config. If omitted, defaults to get_v1_conservative()
            which is safe for small models (<1B).

        verbose:
            If True, prints layer-by-layer injection info.

        **kwargs:
            Forwarded to AutoModelForCausalLM.from_pretrained() if `model`
            is a string.

    Returns:
        The model modified with DRAI V1 attractor-memory layers.
    """

    # Load if model is a name
    if isinstance(model, str):
        model = _load_model_from_name(model, **kwargs)

    # Ensure DRAI V1 isn't applied twice
    _validate_not_already_injected(model)

    # Default config
    if config is None:
        config = get_v1_conservative()

    # Inject DRAI V1 (corrected function name)
    model = apply_v1_neox_injection(model, config, verbose=verbose)

    # Mark as patched
    _mark_injected(model)

    return model


# ============================================================================
# Public API: memory-state lifecycle
# ============================================================================

def reset_v1_state(model: PreTrainedModel) -> int:
    """Reset all DRAI V1 attractor buffers on a model.

    Returns the number of DRAI V1 layers reset. Models without DRAI V1 layers
    return zero, which makes this safe to call from shared evaluation code.
    """

    layers = [
        module
        for module in model.modules()
        if isinstance(module, DRAIV1ResonanceLayer)
    ]
    for layer in layers:
        layer.reset_state()
    return len(layers)


def get_v1_state_size_bytes(model: PreTrainedModel) -> int:
    """Return allocated recurrent-state bytes across DRAI V1 layers."""

    return sum(
        module.state_size_bytes()
        for module in model.modules()
        if isinstance(module, DRAIV1ResonanceLayer)
    )


# ============================================================================
# Public API: get_v1_stats
# ============================================================================

def get_v1_stats(model: PreTrainedModel) -> Dict[str, Any]:
    """
    Extract attractor statistics from a DRAI-V1-modified model.

    Returns:
        {
            "num_drai_layers": int,
            "total_active": int,
            "total_strength": float,
            "mean_strength": float,
            "layers": [
                {
                    "layer_idx": int,
                    "num_alive": int,
                    "total_strength": float,
                    "mean_strength": float,
                    ...
                },
                ...
            ]
        }

    Notes:
        - Does NOT modify the model.
        - If the model has no DRAI layers, returns zeros.
    """

    layers = _get_layers_for_stats(model)

    stats = {
        "num_drai_layers": 0,
        "total_active": 0,
        "total_strength": 0.0,
        "mean_strength": 0.0,
        "layers": [],
    }

    for idx, layer in enumerate(layers):
        attn = getattr(layer, "attention", None)

        # Correct NeoX wrapper class name
        if isinstance(attn, DRAIV1NeoXAttention):
            layer_stats = attn.get_v1_statistics()

            stats["num_drai_layers"] += 1
            stats["total_active"] += layer_stats.get("num_alive", 0)
            stats["total_strength"] += layer_stats.get("total_strength", 0.0)

            stats["layers"].append(layer_stats)

    if stats["num_drai_layers"] > 0:
        stats["mean_strength"] = (
            stats["total_strength"] / stats["num_drai_layers"]
        )

    return stats
