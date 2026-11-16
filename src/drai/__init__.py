"""
DRAI (Dynamic Resonance AI) — V1 Production API
-----------------------------------------------

Stable, training-free dynamic resonance memory for transformer models.

This module exposes the public V1 API:
    • apply_v1()                     – patch a model with DRAI V1
    • get_v1_stats()                 – inspect attractor state
    • V1Config / V1Hyperparameters
    • Presets: get_v1_conservative(), get_v1_standard()
    • DRAIV1ResonanceLayer           – core attractor engine
    • DRAIV1NeoXAttention            – GPT-NeoX attention wrapper
    • apply_v1_neox_injection        – NeoX injection utility
"""

# Core V1 attractor engine
from .resonance_layer_v1 import DRAIV1ResonanceLayer

# GPT-NeoX integration components
from .neox_integration_v1 import (
    DRAIV1NeoXAttention,
    apply_v1_neox_injection,
)

# Configuration system
from .config import (
    V1Config,
    V1Hyperparameters,
    get_v1_conservative,
    get_v1_standard,
)

# High-level application API
from .apply_v1 import apply_v1, get_v1_stats, reset_v1_state, get_v1_state_size_bytes

__version__ = "1.1.0"
__author__ = "Halcyon AI Research"

__all__ = [
    # High-level API
    "apply_v1",
    "get_v1_stats",
    "reset_v1_state",
    "get_v1_state_size_bytes",

    # Configuration system
    "V1Config",
    "V1Hyperparameters",
    "get_v1_conservative",
    "get_v1_standard",

    # Core engine + integrations
    "DRAIV1ResonanceLayer",
    "DRAIV1NeoXAttention",
    "apply_v1_neox_injection",
]

