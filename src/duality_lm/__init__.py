"""Native decoder-only language model built around the Duality memory idea.

The original project patches pretrained Hugging Face models with DRAI V1.  The
``duality_lm`` package is the next layer: a small, trainable decoder-only model
whose attention blocks can read and update a bounded attractor memory state.
"""

from .config import DualityLMConfig
from .cognition import (
    ContinuityEngine,
    MetaState,
    ReflectionContext,
    ReflectionResult,
    measure_metastate,
)
from .memory import DualityMemory, MemoryState
from .model import DualityLM, DualityLMOutput
from .persistence import MemoryKind, MemoryRecord, PersistentMemoryStore, SelfModel
from .tokenizer import ByteTokenizer

__all__ = [
    "ByteTokenizer",
    "ContinuityEngine",
    "DualityLM",
    "DualityLMConfig",
    "DualityLMOutput",
    "DualityMemory",
    "MemoryState",
    "MemoryKind",
    "MemoryRecord",
    "MetaState",
    "PersistentMemoryStore",
    "ReflectionContext",
    "ReflectionResult",
    "SelfModel",
    "measure_metastate",
]
