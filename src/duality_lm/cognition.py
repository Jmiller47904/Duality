"""Bounded metacognition and continuity loops for DualityLM.

This module measures internal state and supplies an explicit reflection cycle.
It implements functional continuity and self-monitoring.  It does not claim
that those mechanisms establish subjective experience.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

import torch
from torch.nn import functional as F

from .persistence import (
    MemoryKind,
    MemoryRecord,
    ModelMemoryState,
    PersistentMemoryStore,
    SelfModel,
)


@dataclass(frozen=True)
class MetaState:
    """Normalized measurements of a model's current attractor field."""

    confidence: float
    uncertainty: float
    stability: float
    occupancy: float
    diversity: float
    active_slots: int
    total_slots: int


@dataclass(frozen=True)
class ReflectionContext:
    """Information made available to one bounded cognition step."""

    identity: str
    focus: str
    cycle: int
    self_model: SelfModel
    recalled_memories: Tuple[MemoryRecord, ...]
    meta_state: Optional[MetaState]


@dataclass
class ReflectionResult:
    """Structured output from a user-supplied reasoning function."""

    content: str
    vector: Sequence[float]
    importance: float = 0.6
    novelty: float = 0.5
    salience: float = 0.0
    resolved_question: Optional[str] = None
    new_question: Optional[str] = None


def measure_metastate(state: Optional[ModelMemoryState]) -> MetaState:
    """Measure confidence, entropy, stability, occupancy, and diversity."""

    layers = [layer for layer in (state or ()) if layer is not None]
    if not layers:
        return MetaState(0.0, 1.0, 0.0, 0.0, 0.0, 0, 0)

    strength_parts = [layer.strengths.detach().float().flatten().cpu() for layer in layers]
    strengths = torch.cat(strength_parts)
    active = strengths > 1e-3
    active_strengths = strengths[active]
    active_slots = int(active.sum().item())
    total_slots = int(strengths.numel())
    if active_slots == 0:
        return MetaState(0.0, 1.0, 0.0, 0.0, 0.0, 0, total_slots)

    confidence = min(1.0, float(active_strengths.max().item()))
    stability = min(1.0, float(active_strengths.mean().item()))
    occupancy = active_slots / max(total_slots, 1)
    probabilities = active_strengths / active_strengths.sum().clamp_min(1e-9)
    entropy = float((-(probabilities * probabilities.clamp_min(1e-9).log()).sum()).item())
    uncertainty = entropy / math.log(active_slots) if active_slots > 1 else 0.0

    vectors: List[torch.Tensor] = []
    for layer in layers:
        layer_active = layer.strengths.detach().float().cpu() > 1e-3
        vectors.extend(layer.attractors.detach().float().cpu()[layer_active].unbind(0))
    if len(vectors) > 1:
        matrix = F.normalize(torch.stack(vectors), dim=-1, eps=1e-6)
        similarities = matrix @ matrix.T
        mask = ~torch.eye(len(vectors), dtype=torch.bool)
        mean_similarity = float(similarities[mask].mean().item())
        diversity = min(1.0, max(0.0, (1.0 - mean_similarity) / 2.0))
    else:
        diversity = 0.0

    return MetaState(
        confidence=confidence,
        uncertainty=min(1.0, max(0.0, uncertainty)),
        stability=stability,
        occupancy=occupancy,
        diversity=diversity,
        active_slots=active_slots,
        total_slots=total_slots,
    )


class ContinuityEngine:
    """Connect a Duality model to durable memory and bounded reflection."""

    def __init__(self, store: PersistentMemoryStore, identity: str) -> None:
        if not identity.strip():
            raise ValueError("identity cannot be empty")
        self.store = store
        self.identity = identity

    def checkpoint(self, state: ModelMemoryState) -> None:
        self.store.save_model_state(self.identity, state)

    def restore(self, device: Optional[torch.device] = None) -> Optional[ModelMemoryState]:
        return self.store.load_model_state(self.identity, device=device)

    def observe(
        self,
        content: str,
        vector: Sequence[float],
        kind: MemoryKind = MemoryKind.EPISODIC,
        importance: float = 0.5,
        novelty: float = 0.5,
        salience: float = 0.0,
        associations: Sequence[str] = (),
    ) -> str:
        record = MemoryRecord(
            content=content,
            vector=list(vector),
            resonance_vector=list(vector),
            kind=kind,
            importance=importance,
            novelty=novelty,
            salience=salience,
            associations=list(associations),
        )
        return self.store.remember(self.identity, record)

    def reflect_once(
        self,
        query_vector: Sequence[float],
        reason: Callable[[ReflectionContext], ReflectionResult],
        model_state: Optional[ModelMemoryState] = None,
        recall_limit: int = 5,
    ) -> ReflectionResult:
        """Run exactly one inspectable cognition step and persist its result."""

        self_model = self.store.load_self_model(self.identity)
        if self_model.unresolved_questions:
            focus = self_model.unresolved_questions[0]
        elif self_model.goals:
            focus = self_model.goals[0]
        else:
            focus = "review continuity, uncertainty, and unresolved observations"

        recalled = self.store.recall(
            self.identity,
            query_vector=query_vector,
            limit=recall_limit,
        )
        context = ReflectionContext(
            identity=self.identity,
            focus=focus,
            cycle=self_model.cycle_count + 1,
            self_model=self_model,
            recalled_memories=tuple(recalled),
            meta_state=measure_metastate(model_state) if model_state is not None else None,
        )
        result = reason(context)
        if not result.content.strip():
            raise ValueError("reflection content cannot be empty")

        association_ids = [memory.id for memory in recalled]
        self.observe(
            result.content,
            result.vector,
            kind=MemoryKind.SELF,
            importance=result.importance,
            novelty=result.novelty,
            salience=result.salience,
            associations=association_ids,
        )

        if result.resolved_question in self_model.unresolved_questions:
            self_model.unresolved_questions.remove(result.resolved_question)
        if result.new_question and result.new_question not in self_model.unresolved_questions:
            self_model.unresolved_questions.append(result.new_question)
        self_model.cycle_count += 1
        self_model.last_reflection = result.content
        self.store.save_self_model(self_model)
        if model_state is not None:
            self.checkpoint(model_state)
        return result
