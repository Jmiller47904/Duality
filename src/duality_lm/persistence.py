"""Durable memory for Duality's recurrent attractor state.

The native model already carries working memory between forward passes.  This
module adds the missing process boundary: attractor state, autobiographical
records, and a small explicit self-model can be checkpointed to SQLite and
restored by a later process.

SQLite is used deliberately.  It is inspectable, transactional, available in
the Python standard library, and keeps this research layer independent from a
particular vector database.
"""

from __future__ import annotations

import json
import math
import sqlite3
import time
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import torch

from .memory import MemoryState

LayerState = Optional[MemoryState]
ModelMemoryState = Tuple[LayerState, ...]


class MemoryKind(str, Enum):
    """Long-term memory channels kept separate for evaluation and recall."""

    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    SELF = "self"


@dataclass
class MemoryRecord:
    """One durable memory and the signals used for resonance recall."""

    content: str
    vector: List[float]
    resonance_vector: List[float]
    kind: MemoryKind = MemoryKind.EPISODIC
    importance: float = 0.5
    novelty: float = 0.5
    salience: float = 0.0
    associations: List[str] = field(default_factory=list)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    activation_count: int = 0
    created_at: float = field(default_factory=time.time)
    last_activated: float = field(default_factory=time.time)
    decay_rate: float = 0.01
    score: float = 0.0


@dataclass
class SelfModel:
    """Inspectable identity continuity rather than an implicit persona claim."""

    identity: str
    version: int = 1
    goals: List[str] = field(default_factory=list)
    unresolved_questions: List[str] = field(default_factory=list)
    beliefs: Dict[str, float] = field(default_factory=dict)
    capabilities: List[str] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)
    cycle_count: int = 0
    last_reflection: Optional[str] = None
    updated_at: float = field(default_factory=time.time)


def _validate_unit_interval(name: str, value: float) -> float:
    value = float(value)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
    return value


def _validate_vector(vector: Sequence[float], name: str = "vector") -> List[float]:
    values = [float(value) for value in vector]
    if not values:
        raise ValueError(f"{name} cannot be empty")
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"{name} must contain only finite values")
    return values


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ValueError("memory and query vectors must have the same dimension")
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


class PersistentMemoryStore:
    """SQLite store for attractor checkpoints and long-term memories."""

    def __init__(self, path: Union[str, Path]) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(str(self.path))
        self.connection.row_factory = sqlite3.Row
        self._migrate()

    def __enter__(self) -> "PersistentMemoryStore":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        self.connection.close()

    def _migrate(self) -> None:
        self.connection.executescript("""
            CREATE TABLE IF NOT EXISTS model_state (
                identity TEXT PRIMARY KEY,
                payload TEXT NOT NULL,
                updated_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS self_model (
                identity TEXT PRIMARY KEY,
                payload TEXT NOT NULL,
                updated_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS memory (
                id TEXT PRIMARY KEY,
                identity TEXT NOT NULL,
                kind TEXT NOT NULL,
                content TEXT NOT NULL,
                vector TEXT NOT NULL,
                resonance_vector TEXT NOT NULL,
                importance REAL NOT NULL,
                novelty REAL NOT NULL,
                salience REAL NOT NULL,
                associations TEXT NOT NULL,
                activation_count INTEGER NOT NULL,
                created_at REAL NOT NULL,
                last_activated REAL NOT NULL,
                decay_rate REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_memory_identity_kind
            ON memory(identity, kind);
            """)
        self.connection.commit()

    def remember(self, identity: str, record: MemoryRecord) -> str:
        """Persist an episodic, semantic, or self memory."""

        if not identity.strip():
            raise ValueError("identity cannot be empty")
        record.vector = _validate_vector(record.vector)
        record.resonance_vector = _validate_vector(record.resonance_vector, "resonance_vector")
        if len(record.vector) != len(record.resonance_vector):
            raise ValueError("vector and resonance_vector must have the same dimension")
        record.importance = _validate_unit_interval("importance", record.importance)
        record.novelty = _validate_unit_interval("novelty", record.novelty)
        record.salience = _validate_unit_interval("salience", record.salience)
        if record.decay_rate < 0.0:
            raise ValueError("decay_rate must be >= 0")

        self.connection.execute(
            """
            INSERT OR REPLACE INTO memory VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.id,
                identity,
                record.kind.value,
                record.content,
                json.dumps(record.vector),
                json.dumps(record.resonance_vector),
                record.importance,
                record.novelty,
                record.salience,
                json.dumps(record.associations),
                record.activation_count,
                record.created_at,
                record.last_activated,
                record.decay_rate,
            ),
        )
        self.connection.commit()
        return record.id

    def recall(
        self,
        identity: str,
        query_vector: Sequence[float],
        limit: int = 5,
        kinds: Optional[Iterable[MemoryKind]] = None,
        association_ids: Sequence[str] = (),
        now: Optional[float] = None,
    ) -> List[MemoryRecord]:
        """Recall records by similarity, resonance, importance, and recency.

        Scoring is intentionally explicit so experiments can ablate each term.
        A memory must first be directionally similar to the current state.  Its
        resonance vector, importance, salience, associations, and temporal
        decay then modulate that base activation.
        """

        query = _validate_vector(query_vector, "query_vector")
        if limit <= 0:
            return []
        parameters: List[Any] = [identity]
        sql = "SELECT * FROM memory WHERE identity = ?"
        if kinds is not None:
            kind_values = [MemoryKind(kind).value for kind in kinds]
            if not kind_values:
                return []
            sql += " AND kind IN ({})".format(",".join("?" for _ in kind_values))
            parameters.extend(kind_values)

        rows = self.connection.execute(sql, parameters).fetchall()
        timestamp = time.time() if now is None else float(now)
        association_set = set(association_ids)
        ranked: List[MemoryRecord] = []
        for row in rows:
            record = self._record_from_row(row)
            similarity = max(0.0, _cosine(query, record.vector))
            resonance = max(0.0, _cosine(query, record.resonance_vector))
            age_days = max(0.0, timestamp - record.last_activated) / 86400.0
            temporal_weight = math.exp(-record.decay_rate * age_days)
            association_matches = len(association_set.intersection(record.associations))
            association_weight = 1.0 + min(association_matches, 4) * 0.15
            importance_weight = 0.25 + 0.75 * record.importance
            salience_weight = 1.0 + 0.20 * record.salience + 0.10 * record.novelty
            record.score = (
                math.sqrt(similarity)
                * (0.5 + 0.5 * resonance)
                * importance_weight
                * salience_weight
                * association_weight
                * temporal_weight
            )
            ranked.append(record)

        ranked.sort(key=lambda item: (item.score, item.importance), reverse=True)
        recalled = ranked[:limit]
        if recalled:
            self.connection.executemany(
                "UPDATE memory SET activation_count = activation_count + 1, "
                "last_activated = ? WHERE id = ?",
                [(timestamp, record.id) for record in recalled],
            )
            self.connection.commit()
            for record in recalled:
                record.activation_count += 1
                record.last_activated = timestamp
        return recalled

    @staticmethod
    def _record_from_row(row: sqlite3.Row) -> MemoryRecord:
        return MemoryRecord(
            id=row["id"],
            kind=MemoryKind(row["kind"]),
            content=row["content"],
            vector=json.loads(row["vector"]),
            resonance_vector=json.loads(row["resonance_vector"]),
            importance=float(row["importance"]),
            novelty=float(row["novelty"]),
            salience=float(row["salience"]),
            associations=json.loads(row["associations"]),
            activation_count=int(row["activation_count"]),
            created_at=float(row["created_at"]),
            last_activated=float(row["last_activated"]),
            decay_rate=float(row["decay_rate"]),
        )

    def save_model_state(self, identity: str, state: ModelMemoryState) -> None:
        """Serialize all memory-enabled transformer layers transactionally."""

        layers: List[Optional[Dict[str, Any]]] = []
        for layer in state:
            if layer is None:
                layers.append(None)
                continue
            detached = layer.detach()
            layers.append(
                {
                    "attractors": detached.attractors.cpu().tolist(),
                    "strengths": detached.strengths.cpu().tolist(),
                    "timestep": detached.timestep.cpu().tolist(),
                    "dtype": str(detached.attractors.dtype).replace("torch.", ""),
                }
            )
        timestamp = time.time()
        self.connection.execute(
            "INSERT OR REPLACE INTO model_state VALUES (?, ?, ?)",
            (identity, json.dumps({"version": 1, "layers": layers}), timestamp),
        )
        self.connection.commit()

    def load_model_state(
        self,
        identity: str,
        device: Optional[Union[str, torch.device]] = None,
    ) -> Optional[ModelMemoryState]:
        """Restore a saved attractor state without executing pickle payloads."""

        row = self.connection.execute(
            "SELECT payload FROM model_state WHERE identity = ?", (identity,)
        ).fetchone()
        if row is None:
            return None
        payload = json.loads(row["payload"])
        if payload.get("version") != 1:
            raise ValueError("unsupported persistent model-state version")

        layers: List[LayerState] = []
        for layer in payload["layers"]:
            if layer is None:
                layers.append(None)
                continue
            dtype = getattr(torch, layer["dtype"], None)
            if not isinstance(dtype, torch.dtype):
                raise ValueError(f"unsupported tensor dtype {layer['dtype']}")
            layers.append(
                MemoryState(
                    attractors=torch.tensor(layer["attractors"], dtype=dtype, device=device),
                    strengths=torch.tensor(layer["strengths"], dtype=dtype, device=device),
                    timestep=torch.tensor(layer["timestep"], dtype=dtype, device=device),
                )
            )
        return tuple(layers)

    def save_self_model(self, model: SelfModel) -> None:
        model.updated_at = time.time()
        self.connection.execute(
            "INSERT OR REPLACE INTO self_model VALUES (?, ?, ?)",
            (model.identity, json.dumps(asdict(model), sort_keys=True), model.updated_at),
        )
        self.connection.commit()

    def load_self_model(self, identity: str) -> SelfModel:
        row = self.connection.execute(
            "SELECT payload FROM self_model WHERE identity = ?", (identity,)
        ).fetchone()
        if row is None:
            return SelfModel(identity=identity)
        payload = json.loads(row["payload"])
        if payload.get("identity") != identity:
            raise ValueError("stored self-model identity does not match lookup identity")
        return SelfModel(**payload)
