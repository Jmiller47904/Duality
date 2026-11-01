"""Regression checks for V1 burn-in status reported by layer statistics."""

import torch

from src.drai.resonance_layer_v1 import DRAIV1ResonanceLayer


def _query():
    return torch.ones(2, 1, 1, 4)


def test_stats_report_strength_burn_in_status():
    layer = DRAIV1ResonanceLayer(
        hidden_size=4,
        num_heads=1,
        head_dim=4,
        burn_in_threshold=0.5,
    )

    assert layer.get_stats()["burn_in_active"] is True

    layer(_query())

    stats = layer.get_stats()
    assert stats["total_strength"] >= 0.5
    assert stats["burn_in_active"] is False


def test_stats_report_token_burn_in_status():
    layer = DRAIV1ResonanceLayer(
        hidden_size=4,
        num_heads=1,
        head_dim=4,
        burn_in_mode="tokens",
        burn_in_tokens=2,
    )

    assert layer.get_stats()["burn_in_active"] is True

    layer(_query())
    assert layer.get_stats()["burn_in_active"] is True

    layer(_query())
    stats = layer.get_stats()
    assert stats["timestep"] == 2.0
    assert stats["burn_in_active"] is False
