"""Tests for DRAI V1 state reset and byte accounting."""

import torch

from src.drai import DRAIV1ResonanceLayer, get_v1_state_size_bytes, reset_v1_state


def test_reset_v1_state_clears_all_layers_and_reports_allocated_bytes():
    layer = DRAIV1ResonanceLayer(
        hidden_size=4,
        num_heads=1,
        head_dim=4,
        max_attractors=3,
        burn_in_threshold=0.0,
    )
    model = torch.nn.Sequential(layer)
    layer(torch.ones(2, 1, 1, 4))

    assert layer.timestep.item() == 1.0
    assert layer.get_stats()["num_alive"] > 0
    assert get_v1_state_size_bytes(model) > 0
    assert reset_v1_state(model) == 1
    assert layer.timestep.item() == 0.0
    assert torch.count_nonzero(layer.attractor_vectors).item() == 0
    assert torch.count_nonzero(layer.attractor_strengths).item() == 0
    assert torch.count_nonzero(layer.attractor_last_used).item() == 0


def test_reset_without_drai_is_a_safe_no_op():
    assert reset_v1_state(torch.nn.Linear(2, 2)) == 0
