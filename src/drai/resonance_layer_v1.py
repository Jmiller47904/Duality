"""
DRAI (Dynamic Resonance AI) — V1 Resonance Layer
------------------------------------------------

Core attractor-memory engine powering the DRAI V1 architecture.

This layer implements:
    • Stable attractor slot memory (Plan A)
    • Robust cosine-similarity matching
    • EMA-like centroid updating with decay
    • Novel-slot assignment with strength init
    • Safe burn-in gating (strength- or token-based)
    • Deterministic synthetic K/V generation

Designed for:
    • correctness and stability in <1B models
    • injection into GPT-NeoX and Pythia-family architectures
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class DRAIV1ResonanceLayer(nn.Module):
    """
    Production-safe, training-free attractor memory used in DRAI V1.

    Input:
        query_layer:
            [seq_len, batch, num_attention_heads, head_dim]

    Output:
        k_reson:
            [seq_len, batch, num_drai_heads, head_dim]

        v_reson:
            [seq_len, batch, num_drai_heads, head_dim]
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 1,
        head_dim: Optional[int] = None,
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        # Hyperparameters
        max_attractors: int = 16,
        theta_match: float = 0.8,
        alpha_update: float = 0.05,
        lambda_decay: float = 0.995,
        strength_init: float = 0.5,
        strength_min: float = 1e-3,
        max_influence_scale: float = 0.15,
        burn_in_threshold: float = 50.0,
        burn_in_mode: str = "strength",
        burn_in_tokens: int = 10,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim or 64

        self.device = device or "cpu"
        self.dtype = dtype or torch.float32

        # Hyperparams
        self.M = max_attractors
        self.d = self.head_dim
        self.theta_match = theta_match
        self.alpha_update = alpha_update
        self.lambda_decay = lambda_decay
        self.strength_init = strength_init
        self.strength_min = strength_min
        self.max_influence_scale = max_influence_scale
        self.burn_in_threshold = burn_in_threshold
        self.burn_in_mode = burn_in_mode
        self.burn_in_tokens = burn_in_tokens

        # Persistent memory buffers
        self.register_buffer(
            "attractor_vectors",
            torch.randn(self.M, self.d, device=self.device, dtype=self.dtype) * 1e-3,
        )
        self.register_buffer(
            "attractor_strengths",
            torch.zeros(self.M, device=self.device, dtype=self.dtype),
        )
        self.register_buffer(
            "attractor_last_used",
            torch.zeros(self.M, device=self.device, dtype=self.dtype),
        )
        self.register_buffer(
            "timestep",
            torch.tensor(0.0, device=self.device, dtype=self.dtype),
        )

    # ======================================================================
    # Forward Pass — Query → Attractor Update → Synthetic K/V
    # ======================================================================
    def forward(self, query_layer: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_len, batch, _, _ = query_layer.shape

        # Collapse multi-head Q → one vector per token
        q = query_layer.mean(dim=2)          # [S, B, D]
        q = q.transpose(0, 1)                # [B, S, D]

        A = self.attractor_vectors
        S = self.attractor_strengths
        L = self.attractor_last_used
        step = self.timestep

        # Flatten for detection
        B, T, D = q.shape
        q_flat = q.reshape(-1, D)

        # (1) Detect matches & novel
        match_idx, match_mask, novel_mask = self._detect(q_flat, A, S)

        # (2) Accumulate updates
        A, S, L = self._accumulate(q_flat, A, S, L, match_idx, match_mask, step)

        # (3) Assign novel patterns
        A, S, L = self._assign_new(q_flat, A, S, L, novel_mask, step)

        # (4) Decay + Evict
        A, S, L = self._decay(A, S, L)
        A, S, L = self._evict(A, S, L)

        # Persist updated memory
        self.attractor_vectors.copy_(A)
        self.attractor_strengths.copy_(S)
        self.attractor_last_used.copy_(L)
        self.timestep.copy_(step + 1.0)

        # (5) Generate synthetic K/V
        k_reson, v_reson = self._gen_kv(q, A, S)

        return k_reson.transpose(0, 1), v_reson.transpose(0, 1)

    # ======================================================================
    # Pattern Detection
    # ======================================================================
    def _detect(self, q_flat, A, S):
        eps = 1e-8

        q_norm = F.normalize(q_flat, p=2, dim=-1, eps=eps)
        A_norm = F.normalize(A, p=2, dim=-1, eps=eps)

        sim = q_norm @ A_norm.T   # [N, M]
        best_scores, best_idx = torch.max(sim, dim=-1)

        strong_enough = S[best_idx] > self.strength_min
        match_mask = (best_scores > self.theta_match) & strong_enough
        novel_mask = ~match_mask

        return best_idx, match_mask, novel_mask

    # ======================================================================
    # Accumulation: Update matched attractors
    # ======================================================================
    def _accumulate(self, q_flat, A, S, L, match_idx, match_mask, step):
        M, d = A.shape

        matched = q_flat * match_mask.unsqueeze(-1)

        counts = torch.zeros(M, device=A.device, dtype=A.dtype)
        counts.scatter_add_(0, match_idx, match_mask.float())

        baseA = A[match_idx]
        deltas = matched - baseA

        delta_sum = torch.zeros_like(A)
        delta_sum.scatter_add_(
            0,
            match_idx.unsqueeze(-1).expand(-1, d),
            deltas,
        )

        counts_safe = torch.clamp(counts, 1.0)
        A_new = A + self.alpha_update * (delta_sum / counts_safe.unsqueeze(-1))

        gamma = 0.01
        S_new = (1 - gamma) * S + counts

        L_new = torch.where(counts > 0, step.expand_as(L), L)

        return A_new, S_new, L_new

    # ======================================================================
    # Assign new attractors for novel patterns
    # ======================================================================
    def _assign_new(self, q_flat, A, S, L, novel_mask, step):
        novel_idx = torch.where(novel_mask)[0]
        if len(novel_idx) == 0:
            return A, S, L

        free_slots = torch.where(S < self.strength_min)[0]
        if len(free_slots) == 0:
            return A, S, L

        k = min(len(novel_idx), len(free_slots))
        q_sel = novel_idx[:k]
        slot_sel = free_slots[:k]

        A_new = A.clone()
        S_new = S.clone()
        L_new = L.clone()

        A_new[slot_sel] = q_flat[q_sel]
        S_new[slot_sel] = self.strength_init
        L_new[slot_sel] = step

        return A_new, S_new, L_new

    # ======================================================================
    # Decay + Eviction
    # ======================================================================
    def _decay(self, A, S, L):
        return A, S * self.lambda_decay, L

    def _evict(self, A, S, L):
        dead = S < self.strength_min
        A = torch.where(dead.unsqueeze(-1), torch.zeros_like(A), A)
        S = torch.where(dead, torch.zeros_like(S), S)
        L = torch.where(dead, torch.zeros_like(L), L)
        return A, S, L

    # ======================================================================
    # Synthetic K/V Generation (Plan A)
    # ======================================================================
    def _gen_kv(self, query, A, S):
        B, T, d = query.shape
        eps = 1e-8

        alive = S > self.strength_min
        if alive.sum() == 0:
            zero = torch.zeros((B, T, self.num_heads, d),
                               device=query.device, dtype=query.dtype)
            return zero, zero

        S_alive = torch.where(alive, S, torch.zeros_like(S))
        A_alive = torch.where(alive.unsqueeze(-1), A, torch.zeros_like(A))

        total_strength = S_alive.sum() + eps

        # Burn-in gating
        if self.burn_in_mode == "strength":
            if total_strength < self.burn_in_threshold:
                zero = torch.zeros((B, T, self.num_heads, d),
                                   device=query.device, dtype=query.dtype)
                return zero, zero
        else:
            if self.timestep < self.burn_in_tokens:
                zero = torch.zeros((B, T, self.num_heads, d),
                                   device=query.device, dtype=query.dtype)
                return zero, zero

        field_vec = (A_alive * S_alive.unsqueeze(-1)).sum(dim=0) / total_strength

        unit = field_vec / (field_vec.norm() + eps)
        scale = self.max_influence_scale * torch.tanh(total_strength)

        field_scaled = scale * unit

        out = field_scaled.view(1, 1, 1, d).expand(B, T, self.num_heads, d)
        return out, out

    # ======================================================================
    # Stats
    # ======================================================================
    def get_stats(self):
        S = self.attractor_strengths
        alive = S > self.strength_min
        total_strength = float(S.sum())
        timestep = float(self.timestep.item())
        burn_in_active = (
            total_strength < self.burn_in_threshold
            if self.burn_in_mode == "strength"
            else timestep < self.burn_in_tokens
        )

        num_alive = int(alive.sum())
        mean = float((S * alive).sum() / (alive.sum() + 1e-8))

        return {
            "num_alive": num_alive,
            "total_strength": total_strength,
            "mean_strength": mean,
            "max_strength": float(S.max()),
            "timestep": timestep,
            "burn_in_active": bool(burn_in_active),
        }
