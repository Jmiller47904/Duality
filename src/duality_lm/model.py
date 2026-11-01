"""Native decoder-only Duality language model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .config import DualityLMConfig
from .memory import DualityMemory, MemoryState


KVCache = Tuple[Tensor, Tensor]
LayerState = Optional[MemoryState]


@dataclass
class DualityLMOutput:
    """Output container for logits, loss, memory state, and KV cache."""

    logits: Tensor
    loss: Optional[Tensor] = None
    memory_state: Optional[Tuple[LayerState, ...]] = None
    past_key_values: Optional[Tuple[Optional[KVCache], ...]] = None


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with an optional KV cache."""

    def __init__(self, config: DualityLMConfig) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.dropout = config.dropout
        self.query = nn.Linear(config.d_model, config.d_model, bias=False)
        self.key = nn.Linear(config.d_model, config.d_model, bias=False)
        self.value = nn.Linear(config.d_model, config.d_model, bias=False)
        self.output = nn.Linear(config.d_model, config.d_model, bias=False)

    def _split_heads(self, tensor: Tensor) -> Tensor:
        batch, sequence_length, _ = tensor.shape
        return tensor.view(batch, sequence_length, self.n_heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, tensor: Tensor) -> Tensor:
        batch, _, sequence_length, _ = tensor.shape
        return tensor.transpose(1, 2).contiguous().view(
            batch, sequence_length, self.n_heads * self.head_dim
        )

    def forward(
        self,
        hidden_states: Tensor,
        past_key_value: Optional[KVCache] = None,
        use_cache: bool = False,
    ) -> Tuple[Tensor, Optional[KVCache]]:
        batch, sequence_length, _ = hidden_states.shape
        query = self._split_heads(self.query(hidden_states))
        key = self._split_heads(self.key(hidden_states))
        value = self._split_heads(self.value(hidden_states))

        past_length = 0
        if past_key_value is not None:
            past_key, past_value = past_key_value
            past_length = past_key.shape[2]
            key = torch.cat((past_key, key), dim=2)
            value = torch.cat((past_value, value), dim=2)

        key_length = key.shape[2]
        scores = torch.matmul(query, key.transpose(-2, -1)) / (self.head_dim**0.5)

        query_positions = torch.arange(
            past_length, past_length + sequence_length, device=hidden_states.device
        )
        key_positions = torch.arange(key_length, device=hidden_states.device)
        allowed = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)
        scores = scores.masked_fill(~allowed.unsqueeze(0).unsqueeze(0), torch.finfo(scores.dtype).min)

        weights = F.softmax(scores, dim=-1)
        weights = F.dropout(weights, p=self.dropout, training=self.training)
        attended = torch.matmul(weights, value)
        output = self.output(self._merge_heads(attended))

        present = (key, value) if use_cache else None
        return output, present


class FeedForward(nn.Module):
    """Gated feed-forward network used inside each decoder block."""

    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.gate = nn.Linear(d_model, d_ff, bias=False)
        self.value = nn.Linear(d_model, d_ff, bias=False)
        self.output = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states: Tensor) -> Tensor:
        gated = F.silu(self.gate(hidden_states)) * self.value(hidden_states)
        return self.dropout(self.output(gated))


class DualityBlock(nn.Module):
    """Pre-norm decoder block with an optional attractor-memory pathway."""

    def __init__(self, config: DualityLMConfig, use_memory: bool) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(config.d_model)
        self.attention = CausalSelfAttention(config)
        self.output_norm = nn.LayerNorm(config.d_model)
        self.feed_forward = FeedForward(config.d_model, config.feed_forward_dim, config.dropout)
        self.dropout = nn.Dropout(config.dropout)
        self.memory = (
            DualityMemory(
                d_model=config.d_model,
                slots=config.memory_slots,
                match_threshold=config.memory_match_threshold,
                alpha=config.memory_alpha,
                decay=config.memory_decay,
                strength_init=config.memory_strength_init,
                strength_min=config.memory_strength_min,
                influence_scale=config.memory_influence_scale,
                temperature=config.memory_temperature,
            )
            if use_memory
            else None
        )

    def forward(
        self,
        hidden_states: Tensor,
        memory_state: LayerState = None,
        past_key_value: Optional[KVCache] = None,
        use_cache: bool = False,
    ) -> Tuple[Tensor, LayerState, Optional[KVCache]]:
        normalized = self.input_norm(hidden_states)
        memory_influence = torch.zeros_like(normalized)
        new_memory_state: LayerState = memory_state
        if self.memory is not None:
            memory_influence, new_memory_state = self.memory(normalized, memory_state)

        attention_output, present = self.attention(
            normalized, past_key_value=past_key_value, use_cache=use_cache
        )
        hidden_states = hidden_states + self.dropout(attention_output + memory_influence)
        hidden_states = hidden_states + self.feed_forward(self.output_norm(hidden_states))
        return hidden_states, new_memory_state, present


class DualityLM(nn.Module):
    """A trainable decoder-only language model with internal Duality memory.

    ``forward`` supports both ordinary teacher-forced training and incremental
    decoding.  Pass the returned ``memory_state`` and ``past_key_values`` back
    into a later call to continue a sequence without rebuilding the prompt.
    """

    def __init__(self, config: DualityLMConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.max_seq_len, config.d_model)
        self.embedding_dropout = nn.Dropout(config.dropout)

        memory_layers = set(config.resolved_memory_layers())
        self.layers = nn.ModuleList(
            [DualityBlock(config, use_memory=index in memory_layers) for index in range(config.n_layers)]
        )
        self.final_norm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if config.tie_embeddings:
            self.lm_head.weight = self.token_embedding.weight

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Use a small GPT-style initialization for stable toy training."""

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _normalize_input_ids(self, input_ids: Tensor) -> Tensor:
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        if input_ids.dtype != torch.long:
            input_ids = input_ids.long()
        if input_ids.numel() and (
            int(input_ids.min().item()) < 0
            or int(input_ids.max().item()) >= self.config.vocab_size
        ):
            raise ValueError("input_ids contain a token outside the configured vocabulary")
        return input_ids

    def _normalize_labels(self, labels: Tensor) -> Tensor:
        """Normalize labels while allowing PyTorch's ``-100`` ignore index."""

        if labels.ndim == 1:
            labels = labels.unsqueeze(0)
        if labels.ndim != 2:
            raise ValueError("labels must have shape [batch, sequence]")
        if labels.dtype != torch.long:
            labels = labels.long()
        valid = labels != -100
        if valid.any() and (
            int(labels[valid].min().item()) < 0
            or int(labels[valid].max().item()) >= self.config.vocab_size
        ):
            raise ValueError("labels contain a token outside the configured vocabulary")
        return labels

    @staticmethod
    def _past_length(
        past_key_values: Optional[Tuple[Optional[KVCache], ...]],
    ) -> int:
        if not past_key_values:
            return 0
        for key_value in past_key_values:
            if key_value is not None:
                return int(key_value[0].shape[2])
        return 0

    def forward(
        self,
        input_ids: Tensor,
        labels: Optional[Tensor] = None,
        memory_state: Optional[Tuple[LayerState, ...]] = None,
        past_key_values: Optional[Tuple[Optional[KVCache], ...]] = None,
        use_cache: bool = False,
        return_memory_state: bool = True,
        persistent_memory: bool = False,
    ) -> DualityLMOutput:
        """Run the model and optionally calculate next-token cross-entropy."""

        input_ids = self._normalize_input_ids(input_ids)
        batch_size, sequence_length = input_ids.shape
        past_length = self._past_length(past_key_values)
        if past_length + sequence_length > self.config.max_seq_len:
            raise ValueError(
                "sequence exceeds max_seq_len; shorten the prompt or use a larger config"
            )

        if memory_state is not None and len(memory_state) != len(self.layers):
            raise ValueError("memory_state must contain one entry per transformer layer")
        if past_key_values is not None and len(past_key_values) != len(self.layers):
            raise ValueError("past_key_values must contain one entry per transformer layer")

        if memory_state is None and persistent_memory:
            memory_state = getattr(self, "_persistent_memory_state", None)

        positions = torch.arange(
            past_length,
            past_length + sequence_length,
            device=input_ids.device,
        )
        hidden_states = self.token_embedding(input_ids) + self.position_embedding(positions)
        hidden_states = self.embedding_dropout(hidden_states)

        next_states: List[LayerState] = []
        next_cache: List[Optional[KVCache]] = []
        for layer_index, layer in enumerate(self.layers):
            previous_state = memory_state[layer_index] if memory_state is not None else None
            previous_cache = past_key_values[layer_index] if past_key_values is not None else None
            hidden_states, state, cache = layer(
                hidden_states,
                memory_state=previous_state,
                past_key_value=previous_cache,
                use_cache=use_cache,
            )
            next_states.append(state)
            next_cache.append(cache)

        logits = self.lm_head(self.final_norm(hidden_states))
        loss = None
        if labels is not None:
            labels = self._normalize_labels(labels)
            if labels.shape != input_ids.shape:
                raise ValueError("labels must have the same shape as input_ids")
            if sequence_length < 2:
                raise ValueError("at least two tokens are required to calculate causal loss")
            loss = F.cross_entropy(
                logits[:, :-1].contiguous().view(-1, self.config.vocab_size),
                labels[:, 1:].contiguous().view(-1),
                ignore_index=-100,
            )

        output_state = tuple(next_states) if return_memory_state else None
        output_cache = tuple(next_cache) if use_cache else None
        if persistent_memory:
            self._persistent_memory_state = (
                tuple(state.detach() if state is not None else None for state in next_states)
            )

        return DualityLMOutput(
            logits=logits,
            loss=loss,
            memory_state=output_state,
            past_key_values=output_cache,
        )

    def reset_memory(self) -> None:
        """Forget the optional state retained by ``persistent_memory=True``."""

        self._persistent_memory_state = None

    def memory_stats(
        self,
        memory_state: Optional[Tuple[LayerState, ...]] = None,
    ) -> List[dict]:
        """Return diagnostics for all memory-enabled blocks."""

        if memory_state is None:
            memory_state = getattr(self, "_persistent_memory_state", None)
        if memory_state is None:
            memory_state = tuple(None for _ in self.layers)
        if len(memory_state) != len(self.layers):
            raise ValueError("memory_state must contain one entry per transformer layer")

        stats = []
        for index, layer in enumerate(self.layers):
            if layer.memory is not None:
                stats.append({"layer": index, **layer.memory.stats(memory_state[index])})
        return stats

    @torch.no_grad()
    def generate(
        self,
        input_ids: Tensor,
        max_new_tokens: int = 32,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: float = 1.0,
        do_sample: bool = True,
        eos_token_id: Optional[int] = None,
    ) -> Tensor:
        """Generate tokens using prompt prefill, memory state, and KV caching."""

        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be >= 0")
        if temperature <= 0.0:
            raise ValueError("temperature must be > 0")
        if not 0.0 < top_p <= 1.0:
            raise ValueError("top_p must be in (0, 1]")

        input_ids = self._normalize_input_ids(input_ids)
        generated = input_ids.clone()
        output = self(
            generated,
            use_cache=True,
            return_memory_state=True,
        )
        memory_state = output.memory_state
        cache = output.past_key_values
        eos_token_id = self.config.eos_token_id if eos_token_id is None else eos_token_id

        for _ in range(max_new_tokens):
            logits = output.logits[:, -1, :] / temperature
            if top_k is not None:
                if top_k <= 0:
                    raise ValueError("top_k must be positive when provided")
                top_k = min(top_k, logits.shape[-1])
                threshold = torch.topk(logits, top_k, dim=-1).values[:, -1].unsqueeze(-1)
                logits = logits.masked_fill(logits < threshold, torch.finfo(logits.dtype).min)

            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                sorted_probs = F.softmax(sorted_logits, dim=-1)
                cumulative = torch.cumsum(sorted_probs, dim=-1)
                remove = cumulative > top_p
                remove[:, 1:] = remove[:, :-1].clone()
                remove[:, 0] = False
                sorted_logits = sorted_logits.masked_fill(remove, torch.finfo(logits.dtype).min)
                logits = torch.full_like(logits, torch.finfo(logits.dtype).min)
                logits.scatter_(1, sorted_indices, sorted_logits)

            if do_sample:
                next_token = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)

            generated = torch.cat((generated, next_token), dim=1)
            if eos_token_id is not None and torch.all(next_token.squeeze(-1) == eos_token_id):
                break

            output = self(
                next_token,
                memory_state=memory_state,
                past_key_values=cache,
                use_cache=True,
                return_memory_state=True,
            )
            memory_state = output.memory_state
            cache = output.past_key_values

        return generated
