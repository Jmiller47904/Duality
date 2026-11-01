"""Tests for the native DualityLM milestone."""

import pytest
import torch

from duality_lm import ByteTokenizer, DualityLM, DualityLMConfig, DualityMemory


def tiny_model() -> DualityLM:
    tokenizer = ByteTokenizer()
    return DualityLM(DualityLMConfig.tiny(vocab_size=tokenizer.vocab_size, max_seq_len=64))


def test_byte_tokenizer_round_trip() -> None:
    tokenizer = ByteTokenizer()
    text = "memory: café"
    encoded = tokenizer.encode(text)
    assert tokenizer.decode(encoded) == text
    assert max(encoded) < tokenizer.vocab_size


def test_memory_updates_and_is_bounded() -> None:
    memory = DualityMemory(d_model=16, slots=4, match_threshold=0.7, influence_scale=0.15)
    hidden = torch.randn(2, 5, 16)
    influence, state = memory(hidden)
    assert influence.shape == hidden.shape
    assert state.attractors.shape == (2, 4, 16)
    assert state.timestep.tolist() == [5.0, 5.0]
    assert torch.isfinite(influence).all()
    assert float(influence.detach().abs().max()) <= 0.15
    assert torch.count_nonzero(state.strengths).item() > 0


def test_forward_loss_and_memory_state() -> None:
    model = tiny_model()
    tokenizer = ByteTokenizer()
    tokens = torch.tensor([tokenizer.encode("Duality memory", add_eos=True)], dtype=torch.long)
    output = model(tokens, labels=tokens)
    assert output.logits.shape == (1, tokens.shape[1], tokenizer.vocab_size)
    assert output.loss is not None
    assert torch.isfinite(output.loss)
    assert output.memory_state is not None
    assert model.memory_stats(output.memory_state)[0]["active_slots"] > 0


def test_incremental_cache_matches_full_logits() -> None:
    torch.manual_seed(3)
    model = tiny_model().eval()
    tokens = torch.randint(0, model.config.vocab_size, (1, 8))

    full = model(tokens, use_cache=False).logits[:, -1]
    prefix = model(tokens[:, :-1], use_cache=True)
    incremental = model(
        tokens[:, -1:],
        memory_state=prefix.memory_state,
        past_key_values=prefix.past_key_values,
        use_cache=True,
    ).logits[:, -1]
    assert torch.allclose(full, incremental, atol=1e-5, rtol=1e-4)


def test_generation_is_deterministic_without_sampling() -> None:
    torch.manual_seed(11)
    model = tiny_model().eval()
    prompt = torch.tensor([[1, 3, 4, 5]], dtype=torch.long)
    first = model.generate(prompt, max_new_tokens=5, do_sample=False)
    second = model.generate(prompt, max_new_tokens=5, do_sample=False)
    assert first.shape == (1, 9)
    assert torch.equal(first, second)


def test_persistent_memory_can_be_reset() -> None:
    model = tiny_model().eval()
    prompt = torch.tensor([[1, 3, 4, 5]], dtype=torch.long)
    first = model(prompt, persistent_memory=True)
    second = model(prompt[:, :1], persistent_memory=True)
    assert first.memory_state is not None
    assert second.memory_state is not None
    assert second.memory_state[2] is not None
    assert second.memory_state[2].timestep.item() > first.memory_state[2].timestep.item()

    model.reset_memory()
    reset = model(prompt[:, :1], persistent_memory=True)
    assert reset.memory_state is not None
    assert reset.memory_state[2] is not None
    assert reset.memory_state[2].timestep.item() == 1.0


def test_persistent_memory_survives_when_state_output_is_suppressed() -> None:
    model = tiny_model().eval()
    prompt = torch.tensor([[1, 3, 4, 5]], dtype=torch.long)

    hidden_output = model(
        prompt,
        persistent_memory=True,
        return_memory_state=False,
    )

    assert hidden_output.memory_state is None
    assert any(item["active_slots"] > 0 for item in model.memory_stats())

    continuation = model(prompt[:, :1], persistent_memory=True)
    assert continuation.memory_state is not None
    assert continuation.memory_state[2] is not None
    assert continuation.memory_state[2].timestep.item() == 5.0

    model.reset_memory()
    assert all(item["active_slots"] == 0 for item in model.memory_stats())


def test_config_rejects_bad_memory_layer() -> None:
    with pytest.raises(ValueError):
        DualityLMConfig(n_layers=2, memory_layers=(2,))


def test_empty_memory_layers_disable_the_pathway() -> None:
    config = DualityLMConfig.tiny(memory_layers=())
    model = DualityLM(config)
    assert config.resolved_memory_layers() == ()
    assert model.memory_stats() == []
