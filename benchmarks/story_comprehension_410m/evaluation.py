"""Deterministic answer helpers for story evaluation."""

import re

EVALUATION_SEED = 42


def seed_model_initialization(torch_module):
    """Seed model loading and added-memory initialization for each arm."""

    torch_module.manual_seed(EVALUATION_SEED)


def normalize_answer(text):
    """Normalize a complete answer for case- and punctuation-insensitive EM."""

    text = re.sub(r"[^\w\s]", " ", str(text).casefold(), flags=re.UNICODE)
    return " ".join(text.split())


def generate_answer(model, tokenizer, prompt):
    """Return the complete first generated line using greedy decoding."""

    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]
    outputs = model.generate(
        input_ids,
        max_new_tokens=20,
        do_sample=False,
        repetition_penalty=1.2,
        pad_token_id=tokenizer.pad_token_id,
    )
    generated = tokenizer.decode(
        outputs[0][input_ids.shape[1]:],
        skip_special_tokens=True,
    ).strip()
    return generated.splitlines()[0].strip() if generated else ""
