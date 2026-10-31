"""Lightweight checks for the story benchmark's generation settings."""

import unittest

from benchmarks.story_comprehension_410m.evaluation import (
    EVALUATION_SEED,
    generate_answer,
    seed_model_initialization,
)


class _InputIds(list):
    shape = (1, 2)


class _Tokenizer:
    pad_token_id = 0

    def __call__(self, prompt, return_tensors):
        assert prompt == "story question"
        assert return_tensors == "pt"
        return {"input_ids": _InputIds([[1, 2]])}

    def decode(self, token_ids, skip_special_tokens):
        assert token_ids == [3]
        assert skip_special_tokens is True
        return "Alice."


class _Model:
    def __init__(self):
        self.generation_options = None

    def generate(self, input_ids, **options):
        self.generation_options = options
        return [[1, 2, 3]]


class _TorchStub:
    def __init__(self):
        self.seed = None

    def manual_seed(self, seed):
        self.seed = seed


class StoryEvaluationDeterminismTests(unittest.TestCase):
    def test_answer_uses_greedy_decoding_without_sampling_options(self):
        model = _Model()
        answer = generate_answer(model, _Tokenizer(), "story question")

        self.assertEqual(answer, "Alice.")
        self.assertIs(model.generation_options["do_sample"], False)
        self.assertEqual(model.generation_options["max_new_tokens"], 20)
        self.assertNotIn("temperature", model.generation_options)
        self.assertNotIn("top_p", model.generation_options)

    def test_model_initialization_seed_is_explicit(self):
        torch_stub = _TorchStub()
        seed_model_initialization(torch_stub)
        self.assertEqual(torch_stub.seed, EVALUATION_SEED)


if __name__ == "__main__":
    unittest.main()
