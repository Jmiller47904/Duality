# Story Comprehension Benchmark - Pythia-410M

This exploratory benchmark measures exact-match answers on synthetic stories.
It remains useful as a simple story-QA check, but it is **not** a memory
isolation evaluation: each question receives the entire story in its prompt.

## Historical result

The values below came from an earlier stochastic run and are retained only as
historical observations:

~~~
Baseline (no DRAI): 85.0% accuracy (51/60 correct)
V1 DRAI:            88.3% accuracy (53/60 correct)
~~~

They are not the expected output of the current greedy-decoding script and do
not establish a general improvement.

## Current protocol

The script generates 20 stories and calculates the question count from the
generated data; it is not fixed at 60. It uses greedy decoding, compares the
complete first answer line after case/punctuation normalization, and resets
DRAI buffers before each independent story while retaining state within a
story's questions. Results can still vary with the model, tokenizer, library
versions, and hardware.

The latest completed scheduled run (November 8, 2026) scored 1/157 for both the
baseline and DRAI arms. The November 15 scheduled run was canceled during the
benchmark step. This near-zero result is not a usable memory result: the model
sees the full story for every question, and this task does not test recall after
history is removed.

For the controlled three-condition comparison, see
[the paired memory evaluation](../memory_evaluation/README.md).

## Run

~~~bash
python benchmarks/story_comprehension_410m/run.py
~~~
