# Paired conversation-memory evaluation

This pilot compares three conditions on the same 40 short, original conversation
cases and the same model checkpoint:

1. **No DRAI** receives the query and answer options only.
2. **DRAI state** reads earlier sessions in order, carries attractor state into
   the query, and clears state before the next independent conversation.
3. **Full history** receives all earlier sessions with the query as a reference
   ceiling.

Each arm scores the same complete answer options by mean token log likelihood.
The benchmark reports accuracy and paired 95% percentile intervals overall and
by category. Intervals resample conversations as clusters. It also reports
context and query-scoring latency, token counts, and allocated DRAI state bytes.
DRAI state is restored to its post-context snapshot before each option is scored,
so one candidate cannot alter the next candidate's result.

The test categories follow LongMemEval's information retrieval, multi-session,
knowledge-update, temporal, and abstention taxonomy. The conversational and
implicit/composed question style follows the evaluation direction of
LOCOMO-CONV. Included cases are original synthetic probes, not copies or a
reproduction of either published dataset.

This is an exploratory behavioral comparison, not a generalization or
consciousness claim. The short contexts fit the default Pythia-410M context
window. The raw pretrained checkpoint is not instruction-tuned, so the result
measures candidate likelihood under this protocol rather than chat quality.

## Run

~~~bash
python -m pip install -e .
python benchmarks/memory_evaluation/run.py \
  --model EleutherAI/pythia-410m \
  --output results/memory-evaluation.json
~~~

Use --revision to pin a Hugging Face commit, --device cuda when available,
--cases path/to/cases.jsonl for normalized custom cases, or
--max-per-category 1 for a quick plumbing check. The report records the
resolved model revision and checks that baseline and DRAI parameter hashes
match before scoring.

Each JSONL record must contain:

~~~json
{"case_id":"example-01","conversation_id":"conversation-01","category":"direct_recall","sessions":["User: Earlier fact.","User: Later turn."],"question":"What was the fact?","options":["correct","wrong one","wrong two"],"answer":"correct"}
~~~

Supported category names are direct_recall, composed_implicit, changed_fact,
conflict_resolution, and abstention.

## Research references

- LongMemEval (ICLR 2025): https://github.com/xiaowu0162/LongMemEval
- LOCOMO-CONV (Findings of EMNLP 2026): https://github.com/MiuLab/LoCoMo-Conv
- Titans: Learning to Memorize at Test Time: https://arxiv.org/abs/2501.00663

The older story benchmark remains available for historical comparison. It puts
the full story in every question prompt, so it does not test recall after prior
history has been withheld.
