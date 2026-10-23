# FN-QA-004 harness

Evaluator code of the case; the self-check (`selfcheck.py`) runs the same stages on the dev split.

## Pipeline (`stages.py`)

```
FN-QA-004 pipeline: numerical questions over financial reports answered by a submitted pipeline around the frozen
Qwen2.5-1.5B-Instruct (evaluator and self-check use this same file).

One stage, "answer" (the frozen model is under models/Qwen2.5-1.5B-Instruct):
  timed   import pipeline.py, load(checkpoint_dir, models_dir, device) -> state                   (killed at 600 s)
  finish  answer(state, items) on the held-out questions in batches of 32, in a fixed shuffled order; each item is
          {id, question, pre_text, post_text, table} and each result an answer value and/or a FinQA program, which
          this stage executes on the item's table (finqa.execute)                                       (≤ 3,600 s)
```

## Files

- `abench_metrics.py`
- `abench_stage.py`
- `baseline/pipeline.py`
- `baseline/src/finqa_exec.py`
- `baseline/src/retrieve.py`
- `baseline/train.py`
- `finqa.py`
- `interface.md`
- `metrics.py`
- `sandbox_guide.md`
- `selfcheck.py`
- `stages.py`
- `submission_rules.json`

The held-out data, keys and judge configuration are never in this directory.
