# C-SPEC-001 harness

Evaluator code of the case; the self-check (`selfcheck.py`) runs the same stages on the dev split.

## Pipeline (`stages.py`)

```
C-SPEC-001 pipeline: end-to-end speed of your decoding method against plain greedy decoding of the target model,
with the output required to be the target's greedy output (evaluator and self-check use this same file).

The evaluator does not train. One stage, "decode":
  timed   import model.py, load(checkpoint_dir, models_dir, device) -> model               (load time, killed at 600 s)
  finish  for each workload and prompt (batch size 1): the evaluator first runs plain greedy decoding of the target
          (greedy.py, fp32) and times it, then calls generate(model, prompt_ids, max_new_tokens) and times it; the
          tokens are compared with the tie-tolerant rule of greedy.py                         (all ≤ 3,000 s)
```

## Files

- `abench_metrics.py`
- `abench_stage.py`
- `baseline/model.py`
- `baseline/train.py`
- `greedy.py`
- `interface.md`
- `metrics.py`
- `sandbox_guide.md`
- `selfcheck.py`
- `stages.py`
- `submission_rules.json`

The held-out data, keys and judge configuration are never in this directory.
