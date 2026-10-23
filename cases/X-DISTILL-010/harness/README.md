# X-DISTILL-010 harness

Evaluator code of the case; the self-check (`selfcheck.py`) runs the same stages on the dev split.

## Pipeline (`stages.py`)

```
X-DISTILL-010 pipeline: a SmolLM2-135M student trained in the sandbox, with SmolLM2-1.7B available as teacher,
scored by its held-out loss (evaluator and self-check).

One stage, "eval" (models/SmolLM2-135M, the student's starting point, and models/SmolLM2-1.7B, the teacher):
  timed   import model.py and call load(checkpoint_dir, models_dir, device); the returned model must have exactly
          SmolLM2-135M's parameter shapes
  finish  mean next-token loss on the held-out tokens of the student, of the starting model and of the teacher
                                                                                              (≤ 1,200 s)
```

## Files

- `abench_metrics.py`
- `abench_stage.py`
- `baseline/model.py`
- `baseline/train.py`
- `interface.md`
- `lmpost.py`
- `metrics.py`
- `sandbox_guide.md`
- `selfcheck.py`
- `stages.py`
- `submission_rules.json`
- `token_data.py`

The held-out data, keys and judge configuration are never in this directory.
