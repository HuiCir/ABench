# X-GRPO-002 harness

Evaluator code of the case; the self-check (`selfcheck.py`) runs the same stages on the dev split.

## Pipeline (`stages.py`)

```
X-GRPO-002 pipeline: the RL-trained Qwen2.5-0.5B-Instruct answers held-out Countdown and GSM8K problems; the
evaluator formats the prompts, decodes and verifies (evaluator and self-check use this same file).

One stage, "eval" (the base model is under models/Qwen2.5-0.5B-Instruct):
  timed   import model.py and call load(checkpoint_dir, models_dir, device), which returns the trained causal LM
  finish  greedy responses (ChatML template, at most 512 new tokens, lmpost.generate) to the held-out prompts of both
          task sets, and the number of parameters stored in checkpoint/                                (≤ 1,800 s)
```

## Files

- `abench_metrics.py`
- `abench_stage.py`
- `baseline/model.py`
- `baseline/src/lora.py`
- `baseline/src/verify_math.py`
- `baseline/train.py`
- `interface.md`
- `lmpost.py`
- `metrics.py`
- `sandbox_guide.md`
- `selfcheck.py`
- `stages.py`
- `submission_rules.json`
- `textpack.py`
- `verify_math.py`

The held-out data, keys and judge configuration are never in this directory.
