# X-SFT-005 harness

Evaluator code of the case; the self-check (`selfcheck.py`) runs the same stages on the dev split.

## Pipeline (`stages.py`)

```
X-SFT-005 pipeline: instruction following of a supervised fine-tune of SmolLM2-360M, its forgetting and its
trained-parameter cost (evaluator and self-check).

One stage, "eval" (the base model is under models/SmolLM2-360M):
  timed   import model.py and call load(checkpoint_dir, models_dir, device), which returns the fine-tuned causal LM
  finish  greedy responses (ChatML template, at most 512 new tokens, lmpost.generate) to the held-out prompts;
          held-out FineWeb-Edu loss of the fine-tuned model and of the base model; numbers stored in checkpoint/
                                                                                                   (≤ 1,800 s)
```

## Files

- `abench_metrics.py`
- `abench_stage.py`
- `baseline/model.py`
- `baseline/src/lora.py`
- `baseline/train.py`
- `ifcheck.py`
- `interface.md`
- `lmpost.py`
- `metrics.py`
- `sandbox_guide.md`
- `selfcheck.py`
- `stages.py`
- `submission_rules.json`
- `textpack.py`

The held-out data, keys and judge configuration are never in this directory.
