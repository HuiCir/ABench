# X-DPO-006 harness

Evaluator code of the case; the self-check (`selfcheck.py`) runs the same stages on the dev split.

## Pipeline (`stages.py`)

```
X-DPO-006 pipeline: preference optimisation of SmolLM2-360M-Instruct, judged by a frozen reward model against the
starting model's own answers (evaluator and self-check).

Two stages, each in a new process:
  generate  (the starting model is under models/SmolLM2-360M-Instruct) timed: import model.py and call
            load(checkpoint_dir, models_dir, device). finish: greedy responses of your model and of the starting
            model to the held-out prompts (ChatML, at most 512 new tokens); log-probabilities of your responses under
            both models (KL estimate); implicit-reward accuracy on the held-out preference pairs; numbers stored in
            checkpoint/. Responses are written to responses.npz for the next stage.
  judge     (evaluator-only reward model; your code is not imported) finish: reward of every (prompt, response)
            for your responses and the starting model's. Without the reward model (self-check) the rewards are NaN.
```

## Files

- `abench_metrics.py`
- `abench_stage.py`
- `baseline/model.py`
- `baseline/src/lora.py`
- `baseline/train.py`
- `interface.md`
- `lmpost.py`
- `metrics.py`
- `sandbox_guide.md`
- `selfcheck.py`
- `stages.py`
- `submission_rules.json`
- `textpack.py`

The held-out data, keys and judge configuration are never in this directory.
