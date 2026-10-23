# B-OPT-001 harness

Evaluator code of the case; the self-check (`selfcheck.py`) runs the same stages on the dev split.

## Pipeline (`stages.py`)

```
B-OPT-001 pipeline: train the fixed language model (lm_model.py) with your update rule, then measure held-out loss.

Stage "train"  (imports optimizer.py)
  setup   map the training token shards, build the model on the GPU with the seed (untimed)
  timed   import optimizer.py and hparam_space.yaml, build RULE with the declared defaults, train STEPS steps
  finish  save model.pt; report the loss curve, divergence and the size of the optimizer state
Stage "eval"   (does not import your code; receives only model.pt)
  timed   load model.pt
  finish  mean next-token loss on the held-out tokens
```

## Files

- `abench_metrics.py`
- `abench_stage.py`
- `agent_tools.md`
- `baselines/adamw/hparam_space.yaml`
- `baselines/adamw/optimizer.py`
- `interface.md`
- `lm_model.py`
- `metrics.py`
- `optim_api.py`
- `reference_rules.py`
- `sandbox_guide.md`
- `selfcheck.py`
- `stages.py`
- `submission_rules.json`
- `token_data.py`

The held-out data, keys and judge configuration are never in this directory.
