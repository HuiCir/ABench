# T-OPT-002 harness

Evaluator code of the case; the self-check (`selfcheck.py`) runs the same stages on the dev split.

## Pipeline (`stages.py`)

```
T-OPT-002 pipeline: train the fixed ViT (vit_model.py) on CIFAR-10 with your update rule, then classify the test set.

Stage "train"  (imports optimizer.py)
  setup   load train.npz to the GPU, build the model with the seed (untimed)
  timed   import optimizer.py and hparam_space.yaml, build RULE with the declared defaults, train STEPS steps
  finish  save model.pt; report the loss curve, divergence and optimizer-state size
Stage "eval"   (does not import your code; receives only model.pt)
  timed   load model.pt
  finish  predicted classes for the held-out test images (labels stay with the evaluator)
```

## Files

- `abench_metrics.py`
- `abench_stage.py`
- `baselines/adamw/hparam_space.yaml`
- `baselines/adamw/optimizer.py`
- `interface.md`
- `metrics.py`
- `optim_api.py`
- `sandbox_guide.md`
- `selfcheck.py`
- `stages.py`
- `submission_rules.json`
- `vit_model.py`

The held-out data, keys and judge configuration are never in this directory.
