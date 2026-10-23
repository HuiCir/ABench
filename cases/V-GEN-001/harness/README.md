# V-GEN-001 harness

Evaluator code of the case; the self-check (`selfcheck.py`) runs the same stages on the dev split.

## Pipeline (`stages.py`)

```
V-GEN-001 pipeline: images from your trained generator, scored by FID against CIFAR-10 (evaluator and self-check).

Stage "sample" (imports model.py; only checkpoint/ is available)
  timed   load(checkpoint_dir, device) -> net, a torch.nn.Module with at most PARAMS["max_params"] parameters
  finish  for each step budget S in (25, 100): N images from generate(net, n, S, generator, device) in batches, each batch
          with its own seeded generator; a forward hook counts calls of net during each generate() (at most S);
          images saved as uint8 for the next stage
Stage "fid" (does not import your code; receives only the image files)
  finish  Inception-v3 pool features (pytorch-fid weights) -> mean and covariance per budget; the evaluator computes
          FID against the CIFAR-10 training-set statistics
```

## Files

- `abench_metrics.py`
- `abench_stage.py`
- `baseline/model.py`
- `baseline/train.py`
- `interface.md`
- `metrics.py`
- `sandbox_guide.md`
- `selfcheck.py`
- `stages.py`
- `submission_rules.json`

The held-out data, keys and judge configuration are never in this directory.
