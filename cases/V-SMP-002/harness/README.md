# V-SMP-002 harness

Evaluator code of the case; the self-check (`selfcheck.py`) runs the same stages on the dev split.

## Pipeline (`stages.py`)

```
V-SMP-002 pipeline: training-free sampling from a frozen CIFAR-10 DDPM at small NFE, scored by FID.

Stage "sample" (imports sampler.py)
  setup   load the frozen UNet to the GPU (untimed)
  timed   import sampler.py
  finish  for each NFE in (5, 10, 20): N images from sample(denoiser, ...) in batches, each batch with its own
          seeded generator; the denoiser refuses calls beyond the NFE budget; images saved as uint8 for the next stage
Stage "fid" (does not import your code; receives only the image files)
  finish  Inception-v3 pool features (pytorch-fid weights) → mean and covariance per NFE; the evaluator computes FID
          against the CIFAR-10 training-set statistics
```

## Files

- `abench_metrics.py`
- `abench_stage.py`
- `baselines/ddim/sampler.py`
- `diffusion.py`
- `interface.md`
- `metrics.py`
- `sandbox_guide.md`
- `selfcheck.py`
- `stages.py`
- `submission_rules.json`

The held-out data, keys and judge configuration are never in this directory.
