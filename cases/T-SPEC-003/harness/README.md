# T-SPEC-003 harness

Evaluator code of the case; the self-check (`selfcheck.py`) runs the same stages on the dev split.

## Pipeline (`stages.py`)

```
T-SPEC-003 pipeline: training-free sampling from the frozen CIFAR-10 DDPM of V-SMP-002 in few sequential rounds,
each round evaluating up to MAX_PARALLEL inputs per image in one batched call; scored by FID.

Stage "sample" (imports sampler.py)
  setup   load the frozen UNet to the GPU (untimed)
  timed   import sampler.py
  finish  for each round budget R in (5, 10): N images from sample(denoiser, n, R, generator, device) in batches, each
          batch with its own seeded generator; the denoiser refuses rounds beyond R and more than MAX_PARALLEL inputs
          per image in one round; images saved as uint8 for the next stage
Stage "fid" (does not import your code; receives only the image files)
  finish  Inception-v3 pool features (pytorch-fid weights) → mean and covariance per round budget; the evaluator computes FID
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
