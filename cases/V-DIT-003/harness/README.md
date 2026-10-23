# V-DIT-003 harness

Evaluator code of the case; the self-check (`selfcheck.py`) runs the same stages on the dev split.

## Pipeline (`stages.py`)

```
V-DIT-003 pipeline: the evaluator trains your latent-diffusion backbone with the fixed recipe of ldm.py on Imagenette
latents, samples with DDIM and classifier-free guidance, and measures FID (evaluator and self-check use this file).

Stage "train" (imports backbone.py):
  setup   load the public latents (untimed)
  timed   net = make_backbone((4, 16, 16), 10); forward FLOPs per latent; ldm.train for ldm.STEPS steps
  finish  with the EMA weights: N_SAMPLES latents with guidance 1.5 and N_SAMPLES without, from the held-out seeds
          (500 per class), saved for the next stage; FLOPs, parameters, loss curve, divergence           (≤ 900 s)
Stage "fid" (does not import your code; receives only the latent files)
  finish  VAE decoding to 128 x 128 images, Inception-v3 pool features (pytorch-fid weights) -> mean and covariance
          per guidance setting; the evaluator computes FID against the Imagenette validation images
The evaluator trains DiT-S/2 (baselines/dit_s) through the same stages on the same seed.
```

## Files

- `abench_metrics.py`
- `abench_stage.py`
- `baseline/backbone.py`
- `baselines/dit_s/backbone.py`
- `interface.md`
- `ldm.py`
- `metrics.py`
- `sandbox_guide.md`
- `selfcheck.py`
- `stages.py`
- `submission_rules.json`

The held-out data, keys and judge configuration are never in this directory.
