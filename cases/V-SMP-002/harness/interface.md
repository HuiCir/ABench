# V-SMP-002 interface

## Entry point: `sampler.py`

```python
def sample(denoiser, n: int, nfe: int, generator: torch.Generator, device: str) -> torch.Tensor:
    """Return n images, float tensor (n, 3, 32, 32) with values in [-1, 1].

    denoiser(x, t) -> ε prediction, same shape as x; x float (B, 3, 32, 32), t a number or tensor () / (B,) in [0, 999]
    denoiser.alphas_cumprod   numpy float64 (1000,), ᾱ_t of the linear β schedule
    nfe                       5, 10 or 20: maximum number of denoiser calls in this call of sample()
    generator                 torch.Generator on `device`; draw all randomness from it
    """
```

- The forward process is x_t = sqrt(ᾱ_t) x₀ + sqrt(1 − ᾱ_t) ε with x₀ in [-1, 1]; t = 0 is the least noisy training step.
- The evaluator calls `sample` with n = 500, 20 times per budget, each time with a fresh `Denoiser` whose budget is `nfe`.

## Evaluator code you can read

- `diffusion.py`: schedule and the counting `Denoiser`; `stages.py`: sampling loop, image conversion (round((x + 1) · 127.5)) and the Inception features.
- `baselines/ddim/sampler.py`: the DDIM baseline the evaluator runs alongside your sampler.
- `metrics.py` (FID), `abench_metrics.py`, `abench_stage.py`, `selfcheck.py`.

## Available software

Python 3, PyTorch (CUDA, Triton), NumPy, SciPy, diffusers, pytorch-fid (offline). Nothing can be installed.
