# U-DLM-005 interface

## Entry point: `sampler.py`

```python
def sample(denoiser, prefix, length: int, nfe: int, gen) -> "torch.LongTensor":
    """Return (B, length) GPT-2 token ids with no mask token left; keep prefix (B, 256) unchanged if given.
    denoiser(x) -> (B, L, 50257) log-probabilities; at most nfe calls per sample() call."""
```

## Evaluator code you can read

`dlm.py` (denoiser, baseline sampler, perplexity and entropy), `stages.py`, `metrics.py`, `abench_metrics.py`,
`abench_stage.py`, `selfcheck.py`, the ancestral baseline under `baselines/ancestral/` and the reference under
`baseline/`.

## Available software

Python 3, PyTorch (CUDA), transformers, NumPy. Nothing can be installed.
