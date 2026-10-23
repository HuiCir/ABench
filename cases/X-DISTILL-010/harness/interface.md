# X-DISTILL-010 interface

## Entry point: `model.py`

```python
def load(checkpoint_dir: str, models_dir: str, device: str):
    """Return the student: a causal LM with SmolLM2-135M's parameter names and shapes (e.g. AutoModelForCausalLM
    from models_dir/SmolLM2-135M with your weights loaded), on `device`, in eval mode."""
```

- The evaluator calls `model(input_ids=...)` on windows of 1,025 tokens (`lmpost.heldout_nll`).

## Evaluator code you can read

`lmpost.py`, `token_data.py`, `stages.py`, `metrics.py`, `abench_metrics.py`, `abench_stage.py`, `selfcheck.py`, and
the reference submission's code under `baseline/`.

## Available software

Python 3, PyTorch (CUDA, Triton), transformers, safetensors, NumPy. Nothing can be installed.
