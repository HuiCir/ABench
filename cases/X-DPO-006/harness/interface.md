# X-DPO-006 interface

## Entry point: `model.py`

```python
def load(checkpoint_dir: str, models_dir: str, device: str):
    """Return the trained causal LM (e.g. AutoModelForCausalLM from models_dir/SmolLM2-360M-Instruct with your
    checkpoint applied), on `device`, in eval mode."""
```

- The evaluator calls `generate()` (greedy, at most 512 new tokens) on left-padded ChatML prompts and `forward()`
  for log-probabilities of answers (`lmpost.response_logprob`).

## Evaluator code you can read

`lmpost.py` (template, decoding, log-probabilities, trained-number count), `stages.py`, `metrics.py`, `textpack.py`,
`abench_metrics.py`, `abench_stage.py`, `selfcheck.py`, and the reference submission's code under `baseline/`.

## Available software

Python 3, PyTorch (CUDA), transformers, safetensors, NumPy. Nothing can be installed.
