# C-SPEC-001 interface

## Entry point: `model.py`

```python
def load(checkpoint_dir: str, models_dir: str, device: str):
    """models_dir holds Qwen2.5-1.5B-Instruct/ (target) and Qwen2.5-0.5B-Instruct/ (draft), Hugging Face format."""

def generate(model, prompt_ids: list[int], max_new_tokens: int) -> list[int]:
    """The target's greedy continuation: new tokens only, ending with an end-of-sequence token
    (generation_config.json) or after max_new_tokens tokens."""
```

- Batch size 1; `max_new_tokens` is 256; `generate` runs under `torch.no_grad()`.
- `greedy.py` shows the evaluator's reference decoding and the tie-tolerant comparison.

## Evaluator code you can read

`greedy.py`, `stages.py` (timing protocol), `metrics.py`, `abench_metrics.py`, `abench_stage.py`, `selfcheck.py`,
and the reference submission's code under `baseline/`.

## Available software

Python 3, PyTorch (CUDA, Triton), transformers, safetensors, NumPy. Nothing can be installed.
