# X-GRPO-002 interface

## Entry point: `model.py`

```python
def load(checkpoint_dir: str, models_dir: str, device: str):
    """Return the trained causal LM (transformers interface); models_dir/Qwen2.5-0.5B-Instruct is the base model."""
```

- The evaluator builds every prompt with `verify_math.countdown_prompt` or `verify_math.gsm8k_prompt`, wraps it in
  the ChatML template (`lmpost.CHAT`) and decodes greedily with the base model's tokenizer.
- `train.py --data "$ABENCH_DATA_DIR" --out checkpoint` must reproduce the checkpoint.

## Evaluator code you can read

`verify_math.py` (generator, prompts, verifiers), `lmpost.py` (template, decoding), `stages.py`, `metrics.py`,
`textpack.py`, `abench_metrics.py`, `abench_stage.py`, `selfcheck.py`, and the reference under `baseline/`.

## Available software

Python 3, PyTorch (CUDA), transformers, safetensors, NumPy. Nothing can be installed.
