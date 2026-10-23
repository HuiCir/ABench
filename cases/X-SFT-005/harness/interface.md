# X-SFT-005 interface

## Entry point: `model.py`

```python
def load(checkpoint_dir: str, models_dir: str, device: str):
    """Return the fine-tuned causal LM (e.g. AutoModelForCausalLM from models_dir/SmolLM2-360M with your
    checkpoint applied), on `device`, in eval mode."""
```

- The evaluator calls `model.generate(input_ids, attention_mask, max_new_tokens=512, do_sample=False, eos_token_id=[<|im_end|>, <|endoftext|>])` on left-padded ChatML prompts and `model(input_ids=...)` for the held-out loss.

## Evaluator code you can read

`ifcheck.py` (constraint types, wording, checkers, generator), `lmpost.py` (template, decoding, loss, trained-number
count), `stages.py`, `metrics.py`, `textpack.py`, `abench_metrics.py`, `abench_stage.py`, `selfcheck.py`, and the
reference submission's code under `baseline/`.

## Available software

Python 3, PyTorch (CUDA), transformers, safetensors, NumPy. Nothing can be installed.
