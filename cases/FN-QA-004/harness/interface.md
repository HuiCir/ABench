# FN-QA-004 interface

## Entry point: `pipeline.py`

```python
def load(checkpoint_dir: str, models_dir: str, device: str):
    """models_dir/Qwen2.5-1.5B-Instruct holds the frozen model (Hugging Face format)."""

def answer(state, items: list[dict]) -> list[dict]:
    """items: {"id", "question", "pre_text": [str], "post_text": [str], "table": [[str]]};
    return one dict per item with "answer" (number or "yes"/"no") and/or "program" (FinQA program)."""
```

## FinQA programs (`finqa.py`)

`subtract(5829, 5735), divide(#0, 5735)`: steps separated by commas; operations add, subtract, multiply, divide,
exp, greater, table_max, table_min, table_sum, table_average (the last four take a table row's header and `none`);
`#i` is the result of step i; constants `const_100`, `const_1000`, `const_m1`; `14%` means 0.14.

## Evaluator code you can read

`finqa.py` (executor and answer comparison), `stages.py`, `metrics.py`, `abench_metrics.py`, `abench_stage.py`,
`selfcheck.py`, and the reference submission under `baseline/`.

## Available software

Python 3, PyTorch (CUDA), transformers, NumPy, SciPy, scikit-learn. Nothing can be installed.
