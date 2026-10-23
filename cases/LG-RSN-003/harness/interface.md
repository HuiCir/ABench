# LG-RSN-003 interface

## Entry point: `pipeline.py`

```python
def load(checkpoint_dir: str, models_dir: str, device: str):
    """models_dir/Qwen2.5-1.5B-Instruct holds the frozen model (Hugging Face format)."""

def predict(state, task: dict, rows: list[dict]) -> list[str]:
    """task: {"name", "prompt_template", "labels", "columns", "train"}; rows: [{column: value}];
    return one label per row, spelled as in task["labels"]."""
```

`legaltasks.fill(task["prompt_template"], row)` gives the LegalBench prompt for a row.

## Evaluator code you can read

`legaltasks.py` (task format, label matching, balanced accuracy), `stages.py`, `metrics.py`, `abench_metrics.py`,
`abench_stage.py`, `selfcheck.py`, and the reference submission under `baseline/`.

## Available software

Python 3, PyTorch (CUDA), transformers, NumPy, SciPy, scikit-learn. Nothing can be installed.
