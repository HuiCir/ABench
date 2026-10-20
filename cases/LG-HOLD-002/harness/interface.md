# LG-HOLD-002 interface

## Entry point: `model.py`

```python
def load(checkpoint_dir: str, models_dir: str | None, device: str):
    """Load your trained holding selector from checkpoint/ and return it (any Python object).

    models_dir   directory containing roberta-base/ (read-only; None only in synthetic plumbing tests)
    device       "cuda" at evaluation
    """

def predict(model, prompts: list[str], holdings: list[list[str]]) -> numpy.ndarray:
    """prompts: M citing contexts; holdings: M lists of 5 candidate holdings. Return float (M, 5) scores."""
```

- `model.py` may import modules under `src/`; the submission directory is on `sys.path`.

## Training program: `train.py`

Write a `train.py` that produces `checkpoint/` from the files in `$ABENCH_DATA_DIR` (`train.npz`: `prompt`, `holdings`, `y` and the official validation split `val_prompt`, `val_holdings`, `val_y`) and, if you use it, `$ABENCH_MODELS_DIR/roberta-base`, for example
`python3 train.py --data "$ABENCH_DATA_DIR" --out "$ABENCH_SUBMISSION_DIR/checkpoint"`. The evaluator does not run it;
it is kept with the submission so the checkpoint can be audited and reproduced. The reference implementation under
`interface/baseline/` (its `model.py` and `train.py`) is a working starting point you may improve on or replace.

## Evaluator code you can read

`stages.py` (pipeline, dev split, time limits), `metrics.py` and `abench_metrics.py` (macro-F1),
`abench_stage.py` (process and timing protocol) and `selfcheck.py`.

## Available software

Python 3, PyTorch (CUDA, Triton), NumPy, Hugging Face transformers / tokenizers / safetensors (offline). Nothing can be installed.
