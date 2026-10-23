# LG-LONG-001 interface

## Entry point: `model.py`

```python
def load(checkpoint_dir: str, models_dir: str, device: str):
    """Load your trained classifier (and, if you use it, the encoder from models_dir/legal-bert-base-uncased)."""

def predict(model, texts: list[str]) -> numpy.ndarray:
    """texts: the facts of each case, paragraphs separated by a blank line. Return int (len(texts), 10) 0/1
    decisions in the order of the labels array: articles 2, 3, 5, 6, 8, 9, 10, 11, 14, P1-1."""
```

- The evaluator calls `load` inside the timed phase and `predict` once, on all held-out cases, after it.
- `model.py` may import modules under `src/`; the submission directory is on `sys.path`.

## Training program: `train.py`

Write a `train.py` that produces `checkpoint/` from `$ABENCH_DATA_DIR/ecthr_a.npz` (and, if you use it,
`$ABENCH_MODELS_DIR/legal-bert-base-uncased`), for example
`python3 train.py --data "$ABENCH_DATA_DIR" --out "$ABENCH_SUBMISSION_DIR/checkpoint"`. The evaluator does not run it;
it is kept with the submission so the checkpoint can be audited and reproduced. The reference implementation under
`interface/baseline/` (its `model.py` and `train.py`) is a working starting point you may improve on or replace.

## Evaluator code you can read

`stages.py` (pipeline, dev split, time limits), `metrics.py` and `abench_metrics.py` (micro- and macro-F1 with the
"no violation" label), `abench_stage.py` (process and timing protocol), `textpack.py` and `selfcheck.py`.

## Available software

Python 3, PyTorch (CUDA, Triton), NumPy, SciPy, transformers, tokenizers, safetensors (offline). Nothing can be installed.
