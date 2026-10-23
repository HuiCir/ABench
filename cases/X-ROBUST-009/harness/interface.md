# X-ROBUST-009 interface

## Entry point: `model.py`

```python
def load(checkpoint_dir: str, models_dir: str, device: str):
    """Return whatever predict() needs; roberta-base is under models_dir/roberta-base."""

def predict(state, texts: list[str]):
    """Return (labels, confidence, abstain), each of length len(texts):
    labels      int 0..149, your best in-scope intent (always given, also when you abstain)
    confidence  float in [0, 1], your probability that `labels` is the right answer
    abstain     bool, True if the assistant should not answer"""
```

- Intent indices follow `$ABENCH_DATA_DIR/intents.json`; label −1 in the public data marks out-of-scope queries.
- `train.py --data "$ABENCH_DATA_DIR" --out checkpoint` must reproduce the checkpoint.

## Evaluator code you can read

`stages.py` (batching, timing), `metrics.py` (AURC, ECE, AUROC, decision accuracy), `textpack.py`,
`abench_metrics.py`, `abench_stage.py`, `selfcheck.py`, and the reference submission's code under `baseline/`.

## Available software

Python 3, PyTorch (CUDA), transformers, NumPy, SciPy, scikit-learn. Nothing can be installed.
