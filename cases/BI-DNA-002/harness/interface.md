# BI-DNA-002 interface

## Entry point: `model.py`

```python
def load(checkpoint_dir: str, device: str):
    """Load your trained classifiers for the three datasets from checkpoint/ (any Python object). device is "cuda" at evaluation."""

def predict(model, name: str, seqs: list[str]) -> numpy.ndarray:
    """name in ("human_nontata_promoters", "human_enhancers_cohn", "human_ocr_ensembl"); seqs over {A, C, G, T, N}.
    Return float (len(seqs), 2) class scores."""
```

- `model.py` may import modules under `src/`; the submission directory is on `sys.path`.

## Training program: `train.py`

Write a `train.py` that produces `checkpoint/` from the files in `$ABENCH_DATA_DIR` (one `<dataset>.npz` per dataset with `seq`, `y` and `dev_idx`), for example
`python3 train.py --data "$ABENCH_DATA_DIR" --out "$ABENCH_SUBMISSION_DIR/checkpoint"`. The evaluator does not run it;
it is kept with the submission so the checkpoint can be audited and reproduced. The reference implementation under
`interface/baseline/` (its `model.py` and `train.py`) is a working starting point you may improve on or replace.

## Evaluator code you can read

`stages.py` (pipeline, dev split, time limits), `metrics.py` and `abench_metrics.py` (accuracy, AUC),
`abench_stage.py` (process and timing protocol) and `selfcheck.py`.

## Available software

Python 3, PyTorch (CUDA, Triton), NumPy, SciPy. Nothing can be installed.
