# RS-SEQ-001 interface

## Entry point: `model.py`

```python
def load(checkpoint_dir: str, device: str):
    """Load your trained recommender from checkpoint/ and return it (any Python object). device is "cuda" at evaluation."""

def recommend(model, histories: dict, users: numpy.ndarray, k: int = 10) -> numpy.ndarray:
    """Return int (len(users), k): distinct item ids in 1..n_items, best first, for each user's next item.

    histories["items"]    int32 (L,), item ids 1..n_items, concatenated user by user, oldest first
    histories["times"]    int64 (L,), Unix timestamps
    histories["offsets"]  int64 (U + 1,), user u's history is items[offsets[u]:offsets[u + 1]]
    histories["n_items"]  int64 scalar
    """
```

- The evaluator calls `load` inside the timed phase and `recommend` once after it, with `users = arange(U)` and the public histories.
- `model.py` may import modules under `src/`; the submission directory is on `sys.path`.

## Training program: `train.py`

Write a `train.py` that produces `checkpoint/` from the files in `$ABENCH_DATA_DIR` (`train.npz`), for example
`python3 train.py --data "$ABENCH_DATA_DIR" --out "$ABENCH_SUBMISSION_DIR/checkpoint"`. The evaluator does not run it;
it is kept with the submission so the checkpoint can be audited and reproduced. The reference implementation under
`interface/baseline/` (its `model.py` and `train.py`) is a working starting point you may improve on or replace.

## Evaluator code you can read

`stages.py` (pipeline, dev split, time limits), `metrics.py` and `abench_metrics.py` (NDCG@10 and Recall@10),
`abench_stage.py` (process and timing protocol) and `selfcheck.py`.

## Available software

Python 3, PyTorch (CUDA, Triton), NumPy. Nothing can be installed.
