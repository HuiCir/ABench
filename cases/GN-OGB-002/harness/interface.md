# GN-OGB-002 interface

## Entry point: `model.py`

```python
def load(checkpoint_dir: str, device: str):
    """Load your trained node classifier from checkpoint/ and return it (any Python object). device is "cuda" at evaluation."""

def predict(model, graph: dict, nodes: numpy.ndarray) -> numpy.ndarray:
    """Return float (len(nodes), 40) class scores for the given node ids (the evaluator passes test_idx).

    graph["x"]           float32 (N, 128)
    graph["edge_index"]  int64 (2, E), directed (source cites target); symmetrise if you want an undirected graph
    graph["year"]        int64 (N,)
    graph["y"]           int64 (N,), 0..39, -1 where the label is not given
    graph["train_idx"], graph["valid_idx"], graph["test_idx"]   int64 node ids
    """
```

- The evaluator calls `load` inside the timed phase and `predict` once after it, with the public graph.
- `model.py` may import modules under `src/`; the submission directory is on `sys.path`.

## Training program: `train.py`

Write a `train.py` that produces `checkpoint/` from the files in `$ABENCH_DATA_DIR` (`graph.npz`), for example
`python3 train.py --data "$ABENCH_DATA_DIR" --out "$ABENCH_SUBMISSION_DIR/checkpoint"`. The evaluator does not run it;
it is kept with the submission so the checkpoint can be audited and reproduced. The reference implementation under
`interface/baseline/` (its `model.py` and `train.py`) is a working starting point you may improve on or replace.

## Evaluator code you can read

`stages.py` (pipeline, dev split, time limits), `metrics.py` and `abench_metrics.py` (accuracy),
`abench_stage.py` (process and timing protocol) and `selfcheck.py`.

## Available software

Python 3, PyTorch (CUDA, Triton), NumPy, SciPy. Nothing can be installed.
