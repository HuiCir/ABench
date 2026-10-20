# U-LOOP-003 interface

## Entry point: `model.py`

```python
def load(checkpoint_dir: str, device: str):
    """Load your trained model from checkpoint/ and return it (any Python object). device is "cuda" at evaluation."""

def predict(model, task: str, inputs: list[numpy.ndarray]) -> list[numpy.ndarray]:
    """task in ("parity", "addition", "copy"). Return one int array per input: the complete output sequence.
    An output of the wrong length is wrong."""
```

- `algo_tasks.py` defines the tasks exactly; you may import it (it is on `sys.path` during the self-check and at evaluation) or copy what you need.
- `model.py` may import modules under `src/`; the submission directory is on `sys.path`.

## Training program: `train.py`

Write a `train.py` that produces `checkpoint/` from examples drawn with `algo_tasks.Sampler(seed).sample(task, n, count)`
(1 <= n <= 20; import it from `$ABENCH_VIEW_DIR/interface`), for example
`python3 train.py --out "$ABENCH_SUBMISSION_DIR/checkpoint"`. This case has no data files. The evaluator does not run it;
it is kept with the submission so the checkpoint can be audited and reproduced. The reference implementation under
`interface/baseline/` (its `model.py` and `train.py`) is a working starting point you may improve on or replace.

## Evaluator code you can read

`algo_tasks.py` (tasks and sampler), `stages.py` (pipeline, self-check problems, time limits), `metrics.py` and
`abench_metrics.py` (exact match), `abench_stage.py` (process and timing protocol) and `selfcheck.py`.

## Available software

Python 3, PyTorch (CUDA, Triton), NumPy. Nothing can be installed.
