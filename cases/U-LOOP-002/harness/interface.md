# U-LOOP-002 interface

## Entry point: `model.py`

```python
def load(checkpoint_dir: str, device: str):
    """Return the trained model (any Python object). Not counted: measure your own costs here if you need them."""

def init(model, task: str, x: numpy.ndarray, budget_flops: float):
    """x: int64 (n, seq_len) problems of one task; budget_flops: the average budget per problem in FLOPs.
    Return a state (any Python object)."""

def step(model, state, active: numpy.ndarray):
    """One recurrent iteration for the problems where active is True. Return the new state."""

def readout(model, state):
    """Return (logits (n, n_classes), halt (n,) bool): current answers for all problems, and which ones stop now.
    Problems that have stopped stay stopped; their answer is the argmax of the logits at the iteration they stopped."""

def n_params(model) -> float:     # optional; otherwise model.parameters() is counted
```

- Loop driven by the evaluator (`stages.run_loop`): `init` once, then up to 64 times `step` then `readout`.
- Compute = FLOPs counted by `torch.utils.flop_counter.FlopCounterMode` over all calls, divided by n; budget b means
  at most b × F_ref(task) per problem (2% tolerance); F_ref is measured on `refblock.RefLayer`.
- `reason_tasks.py` defines the tasks exactly; it is on `sys.path` during the self-check and at evaluation.

## Training program: `train.py`

Write a `train.py` that produces `checkpoint/` from problems drawn with `reason_tasks.Sampler(seed).sample(task, d, count)`.
The evaluator does not run it; it is kept for audit. The reference implementation under `interface/baseline/`
(`model.py`, `train.py`, `src/looped.py`) is a working starting point.

## Evaluator code you can read

`reason_tasks.py`, `refblock.py`, `stages.py` (the loop, compute counting, budgets), `metrics.py`,
`abench_metrics.py`, `abench_stage.py` and `selfcheck.py`.

## Available software

Python 3, PyTorch (CUDA, Triton), NumPy, SciPy. Nothing can be installed.
