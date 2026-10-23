# T-FM-001 interface

## Entry point: `policy.py`

```python
def load(checkpoint_dir: str, device: str):
    """Load your trained policy model from checkpoint/ and return it (any Python object). device is "cuda" at evaluation;
    seed any sampling in the policy from ABENCH_SEED (set by the evaluator, different for each run)."""

def make_policy(model):
    """Return an object with reset() and act(obs) -> numpy (2,) target position in [0, 512].
    obs = {"agent_pos": float32 (2,), "environment_state": float32 (16,)}"""
```

- `make_policy` is called once per episode, `reset()` at the start of each episode, `act()` once per environment step (10 Hz control); every `act()` call is timed with CUDA synchronised.
- The episode ends when coverage exceeds 95% or after 300 steps.

## Training program: `train.py`

Write a `train.py` that produces `checkpoint/` from the files in `$ABENCH_DATA_DIR` (`demos.npz`: `agent_pos` float32 (N, 2), `environment_state` float32 (N, 16), `action` float32 (N, 2), `episode` and `frame` int64 (N,)), for example
`python3 train.py --data "$ABENCH_DATA_DIR" --out "$ABENCH_SUBMISSION_DIR/checkpoint"`. The evaluator does not run it;
it is kept with the submission so the checkpoint can be audited and reproduced. The reference implementation under
`interface/baseline/` (its `policy.py` and `train.py`) is a working starting point you may improve on or replace.

## Evaluator code you can read

`stages.py` (loading, the rollout loop and the act() timing), `metrics.py` and `abench_metrics.py`, `abench_stage.py` (process
and timing protocol) and `selfcheck.py`.

## Available software

Python 3, PyTorch (CUDA, Triton), NumPy, gymnasium, gym-pusht. Nothing can be installed.
