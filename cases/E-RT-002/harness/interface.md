# E-RT-002 interface

## Entry point: `policy.py`

```python
def load_planner(checkpoint_dir: str, device: str):
    """Runs in its own process on the GPU. Methods:
       reset(instruction: str) -> None
       plan(obs: dict, t: int, request: dict) -> array (H, 7)   # chunk[k] is meant for step t + k"""

def make_controller(checkpoint_dir: str):
    """Runs in its own process on the CPU. Methods:
       reset(instruction: str) -> None
       request(t: int) -> None | dict     # ask for a plan from the current observation (only when none is in flight)
       receive(chunk, t_issue: int, t_now: int) -> None
       act(t: int) -> 7 numbers           # executed at step t, clipped to [-1, 1]"""
```

- `obs`: `agentview_rgb`, `eye_in_hand_rgb` (uint8, 128 x 128 x 3), `state` (8 numbers: end-effector position,
  axis-angle orientation, gripper fingers) and `instruction`; see `libero_eval.py`.
- A chunk requested at step t arrives at step t + injected latency + ceil(planning ms / 50). The first chunk of an
  episode is planned from the initial observation and delivered before step 0.
- `train.py --data "$ABENCH_DATA_DIR" --out checkpoint` must reproduce the checkpoint.

## Evaluator code you can read

`libero_eval.py` (environment, observation, episode loops, process isolation), `stages.py`, `metrics.py`,
`abench_metrics.py`, `abench_stage.py`, `selfcheck.py`, and the reference under `baseline/`.

## Available software

Python 3, PyTorch (CUDA), NumPy, h5py, safetensors, LIBERO (robosuite, MuJoCo). Nothing can be installed.
