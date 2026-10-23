# E-VLA-003 interface

## Entry point: `policy.py`

```python
def load_policy(checkpoint_dir: str, models_dir: str, device: str, vlm, processor):
    """Runs in its own process. vlm: the frozen SmolVLM2-500M-Video-Instruct (bfloat16, eval mode, no gradients);
    processor: its AutoProcessor. Returns an object with
       reset(instruction: str) -> None
       act(obs: dict) -> 7 numbers        # called every control step (20 Hz), clipped to [-1, 1]"""
```

- `obs`: `agentview_rgb`, `eye_in_hand_rgb` (uint8, 128 x 128 x 3), `state` (8 numbers: end-effector position,
  axis-angle orientation, gripper fingers) and `instruction`; see `libero_eval.py`.
- The time of every `act` call counts toward the 50 ms mean per step; an action chunk may be computed in one call
  and returned over the following ones.
- `train.py --data "$ABENCH_DATA_DIR" --models "$ABENCH_MODELS_DIR" --out checkpoint` must reproduce the checkpoint.

## Evaluator code you can read

`libero_eval.py` (environment, observation, episode loop, process isolation), `vla_frozen.py` (how the backbone is
loaded and checked), `stages.py`, `metrics.py`, `abench_metrics.py`, `abench_stage.py`, `selfcheck.py`, and the
reference under `baseline/`.

## Available software

Python 3, PyTorch (CUDA), transformers, safetensors, NumPy, h5py, Pillow, lerobot, LIBERO (robosuite, MuJoCo).
Nothing can be installed.
