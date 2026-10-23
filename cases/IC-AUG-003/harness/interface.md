# IC-AUG-003 interface

## Entry point: `policy.py`

```python
def augment(x, ops, gen, epoch: int, n_epochs: int):
    """x: float tensor (N, 3, 32, 32) in [0, 1] on the GPU (the whole training set); ops: the aug_ops module;
    gen: torch.Generator for every random draw. Return an augmented tensor of the same shape (values are clamped
    to [0, 1] by the recipe)."""
```

- The recipe calls `augment` once per epoch on all training images, then normalises them with the training-set channel statistics and trains one epoch on them.
- `policy.py` may import only `math` and `typing`; the evaluator checks this (and the absence of torch, NumPy, file and reflection names) before importing it.

## Primitives: `aug_ops.py`

Flip, translation, rotation, shear, scaling, cutout, brightness, contrast, saturation, greyscale, solariser, posteriser, inversion and sharpness, plus `bernoulli`, `uniform`, `choice` and `where` to apply operations to random subsets with random magnitudes. Every function works on a whole batch and draws from `gen`; see the module docstring.

## Evaluator code you can read

- `recipe.py`: the fixed network and training recipe; `stages.py`: the stage, the policy check and the dev split.
- `baselines/flipcrop/policy.py`: the baseline trained alongside your policy.
- `metrics.py`, `abench_metrics.py`, `abench_stage.py`, `selfcheck.py`.

## Available software

Python 3, PyTorch (CUDA, Triton), NumPy (for your search scripts; not inside `policy.py`). Nothing can be installed.
