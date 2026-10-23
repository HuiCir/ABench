# IC-RES-002 harness

Evaluator code of the case; the self-check (`selfcheck.py`) runs the same stages on the dev split.

## Pipeline (`stages.py`)

```
IC-RES-002 pipeline: the evaluator trains your image backbone at two compute budgets on Imagenette with a fixed
recipe and measures test accuracy (evaluator and self-check use this same file).

For each variant v in VARIANTS ("small", "large"), one stage "train_<v>" (imports model.py):
  setup   load the training images (untimed)
  timed   build(v, num_classes=10) with the seed; train with RECIPE (img_train.py: SGD with Nesterov momentum,
          warm-up and cosine decay, label smoothing, random resized crops and flips at 128 pixels)
  finish  forward FLOPs per image at 128 pixels (FlopCounterMode) against the variant's budget; the test images arrive
          (held_out.npz) and their logits are returned
The evaluator runs the ResNet baseline (baselines/resnet: ResNet-18 and ResNet-50) through the same stages and seeds.
```

## Files

- `abench_metrics.py`
- `abench_stage.py`
- `baselines/resnet/model.py`
- `img_train.py`
- `interface.md`
- `metrics.py`
- `sandbox_guide.md`
- `selfcheck.py`
- `stages.py`
- `submission_rules.json`

The held-out data, keys and judge configuration are never in this directory.
