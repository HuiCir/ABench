# V-BB-006 harness

Evaluator code of the case; the self-check (`selfcheck.py`) runs the same stages on the dev split.

## Pipeline (`stages.py`)

```
V-BB-006 pipeline: the evaluator trains your vision-transformer backbone at two compute budgets on Imagenette with a fixed
recipe and measures test accuracy (evaluator and self-check use this same file).

For each variant v in VARIANTS ("small", "large"), one stage "train_<v>" (imports model.py):
  setup   load the training images (untimed)
  timed   build(v, num_classes=10) with the seed; train with RECIPE (img_train.py:
          AdamW, warm-up and cosine decay, label smoothing, clipping, random resized crops and flips at 160 pixels)
  finish  forward FLOPs per image at 160 pixels (FlopCounterMode) against the variant's budget; the test images arrive
          (held_out.npz) and their logits are returned
The evaluator runs the ViT baseline (baselines/vit: ViT-Ti/16 and ViT-S/16) through the same stages and seeds.
```

## Files

- `abench_metrics.py`
- `abench_stage.py`
- `baselines/vit/model.py`
- `img_train.py`
- `interface.md`
- `metrics.py`
- `sandbox_guide.md`
- `selfcheck.py`
- `stages.py`
- `submission_rules.json`

The held-out data, keys and judge configuration are never in this directory.
