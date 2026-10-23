# IC-AUG-003 harness

Evaluator code of the case; the self-check (`selfcheck.py`) runs the same stages on the dev split.

## Pipeline (`stages.py`)

```
IC-AUG-003 pipeline: the evaluator trains the fixed CIFAR-10 recipe (recipe.py) with your augmentation policy and
measures test accuracy (evaluator and self-check use this same file).

One stage, "train" (imports policy.py after checking that it uses only the primitives of aug_ops.py):
  setup   load the training images to the GPU (untimed)
  timed   check and import policy.py; train the fixed recipe for 40 epochs, augmenting the whole training set with
          policy.augment at the start of every epoch
  finish  class probabilities for the CIFAR-10 test images and the CIFAR-10.1 images (with mirror averaging)
The evaluator runs the flip-and-translate baseline (baselines/flipcrop) through the same stage on the same seeds.
```

## Files

- `abench_metrics.py`
- `abench_stage.py`
- `aug_ops.py`
- `baselines/flipcrop/policy.py`
- `interface.md`
- `metrics.py`
- `recipe.py`
- `sandbox_guide.md`
- `selfcheck.py`
- `stages.py`
- `submission_rules.json`

The held-out data, keys and judge configuration are never in this directory.
