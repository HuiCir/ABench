# MD-SEG-001 harness

Evaluator code of the case; the self-check (`selfcheck.py`) runs the same stages on the dev split.

## Pipeline (`stages.py`)

```
MD-SEG-001 pipeline: the evaluator trains your 3D segmentation network with the fixed recipe of seg3d.py on four
Medical Segmentation Decathlon tasks, one stage per task, and segments their held-out volumes (evaluator and
self-check use this same file).

Stages "hippocampus", "prostate", "heart" and "unseen" (imports network.py):
  setup   load the task's preprocessed training volumes: the public files for the first three; for "unseen" they and
          its configuration arrive in ctx.pre_dir (held_pre.npz) (untimed)
  timed   net = make_network(in_channels, num_classes, patch_size, spacing); seg3d.train with the task's iterations
  finish  sliding-window segmentation of the task's held-out volumes (given after the timed phase); parameters,
          peak GPU memory, divergence                                                         (≤ 900 s per task)
The evaluator trains the nnU-Net-style U-Net of baselines/unet through the same stages on the same seed.
```

## Files

- `abench_metrics.py`
- `abench_stage.py`
- `baseline/network.py`
- `baselines/unet/network.py`
- `interface.md`
- `metrics.py`
- `sandbox_guide.md`
- `seg3d.py`
- `selfcheck.py`
- `stages.py`
- `submission_rules.json`

The held-out data, keys and judge configuration are never in this directory.
