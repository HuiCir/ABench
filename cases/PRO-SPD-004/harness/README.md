# PRO-SPD-004 harness

Evaluator code of the case; the self-check (`selfcheck.py`) runs the same stages on the dev split.

## Pipeline (`stages.py`)

```
PRO-SPD-004 pipeline: time-to-target pretraining on one 4 × H100 node (evaluator and self-check use this file).

Stage "train" (runs your train.py under torchrun on all GPUs)
  timed   torchrun --standalone --nproc_per_node=<GPUs> train.py --data <public> --out <dir> --seed <seed>; the clock
          runs from the launch until every training process has exited, so compilation and data loading count
  finish  check out/model.pt and count its parameters
Stage "eval" (imports your model.py; receives only the out/ directory)
  timed   build the model and load out/model.pt on one GPU
  finish  mean next-token loss on the first 10,485,760 tokens of the FineWeb validation shard
```

## Files

- `abench_metrics.py`
- `abench_stage.py`
- `baseline/model.py`
- `baseline/train.py`
- `interface.md`
- `metrics.py`
- `sandbox_guide.md`
- `selfcheck.py`
- `stages.py`
- `submission_rules.json`
- `token_data.py`

The held-out data, keys and judge configuration are never in this directory.
