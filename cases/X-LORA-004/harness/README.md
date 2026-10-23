# X-LORA-004 harness

Evaluator code of the case; the self-check (`selfcheck.py`) runs the same stages on the dev split.

## Pipeline (`stages.py`)

```
X-LORA-004 pipeline: score parameter-efficient fine-tuning states for frozen roberta-base on four GLUE tasks
(SST-2, CoLA, MNLI-matched, QNLI). Texts are stored packed (UTF-8 bytes + offsets, textpack.py).

The evaluator does not train. One stage, "eval" (imports peft.py; the provided roberta-base is available):
  timed   load checkpoint/states.pt ({task: {name: tensor}}, task-specific parameters only), import peft.py
  finish  count the elements per task; predict(models_dir, task, state, sentence1, sentence2, device) -> labels,
          for each task's held-out split                                                           (≤ 600 s)
```

## Files

- `abench_metrics.py`
- `abench_stage.py`
- `baseline/peft.py`
- `baseline/train.py`
- `interface.md`
- `metrics.py`
- `sandbox_guide.md`
- `selfcheck.py`
- `stages.py`
- `submission_rules.json`
- `textpack.py`

The held-out data, keys and judge configuration are never in this directory.
