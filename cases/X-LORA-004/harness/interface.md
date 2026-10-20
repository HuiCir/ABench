# X-LORA-004 interface

## Entry point: `peft.py`

```python
def predict(models_dir: str, task: str, state: dict, sentence1: list[str], sentence2: list[str] | None,
            device: str) -> numpy.ndarray:
    """Rebuild the model from the frozen base and `state` (this task's entry of checkpoint/states.pt, on CPU)
    and return int labels (len(sentence1),).

    task in ("rte", "mrpc", "cola", "sst2"); models_dir contains roberta-base/ (frozen)
    """
```

- The element count of each task's `state` is the task-specific parameter count checked against the cap.
- `peft.py` may import modules under `src/`; the submission directory is on `sys.path`.

## Training program: `train.py`

Write a `train.py` that produces `checkpoint/` from the files in `$ABENCH_DATA_DIR` (`glue_train.npz`: `<task>_<column>`, `<task>_label` and `<task>_dev_idx`) and `$ABENCH_MODELS_DIR/roberta-base`; it must save `{task: {name: torch.Tensor}}` to `checkpoint/states.pt`, for example
`python3 train.py --data "$ABENCH_DATA_DIR" --out "$ABENCH_SUBMISSION_DIR/checkpoint"`. The evaluator does not run it;
it is kept with the submission so the checkpoint can be audited and reproduced. The reference implementation under
`interface/baseline/` (its `peft.py` and `train.py`) is a working starting point you may improve on or replace.

## Evaluator code you can read

`stages.py` (the evaluation stage, the dev split, time limits), `metrics.py` and `abench_metrics.py` (GLUE metrics),
`abench_stage.py` (process and timing protocol) and `selfcheck.py`.

## Available software

Python 3, PyTorch (CUDA, Triton), NumPy, transformers, tokenizers, safetensors (offline). Nothing can be installed.
