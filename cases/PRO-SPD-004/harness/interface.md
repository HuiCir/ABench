# PRO-SPD-004 interface

## `train.py`

```
torchrun --standalone --nproc_per_node=4 train.py --data DATA_DIR --out OUT_DIR --seed SEED [--quick]
```

- `DATA_DIR` contains `fineweb_train_000001.bin` … `fineweb_train_000103.bin` (format in `token_data.py`).
- Before exiting, rank 0 must write `OUT_DIR/model.pt`: a state dict (tensors only) loadable by `model.build_model()`.
- `--quick` is passed only by `selfcheck.py --quick`; use it for a short smoke run.
- The clock covers the whole torchrun invocation; there is no separate warm-up allowance.

## `model.py`

```python
EVAL_SEQ_LEN = 1024          # optional: window length for evaluation (consecutive, non-overlapping windows)

def build_model() -> torch.nn.Module:
    """Return the architecture with freshly initialised weights; the evaluator then loads OUT_DIR/model.pt
    with load_state_dict and calls model(idx: LongTensor (1, T)) -> logits (1, T, vocab) under bf16 autocast."""
```

## Evaluator code you can read

`stages.py` (the torchrun invocation, the timing and the held-out loss computation), `token_data.py`, `metrics.py`,
`abench_metrics.py`, `abench_stage.py` and `selfcheck.py`.

## Available software

Python 3, PyTorch (CUDA, Triton, torch.distributed, torch.compile), NumPy. Nothing can be installed.
