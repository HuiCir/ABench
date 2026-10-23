# PRO-MOE-003 interface

## Deliverable

```python
EVAL_SEQ_LEN = 1024          # optional

def build_model() -> "torch.nn.Module":
    """Freshly initialised architecture; the evaluator loads checkpoint/model.pt with load_state_dict and calls
    model(idx: LongTensor (1, T)) -> logits (1, T, 50257) under bf16 autocast (FLOPs are counted without autocast)."""
```

## Supervised runs (`pro_audit.py`)

```
python3 pro_audit.py run --name NAME --code DIR -- torchrun --standalone --nproc_per_node=4 train.py
```

Environment of the run: `ABENCH_DATA_DIR`, `ABENCH_CKPT_DIR` (write `step_<n>/model/model.pt`, resume state,
then `DONE`), `ABENCH_RESUME_FROM` (only when resuming).

## Evaluator code you can read

`pro_audit.py`, `pro_stages.py`, `stages.py` (FLOPs count and held-out loss), `metrics.py`, `token_data.py`,
`abench_metrics.py`, `abench_stage.py`, `selfcheck.py`, and the reference under `baseline/`.

## Available software

Python 3, PyTorch (CUDA, Triton, torch.distributed, torch.compile), NumPy; kernels from the allowlisted mirror.
