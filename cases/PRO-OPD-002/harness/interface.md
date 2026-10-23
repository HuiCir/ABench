# PRO-OPD-002 interface

## Deliverable

- `checkpoint/`: Hugging Face model directory with the architecture of `Qwen3-1.7B-Base`, byte-identical to
  `model/` of a registered checkpoint, with a tokenizer and chat template (the evaluator applies it with
  thinking enabled when the template supports it). Loaded with vLLM (bfloat16).
- The code of the registered run; it must resume from `$ABENCH_RESUME_FROM` when that is set and print
  `ABENCH step=<n> loss=<float>` lines.

## Supervised runs (`pro_audit.py`)

```
python3 pro_audit.py run --name NAME --code DIR -- torchrun --standalone --nproc_per_node=2 train.py
```

Environment of the run: `ABENCH_DATA_DIR`, `ABENCH_MODELS_DIR` (student and teacher), `ABENCH_CKPT_DIR`,
`ABENCH_RESUME_FROM` (only when resuming).

## Evaluator code you can read

`math_tool.py` (prompt, sampling, answer check), `pro_audit.py`, `pro_stages.py`, `stages.py`, `metrics.py`,
`abench_metrics.py`, `abench_stage.py`, `selfcheck.py`, and the reference under `baseline/`.

## Available software

Python 3, PyTorch (CUDA), vLLM, transformers, safetensors, NumPy, pandas; training frameworks from the allowlisted
mirror.
