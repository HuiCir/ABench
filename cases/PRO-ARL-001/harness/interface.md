# PRO-ARL-001 interface

## Deliverable

- `checkpoint/`: Hugging Face model directory (config, safetensors weights, tokenizer) with the architecture of
  `Qwen3-4B-Base`, byte-identical to `model/` of a registered checkpoint. The evaluator loads it with vLLM
  (bfloat16) and uses its tokenizer's chat template with thinking enabled.
- The code of the registered run; it must resume from `$ABENCH_RESUME_FROM` when that is set and print
  `ABENCH step=<n> reward=<float>` lines.

## Supervised runs (`pro_audit.py`)

```
python3 pro_audit.py run --name NAME --code DIR -- torchrun --standalone --nproc_per_node=3 train.py
```

Environment of the run: `ABENCH_DATA_DIR`, `ABENCH_MODELS_DIR`, `ABENCH_CKPT_DIR` (write `step_<n>/model/`,
resume state, then `DONE`), `ABENCH_RESUME_FROM` (only when resuming).

## Evaluator code you can read

`math_tool.py` (prompt, tool loop, sampling, answer check), `pro_audit.py`, `pro_stages.py`, `stages.py`,
`metrics.py`, `abench_metrics.py`, `abench_stage.py`, `selfcheck.py`, and the reference under `baseline/`.

## Available software

Python 3, PyTorch (CUDA), vLLM, transformers, safetensors, NumPy, pandas; RL frameworks from the allowlisted mirror.
