# PRO-SPEC-006 interface

## Entry point: `serve.py`

```
ABENCH_DRAFT_DIR=<submission>/checkpoint python3 serve.py --model MODEL_DIR --port PORT
GET  /health    -> 200 when ready
POST /generate  {"input_ids": [...], "max_new_tokens": 512, "ignore_eos": true}
                -> stream of {"ids": [...]}\n lines, then {"done": true}\n   (greedy decoding)
```

## Supervised runs (`pro_audit.py`)

```
python3 pro_audit.py run --name NAME --code DIR -- python3 train.py
```

Environment of the run: `ABENCH_DATA_DIR`, `ABENCH_MODELS_DIR`, `ABENCH_CKPT_DIR`, `ABENCH_RESUME_FROM`.

## Evaluator code you can read

`serve_bench.py`, `baselines/vanilla/serve.py` (the plain-decoding server), `pro_audit.py`, `pro_stages.py`,
`stages.py`, `metrics.py`, `abench_metrics.py`, `abench_stage.py`, `selfcheck.py`, and the reference under
`baseline/`.

## Available software

Python 3, PyTorch (CUDA, Triton), transformers, safetensors, NumPy, pandas; serving engines from the allowlisted
mirror.
