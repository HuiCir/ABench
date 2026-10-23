# PRO-SRV-005 interface

## Entry point: `serve.py`

```
python3 serve.py --model MODEL_DIR --port PORT
GET  /health    -> 200 when ready
POST /generate  {"input_ids": [...], "max_new_tokens": n, "ignore_eos": true}
                -> stream of {"ids": [...]}\n lines, then {"done": true}\n   (greedy decoding)
```

## Evaluator code you can read

`serve_bench.py` (protocol client, replay, latency statistics, greedy check), `stages.py` (rate ladder),
`metrics.py`, `abench_metrics.py`, `abench_stage.py`, `selfcheck.py`, and the reference under `baseline/`.

## Available software

Python 3, PyTorch (CUDA, Triton), transformers, NumPy; serving engines and kernels from the allowlisted mirror.
