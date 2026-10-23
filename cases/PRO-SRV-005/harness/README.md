# PRO-SRV-005 harness

Evaluator code of the case; the self-check (`selfcheck.py`) runs the same stages on the dev split.

## Pipeline (`stages.py`)

```
PRO-SRV-005 pipeline: your serving system for Qwen3-8B is started on the 4-GPU node and loaded with replayed
production traffic at increasing rates; a sample of its answers is then checked against greedy decoding
(evaluator and self-check use this file; the evaluator runs the whole pipeline twice).

Stage "serve" (starts serve.py; the load generator is evaluator code, serve_bench.py):
  timed   python3 serve.py --model <models>/Qwen3-8B --port PORT, until GET /health answers 200 (≤ 1,800 s)
  finish  for each held-out trace (conversation, code): the trace's requests replayed with their inter-arrival times
          divided by each rate factor of LADDER in turn (stopping after the first factor at which fewer than 90%
          of requests meet the SLO); per request TTFT, TPOT, success; a fixed sample of the answers is saved for
          the next stage; the server is stopped                                                   (≤ 7,200 s)
Stage "check" (evaluator code only, after the server has stopped):
  serve_bench.greedy_check of the saved sample with the evaluator's own forward pass of Qwen3-8B
```

## Files

- `abench_metrics.py`
- `abench_stage.py`
- `baseline/serve.py`
- `interface.md`
- `metrics.py`
- `sandbox_guide.md`
- `selfcheck.py`
- `serve_bench.py`
- `stages.py`
- `submission_rules.json`

The held-out data, keys and judge configuration are never in this directory.
