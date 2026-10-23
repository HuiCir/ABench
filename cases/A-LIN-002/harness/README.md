# A-LIN-002 harness

Evaluator code of the case; the self-check (`selfcheck.py`) runs the same stages on the dev split.

## Pipeline (`stages.py`)

```
A-LIN-002 pipeline: train the fixed two-layer model (lin_model.py) with your Mixer on MQAR using the fixed recipe
below, then test recall on longer sequences and more key-value pairs, and measure training throughput.

The recipe is fixed so that differences in recall come from the mixer, not from training compute. One stage, "fit"
(imports mixer.py):
  timed   build the model with the seed, train PARAMS["steps"] steps on MQAR batches from mqar.TRAIN_GRID
  finish  (1) predictions at the query positions of the held-out MQAR sequences (scored by the evaluator)
          (2) causality check: changing tokens after position t must not change outputs up to t
          (3) forward+backward time per token at T = 1,024 and T = 4,096 (B·T fixed), giving throughput and the
              empirical scaling exponent of time in T
The evaluator runs the GLA baseline (baselines/gla) through the same stage on the same seeds.
```

## Files

- `abench_metrics.py`
- `abench_stage.py`
- `baselines/gla/mixer.py`
- `interface.md`
- `lin_model.py`
- `metrics.py`
- `mqar.py`
- `sandbox_guide.md`
- `selfcheck.py`
- `stages.py`
- `submission_rules.json`

The held-out data, keys and judge configuration are never in this directory.
