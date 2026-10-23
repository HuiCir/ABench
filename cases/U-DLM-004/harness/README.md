# U-DLM-004 harness

Evaluator code of the case; the self-check (`selfcheck.py`) runs the same stages on the dev split.

## Pipeline (`stages.py`)

```
U-DLM-004 pipeline: the evaluator trains the fixed bidirectional denoiser (dlm.build_denoiser) with your noise
schedule and training objective at two scales and computes the masked-diffusion likelihood bound under your schedule
on held-out text (evaluator and self-check use this same file).

For each scale in SCALES, one stage "run_<scale>" (imports recipe.py):
  setup   map the training shards (untimed)
  timed   check the schedule (dlm.check_schedule); build the denoiser with the seed; train STEPS steps of 32 x 1,024
          tokens with the fixed recipe of lm_train.py (AdamW lr 1e-3, warm-up and cosine decay, bf16, clipping 1.0),
          the loss being your recipe.loss(model, x)
  finish  NELBO per token on the held-out windows under your schedule recipe.alpha (dlm.nelbo_per_token, 8 stratified
          times per window), its perplexity bound; divergence
The evaluator runs MDLM (linear schedule, baselines/mdlm) through the same stages on the same seeds; the gap to an
autoregressive model of the same size and data (AR_REFERENCE, measured by the operator at activation with
lm_model.LM and the same recipe) is reported.
```

## Files

- `abench_metrics.py`
- `abench_stage.py`
- `baselines/mdlm/recipe.py`
- `dlm.py`
- `interface.md`
- `lm_model.py`
- `lm_train.py`
- `metrics.py`
- `sandbox_guide.md`
- `selfcheck.py`
- `stages.py`
- `submission_rules.json`
- `token_data.py`

The held-out data, keys and judge configuration are never in this directory.
