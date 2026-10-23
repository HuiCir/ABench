# U-DLM-005 harness

Evaluator code of the case; the self-check (`selfcheck.py`) runs the same stages on the dev split.

## Pipeline (`stages.py`)

```
U-DLM-005 pipeline: a submitted sampler drives the frozen MDLM through the counting denoiser at three budgets,
unconditionally and continuing held-out prefixes; the samples are then scored by an autoregressive model the
submission never sees (evaluator and self-check use this same file).

Stage "sample" (imports sampler.py; the MDLM is under models/mdlm-owt):
  timed   import sampler.py, load the MDLM (untimed for the sampler, but inside this process)
  finish  for NFE in NFES: N_UNCOND unconditional samples of 1,024 tokens and one 256-token continuation of each
          held-out 256-token prefix, in batches of BATCH, each batch with its own Denoiser(budget = NFE) and seed;
          wall-clock seconds per budget; samples saved to samples.npz
Stage "score" (does not import your code; GPT-2 Large under judge_models/, or GPT-2 under models/ in the self-check)
  finish  generative NLL per sample (continuations conditioned on their prefix) and unigram entropy per sample;
          the same for the held-out reference texts (1,024-token windows whose first 256 tokens are the prefixes)
```

## Files

- `abench_metrics.py`
- `abench_stage.py`
- `baseline/sampler.py`
- `baselines/ancestral/sampler.py`
- `dlm.py`
- `interface.md`
- `metrics.py`
- `sandbox_guide.md`
- `selfcheck.py`
- `stages.py`
- `submission_rules.json`

The held-out data, keys and judge configuration are never in this directory.
