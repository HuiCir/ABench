# B-OPT-002 harness

Evaluator code of the case; the self-check (`selfcheck.py`) runs the same stages on the dev split.

## Pipeline (`stages.py`)

```
B-OPT-002 pipeline: a program in the primitive language (primitive_dsl.py) is checked for novelty and stability on
a fixed gradient stream, then used to train the fixed language model of B-OPT-001, whose held-out loss is measured
(evaluator and self-check use this same file).

Stage "screen"  (reads program.yaml; CPU work)
  timed   parse and validate program.yaml (no Python from the submission is imported or run)
  finish  behaviour.screen on a gradient stream drawn from the held-out screen seed: similarity to known optimizers,
          descent direction, finiteness at gradient scales 1e-6 and 1e6
Stage "train"  (reads program.yaml)
  setup   map the training token shards, build the model on the GPU with the seed (untimed)
  timed   compile the program into a FunctionalRule and train STEPS steps with its learning rate (B-OPT-001 recipe)
  finish  save model.pt; report the loss curve, divergence and the size of the optimizer state
Stage "eval"   (does not read your program; receives only model.pt)
  timed   load model.pt
  finish  mean next-token loss on the held-out tokens
```

## Files

- `abench_metrics.py`
- `abench_stage.py`
- `baseline/program.yaml`
- `baseline/search.py`
- `baselines/adamw/program.yaml`
- `behaviour.py`
- `interface.md`
- `lm_model.py`
- `metrics.py`
- `optim_api.py`
- `primitive_dsl.py`
- `reference_rules.py`
- `sandbox_guide.md`
- `selfcheck.py`
- `stages.py`
- `submission_rules.json`
- `token_data.py`

The held-out data, keys and judge configuration are never in this directory.
