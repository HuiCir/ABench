# U-LOOP-002 harness

Evaluator code of the case; the self-check (`selfcheck.py`) runs the same stages on the dev split.

## Pipeline (`stages.py`)

```
U-LOOP-002 pipeline: accuracy of a looped model at three test-time compute budgets on three reasoning tasks
(evaluator and self-check use this same file).

The evaluator does not train. One stage, "predict":
  timed   import model.py, load(checkpoint_dir, device) -> model                     (load time, killed at 120 s)
  finish  for each task and each budget B in PARAMS["budgets"] (in reference-layer units, see below), the evaluator
          drives the loop itself:
            state = init(model, task, x, budget_flops)          # embed the problems
            repeat at most PARAMS["max_iters"] times:
              state = step(model, state, active)                 # one recurrent iteration (active: bool mask)
              logits, halt = readout(model, state)               # answers so far, and which problems stop now
              a problem's answer is argmax(logits) at the iteration where it halts (or at the last iteration)
          Every call runs under torch.utils.flop_counter.FlopCounterMode; the floating-point operations of all calls
          divided by the number of problems is the test-time compute per problem.             (all ≤ 600 s)
Budget unit: F_ref(task), the FLOPs of one application of the reference layer (refblock.py: a pre-norm transformer
layer of width 256 with 4 heads and an MLP of 1,024) to one problem of the task's sequence length.
```

## Files

- `abench_metrics.py`
- `abench_stage.py`
- `baseline/model.py`
- `baseline/src/looped.py`
- `baseline/train.py`
- `interface.md`
- `metrics.py`
- `reason_tasks.py`
- `refblock.py`
- `sandbox_guide.md`
- `selfcheck.py`
- `stages.py`
- `submission_rules.json`

The held-out data, keys and judge configuration are never in this directory.
