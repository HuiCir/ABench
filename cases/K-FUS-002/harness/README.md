# K-FUS-002 harness

Evaluator code of the case; the self-check (`selfcheck.py`) runs the same stages on the dev split.

## Pipeline (`stages.py`)

```
K-FUS-002 pipeline: kernels for 20 KernelBench Level-2 problems, correct on five input distributions, timed against
PyTorch eager and torch.compile (evaluator and self-check; K-KB-001 uses the same file with its own problem list).

Two stages, each in a new process on the same GPU:
  reference  nothing timed; finish: for every problem, Model in PyTorch eager and torch.compile(Model) (default mode),
             timed on the problem's own inputs (kb_bench.time_call), and eager peak memory. Your code is not imported.
  bench      setup: a snapshot of torch's entry points; timed: import every kernels/<problem>.py that exists (module-level
             compilation, load_inline builds and autotuning count here); finish: for every problem, correctness on
             the five input distributions (a ModelNew that modifies its inputs fails), timing and peak memory of
             ModelNew, a correctness check of the timed instance on fresh inputs and again after those buffers are
             overwritten in place, and a check that torch's entry points are unchanged.
Problem files, shape overrides and the order of problems come from `$ABENCH_DATA_DIR/problems.json`.
```

## Files

- `abench_metrics.py`
- `abench_stage.py`
- `interface.md`
- `kb_bench.py`
- `metrics.py`
- `sandbox_guide.md`
- `selfcheck.py`
- `stages.py`
- `submission_rules.json`

The held-out data, keys and judge configuration are never in this directory.
