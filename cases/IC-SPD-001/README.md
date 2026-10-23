# ABench case IC-SPD-001: CIFAR-10 speedrun to 94%

Design an image classification network and its training recipe that reaches 94% CIFAR-10 test accuracy from random
initialization in as little training time as possible on one 24GB reference GPU. Anchor: Keller Jordan, *94% on
CIFAR-10 in 3.29 Seconds on a Single GPU* (arXiv:2404.00498).

## Status

- Judge: `timed_train_classification` (`harness/speedrun_runner.py`): the evaluator trains the submission from scratch
  once per hidden seed, timed by the parent process, and measures accuracy on the CIFAR-10 test set and CIFAR-10.1.
  Activation **awaiting_data**: the operator freezes the data with `tools/prepare_ic_spd.py`.
- Reference submission (`example_submission/`): airbench-style `method.py` with its design report.

## Files

- `case/agent/task.md` — the task statement the agent sees (handcrafted); `case/oj.yaml` — judge configuration.
- `harness/` — agent-visible interface: `interface.md`, `gpu_sandbox.md`, `speedrun_harness.py` (the timed child),
  `speedrun_runner.py` (the timing parent, shared by the evaluator and the self-check) and `selfcheck.py`.
- `example_submission/` — reference submission (`method.py`, `report/index.html`, `references.yaml`).
