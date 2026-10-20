# ABench Case V-DST-004: Few-step distillation scheme

- Track: V · Vision · method
- Anchor: Consistency Models (arXiv: 2303.01469)
- Subset: main; difficulty: medium; resource tier: high; status: **deferred**
- Task: Few-step distillation scheme
- Metrics: 1–4 step FID; diversity
- Proxy: INFER, SUBSET; plan: Fixed pretrained HF teacher (CIFAR / ImageNet-64), train the student only; 1–4 step FID

## Current deliverables

This directory only sets up the bench contract scaffold: task card, evaluation protocol, resource registry, scoring contract and directory-level notes. There is no simulated training, no simulated reference run, no simulated score and no fabricated leaderboard result. Formal runs require the benchmark operator to inject real resources, reference runs, the hidden set and server-generated run records.

## Blockers and reservations

This case is held back from the formal leaderboard for now; only the protocol and interfaces are kept.

Blocking inputs:

- `frozen_weights_or_dataset_snapshot`
- `fid_or_scene_metric_calibration`
- `reference_runs_and_fixed_hardware`
- `fixed_checkpoint_and_runtime_stack`

Until these inputs are frozen, only interfaces/contracts are added; no excessive environment setup or simulated substitutes.

## Files

- `case/anchor_card.yaml`: task anchor and acceptance boundary.
- `case/task_prompt.md`: the task given to the agent under test.
- `case/eval_protocol.yaml`: proxy ladder, budget, hidden evaluation and output constraints.
- `case/resources.yaml`: registry of data, models, software, hardware and private resources.
- `case/scoring_config.yaml`: metrics and thresholds pending calibration.
- `harness/README.md`: execution interface conventions.
- `scoring/README.md`: scoring input/output conventions.
- `example_submission/README.md`: example submission directory conventions.
