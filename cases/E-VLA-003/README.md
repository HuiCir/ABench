# ABench Case E-VLA-003: Lightweight fine-tuning or action-head replacement for an existing VLA

- Track: E · Embodied AI · method
- Anchor: OpenVLA (arXiv: 2406.09246)
- Subset: main; difficulty: medium; resource tier: high; status: **deferred**
- Task: Lightweight fine-tuning or action-head replacement for an existing VLA
- Metrics: SimplerEnv success rate; fine-tuning GPU-h
- Proxy: FROZEN; plan: Switch to a small VLA of ≤ 0.5B (such as SmolVLA), freeze the VLM and only replace or fine-tune the action head; 7B OpenVLA fine-tuning is not in v1

## Current deliverables

This directory only sets up the bench contract scaffold: task card, evaluation protocol, resource registry, scoring contract and directory-level notes. There is no simulated training, no simulated reference run, no simulated score and no fabricated leaderboard result. Formal runs require the benchmark operator to inject real resources, reference runs, the hidden set and server-generated run records.

## Blockers and reservations

This case is held back from the formal leaderboard for now; only the protocol and interfaces are kept.

Blocking inputs:

- `LIBERO_or_SimplerEnv_version_and_assets`
- `simulator_determinism_and_30Hz_measurement`
- `perturbation_holdout_tasks`

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
