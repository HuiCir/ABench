# ABench Case V-T7-007: Judge whether the FID gain of a given generative method comes from memorization or metric bias

- Track: V · Vision · method
- Anchor: Analysis of the limits of generative evaluation metrics (specific paper to be chosen) (to be verified)
- Subset: calibration; difficulty: medium; resource tier: low; status: **scaffold**
- Task: Judge whether the FID gain of a given generative method comes from memorization or metric bias
- Metrics: Judgment accuracy
- Proxy: JUDGE; plan: Judgment question + one minimal verification experiment at μ1 or equivalent scale; no new large-scale training

## Current deliverables

This directory only sets up the bench contract scaffold: task card, evaluation protocol, resource registry, scoring contract and directory-level notes. There is no simulated training, no simulated reference run, no simulated score and no fabricated leaderboard result. Formal runs require the benchmark operator to inject real resources, reference runs, the hidden set and server-generated run records.

## Blockers and reservations

This case can enter the next implementation batch of the shared runner, but real scores still cannot be published on this basis.

Blocking inputs:

- `frozen_weights_or_dataset_snapshot`
- `fid_or_scene_metric_calibration`
- `reference_runs_and_fixed_hardware`

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
