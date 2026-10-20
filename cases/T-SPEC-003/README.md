# ABench Case T-SPEC-003: Apply the draft-and-verify idea to few-step diffusion sampling

- Track: T · Cross-domain transfer · method
- Anchor: Speculative decoding → diffusion sampling (arXiv: 2211.17192)
- Subset: primitive; difficulty: high; resource tier: medium; status: **deferred**
- Task: Apply the draft-and-verify idea to few-step diffusion sampling
- Metrics: FID at NFE + source claims
- Proxy: INFER; plan: Fixed pretrained diffusion weights (same as V-SMP-002)

## Current deliverables

This directory only sets up the bench contract scaffold: task card, evaluation protocol, resource registry, scoring contract and directory-level notes. There is no simulated training, no simulated reference run, no simulated score and no fabricated leaderboard result. Formal runs require the benchmark operator to inject real resources, reference runs, the hidden set and server-generated run records.

## Blockers and reservations

This case is held back from the formal leaderboard for now; only the protocol and interfaces are kept.

Blocking inputs:

- `source_case_contracts_and_cross_domain_holdouts`
- `domain_metric_calibration`
- `reference_runs`
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
